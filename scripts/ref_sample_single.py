"""
*******************************************************************************
Copyright 2024 Adobe
All Rights Reserved.
NOTICE: All information contained herein is, and remains
the property of Adobe and its suppliers, if any. The intellectual
and technical concepts contained herein are proprietary to Adobe
and its suppliers and are protected by all applicable intellectual
property laws, including trade secret and copyright laws.
Dissemination of this information or reproduction of this material
is strictly forbidden unless prior written permission is obtained
from Adobe.
*******************************************************************************

Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..")))

import math
import time

import cv2
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize
from torchvision.utils import make_grid, save_image

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import create_dataloader
from guided_diffusion.script_util import (NUM_CLASSES, add_dict_to_argparser,
                                          args_to_dict,
                                          create_model_and_diffusion,
                                          model_and_diffusion_defaults)


def img2tensor(imgs, bgr2rgb=True, float32=True):

  def _totensor(img, bgr2rgb, float32):
    if img.shape[2] == 3 and bgr2rgb:
      if img.dtype == 'float64':
        img = img.astype('float32')
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = th.from_numpy(img.transpose(2, 0, 1))
    if float32:
      img = img.float()
    return img

  if isinstance(imgs, list):
    return [_totensor(img, bgr2rgb, float32) for img in imgs]
  else:
    return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
  """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
  if not (th.is_tensor(tensor) or
          (isinstance(tensor, list) and all(th.is_tensor(t) for t in tensor))):
    raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

  if th.is_tensor(tensor):
    tensor = [tensor]
  result = []
  for _tensor in tensor:
    _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
    _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

    n_dim = _tensor.dim()
    if n_dim == 4:
      img_np = make_grid(_tensor,
                         nrow=int(math.sqrt(_tensor.size(0))),
                         normalize=False).numpy()
      img_np = img_np.transpose(1, 2, 0)
      if rgb2bgr:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elif n_dim == 3:
      img_np = _tensor.numpy()
      img_np = img_np.transpose(1, 2, 0)
      if img_np.shape[2] == 1:  # gray image
        img_np = np.squeeze(img_np, axis=2)
      else:
        if rgb2bgr:
          img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elif n_dim == 2:
      img_np = _tensor.numpy()
    else:
      raise TypeError('Only support 4D, 3D or 2D tensor. '
                      f'But received with dimension: {n_dim}')
    if out_type == np.uint8:
      # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
      img_np = (img_np * 255.0).round()
    img_np = img_np.astype(out_type)
    result.append(img_np)
  if len(result) == 1:
    result = result[0]
  return result


def set_realesrgan():
  from basicsr.archs.rrdbnet_arch import RRDBNet
  from basicsr.utils.realesrgan_utils import RealESRGANer

  use_half = False
  if th.cuda.is_available():  # set False in CPU/MPS mode
    no_half_gpu_list = ['1650',
                        '1660']  # set False for GPUs that don't support f16
    if not True in [
        gpu in th.cuda.get_device_name(0) for gpu in no_half_gpu_list
    ]:
      use_half = True

  model = RRDBNet(
      num_in_ch=3,
      num_out_ch=3,
      num_feat=64,
      num_block=23,
      num_grow_ch=32,
      scale=2,
  )
  upsampler = RealESRGANer(
      scale=2,
      model_path=
      "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
      model=model,
      tile=400,  #args.bg_tile,
      tile_pad=40,
      pre_pad=0,
      half=use_half)

  # if not gpu_is_available():  # CPU
  #     import warnings
  #     warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA.'
  #                     'The unoptimized RealESRGAN is slow on CPU. '
  #                     'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
  #                     category=RuntimeWarning)
  return upsampler


def main():
  args = create_argparser().parse_args()

  dist_util.setup_dist()
  # logger.configure()

  start_time = time.time()

  logger.log("creating data loader...")
  dataloader = create_dataloader(
      data_dir=args.data_dir,
      batch_size=args.batch_size,
      image_size=args.image_size,
      class_cond=args.class_cond,
  )

  end_time = time.time()
  print('create dataloader:' + str(end_time - start_time))

  logger.log("creating model and diffusion...")
  start_time = time.time()
  model, diffusion = create_model_and_diffusion(
      **args_to_dict(args,
                     model_and_diffusion_defaults().keys()))
  model.to(dist_util.dev())
  end_time = time.time()
  print('create model:' + str(end_time - start_time))

  start_time = time.time()
  model.load_state_dict(dist_util.load_state_dict(args.model_path))

  end_time = time.time()
  print('load model:' + str(end_time - start_time))

  if args.use_fp16:
    model.convert_to_fp16()
  model.eval()

  logger.log("sampling...")

  os.makedirs(args.outputdir, exist_ok=True)
  tot = 0
  TOT = len(dataloader)

  bg_upsampler = None
  face_upsampler = None

  if args.bg_upsample:
    bg_upsampler = set_realesrgan()

  start_time = time.time()

  face_helper = FaceRestoreHelper(args.upscale,
                                  face_size=args.image_size,
                                  crop_ratio=(1, 1),
                                  det_model='retinaface_resnet50',
                                  save_ext='png',
                                  use_parse=True,
                                  device='cuda')

  end_time = time.time()
  print('create face_helper:' + str(end_time - start_time))
  assert args.batch_size == 1, 'batch size must be 1 now'

  for data, dic in dataloader:
    face_helper.clean_all()

    data = data.to(dist_util.dev())
    # if data.shape[0] == args.batch_size:
    #     continue

    if not args.input_aligned:
      start_time = time.time()
      face_helper.read_image(dic['path'][0])
      num_det_faces = face_helper.get_face_landmarks_5(only_center_face=False,
                                                       resize=640,
                                                       eye_dist_threshold=5)
      print(f'\tdetect {num_det_faces} faces')
      # align and warp each face
      face_helper.align_warp_face()
      cropped_face = face_helper.cropped_faces[0]
      cropped_face_t = img2tensor(cropped_face / 255.,
                                  bgr2rgb=True,
                                  float32=True)
      normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
      data = cropped_face_t.unsqueeze(0).to('cuda')

      end_time = time.time()
      print('align face:' + str(end_time - start_time))

    finish = True
    for i in range(data.shape[0]):
      if not os.path.exists(os.path.join(args.outputdir, dic['filename'][i])):
        finish = False
        break

    if finish:
      print(os.path.join(args.outputdir, dic['filename'][0]))
      continue
    else:
      print('running ' + os.path.join(args.outputdir, dic['filename'][0]))

    model_kwargs = {}

    sample_fn = (diffusion.p_sample_loop
                 if not args.use_ddim else diffusion.ddim_sample_loop)
    sample = sample_fn(
        model,
        (data.shape[0], 3, args.image_size, args.image_size),
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        ref_img=data if data.shape[-1] == args.image_size else F.interpolate(
            data, (args.image_size, args.image_size)),
        last=args.last,
        skip=args.skip,
        scale=args.scale,
        noise_step=args.noise_step,
        first=args.first,
    )

    if not args.input_aligned:

      restored_face = tensor2img(sample[0], rgb2bgr=True, min_max=(-1, 1))
      restored_face = restored_face.astype('uint8')
      face_helper.add_restored_face(restored_face, cropped_face)

      # upsample the background
      start_time = time.time()
      if bg_upsampler is not None:
        # Now only support RealESRGAN for upsampling background
        bg_img = bg_upsampler.enhance(cv2.imread(dic['path'][0],
                                                 cv2.IMREAD_COLOR),
                                      outscale=args.upscale)[0]
      else:
        bg_img = None
      end_time = time.time()
      print('upsample bg:' + str(end_time - start_time))

      start_time = time.time()
      face_helper.get_inverse_affine(None)
      # paste each restored face to the input image
      if args.face_upsample and face_upsampler is not None:
        restored_img = face_helper.paste_faces_to_input_image(
            upsample_img=bg_img,
            draw_box=args.draw_box,
            face_upsampler=face_upsampler)
      else:
        restored_img = face_helper.paste_faces_to_input_image(
            upsample_img=bg_img, draw_box=args.draw_box)

      end_time = time.time()
      print('paste back:' + str(end_time - start_time))

      cropped_face_t = img2tensor(restored_img / 255.,
                                  bgr2rgb=True,
                                  float32=True)
      normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
      sample = cropped_face_t.unsqueeze(0)

    # for i in range(sample.shape[0]):
    save_image((sample[0:1] + 1) / 2., os.path.join(args.outputdir,
                                                    'output.png'))
    # tot = tot + 1

    # print(tot, args.num)
    if args.num != -1 and tot >= args.num:
      break

    # logger.log(f"created {tot}/{TOT} samples")


def create_argparser():
  defaults = dict(
      clip_denoised=True,
      batch_size=1,
      use_ddim=False,
      model_path="",
      outputdir="",
      data_dir="",
      last=200,
      skip=10,
      scale=0.3,
      noise_step=0,
      step=5000,
      ans=0,
      askip=0,
      first=0,
      reg_ratio=0.0,
      mode="",
      ftsteps=-1,
      num=-1,
      input_aligned=True,
      draw_box=False,
      face_upsample=False,
      bg_upsample=False,
      upscale=2,
  )
  defaults.update(model_and_diffusion_defaults())
  parser = argparse.ArgumentParser()
  add_dict_to_argparser(parser, defaults)
  return parser


if __name__ == "__main__":
  main()
