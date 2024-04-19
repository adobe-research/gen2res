import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch as th
import os

import random


def create_dataloader(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=True,
    random_crop=False,
    random_flip=False,
):

    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)

    # all_files = all_files[160:161] + all_files[197:198]

    # shuffle files
    random.seed(6)
    random.shuffle(all_files)

    # all_files = all_files[7::8]
    # all_files = all_files[1::2]

    print(len(all_files))

    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        return_filename=True,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False
    )
    return loader


def load_data_local(
    *,
    data_dir,
    batch_size,
    resolution,
):
    all_files = _list_image_files_recursively(data_dir)

    imgs = []
    for path in all_files:
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        arr = np.array(pil_image)
        # arr = center_crop_arr(pil_image, resolution)
        arr = arr.astype(np.float32) / 127.5 - 1
        arr = np.transpose(arr, [2, 0, 1])
        imgs.append(th.tensor(arr))

    data = th.stack(imgs, dim=0)
    length = data.shape[0]

    print('load_data_local: ', length)

    while True:
        if batch_size >= length:
            idxs = np.random.choice(length, batch_size, replace=True)
        else:
            idxs = np.random.choice(length, batch_size, replace=False)
        yield data[idxs], {}



def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
        )
    while True:
        yield from loader


def load_data_reg(
    *,
    data_dir,
    batch_size,
    person_data_dir,
    resolution,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    img_name="",
    reg_ratio=0.0,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)

    all_files_person = _list_image_files_recursively(person_data_dir)


    dataset = ImageDataset_reg(
        resolution,
        all_files,
        all_files_person,
        random_crop=random_crop,
        random_flip=random_flip,
        reg_ratio=reg_ratio,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset_reg(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        person_image_paths,
        classes=None,
        random_crop=False,
        random_flip=True,
        return_filename=False,
        reg_ratio=0.0,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths
        self.local_images_person = person_image_paths
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.return_filename = return_filename
        self.reg_ratio = reg_ratio

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        # print(random.random(), self.reg_ratio)
        if random.random() < self.reg_ratio:
            path = self.local_images[idx]
            # print(path)
        else:
            idx = random.randint(0, len(self.local_images_person) - 1)
            path = self.local_images_person[idx]
            # print(path)
        # exit()

        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if pil_image.size[0] != self.resolution:
            pil_image = pil_image.resize((self.resolution, self.resolution), resample=Image.BICUBIC)

        arr = np.array(pil_image)

        # if self.random_crop:
        #     arr = random_crop_arr(pil_image, self.resolution)
        # else:
        #     arr = center_crop_arr(pil_image, self.resolution)

        # if self.random_flip and random.random() < 0.5:
        #     arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.return_filename:
            out_dict["filename"] = path.split('/')[-1]
            out_dict["path"] = path
        return np.transpose(arr, [2, 0, 1]), out_dict




class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        return_filename=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.return_filename = return_filename

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        arr = np.array(pil_image)

        # if self.random_crop:
        #     arr = random_crop_arr(pil_image, self.resolution)
        # else:
        #     arr = center_crop_arr(pil_image, self.resolution)

        # if self.random_flip and random.random() < 0.5:
        #     arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        if self.return_filename:
            out_dict["filename"] = path.split('/')[-1]
            out_dict["path"] = path
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
