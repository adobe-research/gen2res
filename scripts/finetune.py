"""
Train a diffusion model on images.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data_local, load_data_reg
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    if args.reg_ratio == 0.0:
        data = load_data_local(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            resolution=args.image_size,
            img_name='' if args.personal else args.img_path.split('/')[-1],
        )
    else:
        data = load_data_reg(
            data_dir=args.reg_data_dir,
            person_data_dir=args.data_dir,
            batch_size=args.batch_size,
            resolution=args.image_size,
            img_name='' if args.personal else args.img_path.split('/')[-1],
            reg_ratio=args.reg_ratio,
        )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        finetune=True,
        img_path=args.img_path,
        ftsteps=args.ftsteps,
        finetune_sample=args.finetune_sample,
        outputdir=args.output_dir,
        noise_steps=args.noise_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=50000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        log_dir="",
        reg_ratio=0.0,
        reg_data_dir="",
        ftsteps=-1,
        finetune_sample=False,
        output_dir="",
        img_path="",
        noise_steps=100,
        personal=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
