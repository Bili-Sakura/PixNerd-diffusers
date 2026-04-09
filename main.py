import argparse
import os
from typing import List, Union

import torch
from omegaconf import OmegaConf

from src.pixnerd_diffusers.pipeline import PixNerdPipeline
from src.pixnerd_diffusers.scheduler import PixNerdFlowMatchScheduler
from src.pixnerd_diffusers.transformer import PixNerdTransformer2DModel
from src.pixnerd_diffusers.training import build_arg_parser as build_train_parser
from src.pixnerd_diffusers.training import train
from src.pixnerd_diffusers.config_utils import to_container


def parse_conditioning_inputs(prompt: str, class_label: str) -> Union[List[str], List[int]]:
    if prompt:
        return [entry.strip() for entry in prompt.split("|||") if entry.strip()]
    if class_label:
        return [int(entry.strip()) for entry in class_label.split(",") if entry.strip()]
    raise ValueError("Either --prompt or --class_label must be provided.")


def run_sample(args: argparse.Namespace) -> None:
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else torch.float32
    config = OmegaConf.load(args.config)
    model_cfg = to_container(config.model)

    if args.checkpoint_path and os.path.isdir(args.checkpoint_path) and os.path.exists(
        os.path.join(args.checkpoint_path, "transformer", "config.json")
    ):
        transformer = PixNerdTransformer2DModel.from_pretrained(
            os.path.join(args.checkpoint_path, "transformer"),
            low_cpu_mem_usage=False,
        )
        scheduler = PixNerdFlowMatchScheduler.from_pretrained(os.path.join(args.checkpoint_path, "scheduler"))
    else:
        transformer = PixNerdTransformer2DModel.from_project_config(model_cfg, use_ema=not args.disable_ema)
        if args.checkpoint_path:
            transformer.load_legacy_checkpoint(args.checkpoint_path)
        scheduler_cfg = model_cfg.get("diffusion_sampler", {}).get("init_args", {})
        scheduler = PixNerdFlowMatchScheduler(
            num_inference_steps=scheduler_cfg.get("num_steps", 25),
            guidance_scale=scheduler_cfg.get("guidance", 4.0),
            timeshift=scheduler_cfg.get("timeshift", 3.0),
            order=scheduler_cfg.get("order", 2),
            guidance_interval_min=scheduler_cfg.get("guidance_interval_min", 0.0),
            guidance_interval_max=scheduler_cfg.get("guidance_interval_max", 1.0),
            last_step=scheduler_cfg.get("last_step", None),
        )

    transformer = transformer.to(dtype=dtype)
    pipeline = PixNerdPipeline(
        vae=transformer.vae,
        conditioner=transformer.conditioner,
        transformer=transformer.get_inference_denoiser(use_ema=not args.disable_ema),
        scheduler=scheduler,
    ).to(args.device)

    conditioning = parse_conditioning_inputs(args.prompt, args.class_label)
    output = pipeline(
        prompt=conditioning,
        num_images_per_prompt=args.num_images_per_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_steps or scheduler.num_inference_steps,
        guidance_scale=args.guidance_scale or scheduler.guidance_scale,
        generator=torch.Generator(device=args.device).manual_seed(args.seed) if args.seed is not None else None,
        timeshift=args.timeshift,
        order=args.order,
        output_type="pil",
    ).images

    os.makedirs(args.output_dir, exist_ok=True)
    for index, image in enumerate(output):
        image.save(os.path.join(args.output_dir, f"sample_{index:04d}.png"))
    print(f"Saved {len(output)} images to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="PixNerd Diffusers-style entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parent = build_train_parser(add_help=False)
    subparsers.add_parser("train", parents=[train_parent], help="Accelerate-based training")

    sample_parser = subparsers.add_parser("sample", help="Run inference through DiffusionPipeline")
    sample_parser.add_argument("--config", type=str, required=True)
    sample_parser.add_argument("--checkpoint_path", type=str, default=None)
    sample_parser.add_argument("--prompt", type=str, default=None, help="Use ||| to separate prompts.")
    sample_parser.add_argument("--class_label", type=str, default=None, help="Comma separated class labels.")
    sample_parser.add_argument("--num_images_per_prompt", type=int, default=1)
    sample_parser.add_argument("--seed", type=int, default=0)
    sample_parser.add_argument("--height", type=int, default=512)
    sample_parser.add_argument("--width", type=int, default=512)
    sample_parser.add_argument("--num_steps", type=int, default=None)
    sample_parser.add_argument("--guidance_scale", type=float, default=None)
    sample_parser.add_argument("--timeshift", type=float, default=None)
    sample_parser.add_argument("--order", type=int, default=None)
    sample_parser.add_argument("--disable_ema", action="store_true")
    sample_parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    sample_parser.add_argument("--device", type=str, default="cuda")
    sample_parser.add_argument("--output_dir", type=str, default="samples")

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "sample":
        run_sample(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()