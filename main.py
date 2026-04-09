import argparse
import os
from typing import List, Union

import torch

from src.pixnerd_diffusers.pipeline import PixNerdPipeline
from src.pixnerd_diffusers.training import build_arg_parser as build_train_parser
from src.pixnerd_diffusers.training import train


def parse_conditioning_inputs(prompt: str, class_label: str) -> Union[List[str], List[int]]:
    if prompt:
        return [entry.strip() for entry in prompt.split("|||") if entry.strip()]
    if class_label:
        return [int(entry.strip()) for entry in class_label.split(",") if entry.strip()]
    raise ValueError("Either --prompt or --class_label must be provided.")


def run_sample(args: argparse.Namespace) -> None:
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else torch.float32
    pipeline = PixNerdPipeline.from_config(
        config_path=args.config,
        checkpoint_path=args.checkpoint_path,
        use_ema=not args.disable_ema,
        torch_dtype=dtype,
        device=args.device,
    )

    conditioning = parse_conditioning_inputs(args.prompt, args.class_label)
    output = pipeline(
        y=conditioning,
        num_images_per_prompt=args.num_images_per_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
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