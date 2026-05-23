#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import torch

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from diffusers import PixNerdPipeline


def parse_conditioning_inputs(prompt: str | None, class_label: list[int] | None):
    if prompt:
        return [entry.strip() for entry in prompt.split("|||") if entry.strip()]
    if class_label:
        return class_label
    raise ValueError("Either --prompt or --class-label must be provided.")


def parse_args():
    parser = argparse.ArgumentParser(description="Sample images with a converted Diffusers-style PixNerd pipeline.")
    parser.add_argument("--model", required=True, help="Path or Hub id of a converted PixNerd pipeline.")
    parser.add_argument("--prompt", default=None, help="Text prompts split by |||.")
    parser.add_argument("--class-label", type=int, action="append", default=None, help="Class id. Repeat for batches.")
    parser.add_argument("--num-images-per-prompt", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--timeshift", type=float, default=3.0)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="samples")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.torch_dtype]
    generator_device = args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    pipe = PixNerdPipeline.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    pipe.scheduler.timeshift = args.timeshift
    pipe.scheduler.order = args.order
    output = pipe(
        class_labels=parse_conditioning_inputs(args.prompt, args.class_label),
        num_images_per_prompt=args.num_images_per_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(output.images):
        image.save(output_dir / f"{index:06d}.png")


if __name__ == "__main__":
    main()
