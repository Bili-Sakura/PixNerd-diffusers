#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from diffusers.models import PixNerdLabelConditioner, PixNerdPixelVAE, PixNerdTransformer2DModel
from diffusers.schedulers import PixNerdFlowMatchScheduler


def _extract_state_dict(checkpoint: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"]
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    return checkpoint


def _infer_architecture(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, int]:
    hidden_size = state_dict[f"{prefix}t_embedder.mlp.0.weight"].shape[0]
    in_channels = state_dict[f"{prefix}final_layer.linear.weight"].shape[0]
    hidden_size_x = state_dict[f"{prefix}x_embedder.embedder.0.weight"].shape[0]
    num_classes = state_dict[f"{prefix}y_embedder.embedding_table.weight"].shape[0] - 1

    patch_sq = state_dict[f"{prefix}s_embedder.proj.weight"].shape[1] // in_channels
    patch_size = int(round(patch_sq**0.5))

    head_dim = state_dict[f"{prefix}blocks.0.attn.k_norm.weight"].shape[0]
    num_groups = hidden_size // head_dim

    block_indices = set()
    for key in state_dict:
        if not key.startswith(f"{prefix}blocks."):
            continue
        stripped = key[len(prefix) :]
        parts = stripped.split(".")
        if len(parts) >= 2 and parts[0] == "blocks":
            block_indices.add(int(parts[1]))
    num_blocks = max(block_indices) + 1
    num_cond_blocks = sum(
        1 for idx in block_indices if f"{prefix}blocks.{idx}.attn.qkv.weight" in state_dict
    )

    nerf_weight = state_dict[f"{prefix}blocks.{num_cond_blocks}.param_generator1.0.weight"]
    nerf_mlpratio = nerf_weight.shape[0] // (2 * hidden_size_x * hidden_size_x)

    return {
        "in_channels": in_channels,
        "patch_size": patch_size,
        "num_groups": num_groups,
        "hidden_size": hidden_size,
        "hidden_size_x": hidden_size_x,
        "num_blocks": num_blocks,
        "num_cond_blocks": num_cond_blocks,
        "nerf_mlpratio": nerf_mlpratio,
        "num_classes": num_classes,
    }


def _build_state_dict_for_transformer(
    raw_state_dict: Dict[str, torch.Tensor],
    source_prefix: str,
) -> Dict[str, torch.Tensor]:
    converted: Dict[str, torch.Tensor] = {}
    for key, value in raw_state_dict.items():
        if not key.startswith(source_prefix):
            continue
        stripped = key[len(source_prefix) :]
        converted[f"denoiser.{stripped}"] = value
        converted[f"ema_denoiser.{stripped}"] = value
    return converted


def _write_model_index(output_dir: Path) -> None:
    model_index = {
        "_class_name": "PixNerdPipeline",
        "_diffusers_version": "0.30.0",
        "conditioner": ["diffusers", "PixNerdLabelConditioner"],
        "scheduler": ["diffusers", "PixNerdFlowMatchScheduler"],
        "transformer": ["diffusers", "PixNerdTransformer2DModel"],
        "vae": ["diffusers", "PixNerdPixelVAE"],
    }
    with (output_dir / "model_index.json").open("w", encoding="utf-8") as handle:
        json.dump(model_index, handle, indent=2, sort_keys=True)
        handle.write("\n")


def convert_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    num_inference_steps: int,
    guidance_scale: float,
    timeshift: float,
    order: int,
    use_ema: bool,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    raw_state_dict = _extract_state_dict(checkpoint)

    has_ema = any(key.startswith("ema_denoiser.") for key in raw_state_dict)
    source_prefix = "ema_denoiser." if use_ema and has_ema else "denoiser."
    if not any(key.startswith(source_prefix) for key in raw_state_dict):
        raise ValueError(f"Could not find weights with prefix '{source_prefix}' in {checkpoint_path}.")

    arch = _infer_architecture(raw_state_dict, source_prefix)
    transformer = PixNerdTransformer2DModel(
        in_channels=arch["in_channels"],
        patch_size=arch["patch_size"],
        num_groups=arch["num_groups"],
        hidden_size=arch["hidden_size"],
        hidden_size_x=arch["hidden_size_x"],
        num_blocks=arch["num_blocks"],
        num_cond_blocks=arch["num_cond_blocks"],
        nerf_mlpratio=arch["nerf_mlpratio"],
        num_classes=arch["num_classes"],
        use_ema=True,
    )

    converted_state_dict = _build_state_dict_for_transformer(raw_state_dict, source_prefix)
    missing_keys, unexpected_keys = transformer.load_state_dict(converted_state_dict, strict=False)
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys during conversion: {unexpected_keys[:10]}")
    if missing_keys:
        raise RuntimeError(f"Missing keys during conversion: {missing_keys[:10]}")

    conditioner = PixNerdLabelConditioner(num_classes=arch["num_classes"])
    vae = PixNerdPixelVAE(scale=1.0)
    scheduler = PixNerdFlowMatchScheduler(
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        timeshift=timeshift,
        order=order,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    transformer.save_pretrained(output_dir / "transformer")
    conditioner.save_pretrained(output_dir / "conditioner")
    vae.save_pretrained(output_dir / "vae")
    scheduler.save_pretrained(output_dir / "scheduler")
    _write_model_index(output_dir)

    metadata = {
        "checkpoint": str(checkpoint_path),
        "source_prefix": source_prefix,
        "architecture": arch,
    }
    with (output_dir / "conversion_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PixNerd raw ckpt to a Diffusers pipeline directory.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to raw .ckpt file.")
    parser.add_argument("--output", type=Path, required=True, help="Output Diffusers checkpoint directory.")
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--timeshift", type=float, default=3.0)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        timeshift=args.timeshift,
        order=args.order,
        use_ema=args.use_ema,
    )
    print(f"Saved Diffusers-style checkpoint to: {args.output}")


if __name__ == "__main__":
    main()
