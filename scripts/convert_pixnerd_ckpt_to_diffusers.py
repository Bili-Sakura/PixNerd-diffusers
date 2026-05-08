#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
import sys
import shutil

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pixnerd_diffusers import PixNerdFlowMatchScheduler, PixNerdTransformer2DModel


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
    patch_size = int(round(patch_sq ** 0.5))

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


def _build_transformer_specs(arch: Dict[str, int]) -> Tuple[Dict, Dict, Dict]:
    denoiser_spec = {
        "class_path": "src.models.transformer.pixnerd_c2i.PixNerDiT",
        "init_args": {
            "in_channels": arch["in_channels"],
            "patch_size": arch["patch_size"],
            "num_groups": arch["num_groups"],
            "hidden_size": arch["hidden_size"],
            "hidden_size_x": arch["hidden_size_x"],
            "num_blocks": arch["num_blocks"],
            "num_cond_blocks": arch["num_cond_blocks"],
            "nerf_mlpratio": arch["nerf_mlpratio"],
            "num_classes": arch["num_classes"],
        },
    }
    conditioner_spec = {
        "class_path": "src.models.conditioner.class_label.LabelConditioner",
        "init_args": {"num_classes": arch["num_classes"]},
    }
    vae_spec = {
        "class_path": "src.models.autoencoder.pixel.PixelAE",
        "init_args": {"scale": 1.0},
    }
    return denoiser_spec, conditioner_spec, vae_spec


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
        "scheduler": ["diffusers_modules.local.scheduling_pixnerd_flow_match", "PixNerdFlowMatchScheduler"],
        "transformer": ["diffusers_modules.local.modeling_pixnerd_transformer_2d", "PixNerdTransformer2DModel"],
    }
    with (output_dir / "model_index.json").open("w", encoding="utf-8") as handle:
        json.dump(model_index, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_source(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _rewrite_transformer_config_for_self_contained(transformer_dir: Path) -> None:
    config_path = transformer_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["denoiser_spec"]["class_path"] = "diffusers_modules.local.modeling_pixnerd_transformer_2d.PixNerDiT"
    config["conditioner_spec"]["class_path"] = "diffusers_modules.local.modeling_pixnerd_transformer_2d.LabelConditioner"
    config["vae_spec"]["class_path"] = "diffusers_modules.local.modeling_pixnerd_transformer_2d.PixelAE"
    _write_text(config_path, json.dumps(config, indent=2, sort_keys=True) + "\n")


def _strip_imports_and_future(source: str) -> str:
    lines = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("from __future__ import"):
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


def _cleanup_runtime_files(output_dir: Path) -> None:
    for path in output_dir.glob("*.py"):
        path.unlink(missing_ok=True)
    shutil.rmtree(output_dir / "src", ignore_errors=True)
    shutil.rmtree(output_dir / "__pycache__", ignore_errors=True)


def _build_modeling_source() -> str:
    header = """from __future__ import annotations

import copy
import importlib
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from torch.nn.functional import scaled_dot_product_attention

"""
    sections = [
        _strip_imports_and_future(_read_source("src/models/autoencoder/base.py")),
        _strip_imports_and_future(
            _read_source("src/models/autoencoder/pixel.py").replace(
                "from src.models.autoencoder.base import BaseAE", ""
            )
        ),
        _strip_imports_and_future(_read_source("src/models/conditioner/base.py")),
        _strip_imports_and_future(
            _read_source("src/models/conditioner/class_label.py").replace(
                "from src.models.conditioner.base import BaseConditioner, resolve_conditioner_device", ""
            )
        ),
        _strip_imports_and_future(_read_source("src/models/transformer/pixnerd_c2i.py")),
        _strip_imports_and_future(_read_source("src/pixnerd_diffusers/config_utils.py")),
        _strip_imports_and_future(
            _read_source("src/pixnerd_diffusers/models/modeling_pixnerd_transformer_2d.py").replace(
                "from src.pixnerd_diffusers.config_utils import instantiate_from_spec, to_container", ""
            )
        ),
    ]
    footer = """
__all__ = [
    "PixNerDiT",
    "LabelConditioner",
    "PixelAE",
    "PixNerdTransformer2DModel",
    "PixNerdTransformer2DModelOutput",
]
"""
    return header + "\n\n".join(sections) + footer


def _build_scheduler_source() -> str:
    header = """from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput

"""
    body = _strip_imports_and_future(_read_source("src/pixnerd_diffusers/schedulers/scheduling_pixnerd_flow_match.py"))
    footer = """
__all__ = [
    "PixNerdFlowMatchScheduler",
    "PixNerdSchedulerOutput",
]
"""
    return header + body + footer


def _build_pipeline_source() -> str:
    header = """from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import BaseOutput
from PIL import Image

from .modeling_pixnerd_transformer_2d import PixNerdTransformer2DModel
from .scheduling_pixnerd_flow_match import PixNerdFlowMatchScheduler

"""
    body = _strip_imports_and_future(
        _read_source("src/pixnerd_diffusers/pipelines/pipeline_pixnerd.py").replace(
            "from src.pixnerd_diffusers.schedulers.scheduling_pixnerd_flow_match import PixNerdFlowMatchScheduler",
            "",
        )
    )
    footer = """
__all__ = [
    "PixNerdPipeline",
    "PixNerdPipelineOutput",
]
"""
    return header + body + footer


def _export_self_contained_runtime(output_dir: Path) -> None:
    _cleanup_runtime_files(output_dir)
    _write_text(output_dir / "modeling_pixnerd_transformer_2d.py", _build_modeling_source())
    _write_text(output_dir / "scheduling_pixnerd_flow_match.py", _build_scheduler_source())
    _write_text(output_dir / "pipeline.py", _build_pipeline_source())


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
    denoiser_spec, conditioner_spec, vae_spec = _build_transformer_specs(arch)
    transformer = PixNerdTransformer2DModel(
        denoiser_spec=denoiser_spec,
        conditioner_spec=conditioner_spec,
        vae_spec=vae_spec,
        diffusion_trainer_spec=None,
        use_ema=True,
    )

    converted_state_dict = _build_state_dict_for_transformer(raw_state_dict, source_prefix)
    missing_keys, unexpected_keys = transformer.load_state_dict(converted_state_dict, strict=False)
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys during conversion: {unexpected_keys[:10]}")
    # Conditioner and PixelAE are parameter-free, so missing keys there are expected.
    if any(not key.startswith(("conditioner.", "vae.")) for key in missing_keys):
        raise RuntimeError(f"Missing non-auxiliary keys during conversion: {missing_keys[:10]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "transformer").mkdir(parents=True, exist_ok=True)
    (output_dir / "scheduler").mkdir(parents=True, exist_ok=True)

    transformer.save_pretrained(output_dir / "transformer")
    scheduler = PixNerdFlowMatchScheduler(
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        timeshift=timeshift,
        order=order,
    )
    scheduler.save_pretrained(output_dir / "scheduler")
    _rewrite_transformer_config_for_self_contained(output_dir / "transformer")
    _export_self_contained_runtime(output_dir)
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
    parser = argparse.ArgumentParser(description="Convert PixNerd raw ckpt to Diffusers-style checkpoint folder.")
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
