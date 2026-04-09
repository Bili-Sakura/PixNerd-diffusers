from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.pixnerd_diffusers.config_utils import instantiate_from_spec, load_symbol, to_container
from src.pixnerd_diffusers.model_wrapper import PixNerdModelWrapper


def collate_batch(batch):
    x, y, metadata = list(zip(*batch))
    stacked_metadata: Dict[str, Any] = {}
    if metadata and isinstance(metadata[0], dict):
        for key in metadata[0].keys():
            values = [m[key] for m in metadata if key in m]
            if not values:
                continue
            if isinstance(values[0], torch.Tensor):
                try:
                    stacked_metadata[key] = torch.stack(values, dim=0)
                    continue
                except Exception:
                    pass
            stacked_metadata[key] = values
    return torch.stack(x, dim=0), list(y), stacked_metadata


def move_tensors_to_device(data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device=device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def parse_precision(precision: Optional[str]) -> str:
    if precision is None:
        return "no"
    precision = str(precision).lower()
    if "bf16" in precision:
        return "bf16"
    if "16" in precision or "fp16" in precision:
        return "fp16"
    return "no"


def build_optimizer(model: PixNerdModelWrapper, optimizer_spec: Dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_spec = to_container(optimizer_spec)
    optimizer_class = load_symbol(optimizer_spec["class_path"])
    optimizer_kwargs = instantiate_from_spec(optimizer_spec.get("init_args", {}))

    param_groups = []
    denoiser_params = [p for p in model.denoiser.parameters() if p.requires_grad]
    if denoiser_params:
        param_groups.append({"params": denoiser_params})

    if model.diffusion_trainer is not None:
        trainer_params = [p for p in model.diffusion_trainer.parameters() if p.requires_grad]
        if trainer_params:
            param_groups.append({"params": trainer_params})

    if model.diffusion_sampler is not None:
        sampler_params = [p for p in model.diffusion_sampler.parameters() if p.requires_grad]
        if sampler_params:
            param_groups.append({"params": sampler_params, "lr": 1e-3})

    if not param_groups:
        raise RuntimeError("No trainable parameters were found in the model wrapper.")

    return optimizer_class(param_groups, **optimizer_kwargs)


def build_lr_scheduler(
    lr_scheduler_spec: Optional[Dict[str, Any]],
    optimizer: torch.optim.Optimizer,
):
    if lr_scheduler_spec is None:
        return None
    lr_scheduler_spec = to_container(lr_scheduler_spec)
    scheduler_class = load_symbol(lr_scheduler_spec["class_path"])
    scheduler_kwargs = instantiate_from_spec(lr_scheduler_spec.get("init_args", {}))
    return scheduler_class(optimizer, **scheduler_kwargs)


def save_checkpoint(
    accelerator: Accelerator,
    model: PixNerdModelWrapper,
    sampler: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    output_dir: str,
    step: int,
) -> None:
    if not accelerator.is_main_process:
        return
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_sampler = accelerator.unwrap_model(sampler)
    unwrapped_model.save_pretrained(checkpoint_dir, safe_serialization=False)
    torch.save(
        {
            "step": step,
            "sampler_state_dict": unwrapped_sampler.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        },
        os.path.join(checkpoint_dir, "training_state.pt"),
    )


def train(args: argparse.Namespace) -> None:
    config = OmegaConf.load(args.config)
    model_config = to_container(config.model)
    data_config = to_container(config.data)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    mixed_precision = args.mixed_precision
    if mixed_precision is None:
        mixed_precision = parse_precision(to_container(config.get("trainer", {})).get("precision"))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    if args.resume_from is not None:
        model = PixNerdModelWrapper.from_pretrained(
            args.resume_from,
            low_cpu_mem_usage=False,
        )
    else:
        model = PixNerdModelWrapper.from_model_config(model_config, use_ema=True)
        if args.legacy_checkpoint is not None:
            model.load_legacy_checkpoint(args.legacy_checkpoint)

    sampler = model.diffusion_sampler
    if sampler is None:
        raise RuntimeError("model.diffusion_sampler is required in config.model.")

    train_dataset = instantiate_from_spec(data_config["train_dataset"])
    train_batch_size = args.train_batch_size or data_config.get("train_batch_size", 32)
    train_num_workers = data_config.get("train_num_workers", 4)
    dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_batch,
    )

    optimizer = build_optimizer(model, model_config["optimizer"])
    lr_scheduler = build_lr_scheduler(model_config.get("lr_scheduler"), optimizer)

    model, sampler, optimizer, dataloader = accelerator.prepare(model, sampler, optimizer, dataloader)
    sampler.train()
    model.train()

    trainer_cfg = to_container(config.get("trainer", {}))
    max_steps = args.max_steps if args.max_steps is not None else trainer_cfg.get("max_steps")
    if max_steps is None:
        raise ValueError("max_steps must be provided via --max_steps or config.trainer.max_steps.")

    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    while global_step < max_steps:
        for pixel_values, labels, metadata in dataloader:
            with accelerator.accumulate(model):
                pixel_values = pixel_values.to(device=accelerator.device, non_blocking=True)
                metadata = move_tensors_to_device(metadata, accelerator.device)
                loss_dict = model.compute_training_loss(pixel_values, labels, metadata=metadata, sampler=sampler)
                loss = loss_dict["loss"]

                accelerator.backward(loss)

                if accelerator.sync_gradients and args.max_grad_norm is not None:
                    params = [p for p in model.parameters() if p.requires_grad]
                    if params:
                        accelerator.clip_grad_norm_(params, args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if lr_scheduler is not None:
                    lr_scheduler.step()

                if accelerator.sync_gradients:
                    accelerator.unwrap_model(model).ema_step()
                    global_step += 1

                    if global_step % args.log_every_n_steps == 0:
                        lr = optimizer.param_groups[0]["lr"]
                        accelerator.print(f"step={global_step} loss={loss.item():.6f} lr={lr:.2e}")

                    if global_step % args.save_every_n_steps == 0:
                        save_checkpoint(
                            accelerator=accelerator,
                            model=model,
                            sampler=sampler,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            output_dir=args.output_dir,
                            step=global_step,
                        )

                    if global_step >= max_steps:
                        break
        if global_step >= max_steps:
            break

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        sampler=sampler,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        output_dir=args.output_dir,
        step=global_step,
    )


def build_arg_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PixNerd in diffusers style.", add_help=add_help)
    parser.add_argument("--config", type=str, required=True, help="Path to OmegaConf yaml config.")
    parser.add_argument("--output_dir", type=str, required=True, help="Checkpoint output directory.")
    parser.add_argument("--legacy_checkpoint", type=str, default=None, help="Legacy .ckpt to initialize from.")
    parser.add_argument("--resume_from", type=str, default=None, help="Directory saved with save_pretrained.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--save_every_n_steps", type=int, default=10000)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
