from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator, __version__ as accelerate_version
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import diffusers
from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.pixnerd_diffusers.config_utils import instantiate_from_spec, load_symbol, to_container
from src.pixnerd_diffusers.models.modeling_pixnerd_transformer_2d import PixNerdTransformer2DModel
from src.pixnerd_diffusers.schedulers.scheduling_pixnerd_flow_match import PixNerdFlowMatchScheduler

logger = get_logger(__name__, log_level="INFO")


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


def build_optimizer(model: PixNerdTransformer2DModel, optimizer_spec: Dict[str, Any]) -> torch.optim.Optimizer:
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
    transformer: PixNerdTransformer2DModel,
    scheduler: PixNerdFlowMatchScheduler,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    output_dir: str,
    step: int,
) -> None:
    if not accelerator.is_main_process:
        return
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    unwrapped_transformer = accelerator.unwrap_model(transformer)
    unwrapped_scheduler = accelerator.unwrap_model(scheduler)
    unwrapped_transformer.save_pretrained(os.path.join(checkpoint_dir, "transformer"), safe_serialization=False)
    unwrapped_scheduler.save_pretrained(os.path.join(checkpoint_dir, "scheduler"))
    if unwrapped_transformer.vae is not None and hasattr(unwrapped_transformer.vae, "save_pretrained"):
        unwrapped_transformer.vae.save_pretrained(os.path.join(checkpoint_dir, "vae"))
    if hasattr(unwrapped_transformer.conditioner, "save_pretrained"):
        unwrapped_transformer.conditioner.save_pretrained(os.path.join(checkpoint_dir, "conditioner"))
    torch.save(
        {
            "step": step,
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        },
        os.path.join(checkpoint_dir, "training_state.pt"),
    )


def train(args: argparse.Namespace) -> None:
    config = OmegaConf.load(args.config)
    model_config = to_container(config.model)
    data_config = to_container(config.data)
    trainer_cfg = to_container(config.get("trainer", {}))

    if args.seed is not None:
        set_seed(args.seed)

    mixed_precision = args.mixed_precision
    if mixed_precision is None:
        mixed_precision = parse_precision(trainer_cfg.get("precision"))

    logging_dir = os.path.join(args.output_dir, "logs")
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        project_config=project_config,
        log_with=args.report_to if args.report_to != "none" else None,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    if args.resume_from is not None and os.path.exists(os.path.join(args.resume_from, "transformer")):
        transformer = PixNerdTransformer2DModel.from_pretrained(
            os.path.join(args.resume_from, "transformer"),
            low_cpu_mem_usage=False,
        )
        scheduler = PixNerdFlowMatchScheduler.from_pretrained(os.path.join(args.resume_from, "scheduler"))
    else:
        transformer = PixNerdTransformer2DModel.from_project_config(model_config, use_ema=True)
        scheduler_cfg = model_config.get("diffusion_sampler", {})
        scheduler = PixNerdFlowMatchScheduler.from_sampler_spec(scheduler_cfg)

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

    optimizer = build_optimizer(transformer, model_config["optimizer"])
    lr_scheduler = build_lr_scheduler(model_config.get("lr_scheduler"), optimizer)

    transformer, optimizer, dataloader = accelerator.prepare(transformer, optimizer, dataloader)
    transformer.train()

    scheduler_to_save = scheduler

    if version.parse(accelerate_version) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                unwrapped_transformer = accelerator.unwrap_model(transformer)
                unwrapped_transformer.save_pretrained(os.path.join(output_dir, "transformer"), safe_serialization=False)
                scheduler_to_save.save_pretrained(os.path.join(output_dir, "scheduler"))
                for _ in range(len(models)):
                    weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                model = models.pop()
                load_model = PixNerdTransformer2DModel.from_pretrained(os.path.join(input_dir, "transformer"))
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    else:
        logger.warning(
            "Accelerate < 0.16.0 detected; save/load hooks are disabled and only explicit checkpoint saving is used."
        )

    if accelerator.is_main_process and args.report_to != "none":
        tracker_config = dict(args.__dict__)
        tracker_config["config"] = args.config
        accelerator.init_trackers("pixnerd-diffusers", tracker_config)

    max_steps = args.max_steps if args.max_steps is not None else trainer_cfg.get("max_steps")
    if max_steps is None:
        raise ValueError("max_steps must be provided via --max_steps or config.trainer.max_steps.")

    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    while global_step < max_steps:
        for pixel_values, labels, metadata in dataloader:
            with accelerator.accumulate(transformer):
                pixel_values = pixel_values.to(device=accelerator.device, non_blocking=True)
                metadata = move_tensors_to_device(metadata, accelerator.device)
                loss_dict = transformer.compute_training_loss(
                    pixel_values,
                    labels,
                    metadata=metadata,
                    scheduler=scheduler,
                )
                loss = loss_dict["loss"]

                accelerator.backward(loss)

                if accelerator.sync_gradients and args.max_grad_norm is not None:
                    params = [p for p in transformer.parameters() if p.requires_grad]
                    if params:
                        accelerator.clip_grad_norm_(params, args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if lr_scheduler is not None:
                    lr_scheduler.step()

                if accelerator.sync_gradients:
                    accelerator.unwrap_model(transformer).ema_step()
                    global_step += 1

                    if global_step % args.log_every_n_steps == 0:
                        lr = optimizer.param_groups[0]["lr"]
                        accelerator.print(f"step={global_step} loss={loss.item():.6f} lr={lr:.2e}")
                        if args.report_to != "none":
                            accelerator.log({"train/loss": loss.item(), "train/lr": lr}, step=global_step)

                    if global_step % args.save_every_n_steps == 0:
                        save_checkpoint(
                            accelerator=accelerator,
                            transformer=transformer,
                            scheduler=scheduler,
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
        transformer=transformer,
        scheduler=scheduler,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        output_dir=args.output_dir,
        step=global_step,
    )
    if args.report_to != "none":
        accelerator.end_training()


def build_arg_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PixNerd in diffusers style.", add_help=add_help)
    parser.add_argument("--config", type=str, required=True, help="Path to OmegaConf yaml config.")
    parser.add_argument("--output_dir", type=str, required=True, help="Checkpoint output directory.")
    parser.add_argument("--resume_from", type=str, default=None, help="Directory saved with save_pretrained.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="none", help="Tracker backend (e.g. tensorboard, wandb, none).")
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
