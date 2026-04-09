from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import torch
from diffusers import DiffusionPipeline, ImagePipelineOutput
from omegaconf import OmegaConf
from PIL import Image

from src.models.autoencoder.base import fp2uint8
from src.pixnerd_diffusers.config_utils import clone_spec, instantiate_from_spec, to_container
from src.pixnerd_diffusers.model_wrapper import PixNerdModelWrapper


ConditioningInput = Union[str, int, Sequence[Union[str, int]]]


class PixNerdPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "model"

    def __init__(self, model: PixNerdModelWrapper, sampler_spec: Dict[str, Any]):
        super().__init__()
        self.register_modules(model=model)
        sampler_spec = to_container(sampler_spec)
        self.register_to_config(sampler_spec=sampler_spec)
        self._default_sampler_init = sampler_spec.get("init_args", {})

    @classmethod
    def from_config(
        cls,
        config_path: str,
        checkpoint_path: Optional[str] = None,
        use_ema: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "PixNerdPipeline":
        config = OmegaConf.load(config_path)
        model = PixNerdModelWrapper.from_model_config(to_container(config.model), use_ema=use_ema)
        if checkpoint_path is not None:
            if os.path.isdir(checkpoint_path) and os.path.exists(
                os.path.join(checkpoint_path, model.config_name)
            ):
                model = PixNerdModelWrapper.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch_dtype,
                )
            else:
                model.load_legacy_checkpoint(checkpoint_path)
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        pipeline = cls(model=model, sampler_spec=to_container(config.model.diffusion_sampler))
        if device is not None:
            pipeline = pipeline.to(device)
        return pipeline

    @staticmethod
    def _to_list(y: ConditioningInput) -> List[Union[str, int]]:
        if isinstance(y, (str, int)):
            return [y]
        return list(y)

    @staticmethod
    def _repeat(values: List[Union[str, int]], repeats: int) -> List[Union[str, int]]:
        if repeats == 1:
            return values
        expanded = []
        for value in values:
            expanded.extend([value] * repeats)
        return expanded

    def _build_sampler(
        self,
        num_inference_steps: Optional[int],
        guidance_scale: Optional[float],
        timeshift: Optional[float],
        order: Optional[int],
    ) -> torch.nn.Module:
        sampler_spec = clone_spec(self.config.sampler_spec)
        init_args = sampler_spec.setdefault("init_args", {})
        if num_inference_steps is not None:
            init_args["num_steps"] = int(num_inference_steps)
        if guidance_scale is not None:
            init_args["guidance"] = float(guidance_scale)
        if timeshift is not None:
            init_args["timeshift"] = float(timeshift)
        if order is not None:
            init_args["order"] = int(order)
        return instantiate_from_spec(sampler_spec)

    def __call__(
        self,
        y: ConditioningInput,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        timeshift: Optional[float] = None,
        order: Optional[int] = None,
        use_ema: bool = True,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, tuple]:
        prompts = self._repeat(self._to_list(y), num_images_per_prompt)
        batch_size = len(prompts)

        channels = int(getattr(self.model.denoiser, "in_channels", 3))
        patch_size = int(getattr(self.model.denoiser, "patch_size", 1))
        height = (height // patch_size) * patch_size
        width = (width // patch_size) * patch_size

        if hasattr(self.model.denoiser, "decoder_patch_scaling_h"):
            self.model.denoiser.decoder_patch_scaling_h = height / 512
            self.model.denoiser.decoder_patch_scaling_w = width / 512
        if self.model.ema_denoiser is not None and hasattr(self.model.ema_denoiser, "decoder_patch_scaling_h"):
            self.model.ema_denoiser.decoder_patch_scaling_h = height / 512
            self.model.ema_denoiser.decoder_patch_scaling_w = width / 512

        if seed is None:
            generator = torch.Generator(device="cpu")
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        device = self._execution_device
        noise = torch.randn(
            (batch_size, channels, height, width),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(device)

        with torch.no_grad():
            condition, uncondition = self.model.get_conditioning(prompts, metadata={})
            sampler = self._build_sampler(num_inference_steps, guidance_scale, timeshift, order).to(device)
            denoiser = self.model.denoiser_for_inference(use_ema=use_ema)
            samples = sampler(denoiser, noise, condition, uncondition)
            decoded = self.model.decode(samples)

        images_uint8 = fp2uint8(decoded).permute(0, 2, 3, 1).cpu().numpy()
        if output_type == "np":
            output = images_uint8
        elif output_type == "pt":
            output = torch.from_numpy(images_uint8)
        elif output_type == "pil":
            output = [Image.fromarray(image) for image in images_uint8]
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

        if not return_dict:
            return (output,)
        return ImagePipelineOutput(images=output)
