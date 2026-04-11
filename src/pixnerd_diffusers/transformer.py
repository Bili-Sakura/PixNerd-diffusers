from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput

from src.pixnerd_diffusers.config_utils import instantiate_from_spec, to_container


@dataclass
class PixNerdTransformer2DModelOutput(BaseOutput):
    sample: torch.FloatTensor


class PixNerdTransformer2DModel(ModelMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        denoiser_spec: Dict[str, Any],
        conditioner_spec: Dict[str, Any],
        vae_spec: Optional[Dict[str, Any]] = None,
        diffusion_trainer_spec: Optional[Dict[str, Any]] = None,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        compile_denoiser: bool = False,
    ) -> None:
        super().__init__()
        self.denoiser = instantiate_from_spec(to_container(denoiser_spec))
        self.conditioner = instantiate_from_spec(to_container(conditioner_spec))
        self.vae = instantiate_from_spec(to_container(vae_spec)) if vae_spec is not None else None
        self.diffusion_trainer = (
            instantiate_from_spec(to_container(diffusion_trainer_spec))
            if diffusion_trainer_spec is not None
            else None
        )

        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.ema_denoiser = copy.deepcopy(self.denoiser) if self.use_ema else None
        if self.ema_denoiser is not None:
            self.ema_denoiser.to(torch.float32)

        if compile_denoiser and hasattr(self.denoiser, "compile"):
            self.denoiser.compile()
            if self.ema_denoiser is not None:
                self.ema_denoiser.compile()

        self._freeze_non_trainable_modules()
        if self.ema_denoiser is not None:
            self.sync_ema()

    @classmethod
    def from_project_config(
        cls,
        model_config: Dict[str, Any],
        use_ema: bool = True,
        compile_denoiser: bool = False,
    ) -> "PixNerdTransformer2DModel":
        model_config = to_container(model_config)
        ema_decay = model_config.get("ema_tracker", {}).get("init_args", {}).get("decay", 0.9999)
        return cls(
            denoiser_spec=model_config["denoiser"],
            conditioner_spec=model_config["conditioner"],
            vae_spec=model_config.get("vae"),
            diffusion_trainer_spec=model_config.get("diffusion_trainer"),
            use_ema=use_ema,
            ema_decay=ema_decay,
            compile_denoiser=compile_denoiser,
        )

    @staticmethod
    def _as_timestep_tensor(
        timestep: Any,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(timestep, torch.Tensor):
            if timestep.ndim == 0:
                return timestep.repeat(batch_size).to(device=device, dtype=torch.float32)
            return timestep.to(device=device, dtype=torch.float32)
        return torch.full((batch_size,), float(timestep), device=device, dtype=torch.float32)

    def _freeze_module(self, module: Optional[torch.nn.Module]) -> None:
        if module is None:
            return
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False

    def _freeze_non_trainable_modules(self) -> None:
        self._freeze_module(self.conditioner)
        self._freeze_module(self.vae)
        self._freeze_module(self.ema_denoiser)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Any,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
    ) -> PixNerdTransformer2DModelOutput | Tuple[torch.Tensor]:
        t = self._as_timestep_tensor(timestep, sample.shape[0], sample.device)
        out = self.denoiser(sample, t, encoder_hidden_states)
        if not return_dict:
            return (out,)
        return PixNerdTransformer2DModelOutput(sample=out)

    def predict_noise(
        self,
        sample: torch.Tensor,
        timestep: Any,
        encoder_hidden_states: torch.Tensor,
        use_ema: bool = False,
    ) -> torch.Tensor:
        t = self._as_timestep_tensor(timestep, sample.shape[0], sample.device)
        denoiser = self.get_inference_denoiser(use_ema=use_ema)
        return denoiser(sample, t, encoder_hidden_states)

    def get_inference_denoiser(self, use_ema: bool = True) -> torch.nn.Module:
        if use_ema and self.ema_denoiser is not None:
            return self.ema_denoiser
        return self.denoiser

    @torch.no_grad()
    def get_conditioning(
        self,
        y: Iterable[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        metadata = {} if metadata is None else metadata
        return self.conditioner(y, metadata)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return x
        return self.vae.encode(x)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return latents
        return self.vae.decode(latents)

    @torch.no_grad()
    def sync_ema(self) -> None:
        if self.ema_denoiser is None:
            return
        self.ema_denoiser.load_state_dict(self.denoiser.state_dict(), strict=True)
        self.ema_denoiser.to(torch.float32)

    @torch.no_grad()
    def ema_step(self, decay: Optional[float] = None) -> None:
        if self.ema_denoiser is None:
            return
        decay = self.ema_decay if decay is None else float(decay)
        for ema_param, param in zip(self.ema_denoiser.parameters(), self.denoiser.parameters()):
            ema_param.mul_(decay).add_(param.detach().float(), alpha=1.0 - decay)

    def compute_training_loss(
        self,
        x: torch.Tensor,
        y: Iterable[Any],
        scheduler: torch.nn.Module,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.diffusion_trainer is None:
            raise RuntimeError("diffusion_trainer is not configured.")
        metadata = {} if metadata is None else metadata

        with torch.no_grad():
            x = self.encode(x)
            condition, uncondition = self.get_conditioning(y, metadata)

        return self.diffusion_trainer(
            self.denoiser,
            self.ema_denoiser if self.ema_denoiser is not None else self.denoiser,
            scheduler,
            x,
            condition,
            uncondition,
            metadata,
        )
