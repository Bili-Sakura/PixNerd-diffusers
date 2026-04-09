from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, Optional

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin

from src.pixnerd_diffusers.config_utils import instantiate_from_spec, to_container


class PixNerdModelWrapper(ModelMixin, ConfigMixin):
    config_name = "model_config.json"

    @register_to_config
    def __init__(
        self,
        vae_spec: Dict[str, Any],
        conditioner_spec: Dict[str, Any],
        denoiser_spec: Dict[str, Any],
        trainer_spec: Optional[Dict[str, Any]] = None,
        sampler_spec: Optional[Dict[str, Any]] = None,
        ema_decay: float = 0.9999,
        use_ema: bool = True,
        compile_denoiser: bool = False,
    ) -> None:
        super().__init__()
        self.vae = instantiate_from_spec(to_container(vae_spec))
        self.conditioner = instantiate_from_spec(to_container(conditioner_spec))
        self.denoiser = instantiate_from_spec(to_container(denoiser_spec))
        self.diffusion_trainer = (
            instantiate_from_spec(to_container(trainer_spec)) if trainer_spec is not None else None
        )
        self.diffusion_sampler = (
            instantiate_from_spec(to_container(sampler_spec)) if sampler_spec is not None else None
        )

        self.ema_decay = ema_decay
        self.ema_denoiser = copy.deepcopy(self.denoiser) if use_ema else None
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
    def from_model_config(cls, model_config: Dict[str, Any], **kwargs: Any) -> "PixNerdModelWrapper":
        model_config = to_container(model_config)
        ema_decay = (
            model_config.get("ema_tracker", {}).get("init_args", {}).get("decay", 0.9999)
            if isinstance(model_config, dict)
            else 0.9999
        )
        return cls(
            vae_spec=model_config["vae"],
            conditioner_spec=model_config["conditioner"],
            denoiser_spec=model_config["denoiser"],
            trainer_spec=model_config.get("diffusion_trainer"),
            sampler_spec=model_config.get("diffusion_sampler"),
            ema_decay=ema_decay,
            **kwargs,
        )

    def _freeze_module(self, module: Optional[torch.nn.Module]) -> None:
        if module is None:
            return
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False

    def _freeze_non_trainable_modules(self) -> None:
        self._freeze_module(self.vae)
        self._freeze_module(self.conditioner)
        self._freeze_module(self.ema_denoiser)

    @staticmethod
    def _as_timestep_tensor(timestep: Any, batch_size: int, device: torch.device) -> torch.Tensor:
        if isinstance(timestep, torch.Tensor):
            if timestep.ndim == 0:
                return timestep.repeat(batch_size).to(device=device, dtype=torch.float32)
            return timestep.to(device=device, dtype=torch.float32)
        return torch.full((batch_size,), float(timestep), device=device, dtype=torch.float32)

    def forward(self, sample: torch.Tensor, timestep: Any, conditioning: torch.Tensor) -> torch.Tensor:
        t = self._as_timestep_tensor(timestep, sample.shape[0], sample.device)
        return self.denoiser(sample, t, conditioning)

    def denoiser_for_inference(self, use_ema: bool = True) -> torch.nn.Module:
        if use_ema and self.ema_denoiser is not None:
            return self.ema_denoiser
        return self.denoiser

    @torch.no_grad()
    def get_conditioning(self, y: Iterable[Any], metadata: Optional[Dict[str, Any]] = None):
        if metadata is None:
            metadata = {}
        return self.conditioner(y, metadata)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
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
        if decay is None:
            decay = self.ema_decay
        for ema_param, param in zip(self.ema_denoiser.parameters(), self.denoiser.parameters()):
            ema_param.mul_(decay).add_(param.detach().float(), alpha=1.0 - decay)

    def compute_training_loss(
        self,
        x: torch.Tensor,
        y: Iterable[Any],
        metadata: Optional[Dict[str, Any]] = None,
        sampler: Optional[torch.nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.diffusion_trainer is None:
            raise RuntimeError("diffusion_trainer is not configured in model wrapper.")
        if sampler is None:
            sampler = self.diffusion_sampler
        if sampler is None:
            raise RuntimeError("No diffusion sampler available for training.")
        if metadata is None:
            metadata = {}
        with torch.no_grad():
            x = self.encode(x)
            condition, uncondition = self.get_conditioning(y, metadata)
        return self.diffusion_trainer(
            self.denoiser,
            self.ema_denoiser if self.ema_denoiser is not None else self.denoiser,
            sampler,
            x,
            condition,
            uncondition,
            metadata,
        )

    @staticmethod
    def _load_prefixed_state(
        module: Optional[torch.nn.Module],
        state_dict: Dict[str, torch.Tensor],
        prefixes: Iterable[str],
    ) -> None:
        if module is None:
            return
        for prefix in prefixes:
            subset = {
                key[len(prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if subset:
                module.load_state_dict(subset, strict=False)
                return

    def load_legacy_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        self._load_prefixed_state(self.denoiser, state_dict, prefixes=["denoiser."])
        self._load_prefixed_state(self.ema_denoiser, state_dict, prefixes=["ema_denoiser."])
        self._load_prefixed_state(self.diffusion_trainer, state_dict, prefixes=["diffusion_trainer."])
