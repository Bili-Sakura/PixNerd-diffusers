from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput

from src.diffusion.flow_matching.adam_sampling import AdamLMSampler
from src.diffusion.flow_matching.scheduling import LinearScheduler


@dataclass
class PixNerdSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor


class PixNerdFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    """
    Diffusers-compatible scheduler wrapper for PixNerd's AdamLM flow-matching sampler.
    """

    config_name = "scheduler_config.json"
    order = 1
    init_noise_sigma = 1.0

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 25,
        guidance_scale: float = 4.0,
        timeshift: float = 3.0,
        order: int = 2,
        guidance_interval_min: float = 0.0,
        guidance_interval_max: float = 1.0,
        last_step: Optional[float] = None,
    ) -> None:
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.timeshift = float(timeshift)
        self.order = int(order)
        self.guidance_interval_min = float(guidance_interval_min)
        self.guidance_interval_max = float(guidance_interval_max)
        self.last_step = last_step
        self._reset_state()

    @classmethod
    def from_sampler_spec(cls, sampler_spec: Dict[str, Any]) -> "PixNerdFlowMatchScheduler":
        init_args = dict(sampler_spec.get("init_args", {}))
        return cls(
            num_inference_steps=int(init_args.get("num_steps", 25)),
            guidance_scale=float(init_args.get("guidance", 4.0)),
            timeshift=float(init_args.get("timeshift", 3.0)),
            order=int(init_args.get("order", 2)),
            guidance_interval_min=float(init_args.get("guidance_interval_min", 0.0)),
            guidance_interval_max=float(init_args.get("guidance_interval_max", 1.0)),
            last_step=init_args.get("last_step"),
        )

    def _reset_state(self) -> None:
        self.timesteps: Optional[torch.Tensor] = None
        self._timedeltas: Optional[torch.Tensor] = None
        self._solver_coeffs = None
        self._model_outputs = []
        self._step_index = 0

    def _build_solver(self, num_inference_steps: int, timeshift: float) -> AdamLMSampler:
        return AdamLMSampler(
            order=self.order,
            scheduler=LinearScheduler(),
            guidance_fn=None,
            num_steps=num_inference_steps,
            guidance=self.guidance_scale,
            timeshift=timeshift,
            guidance_interval_min=self.guidance_interval_min,
            guidance_interval_max=self.guidance_interval_max,
            last_step=self.last_step,
        )

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timeshift: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        order: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if num_inference_steps is not None:
            self.num_inference_steps = int(num_inference_steps)
        if timeshift is not None:
            self.timeshift = float(timeshift)
        if guidance_scale is not None:
            self.guidance_scale = float(guidance_scale)
        if order is not None:
            self.order = int(order)

        solver = self._build_solver(self.num_inference_steps, self.timeshift)
        # AdamLMSampler stores num_steps + 1 endpoints; model evaluations happen for the first num_steps only.
        self.timesteps = solver.timesteps[:-1].to(device=device)
        self._timedeltas = solver.timedeltas.to(device=device)
        self._solver_coeffs = solver.solver_coeffs
        self._model_outputs = []
        self._step_index = 0

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        return sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        sample: torch.Tensor,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Union[PixNerdSchedulerOutput, Tuple[torch.Tensor]]:
        if self.timesteps is None or self._timedeltas is None or self._solver_coeffs is None:
            raise RuntimeError("`set_timesteps` must be called before `step`.")
        if self._step_index >= len(self._solver_coeffs):
            raise RuntimeError("Scheduler step index exceeded configured timesteps.")

        coeffs = self._solver_coeffs[self._step_index]
        self._model_outputs.append(model_output)
        order = len(coeffs)
        pred = torch.zeros_like(model_output)
        recent = self._model_outputs[-order:]
        for coeff, output in zip(coeffs, recent):
            pred = pred + coeff * output

        prev_sample = sample + pred * self._timedeltas[self._step_index]
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return PixNerdSchedulerOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha = timesteps.view(-1, 1, 1, 1)
        sigma = (1.0 - timesteps).view(-1, 1, 1, 1)
        return alpha * original_samples + sigma * noise
