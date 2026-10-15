"""Flow Matching Scheduler for OneWM.

Based on Fast-WAM's WanContinuousFlowMatchScheduler and LingBo-VA's FlowMatchScheduler.
Supports shifted noise schedules for both video and action modalities.
"""

import torch
from torch import Tensor


class FlowMatchScheduler:
    """Continuous Flow Matching scheduler matching Fast-WAM's Wan scheduler."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.0,
        eps: float = 1e-10,
        stratified_timestep_sampling: bool = False,
    ):
        if num_train_timesteps <= 0:
            raise ValueError(
                f"num_train_timesteps must be positive, got {num_train_timesteps}"
            )
        if shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.eps = float(eps)
        self.stratified_timestep_sampling = bool(stratified_timestep_sampling)
        self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()

    @staticmethod
    def _phi(u: Tensor, shift: float) -> Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def _apply_shift(self, sigmas: Tensor) -> Tensor:
        return self._phi(sigmas, self.shift)

    def _precompute_training_weight_stats(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        t_grid = self._phi(u_grid, self.shift) * float(steps)
        y_grid = torch.exp(-2.0 * ((t_grid - (steps / 2.0)) / steps) ** 2)
        y_min = float(y_grid.min().item())
        norm_const = float((y_grid - y_min).mean().item())
        return y_min, norm_const

    def sample_timesteps(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self.stratified_timestep_sampling and batch_size > 1:
            # One random draw per equal-width stratum preserves the uniform
            # marginal objective while reducing Monte-Carlo variance. Shuffle
            # so sample order is not correlated with noise level.
            u = (
                torch.arange(batch_size, device=device, dtype=torch.float32)
                + torch.rand((batch_size,), device=device, dtype=torch.float32)
            ) / float(batch_size)
            u = u[torch.randperm(batch_size, device=device)]
        else:
            u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigma = self._phi(u, self.shift)
        timestep = sigma * float(self.num_train_timesteps)
        return timestep.to(dtype=dtype)

    def timestep_to_sigma(self, timesteps: Tensor) -> Tensor:
        return timesteps.to(dtype=torch.float32) / float(self.num_train_timesteps)

    def add_noise(
        self, original_samples: Tensor, noise: Tensor, timesteps: Tensor
    ) -> Tensor:
        sigma = self.timestep_to_sigma(timesteps).to(
            original_samples.device, dtype=original_samples.dtype
        )
        if sigma.ndim == 0:
            return (1.0 - sigma) * original_samples + sigma * noise
        sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1.0 - sigma) * original_samples + sigma * noise

    def training_target(
        self, sample: Tensor, noise: Tensor, timesteps: Tensor | None = None
    ) -> Tensor:
        del timesteps
        return noise - sample

    def training_weight(self, timesteps: Tensor) -> Tensor:
        t = timesteps.to(dtype=torch.float32)
        steps = float(self.num_train_timesteps)
        y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
        weight = (y - self._y_min) / (self._weight_norm_const + self.eps)
        if weight.numel() == 1:
            return weight.reshape(())
        return weight

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        if num_inference_steps <= 0:
            raise ValueError(
                f"num_inference_steps must be positive, got {num_inference_steps}"
            )
        shift = self.shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}")
        u_steps = torch.linspace(
            1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32
        )
        sigma_steps = self._phi(u_steps, shift)
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def step(model_output: Tensor, delta: float | Tensor, sample: Tensor) -> Tensor:
        delta = (
            delta.to(sample.device, dtype=sample.dtype)
            if isinstance(delta, Tensor)
            else torch.as_tensor(delta, device=sample.device, dtype=sample.dtype)
        )
        if delta.ndim == 0:
            return sample + model_output * delta
        delta = delta.view(-1, *([1] * (sample.ndim - 1)))
        return sample + model_output * delta
