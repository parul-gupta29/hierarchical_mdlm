"""Noise schedules for masked diffusion.

Provides the LogLinear schedule used by MDLM, mapping timestep t in [0, 1]
to a noise level sigma(t).  The masking probability at time t is:
    move_chance(t) = 1 - exp(-sigma(t))
"""

import abc
import torch
import torch.nn as nn


class Noise(abc.ABC, nn.Module):
    """Base class for noise schedules."""

    @abc.abstractmethod
    def rate_noise(self, t: torch.Tensor) -> torch.Tensor:
        """dsigma/dt — instantaneous noise rate."""

    @abc.abstractmethod
    def total_noise(self, t: torch.Tensor) -> torch.Tensor:
        """sigma(t) — cumulative noise."""

    def forward(self, t: torch.Tensor):
        """Returns (sigma, dsigma) for a batch of timesteps."""
        return self.total_noise(t), self.rate_noise(t)


class LogLinearNoise(Noise):
    """Log-linear noise schedule (default for MDLM).

    sigma(t) = -log(1 - (1 - eps) * t)
    """

    def __init__(self, sigma_min: float = 1e-4, sigma_max: float = 20.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def rate_noise(self, t: torch.Tensor) -> torch.Tensor:
        eps = self.sigma_min
        return (1 - eps) / (1 - (1 - eps) * t)

    def total_noise(self, t: torch.Tensor) -> torch.Tensor:
        eps = self.sigma_min
        return -torch.log1p(-(1 - eps) * t)

    def importance_sampling_transformation(self, t: torch.Tensor) -> torch.Tensor:
        eps = self.sigma_min
        return -torch.log1p(-(1 - eps) * t)


def get_noise_schedule(name: str = "loglinear", **kwargs) -> Noise:
    """Factory for noise schedules."""
    if name == "loglinear":
        return LogLinearNoise(**kwargs)
    raise ValueError(f"Unknown noise schedule: {name}")
