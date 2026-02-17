"""Noise schedules for masked diffusion.

Provides the LogLinear schedule used by MDLM, mapping timestep t in [0, 1]
to a noise level sigma(t).  The masking probability at time t is:
    move_chance(t) = 1 - exp(-sigma(t))

Also provides ``HierarchicalNoiseSchedule`` which wraps a base schedule and
applies per-level scaling factors so that different hierarchy levels receive
different amounts of noise at the same timestep t.
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


class HierarchicalNoiseSchedule(nn.Module):
    """Per-level noise scaling on top of a base schedule.

    Given a base noise schedule that produces sigma(t), this module scales
    sigma independently for each hierarchy level:

        sigma_k(t) = scale_k * sigma(t)

    A scale < 1 gives *less* noise (lower masking rate) → used for summaries.
    A scale > 1 gives *more* noise (higher masking rate) → used for content.

    The corresponding per-level masking probability is:
        move_chance_k(t) = 1 - exp(-sigma_k(t))

    And the per-level ELBO weight is:
        weight_k(t) = dsigma_k / expm1(sigma_k)
                     = (scale_k * dsigma) / expm1(scale_k * sigma)

    Args:
        base_noise: the underlying ``Noise`` schedule.
        level_scales: list of K floats, one per hierarchy level.
            Example for K=2: [0.5, 1.0] → summary gets half the noise.
    """

    def __init__(
        self,
        base_noise: Noise,
        level_scales: list[float],
    ):
        super().__init__()
        self.base_noise = base_noise
        self.num_levels = len(level_scales)
        # Register as buffer so it moves with .to(device)
        self.register_buffer(
            "level_scales",
            torch.tensor(level_scales, dtype=torch.float32),
        )

    def forward(self, t: torch.Tensor):
        """Return base (sigma, dsigma) — still needed by the loss."""
        return self.base_noise(t)

    def get_per_level_move_chance(
        self, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-level masking probability.

        Args:
            sigma: (B,) base sigma from the noise schedule.

        Returns:
            (B, K) masking probabilities, one per level.
        """
        # sigma: (B,) → (B, 1);  level_scales: (K,) → (1, K)
        scaled_sigma = sigma.unsqueeze(1) * self.level_scales.unsqueeze(0)  # (B, K)
        return 1.0 - torch.exp(-scaled_sigma)  # (B, K)

    def get_per_level_loss_weight(
        self, sigma: torch.Tensor, dsigma: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-level ELBO loss weight: scale_k * dsigma / expm1(scale_k * sigma).

        The raw weight approximates 1/t for the loglinear schedule, which
        diverges as t → 0.  We clamp the denominator to avoid extreme
        spikes that cause high-variance loss oscillation (especially with
        small batch sizes).

        Args:
            sigma:  (B,) base sigma.
            dsigma: (B,) base dsigma.

        Returns:
            (B, K) loss weights.
        """
        scales = self.level_scales.unsqueeze(0)             # (1, K)
        scaled_sigma = sigma.unsqueeze(1) * scales          # (B, K)
        scaled_dsigma = dsigma.unsqueeze(1) * scales        # (B, K)
        # Clamp denominator to avoid near-zero division at small timesteps
        denom = torch.expm1(scaled_sigma).clamp(min=1e-4)   # (B, K)
        return scaled_dsigma / denom                         # (B, K)


def get_noise_schedule(name: str = "loglinear", **kwargs) -> Noise:
    """Factory for base noise schedules."""
    if name == "loglinear":
        return LogLinearNoise(**kwargs)
    raise ValueError(f"Unknown noise schedule: {name}")
