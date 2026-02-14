"""Inference / sampling for the hierarchical MDLM.

Generates text given:
    - A title (always unmasked, prefix of every sequence).
    - Hierarchy probabilities per position — can come from:
        (a) ground-truth labels (one-hot) for teacher-forced generation, or
        (b) a hierarchy predictor model that outputs soft probs.

The sampler uses DDPM-caching (default) or plain DDPM, following MDLM.
For each reverse-diffusion step, each output head contributes to the
denoising distribution only at positions assigned to its level.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hierarchical_generator import HierarchicalGenerator
from models.hierarchy_embedding import HierarchyEmbedding
from noise_schedule import Noise, get_noise_schedule


def _sample_categorical(probs: torch.Tensor) -> torch.Tensor:
    """Gumbel-max trick for sampling from a categorical distribution."""
    gumbel_noise = 1e-10 - (torch.rand_like(probs) + 1e-10).log()
    return (probs / gumbel_noise).argmax(dim=-1)


class HierarchicalSampler:
    """DDPM-based sampler for the hierarchical generator.

    Args:
        model: trained HierarchicalGenerator.
        noise: the noise schedule used during training.
        num_steps: number of reverse-diffusion steps.
        device: torch device.
    """

    def __init__(
        self,
        model: HierarchicalGenerator,
        noise: Noise | None = None,
        num_steps: int = 1000,
        device: str = "cuda",
    ):
        self.model = model
        self.model.eval()
        self.noise = noise or get_noise_schedule("loglinear")
        self.noise.to(device)
        self.num_steps = num_steps
        self.device = device
        self.mask_index = model.mask_index

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        title_ids: torch.Tensor,
        hierarchy_probs: torch.Tensor,
        seq_length: int | None = None,
    ) -> torch.Tensor:
        """Generate a full sequence.

        Args:
            title_ids:       (B, T) token ids for the title prefix.
            hierarchy_probs: (B, L, K) soft hierarchy distribution for every
                             position in the *full* sequence (including title).
            seq_length:      total sequence length L.  Inferred from
                             ``hierarchy_probs`` if not given.

        Returns:
            (B, L) generated token ids (title positions unchanged).
        """
        B, T = title_ids.shape
        L = seq_length or hierarchy_probs.shape[1]
        K = hierarchy_probs.shape[2]

        # Start from fully masked body; title is already known
        x = torch.full((B, L), self.mask_index, dtype=torch.long, device=self.device)
        x[:, :T] = title_ids.to(self.device)
        hierarchy_probs = hierarchy_probs.to(self.device)

        # Title mask: True for title positions
        title_mask = torch.zeros(B, L, device=self.device)
        title_mask[:, :T] = 1.0

        eps = 1e-5
        timesteps = torch.linspace(1, eps, self.num_steps + 1, device=self.device)
        dt = (1 - eps) / self.num_steps

        p_x0_cache = None

        for i in range(self.num_steps):
            t = timesteps[i] * torch.ones(B, device=self.device)
            p_x0_cache, x = self._ddpm_caching_step(
                x, t, dt, hierarchy_probs, title_mask, p_x0_cache
            )

        # Final denoising step: argmax over combined logits
        x = self._denoise_final(x, timesteps[-1], hierarchy_probs, title_mask)

        return x

    # ------------------------------------------------------------------
    #  Internal reverse-diffusion steps
    # ------------------------------------------------------------------

    def _get_combined_logits(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        hierarchy_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward through model and combine per-level logits.

        For each position, the combined probability is the weighted sum
        of per-head probabilities according to the hierarchy distribution:

            p(v | i) = sum_k  hier_prob_{i,k} * softmax(logits_k[i])_v

        We return log-probabilities: (B, L, V).
        """
        logits_per_level = self.model(x, sigma, hierarchy_probs)  # list of K (B,L,V)

        # Apply subs parameterization per head -> log probs
        log_probs_per_level = []
        for logits in logits_per_level:
            lp = logits.clone()
            lp[:, :, self.mask_index] += self.model.NEG_INF
            lp = lp - torch.logsumexp(lp, dim=-1, keepdim=True)
            unmasked = x != self.mask_index
            lp[unmasked] = self.model.NEG_INF
            lp[unmasked, x[unmasked]] = 0.0
            log_probs_per_level.append(lp)

        # Weighted combination in probability space
        # hierarchy_probs: (B, L, K)
        combined_prob = torch.zeros_like(log_probs_per_level[0]).exp() * 0
        for k, lp in enumerate(log_probs_per_level):
            w = hierarchy_probs[:, :, k : k + 1]  # (B, L, 1)
            combined_prob = combined_prob + w * lp.exp()

        combined_prob = combined_prob.clamp(min=1e-30)
        return combined_prob  # (B, L, V) — probabilities

    def _ddpm_caching_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        hierarchy_probs: torch.Tensor,
        title_mask: torch.Tensor,
        p_x0_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """One DDPM-caching reverse step."""
        move_chance_t = t[:, None, None]       # (B, 1, 1)
        move_chance_s = (t - dt)[:, None, None]

        if p_x0_cache is None:
            sigma_t = self.noise.total_noise(t)
            p_x0 = self._get_combined_logits(x, sigma_t, hierarchy_probs)
        else:
            p_x0 = p_x0_cache

        # Transition probabilities
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        x_new = _sample_categorical(q_xs)

        # Keep unmasked positions (and title) unchanged
        copy_flag = (x != self.mask_index).to(x.dtype)
        x_new = (copy_flag * x + (1 - copy_flag) * x_new).long()

        # Invalidate cache if x changed
        new_cache: torch.Tensor | None = p_x0
        if not torch.equal(x_new, x):
            new_cache = None

        return new_cache, x_new

    def _denoise_final(
        self,
        x: torch.Tensor,
        t_final: torch.Tensor,
        hierarchy_probs: torch.Tensor,
        title_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Final noise-removal step: argmax of combined probabilities."""
        sigma = self.noise.total_noise(
            t_final * torch.ones(x.shape[0], device=self.device)
        )
        p_x0 = self._get_combined_logits(x, sigma, hierarchy_probs)
        x_denoised = p_x0.argmax(dim=-1)

        # Preserve title
        is_title = title_mask.bool()
        x_denoised[is_title] = x[is_title]
        return x_denoised
