"""Two-phase hierarchical MDLM trainer.

Phase 1 — **Hierarchy-only fine-tuning**
    Freeze the entire DDiT backbone.  Train only:
        • HierarchyEmbedding  (new parameters)
        • MultiHeadVocabOutput (new parameters)

Phase 2 — **Full LoRA fine-tuning**
    Attach LoRA adapters to every attention / MLP projection in the backbone
    and unfreeze them together with the hierarchy + output-head parameters.
    The original backbone weights remain frozen; only the low-rank residuals
    and the new modules are updated.

Both phases share the same diffusion training loop:
    1. Sample t ~ U(eps, 1)
    2. Compute sigma(t), move_chance = 1 - exp(-sigma)
    3. Mask tokens with probability move_chance (title tokens are exempt)
    4. Forward through the model → per-level logits
    5. Compute the hierarchical SUBS loss (only on masked, non-title tokens)
    6. Backprop + optimizer step
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.hierarchical_generator import HierarchicalGenerator
from models.hierarchy_embedding import HierarchyEmbedding
from noise_schedule import Noise, get_noise_schedule
from lora.lora import (
    apply_lora_to_model,
    freeze_non_lora,
    get_trainable_parameters,
    print_trainable_summary,
    unfreeze_all_lora,
)


# ---------------------------------------------------------------------------
#  Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    # --- model ---
    pretrained_path: str = "kuleshov-group/mdlm-owt"
    vocab_size: int = 50258
    num_levels: int = 2
    hidden_size: int = 768
    cond_dim: int = 128
    max_length: int = 1024

    # --- noise ---
    noise_type: str = "loglinear"
    sampling_eps: float = 1e-3
    antithetic_sampling: bool = True

    # --- LoRA ---
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        r".*\.attn_qkv$",
        r".*\.attn_out$",
        r".*\.mlp\.0$",
        r".*\.mlp\.2$",
    ])

    # --- phase 1 ---
    phase1_lr: float = 1e-4
    phase1_weight_decay: float = 0.0
    phase1_max_steps: int = 10_000
    phase1_batch_size: int = 16
    phase1_grad_clip: float = 1.0

    # --- phase 2 ---
    phase2_lr: float = 3e-5
    phase2_weight_decay: float = 0.01
    phase2_max_steps: int = 50_000
    phase2_batch_size: int = 16
    phase2_grad_clip: float = 1.0

    # --- general ---
    device: str = "cuda"
    log_every: int = 100
    save_every: int = 5000
    save_dir: str = "checkpoints"
    seed: int = 42


# ---------------------------------------------------------------------------
#  Trainer
# ---------------------------------------------------------------------------

class HierarchicalMDLMTrainer:
    """Orchestrates the two-phase training of HierarchicalGenerator."""

    def __init__(self, config: TrainerConfig):
        self.config = config
        torch.manual_seed(config.seed)

        # ── build model ────────────────────────────────────────────────
        self.model = HierarchicalGenerator(
            pretrained_path=config.pretrained_path,
            vocab_size=config.vocab_size,
            num_levels=config.num_levels,
            hidden_size=config.hidden_size,
            cond_dim=config.cond_dim,
        )
        self.model.to(config.device)
        self.mask_index = self.model.mask_index

        # ── noise schedule ─────────────────────────────────────────────
        self.noise: Noise = get_noise_schedule(config.noise_type)
        self.noise.to(config.device)

        os.makedirs(config.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    #  Diffusion helpers
    # ------------------------------------------------------------------

    def _sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps with optional antithetic sampling."""
        eps_t = torch.rand(batch_size, device=device)
        if self.config.antithetic_sampling:
            offset = torch.arange(batch_size, device=device).float() / batch_size
            eps_t = (eps_t / batch_size + offset) % 1
        return (1 - self.config.sampling_eps) * eps_t + self.config.sampling_eps

    def _q_xt(
        self,
        x0: torch.Tensor,
        move_chance: torch.Tensor,
        title_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Corrupt clean tokens by masking — title tokens are never masked.

        Args:
            x0:         (B, L) clean token ids.
            move_chance: (B, 1) per-sample masking probability.
            title_mask:  (B, L) float, 1.0 at title positions.

        Returns:
            xt: (B, L) noisy tokens.
        """
        rand = torch.rand_like(x0.float())
        move_indices = rand < move_chance                  # (B, L)
        move_indices = move_indices & (title_mask < 0.5)   # protect title
        return torch.where(move_indices, self.mask_index, x0)

    # ------------------------------------------------------------------
    #  Single training step
    # ------------------------------------------------------------------

    def _train_step(
        self,
        batch: dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
    ) -> float:
        """Execute one gradient-update step.

        Returns the scalar loss value.
        """
        self.model.train()
        device = self.config.device

        x0 = batch["input_ids"].to(device)                # (B, L)
        hierarchy_labels = batch["hierarchy_labels"].to(device)  # (B, L)
        title_mask = batch["title_mask"].to(device)        # (B, L)
        attention_mask = batch["attention_mask"].to(device)  # (B, L)

        B = x0.shape[0]

        # 1. Sample timestep
        t = self._sample_t(B, device)                      # (B,)
        sigma, dsigma = self.noise(t)                      # (B,), (B,)
        move_chance = 1.0 - torch.exp(-sigma)              # (B,)

        # 2. Corrupt (title tokens stay clean)
        xt = self._q_xt(x0, move_chance.unsqueeze(1), title_mask)

        # 3. Build hierarchy probs (one-hot during training)
        hier_probs = HierarchyEmbedding.labels_to_onehot(
            hierarchy_labels, self.config.num_levels
        )  # (B, L, K)

        # 4. Forward
        logits_per_level = self.model(xt, sigma, hier_probs)

        # 5. Loss (only on masked non-title tokens)
        loss = self.model.compute_loss(
            logits_per_level=logits_per_level,
            xt=xt,
            x0=x0,
            hierarchy_labels=hierarchy_labels,
            title_mask=title_mask,
            dsigma=dsigma,
            sigma=sigma,
        )

        # 6. Backward
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(
                get_trainable_parameters(self.model), grad_clip
            )
        optimizer.step()
        return loss.item()

    # ------------------------------------------------------------------
    #  Phase loops
    # ------------------------------------------------------------------

    def _run_phase(
        self,
        phase_name: str,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        max_steps: int,
        grad_clip: float,
    ) -> None:
        step = 0
        epoch = 0
        while step < max_steps:
            epoch += 1
            for batch in dataloader:
                if step >= max_steps:
                    break
                loss_val = self._train_step(batch, optimizer, grad_clip)
                step += 1
                if step % self.config.log_every == 0:
                    print(f"[{phase_name}] step {step}/{max_steps}  loss={loss_val:.4f}")
                if step % self.config.save_every == 0:
                    self._save_checkpoint(phase_name, step)

        self._save_checkpoint(phase_name, step)
        print(f"[{phase_name}] Finished — {step} steps, {epoch} epochs.")

    # ------------------------------------------------------------------
    #  Phase 1: train hierarchy embedding + output heads only
    # ------------------------------------------------------------------

    def run_phase1(self, dataloader: DataLoader) -> None:
        """Freeze backbone; train hierarchy embedding + output heads."""
        print("=" * 60)
        print("PHASE 1: Hierarchy embedding + output heads")
        print("=" * 60)

        # Freeze everything
        for param in self.model.parameters():
            param.requires_grad_(False)

        # Unfreeze new components
        for param in self.model.hierarchy_embedding.parameters():
            param.requires_grad_(True)
        for param in self.model.output_heads.parameters():
            param.requires_grad_(True)

        print_trainable_summary(self.model, "Phase 1")

        optimizer = torch.optim.AdamW(
            get_trainable_parameters(self.model),
            lr=self.config.phase1_lr,
            weight_decay=self.config.phase1_weight_decay,
        )

        self._run_phase(
            "Phase1",
            dataloader,
            optimizer,
            self.config.phase1_max_steps,
            self.config.phase1_grad_clip,
        )

    # ------------------------------------------------------------------
    #  Phase 2: LoRA fine-tune entire model
    # ------------------------------------------------------------------

    def run_phase2(self, dataloader: DataLoader) -> None:
        """Add LoRA to backbone; fine-tune LoRA + hierarchy + output heads."""
        print("=" * 60)
        print("PHASE 2: LoRA fine-tuning of full model")
        print("=" * 60)

        # Attach LoRA to backbone blocks
        apply_lora_to_model(
            self.model.backbone,
            target_modules=self.config.lora_target_modules,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
        )
        self.model.to(self.config.device)

        # Freeze original weights, unfreeze LoRA + new modules
        freeze_non_lora(self.model)
        unfreeze_all_lora(self.model)
        for param in self.model.hierarchy_embedding.parameters():
            param.requires_grad_(True)
        for param in self.model.output_heads.parameters():
            param.requires_grad_(True)

        print_trainable_summary(self.model, "Phase 2")

        optimizer = torch.optim.AdamW(
            get_trainable_parameters(self.model),
            lr=self.config.phase2_lr,
            weight_decay=self.config.phase2_weight_decay,
        )

        self._run_phase(
            "Phase2",
            dataloader,
            optimizer,
            self.config.phase2_max_steps,
            self.config.phase2_grad_clip,
        )

    # ------------------------------------------------------------------
    #  Convenience: run both phases sequentially
    # ------------------------------------------------------------------

    def train(self, dataloader: DataLoader) -> None:
        """Run Phase 1 then Phase 2."""
        self.run_phase1(dataloader)
        self.run_phase2(dataloader)

    # ------------------------------------------------------------------
    #  Checkpoint I/O
    # ------------------------------------------------------------------

    def _save_checkpoint(self, phase: str, step: int) -> None:
        path = os.path.join(self.config.save_dir, f"{phase}_step{step}.pt")
        torch.save(
            {
                "phase": phase,
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
            },
            path,
        )
        print(f"  → checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> dict:
        ckpt = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded checkpoint from {path} (phase={ckpt['phase']}, step={ckpt['step']})")
        return ckpt
