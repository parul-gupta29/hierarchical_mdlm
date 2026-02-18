"""Two-phase hierarchical MDLM trainer.

Phase 1 — **New-component training**
    Freeze the entire DDiT backbone.  Fully fine-tune the randomly-initialised
    output heads and hierarchy embedding:
        • MultiHeadVocabOutput  (full — all parameters)
        • HierarchyEmbedding   (full — only 1 536 params)

Phase 2 — **Backbone LoRA + continued full fine-tuning**
    Attach LoRA adapters to backbone attention / MLP projections.  Train:
        • Backbone LoRA A/B matrices
        • MultiHeadVocabOutput  (full — continued from Phase 1)
        • HierarchyEmbedding   (full)

Both phases share the same diffusion training loop with **per-level noise**:
    1. Sample t ~ U(eps, 1)
    2. Compute base sigma(t) and per-level masking probabilities
       (level 0 / summary = low noise, level 1 / content = high noise)
    3. Mask tokens per-level (title tokens are exempt)
    4. Forward through the model → per-level logits
    5. Compute the hierarchical SUBS loss with per-level ELBO weights
    6. Backprop + optimizer step
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field

import math

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from models.hierarchical_generator import HierarchicalGenerator
from models.hierarchy_embedding import HierarchyEmbedding
from noise_schedule import Noise, HierarchicalNoiseSchedule, get_noise_schedule
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
    # Per-level noise scales:  level 0 = summary (low), level 1 = text (high)
    noise_level_scales: list[float] = field(default_factory=lambda: [0.5, 1.0])

    # --- LoRA ---
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    # Phase 2 targets: backbone attention + MLP layers (LoRA)
    phase2_lora_target_modules: list[str] = field(default_factory=lambda: [
        r".*\.attn_qkv$",
        r".*\.attn_out$",
        r".*\.mlp\.0$",
        r".*\.mlp\.2$",
    ])

    # --- phase 1 ---
    phase1_lr: float = 1e-4
    phase1_weight_decay: float = 0.0
    phase1_max_steps: int = 10_000
    phase1_batch_size: int = 8
    phase1_grad_clip: float = 1.0
    phase1_warmup_steps: int = 500
    phase1_min_lr: float = 1e-6
    phase1_unfreeze_last_n_blocks: int = 4  # unfreeze last N DDiT blocks

    # --- phase 2 ---
    phase2_lr: float = 3e-5
    phase2_weight_decay: float = 0.01
    phase2_max_steps: int = 50_000
    phase2_batch_size: int = 4
    phase2_grad_clip: float = 1.0
    phase2_warmup_steps: int = 1000
    phase2_min_lr: float = 1e-6

    # --- memory / mixed-precision ---
    use_amp: bool = True                # automatic mixed precision (fp16)
    grad_accum_steps: int = 4           # gradient accumulation steps

    # --- general ---
    device: str = "cuda"
    log_every: int = 100
    save_every: int = 5000
    save_dir: str = "checkpoints"
    seed: int = 42

    # --- wandb ---
    wandb_project: str = "hierarchical-mdlm"
    wandb_entity: str = ""
    wandb_run_name: str = ""
    wandb_mode: str = "online"  # "online", "offline", or "disabled"


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

        # ── noise schedule (per-level) ─────────────────────────────────
        base_noise: Noise = get_noise_schedule(config.noise_type)
        self.hier_noise = HierarchicalNoiseSchedule(
            base_noise=base_noise,
            level_scales=config.noise_level_scales,
        )
        self.hier_noise.to(config.device)

        os.makedirs(config.save_dir, exist_ok=True)

        # ── mixed-precision scaler ────────────────────────────────────
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)

        # ── wandb ─────────────────────────────────────────────────────
        self._init_wandb()

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases run."""
        cfg = self.config
        init_kwargs: dict = {
            "project": cfg.wandb_project,
            "config": asdict(cfg),
            "mode": cfg.wandb_mode,
        }
        if cfg.wandb_entity:
            init_kwargs["entity"] = cfg.wandb_entity
        if cfg.wandb_run_name:
            init_kwargs["name"] = cfg.wandb_run_name
        wandb.init(**init_kwargs)

    def finish(self) -> None:
        """Finish the current wandb run."""
        wandb.finish()

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

    def _q_xt_per_level(
        self,
        x0: torch.Tensor,
        hierarchy_labels: torch.Tensor,
        per_level_move_chance: torch.Tensor,
        title_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Corrupt clean tokens with **per-level** masking probabilities.

        Each token is masked independently with probability determined by
        its hierarchy level.  Title tokens are never masked.

        Args:
            x0:                    (B, L) clean token ids.
            hierarchy_labels:      (B, L) int level index per token.
            per_level_move_chance: (B, K) masking probability per level.
            title_mask:            (B, L) float, 1.0 at title positions.

        Returns:
            xt: (B, L) noisy tokens.
        """
        # Gather the move_chance for each token based on its level
        token_move_chance = torch.gather(
            per_level_move_chance,
            dim=1,
            index=hierarchy_labels.long(),
        )  # (B, L)

        rand = torch.rand_like(x0.float())
        move_indices = rand < token_move_chance              # (B, L)
        move_indices = move_indices & (title_mask < 0.5)     # protect title
        return torch.where(move_indices, self.mask_index, x0)

    # ------------------------------------------------------------------
    #  Single training step
    # ------------------------------------------------------------------

    def _forward_loss(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute forward pass and return the scalar loss (inside AMP context)."""
        device = self.config.device

        x0 = batch["input_ids"].to(device)                # (B, L)
        hierarchy_labels = batch["hierarchy_labels"].to(device)  # (B, L)
        title_mask = batch["title_mask"].to(device)        # (B, L)

        B = x0.shape[0]

        # 1. Sample timestep → base sigma
        t = self._sample_t(B, device)                             # (B,)
        sigma, dsigma = self.hier_noise(t)                        # (B,), (B,)

        # 2. Per-level masking probabilities
        per_level_mc = self.hier_noise.get_per_level_move_chance(sigma)  # (B, K)

        # 3. Corrupt with per-level noise (title stays clean)
        xt = self._q_xt_per_level(x0, hierarchy_labels, per_level_mc, title_mask)

        # 4. Build hierarchy probs (one-hot during training)
        hier_probs = HierarchyEmbedding.labels_to_onehot(
            hierarchy_labels, self.config.num_levels
        )  # (B, L, K)

        # 5. Forward
        logits_per_level = self.model(xt, sigma, hier_probs)

        # 6. Per-level ELBO weights
        per_level_weight = self.hier_noise.get_per_level_loss_weight(
            sigma, dsigma
        )  # (B, K)

        # 7. Loss (only on masked non-title tokens, weighted per level)
        return self.model.compute_loss(
            logits_per_level=logits_per_level,
            xt=xt,
            x0=x0,
            hierarchy_labels=hierarchy_labels,
            title_mask=title_mask,
            per_level_weight=per_level_weight,
        )

    # ------------------------------------------------------------------
    #  Phase loops
    # ------------------------------------------------------------------

    @staticmethod
    def _build_warmup_cosine_scheduler(
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float,
    ) -> torch.optim.lr_scheduler.LambdaLR | None:
        """Linear warmup then cosine decay to *min_lr*.

        Returns ``None`` when *warmup_steps* <= 0 (no scheduling).
        """
        if warmup_steps <= 0:
            return None

        base_lr = optimizer.param_groups[0]["lr"]

        def lr_lambda(current_step: int) -> float:
            # Linear warmup
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            # Cosine decay from base_lr → min_lr
            progress = (current_step - warmup_steps) / max(1, max_steps - warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Scale so that lr decays from base_lr to min_lr
            return max(min_lr / base_lr, cosine_decay)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _run_phase(
        self,
        phase_name: str,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        max_steps: int,
        grad_clip: float,
        scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    ) -> None:
        accum = self.config.grad_accum_steps
        step = 0          # optimizer steps (after accumulation)
        micro_step = 0    # micro-batch counter within an accumulation window
        epoch = 0
        accum_loss = 0.0

        optimizer.zero_grad()

        while step < max_steps:
            epoch += 1
            for batch in dataloader:
                if step >= max_steps:
                    break

                t0 = time.monotonic()
                self.model.train()

                # --- micro-step: forward + scaled backward (no optimizer step yet) ---
                with torch.amp.autocast("cuda", enabled=self.config.use_amp):
                    loss = self._forward_loss(batch)
                    loss_for_accum = loss / accum

                self.scaler.scale(loss_for_accum).backward()
                accum_loss += loss.item()
                micro_step += 1

                # --- optimizer step after `accum` micro-steps ---
                if micro_step % accum == 0:
                    self.scaler.unscale_(optimizer)
                    trainable_params = list(get_trainable_parameters(self.model))
                    if grad_clip > 0:
                        grad_norm = nn.utils.clip_grad_norm_(trainable_params, grad_clip).item()
                    else:
                        grad_norm = nn.utils.clip_grad_norm_(trainable_params, float("inf")).item()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()

                    if scheduler is not None:
                        scheduler.step()

                    step += 1
                    avg_loss = accum_loss / accum
                    step_time = time.monotonic() - t0
                    accum_loss = 0.0

                    lr = optimizer.param_groups[0]["lr"]
                    if step % self.config.log_every == 0:
                        print(f"[{phase_name}] step {step}/{max_steps}  "
                              f"loss={avg_loss:.4f}  lr={lr:.2e}")
                        wandb.log({
                            f"{phase_name}/loss": avg_loss,
                            f"{phase_name}/grad_norm": grad_norm,
                            f"{phase_name}/lr": lr,
                            f"{phase_name}/step_time_s": step_time,
                            f"{phase_name}/epoch": epoch,
                            "global_step": step,
                        }, step=step)

                    if step % self.config.save_every == 0:
                        self._save_checkpoint(phase_name, step)

        self._save_checkpoint(phase_name, step)
        print(f"[{phase_name}] Finished — {step} steps, {epoch} epochs.")

    # ------------------------------------------------------------------
    #  Phase 1: LoRA fine-tune hierarchy layers (output heads + embedding)
    # ------------------------------------------------------------------

    def run_phase1(self, dataloader: DataLoader) -> None:
        """Fully train output heads + hierarchy embedding; backbone frozen."""
        print("=" * 60)
        print("PHASE 1: Full fine-tune output heads + hierarchy embedding")
        print("=" * 60)

        # Freeze everything first
        for param in self.model.parameters():
            param.requires_grad_(False)

        # Output heads are randomly initialized — full fine-tune, not LoRA
        for param in self.model.output_heads.parameters():
            param.requires_grad_(True)
        # Hierarchy embedding is also randomly initialized (only 1536 params)
        for param in self.model.hierarchy_embedding.parameters():
            param.requires_grad_(True)

        # Unfreeze the last N DDiT blocks so the backbone can begin
        # adapting its representations to the hierarchy structure.
        n = self.config.phase1_unfreeze_last_n_blocks
        if n > 0:
            total_blocks = len(self.model.backbone.blocks)
            for block in self.model.backbone.blocks[total_blocks - n :]:
                for param in block.parameters():
                    param.requires_grad_(True)

        print_trainable_summary(self.model, "Phase 1")

        optimizer = torch.optim.AdamW(
            get_trainable_parameters(self.model),
            lr=self.config.phase1_lr,
            weight_decay=self.config.phase1_weight_decay,
        )

        scheduler = self._build_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=self.config.phase1_warmup_steps,
            max_steps=self.config.phase1_max_steps,
            min_lr=self.config.phase1_min_lr,
        )

        self._run_phase(
            "Phase1",
            dataloader,
            optimizer,
            self.config.phase1_max_steps,
            self.config.phase1_grad_clip,
            scheduler=scheduler,
        )

    # ------------------------------------------------------------------
    #  Phase 2: LoRA fine-tune entire architecture
    # ------------------------------------------------------------------

    def run_phase2(self, dataloader: DataLoader) -> None:
        """Add LoRA to backbone; keep output heads + hierarchy embedding fully trainable."""
        print("=" * 60)
        print("PHASE 2: Backbone LoRA + full fine-tune output heads")
        print("=" * 60)

        # Attach LoRA to backbone attention + MLP layers
        apply_lora_to_model(
            self.model.backbone,
            target_modules=self.config.phase2_lora_target_modules,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
        )
        self.model.to(self.config.device)

        # Freeze all original backbone weights, unfreeze LoRA adapters
        freeze_non_lora(self.model)
        unfreeze_all_lora(self.model)
        # Output heads are randomly initialized — full fine-tune
        for param in self.model.output_heads.parameters():
            param.requires_grad_(True)
        # Hierarchy embedding stays fully trainable
        for param in self.model.hierarchy_embedding.parameters():
            param.requires_grad_(True)

        print_trainable_summary(self.model, "Phase 2")

        optimizer = torch.optim.AdamW(
            get_trainable_parameters(self.model),
            lr=self.config.phase2_lr,
            weight_decay=self.config.phase2_weight_decay,
        )

        scheduler = self._build_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=self.config.phase2_warmup_steps,
            max_steps=self.config.phase2_max_steps,
            min_lr=self.config.phase2_min_lr,
        )

        self._run_phase(
            "Phase2",
            dataloader,
            optimizer,
            self.config.phase2_max_steps,
            self.config.phase2_grad_clip,
            scheduler=scheduler,
        )

    # ------------------------------------------------------------------
    #  Test-set perplexity evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_perplexity(
        self,
        dataloader: DataLoader,
        num_t_samples: int = 1000,
        flat: bool = False,
    ) -> dict[str, float]:
        """Compute test-set perplexity via Monte Carlo ELBO integration.

        Estimates the negative log-likelihood upper bound by averaging the
        ELBO loss over a uniform grid of timesteps, then exponentiates.

        Args:
            dataloader: test-set DataLoader (should NOT shuffle or drop_last).
            num_t_samples: number of timesteps in the uniform grid [eps, 1].
                More samples → tighter estimate, but linearly more compute.
            flat: if True, use uniform noise scales (1.0 for all levels)
                instead of the per-level scales.  This makes the result
                directly comparable to standard (non-hierarchical) MDLM.

        Returns:
            Dict with keys:
                ``"ppl"``       — overall perplexity (exp of mean NLL)
                ``"nll"``       — mean NLL per token
                ``"ppl_level0"``— perplexity on level-0 (summary) tokens only
                ``"ppl_level1"``— perplexity on level-1 (text) tokens only
                ``"nll_level0"``— NLL on level-0 tokens
                ``"nll_level1"``— NLL on level-1 tokens
                ``"num_tokens"``— total valid tokens evaluated
        """
        device = self.config.device
        eps = self.config.sampling_eps
        K = self.config.num_levels

        self.model.eval()

        # Build the noise schedule for evaluation
        if flat:
            flat_noise = HierarchicalNoiseSchedule(
                base_noise=self.hier_noise.base_noise,
                level_scales=[1.0] * K,
            )
            flat_noise.to(device)
            eval_noise = flat_noise
        else:
            eval_noise = self.hier_noise

        # Uniform timestep grid
        t_grid = torch.linspace(eps, 1.0, num_t_samples, device=device)

        # Accumulators: total weighted NLL and token counts
        total_nll = 0.0
        total_tokens = 0
        per_level_nll = [0.0] * K
        per_level_tokens = [0] * K

        num_batches = 0
        for batch in dataloader:
            x0 = batch["input_ids"].to(device)
            hierarchy_labels = batch["hierarchy_labels"].to(device)
            title_mask = batch["title_mask"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            B, L = x0.shape

            # Valid tokens: non-pad and non-title
            valid_mask = attn_mask * (1.0 - title_mask)  # (B, L)

            # Per-level valid masks
            level_masks = []
            for k in range(K):
                lm = valid_mask * (hierarchy_labels == k).float()
                level_masks.append(lm)

            # Accumulate ELBO across timestep grid
            batch_nll_sum = 0.0
            batch_level_nll = [0.0] * K

            for t_scalar in t_grid:
                t = t_scalar.expand(B)
                sigma, dsigma = eval_noise(t)

                per_level_mc = eval_noise.get_per_level_move_chance(sigma)
                xt = self._q_xt_per_level(
                    x0, hierarchy_labels, per_level_mc, title_mask,
                )

                hier_probs = HierarchyEmbedding.labels_to_onehot(
                    hierarchy_labels, K,
                )

                with torch.amp.autocast("cuda", enabled=self.config.use_amp):
                    logits_per_level = self.model(xt, sigma, hier_probs)

                per_level_weight = eval_noise.get_per_level_loss_weight(
                    sigma, dsigma,
                )

                # --- per-token NLL (without reducing to scalar) ---
                x0_idx = x0.unsqueeze(-1)
                is_masked = (xt == self.mask_index)
                V = logits_per_level[0].shape[-1]
                mask_bias = logits_per_level[0].new_zeros(V)
                mask_bias[self.mask_index] = self.model.NEG_INF

                log_probs_list = []
                for logits in logits_per_level:
                    logit_at_x0 = torch.gather(
                        logits, dim=-1, index=x0_idx,
                    ).squeeze(-1)
                    lse = torch.logsumexp(logits + mask_bias, dim=-1)
                    lp = torch.where(
                        is_masked,
                        logit_at_x0 - lse,
                        torch.zeros_like(lse),
                    )
                    log_probs_list.append(lp)

                log_probs_at_x0 = torch.stack(log_probs_list, dim=-1)
                level_idx = hierarchy_labels.unsqueeze(-1).long()
                log_p = torch.gather(
                    log_probs_at_x0, dim=-1, index=level_idx,
                ).squeeze(-1)

                token_weight = torch.gather(
                    per_level_weight, dim=1, index=hierarchy_labels.long(),
                )

                per_token_nll = -log_p * token_weight  # (B, L)

                # Overall
                batch_nll_sum += (per_token_nll * valid_mask).sum().item()

                # Per-level
                for k in range(K):
                    batch_level_nll[k] += (
                        per_token_nll * level_masks[k]
                    ).sum().item()

            # Average over timestep grid, accumulate
            n_valid = valid_mask.sum().item()
            total_nll += batch_nll_sum / num_t_samples
            total_tokens += n_valid

            for k in range(K):
                n_k = level_masks[k].sum().item()
                per_level_nll[k] += batch_level_nll[k] / num_t_samples
                per_level_tokens[k] += n_k

            num_batches += 1
            if num_batches % 10 == 0:
                running_nll = total_nll / max(total_tokens, 1)
                print(f"  [eval] {num_batches} batches | "
                      f"running NLL={running_nll:.4f}  "
                      f"PPL={math.exp(running_nll):.2f}")

        # Final metrics
        avg_nll = total_nll / max(total_tokens, 1)
        results = {
            "ppl": math.exp(avg_nll),
            "nll": avg_nll,
            "num_tokens": total_tokens,
        }
        for k in range(K):
            nll_k = per_level_nll[k] / max(per_level_tokens[k], 1)
            results[f"nll_level{k}"] = nll_k
            results[f"ppl_level{k}"] = math.exp(nll_k)

        return results

    # ------------------------------------------------------------------
    #  Convenience: evaluate both hierarchical and flat PPL
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataloader: DataLoader,
        num_t_samples: int = 1000,
        log_to_wandb: bool = True,
    ) -> dict[str, dict[str, float]]:
        """Compute both hierarchical and flat perplexity and report results.

        Args:
            dataloader: test-set DataLoader.
            num_t_samples: timestep grid size for ELBO integration.
            log_to_wandb: if True, log both result sets to wandb.

        Returns:
            ``{"hierarchical": {...}, "flat": {...}}`` where each value
            is the dict returned by :meth:`compute_perplexity`.
        """
        print("=" * 60)
        print("EVALUATION: Hierarchical PPL (per-level noise weights)")
        print("=" * 60)
        hier_results = self.compute_perplexity(
            dataloader, num_t_samples=num_t_samples, flat=False,
        )
        print(f"  Hierarchical PPL = {hier_results['ppl']:.2f}  "
              f"(NLL = {hier_results['nll']:.4f})")
        for k in range(self.config.num_levels):
            print(f"    level {k}: PPL = {hier_results[f'ppl_level{k}']:.2f}  "
                  f"NLL = {hier_results[f'nll_level{k}']:.4f}")

        print()
        print("=" * 60)
        print("EVALUATION: Flat PPL (uniform weights, comparable to MDLM)")
        print("=" * 60)
        flat_results = self.compute_perplexity(
            dataloader, num_t_samples=num_t_samples, flat=True,
        )
        print(f"  Flat PPL = {flat_results['ppl']:.2f}  "
              f"(NLL = {flat_results['nll']:.4f})")
        for k in range(self.config.num_levels):
            print(f"    level {k}: PPL = {flat_results[f'ppl_level{k}']:.2f}  "
                  f"NLL = {flat_results[f'nll_level{k}']:.4f}")

        if log_to_wandb:
            wandb.log({
                "eval/hier_ppl": hier_results["ppl"],
                "eval/hier_nll": hier_results["nll"],
                "eval/flat_ppl": flat_results["ppl"],
                "eval/flat_nll": flat_results["nll"],
                **{f"eval/hier_ppl_level{k}": hier_results[f"ppl_level{k}"]
                   for k in range(self.config.num_levels)},
                **{f"eval/hier_nll_level{k}": hier_results[f"nll_level{k}"]
                   for k in range(self.config.num_levels)},
                **{f"eval/flat_ppl_level{k}": flat_results[f"ppl_level{k}"]
                   for k in range(self.config.num_levels)},
                **{f"eval/flat_nll_level{k}": flat_results[f"nll_level{k}"]
                   for k in range(self.config.num_levels)},
                "eval/num_tokens": hier_results["num_tokens"],
            })

        return {"hierarchical": hier_results, "flat": flat_results}

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
