"""Hierarchical MDLM Generator.

Combines:
    1. DDiTBackbone      – pretrained MDLM-small encoder
    2. HierarchyEmbedding – per-token level embedding added to input
    3. MultiHeadVocabOutput – K=2 independent output heads

Forward flow
────────────
    token_ids   (B, L)   noisy token indices
    sigma       (B,)     diffusion noise level
    hier_probs  (B, L, K) hierarchy probabilities (one-hot at train time)
                         ↓
    tok_emb = backbone.vocab_embed(token_ids)        (B, L, D)
    hier_emb = hierarchy_embedding(hier_probs)       (B, L, D)
    x = tok_emb + hier_emb
    hidden = backbone.encode_from_embeddings(x, sigma) (B, L, D)
    c = backbone.get_timestep_conditioning(sigma)      (B, D_c)
    logits_k = output_heads[k](hidden, c)  for k in 0..K-1

Training loss
─────────────
    For every masked token i with GT level k_i:
        loss_i = CE(logits_{k_i}[i], x0[i])
    Title tokens are never masked so they never contribute to the loss.

SUBS parameterization is applied per-head: mask-token logit → -inf, then
renormalize; unmasked positions get identity logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import DDiTBackbone
from models.hierarchy_embedding import HierarchyEmbedding
from models.output_heads import MultiHeadVocabOutput


class HierarchicalGenerator(nn.Module):
    """Full hierarchical masked diffusion language model.

    Args:
        pretrained_path: HuggingFace model id for the MDLM backbone.
        vocab_size: vocabulary size *including* the mask token.
        num_levels: number of hierarchy levels K.
        hidden_size: DDiT hidden dimension D.
        cond_dim: DDiT conditioning dimension.
        config_overrides: optional dict merged into the backbone config.
    """

    NEG_INF = -1_000_000.0

    def __init__(
        self,
        pretrained_path: str = "kuleshov-group/mdlm-owt",
        vocab_size: int = 50258,
        num_levels: int = 2,
        hidden_size: int = 768,
        cond_dim: int = 128,
        config_overrides: dict | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_levels = num_levels
        self.mask_index = vocab_size - 1  # last token is [MASK]

        # ── sub-modules ────────────────────────────────────────────────────
        self.backbone = DDiTBackbone(
            pretrained_path=pretrained_path,
            vocab_size=vocab_size,
            config_overrides=config_overrides,
        )
        self.hierarchy_embedding = HierarchyEmbedding(
            num_levels=num_levels,
            hidden_size=hidden_size,
        )
        self.output_heads = MultiHeadVocabOutput(
            num_levels=num_levels,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            cond_dim=cond_dim,
        )

    # ------------------------------------------------------------------
    #  Forward: raw logits
    # ------------------------------------------------------------------

    def forward(
        self,
        token_ids: torch.Tensor,
        sigma: torch.Tensor,
        hierarchy_probs: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Produce per-level logits for every token position.

        Args:
            token_ids:      (B, L) noisy token indices (may contain mask_index).
            sigma:          (B,) or (B, 1) noise levels.
            hierarchy_probs: (B, L, K) hierarchy probability distribution.

        Returns:
            List of K tensors, each (B, L, V) — raw logits.
        """
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)

        # 1. Token embedding + hierarchy embedding
        tok_emb = self.backbone.vocab_embed(token_ids)       # (B, L, D)
        hier_emb = self.hierarchy_embedding(hierarchy_probs)  # (B, L, D)
        x = tok_emb + hier_emb

        # 2. Transformer encoder
        hidden = self.backbone.encode_from_embeddings(x, sigma)  # (B, L, D)

        # 3. Timestep conditioning for output heads
        c = self.backbone.get_timestep_conditioning(sigma)       # (B, D_c)

        # 4. Per-level logits
        return self.output_heads(hidden, c)  # list of K × (B, L, V)

    # ------------------------------------------------------------------
    #  SUBS parameterization (applied per head)
    # ------------------------------------------------------------------

    def subs_parameterization(
        self, logits: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        """Apply the MDLM *subs* log-probability parameterization.

        Sets mask-token logit to -inf, renormalizes to log-probs, and forces
        unmasked positions to be deterministic (delta at the observed token).

        Args:
            logits: (B, L, V) raw logits from one head.
            xt:     (B, L)    noisy input tokens.
        Returns:
            (B, L, V) log-probabilities.
        """
        logits[:, :, self.mask_index] += self.NEG_INF
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        unmasked = xt != self.mask_index
        logits[unmasked] = self.NEG_INF
        logits[unmasked, xt[unmasked]] = 0.0
        return logits

    # ------------------------------------------------------------------
    #  Training loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        logits_per_level: list[torch.Tensor],
        xt: torch.Tensor,
        x0: torch.Tensor,
        hierarchy_labels: torch.Tensor,
        title_mask: torch.Tensor,
        per_level_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the hierarchical MDLM training loss with per-level weights.

        Each hierarchy level has its own noise scale, producing a different
        ELBO weight.  For each masked non-title token i with GT level k_i:

            loss_i = -per_level_weight[b, k_i] * log p_{k_i}(x0_i | x_t, sigma)

        Args:
            logits_per_level: list of K tensors (B, L, V), raw logits.
            xt:               (B, L) noisy tokens.
            x0:               (B, L) clean tokens.
            hierarchy_labels: (B, L) integer level labels in {0, ..., K-1}.
            title_mask:       (B, L) bool/float — 1 for title tokens (excluded).
            per_level_weight: (B, K) ELBO coefficient per level (from
                              HierarchicalNoiseSchedule.get_per_level_loss_weight).

        Returns:
            Scalar loss (mean over valid tokens).
        """
        B, L = xt.shape

        # Positions that contribute to loss: masked AND not title
        is_masked = (xt == self.mask_index).float()               # (B, L)
        loss_mask = is_masked * (1.0 - title_mask.float())        # (B, L)

        # Compute log p(x0 | xt) per head *without* materialising full (B,L,V)
        # log-prob tensors.  For masked positions the subs parameterization is:
        #   log p(x0) = logits[x0] - logsumexp(logits with mask-logit = -inf)
        # For unmasked positions the loss_mask zeros them out anyway, so we
        # just need a safe finite value (0.0).
        x0_idx = x0.unsqueeze(-1)                                # (B, L, 1)
        is_masked = (xt == self.mask_index)                       # (B, L)

        log_probs_list: list[torch.Tensor] = []
        for logits in logits_per_level:
            # Gather raw logit at x0 position: (B, L)
            logit_at_x0 = torch.gather(logits, dim=-1, index=x0_idx).squeeze(-1)
            # Set mask-token logit to -inf for logsumexp
            logits[:, :, self.mask_index] += self.NEG_INF
            lse = torch.logsumexp(logits, dim=-1)                 # (B, L)
            # log p(x0) at masked positions; 0 elsewhere (masked out later)
            lp = torch.where(is_masked, logit_at_x0 - lse, torch.zeros_like(lse))
            log_probs_list.append(lp)

        log_probs_at_x0 = torch.stack(log_probs_list, dim=-1)    # (B, L, K)

        # Select the log-prob from the head matching the GT level
        level_idx = hierarchy_labels.unsqueeze(-1).long()         # (B, L, 1)
        log_p = torch.gather(log_probs_at_x0, dim=-1, index=level_idx).squeeze(-1)
        # log_p: (B, L)

        # Per-token ELBO weight based on its level
        # per_level_weight: (B, K) → gather per token
        token_weight = torch.gather(
            per_level_weight, dim=1, index=hierarchy_labels.long()
        )  # (B, L)

        per_token_loss = -log_p * token_weight                    # (B, L)

        # Mask and average
        per_token_loss = per_token_loss * loss_mask
        num_valid = loss_mask.sum().clamp(min=1.0)
        return per_token_loss.sum() / num_valid
