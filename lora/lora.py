"""LoRA (Low-Rank Adaptation) utilities for hierarchical MDLM.

Two training phases:
    Phase 1 — train *only* the new components (hierarchy embedding + output
              heads) while the backbone is completely frozen.
    Phase 2 — attach LoRA adapters to the backbone and fine-tune everything.

We implement a lightweight LoRA layer that wraps ``nn.Linear`` modules.
This avoids a hard dependency on the ``peft`` library while remaining
compatible with it (``peft`` can be swapped in via config if desired).
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Core LoRA linear layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with a low-rank residual.

        out = W x + b  +  (alpha / r) * B @ A @ x

    Only A and B are trainable; W and b are frozen.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Frozen original weight and bias
        self.weight = original.weight
        self.weight.requires_grad_(False)
        self.bias = original.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)

        # Low-rank trainable matrices
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(
            F.linear(self.lora_dropout(x), self.lora_A),
            self.lora_B,
        ) * self.scaling
        return base + lora

    @property
    def lora_parameters(self) -> list[nn.Parameter]:
        return [self.lora_A, self.lora_B]


# ---------------------------------------------------------------------------
#  Application helpers
# ---------------------------------------------------------------------------

def apply_lora_to_model(
    model: nn.Module,
    target_modules: list[str] | None = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> list[LoRALinear]:
    """Replace matching ``nn.Linear`` layers with ``LoRALinear``.

    Args:
        model: the nn.Module to patch (e.g. backbone.blocks).
        target_modules: list of regex patterns matching parameter *names*
            (relative to ``model``).  Defaults to attention + MLP projections
            inside DDiTBlock.
        rank: LoRA rank r.
        alpha: LoRA scaling factor.
        dropout: LoRA dropout.

    Returns:
        List of all newly created LoRALinear modules (handy for optimizer
        param groups).
    """
    if target_modules is None:
        target_modules = [
            r".*\.attn_qkv$",
            r".*\.attn_out$",
            r".*\.mlp\.0$",   # first linear in MLP
            r".*\.mlp\.2$",   # second linear in MLP
        ]

    lora_modules: list[LoRALinear] = []

    # Collect (parent, attr_name, module) triples for eligible linears
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(re.match(pat, name) for pat in target_modules):
            # Split name into parent-path and attribute
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(model.named_modules())[parts[0]]
                attr = parts[1]
            else:
                parent = model
                attr = parts[0]
            replacements.append((parent, attr, module))

    for parent, attr, linear in replacements:
        lora_layer = LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr, lora_layer)
        lora_modules.append(lora_layer)

    return lora_modules


def freeze_non_lora(model: nn.Module) -> None:
    """Freeze every parameter that is NOT a LoRA A/B matrix."""
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad_(False)


def unfreeze_all_lora(model: nn.Module) -> None:
    """Ensure all LoRA parameters are trainable."""
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)


def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Collect all LoRA A/B parameters from the model."""
    params = []
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            params.append(param)
    return params


def get_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return only parameters with requires_grad=True."""
    return [p for p in model.parameters() if p.requires_grad]


def print_trainable_summary(model: nn.Module, label: str = "") -> None:
    """Print a summary of total vs trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total > 0 else 0.0
    print(
        f"[{label}] Trainable: {trainable:,} / {total:,} "
        f"({pct:.2f}%)"
    )
