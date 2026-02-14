"""Multi-head vocabulary output layers.

For K=2 hierarchy levels we maintain two independent output heads.  Each head
receives the DDiT hidden state and the timestep conditioning vector and
produces logits over the full vocabulary.

Architecture per head (mirrors the original DDitFinalLayer):
    AdaLN(hidden, c)  -->  Linear(D, D)  -->  GELU  -->  Linear(D, V)

During training the loss for token i is computed **only** through the head
that matches its ground-truth hierarchy level k_i.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import LayerNorm, modulate_fused


class VocabHead(nn.Module):
    """Single output head: AdaLN + MLP -> vocab logits.

    This is a richer version of the original DDitFinalLayer, adding a hidden
    non-linear layer between the adaptive norm and the final projection so
    that each head can specialise to its hierarchy level.
    """

    def __init__(self, hidden_size: int, vocab_size: int, cond_dim: int):
        super().__init__()
        self.norm = LayerNorm(hidden_size)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, vocab_size, bias=True),
        )
        # Zero-init the final projection so the head starts as identity-ish
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) hidden states from DDiT encoder.
            c: (B, D_cond) timestep conditioning vector.
        Returns:
            (B, L, V) logits over vocabulary.
        """
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm(x), shift, scale)
        return self.mlp(x)


class MultiHeadVocabOutput(nn.Module):
    """Container for K independent VocabHead modules.

    Args:
        num_levels: K — number of hierarchy levels.
        hidden_size: D — DDiT hidden dimension.
        vocab_size: V — vocabulary size (including mask token).
        cond_dim: dimension of the timestep conditioning vector.
    """

    def __init__(
        self,
        num_levels: int = 2,
        hidden_size: int = 768,
        vocab_size: int = 50258,
        cond_dim: int = 128,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.heads = nn.ModuleList(
            [VocabHead(hidden_size, vocab_size, cond_dim) for _ in range(num_levels)]
        )

    def forward(
        self, hidden: torch.Tensor, c: torch.Tensor
    ) -> list[torch.Tensor]:
        """Run all K heads.

        Args:
            hidden: (B, L, D)
            c:      (B, D_cond)

        Returns:
            List of K tensors, each (B, L, V).
        """
        return [head(hidden, c) for head in self.heads]

    def forward_level(
        self, hidden: torch.Tensor, c: torch.Tensor, level: int
    ) -> torch.Tensor:
        """Run a single head for the given level."""
        return self.heads[level](hidden, c)
