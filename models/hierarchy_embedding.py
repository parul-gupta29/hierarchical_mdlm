"""Hierarchy embedding layer.

Each token in the sequence belongs to one of K hierarchy levels.
This module produces a per-token embedding that is *added* to the DDiT token
embedding before the transformer blocks.

During **training** the ground-truth hierarchy labels are available, so
``hierarchy_probs`` is a one-hot tensor of shape (B, L, K).  The embedding
lookup is therefore exact.

During **inference** the hierarchy predictor supplies soft probabilities
over the K levels, and the embedding is the probability-weighted average
of all K level embeddings.

    h_embed_i = sum_k  prob_{i,k} * E_k          (B, L, D)
"""

import torch
import torch.nn as nn


class HierarchyEmbedding(nn.Module):
    """Learnable embedding table for hierarchy levels.

    Args:
        num_levels: Number of hierarchy levels (K). Default 2.
        hidden_size: Dimension of each level embedding (must match DDiT D).
    """

    def __init__(self, num_levels: int = 2, hidden_size: int = 768):
        super().__init__()
        self.num_levels = num_levels
        self.hidden_size = hidden_size
        # (K, D)  — one embedding vector per level
        self.level_embeddings = nn.Embedding(num_levels, hidden_size)
        nn.init.normal_(self.level_embeddings.weight, std=0.02)

    # ----- helpers for building probability tensors -------------------------

    @staticmethod
    def labels_to_onehot(
        labels: torch.Tensor, num_levels: int
    ) -> torch.Tensor:
        """Convert integer labels (B, L) to one-hot (B, L, K).

        Used at training time when ground-truth hierarchy labels are known.
        """
        return torch.nn.functional.one_hot(
            labels.long(), num_classes=num_levels
        ).float()

    # ----- forward ----------------------------------------------------------

    def forward(self, hierarchy_probs: torch.Tensor) -> torch.Tensor:
        """Compute the weighted-average hierarchy embedding per token.

        Args:
            hierarchy_probs: (B, L, K) soft (or one-hot) distribution over
                hierarchy levels for every token in the sequence.

        Returns:
            (B, L, D) hierarchy embedding to be *added* to the token embedding.
        """
        # level_embeddings.weight is (K, D)
        # hierarchy_probs is (B, L, K)
        # result: (B, L, D)
        return torch.matmul(hierarchy_probs, self.level_embeddings.weight)
