"""WikiHow dataset loader for hierarchical MDLM.

Loads ``gursi26/wikihow-cleaned`` from HuggingFace and maps it to the
hierarchical format:

    title   → prompt (never masked during training / inference)
    summary → hierarchy level 0  (low noise)
    text    → hierarchy level 1  (high noise)

The flat sequence layout is:

    [title tokens] [summary tokens] [text tokens] [PAD ...]

No separator tokens are inserted between segments.
"""

from __future__ import annotations

import logging
import torch
from torch.utils.data import Dataset
import transformers
from datasets import load_dataset

# Suppress the "Token indices sequence length is longer than ..." warning
# emitted by HuggingFace tokenizers.  Our code truncates *after* encoding.
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


class WikiHowHierarchyDataset(Dataset):
    """Torch dataset wrapping ``gursi26/wikihow-cleaned``.

    Each sample is tokenised into:
        input_ids        (L,)  clean token ids
        hierarchy_labels (L,)  0 for title/summary tokens, 1 for text tokens
        title_mask       (L,)  1.0 at title positions
        attention_mask   (L,)  1.0 at non-pad positions

    Args:
        split: HuggingFace split name (default ``"train"``).
        tokenizer: a PreTrainedTokenizer (GPT-2 by default).
        max_length: maximum sequence length in tokens.
        max_samples: optional cap on dataset size (for debugging).
    """

    NUM_LEVELS = 2   # 0 = summary, 1 = text

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        max_length: int = 1024,
        split: str = "train",
        max_samples: int | None = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"[WikiHowHierarchyDataset] Loading gursi26/wikihow-cleaned "
              f"(split={split}) ...")
        ds = load_dataset("gursi26/wikihow-cleaned", split=split)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        self.data = ds
        print(f"[WikiHowHierarchyDataset] Loaded {len(self.data)} samples.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]
        return self._encode(
            title=row["title"],
            summary=row["summary"],
            text=row["text"],
        )

    # ------------------------------------------------------------------

    def _encode(self, title: str, summary: str, text: str) -> dict:
        """Tokenise one WikiHow article into the hierarchical format."""
        # Some rows may have None fields — coerce to empty string.
        title = title or ""
        summary = summary or ""
        text = text or ""
        title_ids = self.tokenizer.encode(title, add_special_tokens=False)
        summary_ids = self.tokenizer.encode(summary, add_special_tokens=False)
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)

        # --- Truncation (preserve title fully, then summary, then text) ---
        budget = self.max_length - len(title_ids)
        if budget < 0:
            title_ids = title_ids[: self.max_length]
            summary_ids, text_ids = [], []
            budget = 0

        if len(summary_ids) > budget:
            summary_ids = summary_ids[:budget]
            text_ids = []
        else:
            remaining = budget - len(summary_ids)
            text_ids = text_ids[:remaining]

        # --- Build flat sequence ---
        all_ids = title_ids + summary_ids + text_ids
        seq_len = len(all_ids)

        n_title = len(title_ids)
        n_summary = len(summary_ids)
        n_text = len(text_ids)

        # hierarchy_labels: title → 0, summary → 0, text → 1
        hierarchy_labels = [0] * n_title + [0] * n_summary + [1] * n_text

        # title_mask: 1 only for actual title positions
        title_mask = [1.0] * n_title + [0.0] * (n_summary + n_text)

        # attention_mask: 1 for all real tokens
        attention_mask = [1.0] * seq_len

        # --- Pad to max_length ---
        pad_len = self.max_length - seq_len
        pad_id = self.tokenizer.pad_token_id or 0
        all_ids += [pad_id] * pad_len
        hierarchy_labels += [0] * pad_len
        title_mask += [0.0] * pad_len
        attention_mask += [0.0] * pad_len

        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "hierarchy_labels": torch.tensor(hierarchy_labels, dtype=torch.long),
            "title_mask": torch.tensor(title_mask, dtype=torch.float32),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
        }
