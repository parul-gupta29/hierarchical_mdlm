"""Dataset and collator for hierarchical MDLM training.

Each sample consists of:
    - **title**:  a short text prefix that is *always visible* (never masked).
    - **body segments**: one or more text segments, each annotated with an
      integer hierarchy level in {0, 1, ..., K-1}.

Example raw record (JSON / dict):
    {
        "title": "Machine Learning Basics",
        "segments": [
            {"text": "Supervised learning maps inputs to outputs.", "level": 0},
            {"text": "Linear regression minimises squared error.", "level": 1}
        ]
    }

The dataset tokenises everything into a single flat sequence:

    [title tokens] [segment-0 tokens] [segment-1 tokens] ... [PAD ...]

with parallel arrays for:
    - ``hierarchy_labels``  (int per token: level index, 0 for title & pad)
    - ``title_mask``        (1 for title positions, 0 elsewhere)
    - ``attention_mask``    (1 for real tokens, 0 for padding)

There is **no** separator token inserted between segments.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
import transformers


class HierarchyDataset(Dataset):
    """Torch dataset that tokenises title + hierarchical body segments.

    Args:
        records: list of dicts, each with "title" (str) and "segments"
                 (list of {"text": str, "level": int}).
        tokenizer: a HuggingFace PreTrainedTokenizer (GPT-2 by default).
        max_length: maximum sequence length (tokens).  Sequences are
                    truncated from the right; the title is never truncated.
        num_levels: K — number of hierarchy levels (for validation only).
    """

    def __init__(
        self,
        records: list[dict],
        tokenizer: transformers.PreTrainedTokenizer,
        max_length: int = 1024,
        num_levels: int = 2,
    ):
        super().__init__()
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_levels = num_levels

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        return self.encode_record(record)

    # ------------------------------------------------------------------

    def encode_record(self, record: dict) -> dict:
        """Tokenise a single record and build all auxiliary tensors.

        Returns dict with keys:
            input_ids        (L,)  int64   token ids (clean, before noise)
            hierarchy_labels (L,)  int64   per-token level index
            title_mask       (L,)  float32 1.0 at title positions
            attention_mask   (L,)  float32 1.0 at non-pad positions
        """
        title_text = record["title"]
        segments = record["segments"]

        # Tokenise title (no special tokens)
        title_ids = self.tokenizer.encode(title_text, add_special_tokens=False)

        # Tokenise each body segment and record per-token level
        body_ids: list[int] = []
        body_levels: list[int] = []
        for seg in segments:
            seg_ids = self.tokenizer.encode(seg["text"], add_special_tokens=False)
            body_ids.extend(seg_ids)
            body_levels.extend([seg["level"]] * len(seg_ids))

        # Truncate body to fit within max_length (title is preserved)
        max_body = self.max_length - len(title_ids)
        if max_body < 0:
            # Title itself exceeds max_length — truncate title
            title_ids = title_ids[: self.max_length]
            body_ids = []
            body_levels = []
            max_body = 0
        body_ids = body_ids[:max_body]
        body_levels = body_levels[:max_body]

        # Build full sequence
        all_ids = title_ids + body_ids
        seq_len = len(all_ids)

        # Hierarchy labels: title gets level 0 (arbitrary, won't affect loss)
        hierarchy_labels = [0] * len(title_ids) + body_levels

        # Title mask: 1 for title positions
        title_mask = [1.0] * len(title_ids) + [0.0] * len(body_ids)

        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = [1.0] * seq_len

        # Pad to max_length
        pad_len = self.max_length - seq_len
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
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


class HierarchyCollator:
    """Simple collator that stacks pre-padded tensors into a batch."""

    def __call__(self, batch: list[dict]) -> dict:
        return {
            key: torch.stack([sample[key] for sample in batch])
            for key in batch[0]
        }


# ---------------------------------------------------------------------------
#  Utility: build a demo / synthetic dataset for smoke testing
# ---------------------------------------------------------------------------

def build_demo_records(n: int = 100, num_levels: int = 2) -> list[dict]:
    """Create *n* synthetic records for quick testing."""
    import random

    titles = [
        "Introduction to Python",
        "Data Structures Overview",
        "Machine Learning Primer",
        "Web Development Guide",
        "Database Fundamentals",
    ]
    level_texts = {
        0: [
            "This section covers the high-level concepts and background.",
            "We begin with a broad overview of the topic.",
            "Understanding the fundamentals is essential.",
        ],
        1: [
            "Here we dive into the specific implementation details.",
            "The algorithm proceeds through the following steps.",
            "Consider the edge cases and error handling.",
        ],
    }
    records = []
    for _ in range(n):
        title = random.choice(titles)
        n_segs = random.randint(1, 4)
        segments = []
        for _ in range(n_segs):
            lvl = random.randint(0, num_levels - 1)
            txt = random.choice(level_texts.get(lvl, level_texts[0]))
            segments.append({"text": txt, "level": lvl})
        records.append({"title": title, "segments": segments})
    return records
