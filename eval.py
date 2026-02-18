#!/usr/bin/env python3
"""Evaluate a hierarchical MDLM checkpoint and report both PPL metrics.

Reports two perplexity numbers:
  1. **Hierarchical PPL** — uses the per-level noise weights from training,
     reflects the actual training objective (ELBO).
  2. **Flat PPL** — uses uniform noise across all levels, directly
     comparable to standard (non-hierarchical) MDLM.

Usage:
    # Evaluate a Phase-2 checkpoint on WikiHow test split
    python eval.py --checkpoint checkpoints/Phase2_step50000.pt

    # Evaluate on a JSON dataset with fewer timestep samples (faster)
    python eval.py --checkpoint ckpt.pt --dataset json --data test.json \
        --num_t_samples 200

    # Use a custom config (overrides checkpoint config)
    python eval.py --checkpoint ckpt.pt --config configs/train_config.yaml

    # Skip wandb logging
    python eval.py --checkpoint ckpt.pt --wandb_mode disabled
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import transformers
import yaml
from torch.utils.data import DataLoader

from data.hierarchy_dataset import HierarchyDataset, HierarchyCollator, build_demo_records
from data.wikihow_dataset import WikiHowHierarchyDataset
from training.trainer import HierarchicalMDLMTrainer, TrainerConfig


def load_config(args: argparse.Namespace) -> TrainerConfig:
    """Build a TrainerConfig from checkpoint, YAML, and CLI overrides.

    Priority (highest → lowest): CLI flags > YAML > checkpoint config.
    """
    cfg_dict: dict = {}

    # Start from checkpoint config if available
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        saved_cfg = ckpt.get("config")
        if saved_cfg is not None:
            from dataclasses import asdict
            cfg_dict = asdict(saved_cfg) if hasattr(saved_cfg, "__dataclass_fields__") else {}

    # Layer on YAML config
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f) or {}
        cfg_dict.update(yaml_cfg)

    # Layer on CLI overrides (skip None values)
    for key in vars(args):
        val = getattr(args, key)
        if val is not None and key in TrainerConfig.__dataclass_fields__:
            cfg_dict[key] = val

    return TrainerConfig(**{
        k: v for k, v in cfg_dict.items()
        if k in TrainerConfig.__dataclass_fields__
    })


def build_eval_dataloader(
    dataset_type: str,
    data_path: str | None,
    tokenizer: transformers.PreTrainedTokenizer,
    config: TrainerConfig,
    batch_size: int,
    wikihow_split: str = "test",
    wikihow_max_samples: int | None = None,
) -> DataLoader:
    """Build a DataLoader for evaluation (no shuffle, no drop_last)."""
    if dataset_type == "wikihow":
        dataset = WikiHowHierarchyDataset(
            tokenizer=tokenizer,
            max_length=config.max_length,
            split=wikihow_split,
            max_samples=wikihow_max_samples,
        )
    elif dataset_type == "json" and data_path and Path(data_path).exists():
        with open(data_path) as f:
            records = json.load(f)
        dataset = HierarchyDataset(
            records=records,
            tokenizer=tokenizer,
            max_length=config.max_length,
            num_levels=config.num_levels,
        )
    else:
        print("[eval] Falling back to synthetic demo data.")
        records = build_demo_records(n=200, num_levels=config.num_levels)
        dataset = HierarchyDataset(
            records=records,
            tokenizer=tokenizer,
            max_length=config.max_length,
            num_levels=config.num_levels,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=HierarchyCollator(),
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hierarchical MDLM — report both hierarchical and flat PPL",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt).")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config (overrides checkpoint config).")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset type: 'wikihow' (default) or 'json'.")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to JSON dataset file (for --dataset json).")
    parser.add_argument("--wikihow_split", type=str, default="test",
                        help="WikiHow split to evaluate on (default: test).")
    parser.add_argument("--wikihow_max_samples", type=int, default=None,
                        help="Cap WikiHow dataset size.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Evaluation batch size.")
    parser.add_argument("--num_t_samples", type=int, default=1000,
                        help="Number of timestep grid points for ELBO integration.")
    parser.add_argument("--noise_level_scales", type=float, nargs="+", default=None,
                        help="Override per-level noise scales (e.g. 0.5 1.0).")

    # wandb
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="disabled",
                        choices=["online", "offline", "disabled"],
                        help="W&B logging mode (default: disabled for eval).")

    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────
    config = load_config(args)
    # Force wandb mode from CLI (default: disabled)
    config.wandb_mode = args.wandb_mode
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.wandb_entity:
        config.wandb_entity = args.wandb_entity
    if args.wandb_run_name:
        config.wandb_run_name = args.wandb_run_name

    # ── Dataset ───────────────────────────────────────────────────────
    dataset_type = args.dataset
    if dataset_type is None:
        if args.config and Path(args.config).exists():
            with open(args.config) as f:
                raw = yaml.safe_load(f) or {}
            dataset_type = raw.get("dataset", "wikihow")
        else:
            dataset_type = "wikihow"

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloader = build_eval_dataloader(
        dataset_type=dataset_type,
        data_path=args.data,
        tokenizer=tokenizer,
        config=config,
        batch_size=args.batch_size,
        wikihow_split=args.wikihow_split,
        wikihow_max_samples=args.wikihow_max_samples,
    )

    # ── Trainer + checkpoint ──────────────────────────────────────────
    trainer = HierarchicalMDLMTrainer(config)
    trainer.load_checkpoint(args.checkpoint)

    # ── Evaluate both metrics ─────────────────────────────────────────
    log_wandb = args.wandb_mode != "disabled"
    results = trainer.evaluate(
        dataloader,
        num_t_samples=args.num_t_samples,
        log_to_wandb=log_wandb,
    )

    # ── Summary ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    hier = results["hierarchical"]
    flat = results["flat"]
    print(f"  Hierarchical PPL : {hier['ppl']:.2f}  (NLL {hier['nll']:.4f})")
    print(f"  Flat PPL         : {flat['ppl']:.2f}  (NLL {flat['nll']:.4f})")
    for k in range(config.num_levels):
        print(f"  Level {k}  hier PPL={hier[f'ppl_level{k}']:.2f}  "
              f"flat PPL={flat[f'ppl_level{k}']:.2f}")
    print(f"  Tokens evaluated : {hier['num_tokens']}")

    trainer.finish()


if __name__ == "__main__":
    main()
