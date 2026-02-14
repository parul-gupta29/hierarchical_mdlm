#!/usr/bin/env python3
"""Entry point for hierarchical MDLM training.

Usage:
    # Train with defaults (both phases)
    python train.py

    # Override via CLI
    python train.py --config configs/train_config.yaml \\
                    --phase1_max_steps 5000 \\
                    --phase2_lr 1e-5

    # Run only phase 1
    python train.py --phase 1

    # Run only phase 2 (assumes phase 1 checkpoint exists)
    python train.py --phase 2 --resume checkpoints/Phase1_step10000.pt
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
from training.trainer import HierarchicalMDLMTrainer, TrainerConfig


def load_config(args: argparse.Namespace) -> TrainerConfig:
    """Build a TrainerConfig from YAML file + CLI overrides."""
    cfg_dict: dict = {}
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            cfg_dict = yaml.safe_load(f)

    # Apply CLI overrides
    for key in vars(args):
        val = getattr(args, key)
        if val is not None and key in TrainerConfig.__dataclass_fields__:
            cfg_dict[key] = val

    return TrainerConfig(**{
        k: v for k, v in cfg_dict.items()
        if k in TrainerConfig.__dataclass_fields__
    })


def build_dataloader(
    data_path: str | None,
    tokenizer: transformers.PreTrainedTokenizer,
    config: TrainerConfig,
    batch_size: int,
) -> DataLoader:
    """Load the dataset from a JSON file or build a demo dataset."""
    if data_path and Path(data_path).exists():
        with open(data_path) as f:
            records = json.load(f)
    else:
        print("[train] No data_path provided — using synthetic demo data.")
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
        shuffle=True,
        collate_fn=HierarchyCollator(),
        num_workers=0,
        drop_last=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Hierarchical MDLM Training")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to JSON dataset file.")
    parser.add_argument("--phase", type=int, default=None,
                        help="Run only this phase (1 or 2). Default: both.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to resume from.")
    # Allow any TrainerConfig field as a CLI override
    for field_name, field_obj in TrainerConfig.__dataclass_fields__.items():
        tp = field_obj.type
        if tp in ("int", int):
            parser.add_argument(f"--{field_name}", type=int, default=None)
        elif tp in ("float", float):
            parser.add_argument(f"--{field_name}", type=float, default=None)
        elif tp in ("str", str):
            parser.add_argument(f"--{field_name}", type=str, default=None)
        elif tp in ("bool", bool):
            parser.add_argument(f"--{field_name}", type=lambda v: v.lower() == "true",
                                default=None)
    args = parser.parse_args()

    config = load_config(args)

    # Tokenizer (GPT-2)
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Trainer
    trainer = HierarchicalMDLMTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Phase dispatch
    if args.phase == 1:
        dl = build_dataloader(args.data, tokenizer, config, config.phase1_batch_size)
        trainer.run_phase1(dl)
    elif args.phase == 2:
        dl = build_dataloader(args.data, tokenizer, config, config.phase2_batch_size)
        trainer.run_phase2(dl)
    else:
        dl = build_dataloader(args.data, tokenizer, config, config.phase1_batch_size)
        trainer.run_phase1(dl)
        dl2 = build_dataloader(args.data, tokenizer, config, config.phase2_batch_size)
        trainer.run_phase2(dl2)

    print("Training complete.")


if __name__ == "__main__":
    main()
