#!/usr/bin/env python3
"""Entry point for hierarchical MDLM training on WikiHow.

Usage:
    # Train on WikiHow (both phases, default config)
    python train.py

    # Train on WikiHow with limited samples for debugging
    python train.py --wikihow_max_samples 1000 --phase1_max_steps 200

    # Train on a custom JSON dataset
    python train.py --dataset json --data my_data.json

    # Override noise scales via CLI
    python train.py --noise_level_scales 0.3 0.8

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
from data.wikihow_dataset import WikiHowHierarchyDataset
from training.trainer import HierarchicalMDLMTrainer, TrainerConfig


def load_config(args: argparse.Namespace) -> TrainerConfig:
    """Build a TrainerConfig from YAML file + CLI overrides."""
    cfg_dict: dict = {}
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            cfg_dict = yaml.safe_load(f) or {}

    # Apply CLI overrides (skip None values)
    for key in vars(args):
        val = getattr(args, key)
        if val is not None and key in TrainerConfig.__dataclass_fields__:
            cfg_dict[key] = val

    # Filter to only valid config fields
    return TrainerConfig(**{
        k: v for k, v in cfg_dict.items()
        if k in TrainerConfig.__dataclass_fields__
    })


def build_dataloader(
    dataset_type: str,
    data_path: str | None,
    tokenizer: transformers.PreTrainedTokenizer,
    config: TrainerConfig,
    batch_size: int,
    wikihow_max_samples: int | None = None,
) -> DataLoader:
    """Build a DataLoader for the chosen dataset.

    Args:
        dataset_type: "wikihow" or "json".
        data_path: path to a JSON file (only used when dataset_type="json").
        tokenizer: GPT-2 tokenizer.
        config: trainer configuration.
        batch_size: batch size.
        wikihow_max_samples: optional cap on WikiHow dataset size.
    """
    if dataset_type == "wikihow":
        dataset = WikiHowHierarchyDataset(
            tokenizer=tokenizer,
            max_length=config.max_length,
            split="train",
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
        print("[train] Falling back to synthetic demo data.")
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
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Hierarchical MDLM Training")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset type: 'wikihow' (default) or 'json'.")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to JSON dataset file (for --dataset json).")
    parser.add_argument("--wikihow_max_samples", type=int, default=None,
                        help="Cap WikiHow dataset size (for debugging).")
    parser.add_argument("--phase", type=int, default=None,
                        help="Run only this phase (1 or 2). Default: both.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to resume from.")
    parser.add_argument("--noise_level_scales", type=float, nargs="+", default=None,
                        help="Per-level noise scales (e.g. 0.5 1.0).")

    # wandb overrides
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name (default: hierarchical-mdlm).")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B team/entity name.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (auto-generated if empty).")
    parser.add_argument("--wandb_mode", type=str, default=None,
                        choices=["online", "offline", "disabled"],
                        help="W&B logging mode.")

    # Allow any scalar TrainerConfig field as a CLI override
    for field_name, field_obj in TrainerConfig.__dataclass_fields__.items():
        if field_name in ("noise_level_scales", "phase1_lora_target_modules",
                          "phase2_lora_target_modules",
                          "wandb_project", "wandb_entity",
                          "wandb_run_name", "wandb_mode"):
            continue  # handled above or complex type
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

    # Load config from YAML + CLI
    config = load_config(args)

    # Determine dataset type from CLI or config YAML
    dataset_type = args.dataset
    if dataset_type is None:
        # Check YAML for 'dataset' key (not a TrainerConfig field)
        if args.config and Path(args.config).exists():
            with open(args.config) as f:
                raw = yaml.safe_load(f) or {}
            dataset_type = raw.get("dataset", "wikihow")
        else:
            dataset_type = "wikihow"

    # WikiHow max samples from CLI or config YAML
    wikihow_max_samples = args.wikihow_max_samples
    if wikihow_max_samples is None and args.config and Path(args.config).exists():
        with open(args.config) as f:
            raw = yaml.safe_load(f) or {}
        wikihow_max_samples = raw.get("wikihow_max_samples")

    # Tokenizer (GPT-2)
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Trainer
    trainer = HierarchicalMDLMTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Phase dispatch
    def make_dl(batch_size: int) -> DataLoader:
        return build_dataloader(
            dataset_type=dataset_type,
            data_path=args.data,
            tokenizer=tokenizer,
            config=config,
            batch_size=batch_size,
            wikihow_max_samples=wikihow_max_samples,
        )

    if args.phase == 1:
        trainer.run_phase1(make_dl(config.phase1_batch_size))
    elif args.phase == 2:
        trainer.run_phase2(make_dl(config.phase2_batch_size))
    else:
        dl = make_dl(config.phase1_batch_size)
        trainer.run_phase1(dl)
        trainer.run_phase2(make_dl(config.phase2_batch_size))

    trainer.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
