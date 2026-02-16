#!/usr/bin/env python3
"""Entry point for hierarchical MDLM sampling / inference.

Usage:
    python sample.py --checkpoint checkpoints/Phase2_step50000.pt \\
                     --title "Machine Learning Basics" \\
                     --levels "0,0,0,1,1,1,1,0,0" \\
                     --num_steps 500

``--levels`` supplies the hierarchy labels for *body* positions.
At inference time these can come from a hierarchy predictor instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import transformers

from models.hierarchical_generator import HierarchicalGenerator
from models.hierarchy_embedding import HierarchyEmbedding
from inference.sampler import HierarchicalSampler
from noise_schedule import get_noise_schedule
from training.trainer import TrainerConfig
from lora.lora import apply_lora_to_model


def main():
    parser = argparse.ArgumentParser(description="Hierarchical MDLM Sampling")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--title", type=str, default="Introduction")
    parser.add_argument(
        "--levels", type=str, default=None,
        help="Comma-separated hierarchy levels for body positions "
             "(e.g. '0,0,1,1'). If omitted, defaults to all-0."
    )
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    config: TrainerConfig = ckpt["config"]

    # Rebuild model
    model = HierarchicalGenerator(
        pretrained_path="",  # don't re-download; will load state dict
        vocab_size=config.vocab_size,
        num_levels=config.num_levels,
        hidden_size=config.hidden_size,
        cond_dim=config.cond_dim,
    )

    # Re-attach LoRA shells before loading weights.
    # Phase 1 checkpoints have LoRA on output heads only.
    # Phase 2 checkpoints have LoRA on output heads + backbone.
    phase = ckpt.get("phase", "")

    # Always re-attach output-head LoRA (present in both phases)
    if phase.startswith("Phase1") or phase.startswith("Phase2"):
        apply_lora_to_model(
            model.output_heads,
            target_modules=config.phase1_lora_target_modules,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=0.0,
        )

    # Phase 2 also has backbone LoRA
    if phase.startswith("Phase2"):
        apply_lora_to_model(
            model.backbone,
            target_modules=config.phase2_lora_target_modules,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=0.0,
        )

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    # Encode title
    title_ids = tokenizer.encode(args.title, add_special_tokens=False)
    T = len(title_ids)
    L = args.seq_length
    B = args.num_samples

    title_tensor = torch.tensor([title_ids] * B, dtype=torch.long, device=device)

    # Build hierarchy probs (B, L, K)
    K = config.num_levels
    if args.levels is not None:
        body_levels = [int(x) for x in args.levels.split(",")]
    else:
        body_levels = [0] * (L - T)

    # Pad or truncate body levels to fill L - T positions
    body_levels = (body_levels + [0] * (L - T))[: L - T]

    all_levels = [0] * T + body_levels  # title gets level 0
    labels = torch.tensor([all_levels] * B, dtype=torch.long, device=device)
    hier_probs = HierarchyEmbedding.labels_to_onehot(labels, K).to(device)

    # Noise schedule (with per-level scaling)
    noise = get_noise_schedule(config.noise_type)

    # Sample
    sampler = HierarchicalSampler(
        model=model,
        noise=noise,
        level_scales=config.noise_level_scales,
        num_steps=args.num_steps,
        device=device,
    )
    generated = sampler.sample(title_tensor, hier_probs, seq_length=L)

    # Decode
    for i in range(B):
        tokens = generated[i].cpu().tolist()
        # Stop at first pad / eos
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"\n--- Sample {i + 1} ---")
        print(text)


if __name__ == "__main__":
    main()
