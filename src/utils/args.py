"""Argument helpers for legacy imports."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(default: bool = False):
    parser = argparse.ArgumentParser(description="DeepANC training arguments")
    parser.add_argument("--data-path", "--data_path", type=Path, default=Path("YOUR/OWN/PATH"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", "--batch", dest="batch_size", type=int, default=32)
    parser.add_argument("--sample-rate", "--sample_rate", "--fs", dest="sample_rate", type=int, default=16000)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=5e-5)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=8)
    parser.add_argument("--filter-dir", "--filter_dir", dest="filter_dir", type=Path, default=Path("src/mat_files"))
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    if default:
        return parser.parse_args([])
    return parser.parse_args()
