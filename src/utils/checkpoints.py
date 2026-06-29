"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    stft: nn.Module | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Save a versioned checkpoint that can be reused by train/infer scripts."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": config or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if stft is not None:
        payload["stft_state_dict"] = stft.state_dict()
        if hasattr(stft, "config"):
            payload["stft_config"] = stft.config()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    stft: nn.Module | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint and support the legacy `state_dict` key."""

    checkpoint = torch.load(path, map_location=map_location)
    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        model_state_dict = checkpoint["state_dict"]
    else:
        raise KeyError("Checkpoint must contain 'model_state_dict' or legacy 'state_dict'")

    model.load_state_dict(model_state_dict)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if stft is not None:
        if "stft_state_dict" in checkpoint:
            stft.load_state_dict(checkpoint["stft_state_dict"])
        elif "stft" in checkpoint and hasattr(checkpoint["stft"], "state_dict"):
            stft.load_state_dict(checkpoint["stft"].state_dict())
    return checkpoint
