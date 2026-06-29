"""Inference helpers used by main.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class InferConfig:
    checkpoint: Path = Path("checkpoints/deepanc_best.pt")
    input_dir: Path = Path("results/noise_samples")
    output_dir: Path = Path("wav_results")
    filter_dir: Path = Path("src/mat_files")
    sample_rate: int = 16000
    batch_size: int = 1
    num_workers: int = 0
    device: str = "auto"


def power_to_db(values: np.ndarray, ref: float = 1.0, amin: float = 1e-10) -> np.ndarray:
    values = np.maximum(amin, values)
    return 10.0 * np.log10(values / ref)


def write_audio(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    try:
        import soundfile as sf
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError("soundfile is required for writing inference WAV files") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.detach().cpu().numpy(), sample_rate)
