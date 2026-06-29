"""Training helpers used by main.py."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import AudioDataset
from .dsp import apply_anc_paths, complex_spectrogram_input
from .stft import STFT
from src.models import DeepANC


@dataclass
class TrainConfig:
    data_path: Path
    epochs: int = 100
    batch_size: int = 32
    sample_rate: int = 16000
    learning_rate: float = 1e-3
    weight_decay: float = 5e-5
    num_workers: int = 8
    filter_dir: Path = Path("src/mat_files")
    checkpoint_dir: Path = Path("checkpoints")
    resume: Path | None = None
    device: str = "auto"
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = AudioDataset(config.data_path, status="train", sr=config.sample_rate)
    valid_dataset = AudioDataset(config.data_path, status="valid", sr=config.sample_rate)
    test_dataset = AudioDataset(config.data_path, status="test", sr=config.sample_rate)

    return (
        DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True,
        ),
        DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
        ),
        DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
        ),
    )


def run_batch(
    model: DeepANC,
    stft: STFT,
    origin_audio: torch.Tensor,
    primary_filter: torch.Tensor,
    secondary_filter: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    origin_audio = origin_audio.to(device)
    real, imag = stft.stft(origin_audio.squeeze(1))
    model_input = complex_spectrogram_input(real, imag, pad_frames=2)
    output = model(model_input)
    anti_noise = stft.istft(output)
    desired, anti, _ = apply_anc_paths(primary_filter, secondary_filter, origin_audio, anti_noise)
    return desired, anti


def train_epoch(
    model: DeepANC,
    stft: STFT,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    primary_filter: torch.Tensor,
    secondary_filter: torch.Tensor,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for origin_audio in tqdm(dataloader, desc="train", leave=False):
        desired, anti = run_batch(model, stft, origin_audio, primary_filter, secondary_filter, device)
        loss = loss_fn(desired, anti)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * origin_audio.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(
    model: DeepANC,
    stft: STFT,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    primary_filter: torch.Tensor,
    secondary_filter: torch.Tensor,
    device: torch.device,
    desc: str,
) -> float:
    model.eval()
    total_loss = 0.0
    for origin_audio in tqdm(dataloader, desc=desc, leave=False):
        desired, anti = run_batch(model, stft, origin_audio, primary_filter, secondary_filter, device)
        loss = loss_fn(desired, anti)
        total_loss += loss.item() * origin_audio.size(0)
    return total_loss / len(dataloader.dataset)
