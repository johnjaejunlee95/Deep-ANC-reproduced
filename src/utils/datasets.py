"""Dataset definitions for DeepANC experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

SPLIT_FILES = {
    "train": "Hard_Index_train.csv",
    "valid": "Hard_Index_val.csv",
    "test": "Hard_Index_test.csv",
}


def _load_torchaudio():
    try:
        import torchaudio
    except Exception as exc:  # pragma: no cover - depends on local binary install
        raise RuntimeError(
            "torchaudio is required for audio loading. Install a torch/torchaudio "
            "pair that matches your Python and platform."
        ) from exc
    return torchaudio


class AudioDataset(Dataset):
    """CSV-backed audio dataset used for train/validation/test splits."""

    def __init__(self, root_dir: str | Path, status: str, sr: int = 16000) -> None:
        if status not in SPLIT_FILES:
            raise ValueError(f"Unknown split '{status}'. Expected one of {sorted(SPLIT_FILES)}")
        self.root_dir = Path(root_dir)
        self.sr = sr
        self.status = status

        csv_path = self.root_dir / SPLIT_FILES[status]
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset split CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        if "File_path" not in self.df.columns:
            raise ValueError(f"{csv_path} must contain a 'File_path' column")
        if self.df.empty:
            raise ValueError(f"{csv_path} is empty")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = self.root_dir / self.df.iloc[idx]["File_path"]
        torchaudio = _load_torchaudio()
        audio, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.sr:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sr)(audio)
        return audio


class WavDataset(Dataset):
    """Directory-backed WAV dataset for inference."""

    def __init__(self, root_dir: str | Path, sr: int = 16000) -> None:
        self.root_dir = Path(root_dir)
        self.sr = sr
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {self.root_dir}")
        self.file_list = sorted(
            path for path in self.root_dir.iterdir() if path.suffix.lower() in {".wav", ".flac"}
        )
        if not self.file_list:
            raise ValueError(f"No .wav or .flac files found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int | torch.Tensor) -> tuple[torch.Tensor, str]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_path = self.file_list[idx]
        torchaudio = _load_torchaudio()
        audio, sample_rate = torchaudio.load(file_path)
        if sample_rate != self.sr:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sr)(audio)

        full_seconds = audio.size(-1) // self.sr
        if full_seconds > 0:
            audio = audio[:, : full_seconds * self.sr]
        return audio, file_path.name


AudioDatasets = AudioDataset
