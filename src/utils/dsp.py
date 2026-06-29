"""DSP helpers shared by training and inference."""

from __future__ import annotations

from pathlib import Path

import scipy.io as sio
import torch


def load_path_filters(
    path: str | Path,
    primary_file: str = "Primary_path.mat",
    secondary_file: str = "Secondary_path.mat",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load primary and secondary path FIR coefficients from MATLAB files."""

    path = Path(path)
    primary_path = path / primary_file
    secondary_path = path / secondary_file
    if not primary_path.exists():
        raise FileNotFoundError(f"Primary path file not found: {primary_path}")
    if not secondary_path.exists():
        raise FileNotFoundError(f"Secondary path file not found: {secondary_path}")

    primary = sio.loadmat(primary_path)
    secondary = sio.loadmat(secondary_path)
    if "Pz1" not in primary:
        raise KeyError(f"{primary_path} must contain variable 'Pz1'")
    if "S" not in secondary:
        raise KeyError(f"{secondary_path} must contain variable 'S'")
    return (
        torch.tensor(primary["Pz1"].squeeze(), dtype=torch.float32),
        torch.tensor(secondary["S"].squeeze(), dtype=torch.float32),
    )


def load_mat_data(path, pri_path_file, sec_path_file):
    primary, secondary = load_path_filters(path, pri_path_file, sec_path_file)
    return primary.numpy(), secondary.numpy()


def lfilter(b: torch.Tensor, a: float | int | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply a 1-D FIR filter to batched audio without mutating coefficients."""

    if x.dim() != 3:
        raise ValueError(f"Expected x with shape [batch, channels, samples], got {tuple(x.shape)}")
    coeff = b.to(device=x.device, dtype=x.dtype).flatten()
    a0 = torch.as_tensor(a, device=x.device, dtype=x.dtype).flatten()[0]
    coeff = torch.flip(coeff / a0, dims=[0])

    channels = x.size(1)
    kernel = coeff.view(1, 1, -1).repeat(channels, 1, 1)
    out = torch.conv1d(x, kernel, padding=coeff.numel() - 1, groups=channels)
    return out[:, :, : x.size(-1)]


def complex_spectrogram_input(
    real: torch.Tensor,
    imag: torch.Tensor,
    pad_frames: int,
) -> torch.Tensor:
    """Prepend zero frames and stack real/imaginary STFT parts for the model."""

    batch_size, _, bins = real.shape
    zero_padding = real.new_zeros(batch_size, pad_frames, bins)
    real = torch.cat([zero_padding, real], dim=1)[:, : real.size(1)]
    imag = torch.cat([zero_padding, imag], dim=1)[:, : imag.size(1)]
    return torch.stack([real, imag], dim=1)


def apply_anc_paths(
    primary_filter: torch.Tensor,
    secondary_filter: torch.Tensor,
    origin_audio: torch.Tensor,
    anti_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return desired signal, secondary-path anti-noise, and residual error."""

    desired = lfilter(primary_filter, 1, origin_audio).squeeze(1)
    anti = lfilter(secondary_filter, 1, anti_noise.unsqueeze(1)).squeeze(1)
    return desired, anti, desired - anti
