"""STFT/ISTFT module used by DeepANC."""

from __future__ import annotations

import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
from torch import nn


class STFT(nn.Module):
    """Differentiable short-time Fourier transform implementation."""

    def __init__(
        self,
        win_size: int = 320,
        hop_size: int = 160,
        requires_grad: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.win_size = win_size
        self.hop_size = hop_size
        self.n_overlap = self.win_size // self.hop_size
        self.requires_grad = requires_grad
        self.device = torch.device(device)

        window = scipy.signal.windows.hamming(self.win_size).astype(np.float32)
        win = torch.from_numpy(window).to(self.device)
        self.register_parameter("win", nn.Parameter(win, requires_grad=requires_grad))

        fourier_basis = np.fft.fft(np.eye(self.win_size))
        fourier_basis_r = np.real(fourier_basis).astype(np.float32)
        fourier_basis_i = np.imag(fourier_basis).astype(np.float32)

        self.register_buffer("fourier_basis_r", torch.from_numpy(fourier_basis_r).to(self.device))
        self.register_buffer("fourier_basis_i", torch.from_numpy(fourier_basis_i).to(self.device))

        idx = torch.tensor(range(self.win_size // 2 - 1, 0, -1), dtype=torch.long)
        self.register_buffer("idx", idx.to(self.device))
        self.eps = torch.finfo(torch.float32).eps

    def config(self) -> dict[str, int | bool]:
        return {
            "win_size": self.win_size,
            "hop_size": self.hop_size,
            "requires_grad": self.requires_grad,
        }

    def kernel_fw(self) -> torch.Tensor:
        fourier_basis_r = torch.matmul(self.fourier_basis_r, torch.diag(self.win))
        fourier_basis_i = torch.matmul(self.fourier_basis_i, torch.diag(self.win))
        fourier_basis = torch.stack([fourier_basis_r, fourier_basis_i], dim=-1)
        return fourier_basis.unsqueeze(dim=1)

    def kernel_bw(self) -> torch.Tensor:
        inv_fourier_basis_r = self.fourier_basis_r / self.win_size
        inv_fourier_basis_i = -self.fourier_basis_i / self.win_size
        inv_fourier_basis = torch.stack([inv_fourier_basis_r, inv_fourier_basis_i], dim=-1)
        return inv_fourier_basis.unsqueeze(dim=1)

    def window(self, n_frames: int) -> torch.Tensor:
        if n_frames < 2:
            raise ValueError("STFT inverse requires at least two frames")
        seg = sum(
            self.win[i * self.hop_size : (i + 1) * self.hop_size]
            for i in range(self.n_overlap)
        )
        seg = seg.unsqueeze(dim=-1).expand((self.hop_size, n_frames - self.n_overlap + 1))
        return seg.contiguous().view(-1).contiguous()

    def stft(self, sig: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_samples = sig.shape
        cutoff = self.win_size // 2 + 1

        sig = sig.view(batch_size, 1, n_samples)
        kernel = self.kernel_fw()
        spec_r = F.conv1d(
            sig,
            kernel[..., 0][:cutoff],
            stride=self.hop_size,
            padding=self.win_size - self.hop_size,
        )
        spec_i = F.conv1d(
            sig,
            kernel[..., 1][:cutoff],
            stride=self.hop_size,
            padding=self.win_size - self.hop_size,
        )
        return spec_r.transpose(-1, -2).contiguous(), spec_i.transpose(-1, -2).contiguous()

    def istft(self, x: torch.Tensor) -> torch.Tensor:
        spec_r = x[:, 0, :, :]
        spec_i = x[:, 1, :, :]

        n_frames = spec_r.shape[1]
        spec_r = torch.cat([spec_r, spec_r.index_select(dim=-1, index=self.idx)], dim=-1)
        spec_i = torch.cat([spec_i, -spec_i.index_select(dim=-1, index=self.idx)], dim=-1)
        spec_r = spec_r.transpose(-1, -2).contiguous()
        spec_i = spec_i.transpose(-1, -2).contiguous()

        kernel = self.kernel_bw()
        kernel_r = kernel[..., 0].transpose(0, -1)
        kernel_i = kernel[..., 1].transpose(0, -1)

        sig = F.conv_transpose1d(
            spec_r,
            kernel_r,
            stride=self.hop_size,
            padding=self.win_size - self.hop_size,
        ) - F.conv_transpose1d(
            spec_i,
            kernel_i,
            stride=self.hop_size,
            padding=self.win_size - self.hop_size,
        )
        sig = sig.squeeze(dim=1)
        return sig / (self.window(n_frames) + self.eps)
