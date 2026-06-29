"""DeepANC encoder-decoder model."""

from __future__ import annotations

import torch
from torch import nn


class DeepANC(nn.Module):
    """Convolutional encoder-decoder with an LSTM bottleneck."""

    def __init__(self, lstm_features: int = 1152) -> None:
        super().__init__()
        self.lstm_features = lstm_features

        self.conv1 = nn.Conv2d(2, 16, kernel_size=(1, 3), stride=(1, 2))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 3), stride=(1, 2))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 2))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(1, 3), stride=(1, 2))

        self.lstm = nn.LSTM(lstm_features, lstm_features, 2, batch_first=True)

        self.conv4_t = nn.ConvTranspose2d(256, 64, kernel_size=(1, 3), stride=(1, 2))
        self.conv3_t = nn.ConvTranspose2d(128, 32, kernel_size=(1, 3), stride=(1, 2))
        self.conv2_t = nn.ConvTranspose2d(
            64,
            16,
            kernel_size=(1, 3),
            stride=(1, 2),
            output_padding=(0, 1),
        )
        self.conv1_t = nn.ConvTranspose2d(32, 2, kernel_size=(1, 3), stride=(1, 2))

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)

        self.bn4_t = nn.BatchNorm2d(64)
        self.bn3_t = nn.BatchNorm2d(32)
        self.bn2_t = nn.BatchNorm2d(16)
        self.bn1_t = nn.BatchNorm2d(2)

        self.elu = nn.ELU()
        self.last_layer = nn.Conv2d(2, 2, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.elu(self.bn1(self.conv1(x)))
        e2 = self.elu(self.bn2(self.conv2(e1)))
        e3 = self.elu(self.bn3(self.conv3(e2)))
        e4 = self.elu(self.bn4(self.conv4(e3)))

        out = e4.transpose(1, 2)
        batch_size, frames, channels, bins = out.shape
        out = out.reshape(batch_size, frames, -1)
        if out.size(-1) != self.lstm_features:
            raise ValueError(
                "DeepANC expected %d LSTM features, got %d. "
                "Check the STFT window/frequency geometry."
                % (self.lstm_features, out.size(-1))
            )

        out, _ = self.lstm(out)
        out = out.reshape(batch_size, frames, channels, bins)
        out = out.transpose(1, 2)

        out = torch.cat([out, e4], dim=1)
        d4 = self.elu(torch.cat([self.bn4_t(self.conv4_t(out)), e3], dim=1))
        d3 = self.elu(torch.cat([self.bn3_t(self.conv3_t(d4)), e2], dim=1))
        d2 = self.elu(torch.cat([self.bn2_t(self.conv2_t(d3)), e1], dim=1))
        return self.last_layer(self.elu(self.bn1_t(self.conv1_t(d2))))


def init_weights(module: nn.Module) -> None:
    """Initialize convolution and batch-normalization layers."""

    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
