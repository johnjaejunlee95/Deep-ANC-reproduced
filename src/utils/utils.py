"""General utility functions for DeepANC."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_acc(logits, label):
    num_correct = sum(row.all().int().item() for row in (logits.ge(0.5) == label))
    return torch.tensor(num_correct / label.size(0))


def minmaxscaler(data):
    min_val = data.min(dim=1, keepdim=True).values
    max_val = data.max(dim=1, keepdim=True).values
    return data / (max_val - min_val)


def plot_results(
    train_loss: list[float],
    valid_loss: list[float],
    test_loss: list[float],
    output_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss", marker="*")
    plt.plot(
        range(1, len(valid_loss) + 1),
        valid_loss,
        label="Validation Loss",
        linestyle="dashed",
        marker="x",
    )
    plt.plot(
        range(1, len(test_loss) + 1),
        test_loss,
        label="Test Loss",
        linestyle="dashdot",
        marker="P",
    )
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(loc="best", frameon=True, shadow=True)
    plt.grid()
    plt.savefig(output_path, dpi=300)
    plt.close()
