from __future__ import annotations

import pytest
import torch

from src.models import DeepANC


def test_deepanc_preserves_default_stft_geometry() -> None:
    model = DeepANC().eval()
    x = torch.randn(2, 2, 6, 161)

    with torch.no_grad():
        output = model(x)

    assert output.shape == x.shape


def test_deepanc_reports_unsupported_frequency_geometry() -> None:
    model = DeepANC().eval()
    x = torch.randn(2, 2, 6, 129)

    with pytest.raises(ValueError, match="STFT window/frequency geometry"):
        model(x)
