from __future__ import annotations

import torch

from src.utils import complex_spectrogram_input, lfilter


def test_lfilter_does_not_mutate_coefficients() -> None:
    coeff = torch.tensor([1.0, 2.0, 3.0])
    original = coeff.clone()
    audio = torch.ones(1, 1, 8)

    output = lfilter(coeff, 1, audio)

    assert output.shape == audio.shape
    assert torch.equal(coeff, original)


def test_complex_spectrogram_input_uses_dynamic_batch_size() -> None:
    real = torch.ones(3, 5, 4)
    imag = torch.zeros(3, 5, 4)

    output = complex_spectrogram_input(real, imag, pad_frames=2)

    assert output.shape == (3, 2, 5, 4)
    assert torch.all(output[:, :, :2] == 0)
