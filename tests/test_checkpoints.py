from __future__ import annotations

import torch

from src.utils import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip_loads_model_state(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, epoch=3, model=model, config={"sample_rate": 16000})

    loaded = torch.nn.Linear(2, 1)
    checkpoint = load_checkpoint(path, loaded)

    assert checkpoint["epoch"] == 3
    for expected, actual in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(expected, actual)


def test_legacy_state_dict_checkpoint_is_supported(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "legacy.pt"
    torch.save({"state_dict": model.state_dict()}, path)

    loaded = torch.nn.Linear(2, 1)
    load_checkpoint(path, loaded)

    for expected, actual in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(expected, actual)
