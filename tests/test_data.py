from __future__ import annotations

import pandas as pd
import pytest

from src.utils import AudioDataset


def test_audio_dataset_validates_required_file_path_column(tmp_path) -> None:
    pd.DataFrame({"wrong": ["a.wav"]}).to_csv(tmp_path / "Hard_Index_train.csv", index=False)

    with pytest.raises(ValueError, match="File_path"):
        AudioDataset(tmp_path, "train")


def test_audio_dataset_reports_missing_split_csv(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Hard_Index_val.csv"):
        AudioDataset(tmp_path, "valid")
