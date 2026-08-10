from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_script():
    path = Path(__file__).parents[2] / "scripts" / "smoke_sam_hf_bbox_prompt.py"
    spec = importlib.util.spec_from_file_location("smoke_sam_hf_bbox_prompt", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load_script()


def test_select_train_row_is_deterministic_and_preregistered() -> None:
    frame = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq", "other"],
            "sample_token": ["seq_2", "seq_1", "other_1"],
        }
    )
    row = smoke.select_train_row(
        frame, sequence_id="seq", expected_sample_token="seq_1"
    )
    assert row["sample_token"] == "seq_1"
    with pytest.raises(ValueError, match="preregistered sample mismatch"):
        smoke.select_train_row(frame, sequence_id="seq", expected_sample_token="seq_2")


def test_endpoint_box_supports_public_single_box_arrays() -> None:
    value = np.asarray(
        [np.asarray([0, 10, 20, 30]), np.asarray([1, 11, 21, 31])], dtype=object
    )
    assert smoke.endpoint_box(value, endpoint_index=1, box_index=0) == [1.0, 11.0, 21.0, 31.0]
    with pytest.raises(IndexError, match="single-box"):
        smoke.endpoint_box(value, endpoint_index=1, box_index=1)


def test_rejects_test_paths(tmp_path: Path) -> None:
    path = tmp_path / "test" / "train.parquet"
    path.parent.mkdir()
    path.touch()
    with pytest.raises(ValueError, match="public train.parquet"):
        smoke._require_public_train(path)
