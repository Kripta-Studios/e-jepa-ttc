from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest
import torch


def _load_script():
    path = Path(__file__).parents[2] / "scripts" / "audit_sam_train_bbox_prompts.py"
    spec = importlib.util.spec_from_file_location("audit_sam_train_bbox_prompts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_script()


def test_evenly_spaced_rows_are_deterministic() -> None:
    frame = pd.DataFrame({"sample_token": [f"s_{index:02d}" for index in range(10)]})
    selected = audit.evenly_spaced_rows(frame.sample(frac=1, random_state=2), 4)
    assert selected["sample_token"].tolist() == ["s_00", "s_03", "s_06", "s_09"]
    with pytest.raises(ValueError, match="fewer"):
        audit.evenly_spaced_rows(frame.iloc[:2], 4)


def test_mask_geometry_measures_prompt_overlap() -> None:
    mask = torch.zeros((10, 20), dtype=torch.bool)
    mask[2:8, 5:15] = True
    result = audit.mask_geometry(mask, [5.0, 2.0, 15.0, 8.0])
    assert result["mask_fraction"] == pytest.approx(0.3)
    assert result["bbox_mask_iou"] == pytest.approx(1.0)
    assert result["bbox_coverage"] == pytest.approx(1.0)
    assert result["touches_image_border"] is False


def test_pair_diagnostics_compute_log_ratios() -> None:
    records = [
        {
            "sample_token": "s",
            "sequence_id": "seq",
            "endpoint_index": 1,
            "mask_area_pixels": 200.0,
            "bbox_area_pixels": 100.0,
            "mask_height_fraction": 0.4,
            "bbox_height_fraction": 0.2,
            "mask_width_fraction": 0.5,
            "bbox_width_fraction": 0.25,
        },
        {
            "sample_token": "s",
            "sequence_id": "seq",
            "endpoint_index": 0,
            "mask_area_pixels": 100.0,
            "bbox_area_pixels": 50.0,
            "mask_height_fraction": 0.2,
            "bbox_height_fraction": 0.1,
            "mask_width_fraction": 0.25,
            "bbox_width_fraction": 0.125,
        },
    ]
    pair = audit.pair_diagnostics(records)[0]
    assert pair["mask_area_log_ratio"] == pytest.approx(math.log(2.0))
    assert pair["bbox_height_log_ratio"] == pytest.approx(math.log(2.0))
