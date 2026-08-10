from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_script():
    path = Path(__file__).parents[2] / "scripts" / "materialize_sam_train_bbox_prompt_cache.py"
    spec = importlib.util.spec_from_file_location("materialize_sam_train_bbox_prompt_cache", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


materialize = _load_script()


def test_packbits_roundtrip_is_exact() -> None:
    mask = np.zeros((128, 128), dtype=bool)
    mask[7:91, 11:101] = True
    packed = materialize.pack_binary_mask(mask, bitorder="little")
    restored = materialize.unpack_binary_mask(packed, bitorder="little")
    assert packed.shape == (2048,)
    assert np.array_equal(restored, mask)


def test_endpoint_filters_preserve_named_reasons() -> None:
    config = {
        "minimum_predicted_iou": 0.5,
        "minimum_mask_fraction": 0.001,
        "maximum_mask_fraction": 0.75,
        "minimum_bbox_mask_iou": 0.25,
        "minimum_mask_inside_bbox_fraction": 0.8,
    }
    metrics = {
        "predicted_iou": 0.4,
        "mask_fraction": 0.0001,
        "bbox_mask_iou": 0.2,
        "mask_inside_bbox_fraction": 0.7,
    }
    assert materialize.endpoint_filter_reasons(metrics, config) == [
        "low_predicted_iou",
        "degenerate_mask_fraction",
        "low_bbox_mask_iou",
        "low_inside_bbox_fraction",
    ]


def test_temporal_sign_filter_ignores_near_zero_bbox_change() -> None:
    assert materialize.temporal_sign_consistent(-1.0, 0.001, 0.002) is True
    assert materialize.temporal_sign_consistent(-0.1, 0.1, 0.002) is False
    assert materialize.temporal_sign_consistent(0.1, 0.1, 0.002) is True
