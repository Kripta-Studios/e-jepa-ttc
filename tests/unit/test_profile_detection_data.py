from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from e_jepa_ttc.detection.monodgp_eap_adapter import PreparedSample
from scripts.profile_detection_data import (
    _peak_working_set_bytes,
    prepared_checksum,
    trace_tokens,
)


def sample(value: float = 1.0) -> PreparedSample:
    return PreparedSample(
        image=torch.full((3, 2, 2), value),
        calibration=torch.eye(3, 4),
        image_size=torch.tensor([2.0, 2.0]),
        target={
            "labels": torch.tensor([1]),
            "velocity_valid": torch.tensor([True]),
        },
        native_ground_truth=[
            {
                "class_name": "car",
                "translation_m": [1.0, 2.0, 3.0],
                "size_lwh_m": [4.0, 2.0, 1.5],
                "yaw_rad": 0.0,
            }
        ],
        report={},
    )


def test_trace_is_seeded_unique_and_bounded():
    tokens = [f"s:{index}" for index in range(20)]
    assert trace_tokens(tokens, seed=7, count=5) == trace_tokens(tokens, seed=7, count=5)
    assert trace_tokens(tokens, seed=7, count=5) != trace_tokens(tokens, seed=8, count=5)
    with pytest.raises(ValueError):
        trace_tokens(tokens + [tokens[0]], seed=7, count=5)


def test_prepared_checksum_covers_pixels_member_and_targets():
    raw = np.zeros((2, 2, 3), dtype=np.uint8)
    base = prepared_checksum(raw_image=raw, member_sha256="a" * 64, prepared=sample())
    assert base == prepared_checksum(
        raw_image=raw.copy(), member_sha256="a" * 64, prepared=sample()
    )
    assert base != prepared_checksum(raw_image=raw, member_sha256="b" * 64, prepared=sample())
    assert base != prepared_checksum(raw_image=raw, member_sha256="a" * 64, prepared=sample(2.0))


@pytest.mark.skipif(os.name != "nt", reason="Windows memory API")
def test_peak_working_set_is_available_on_windows():
    value = _peak_working_set_bytes()
    assert value is not None and value > 0
