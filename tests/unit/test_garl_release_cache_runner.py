from __future__ import annotations

import pytest
import torch

from scripts.train_garl_release_cache import _cache_fields, _cache_input


def _batch() -> dict[str, torch.Tensor]:
    return {
        "event_roi": torch.zeros(2, 40, 128, 128),
        "rgb_pair": torch.zeros(2, 2, 3, 128, 128),
    }


@pytest.mark.parametrize(
    ("variant", "channels"),
    (("event_only", 40), ("visual_only", 6), ("rgbe_late_fusion", 46)),
)
def test_cache_runner_builds_official_input_channels(variant: str, channels: int) -> None:
    assert _cache_input(_batch(), variant).shape == (2, channels, 128, 128)


def test_cache_runner_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="Unsupported Garl cache variant"):
        _cache_input(_batch(), "unknown")


@pytest.mark.parametrize(
    ("variant", "required", "excluded"),
    (
        ("event_only", "event_q", "rgb_f16"),
        ("visual_only", "rgb_f16", "event_q"),
        ("rgbe_late_fusion", "event_q", "never"),
    ),
)
def test_cache_runner_selects_only_variant_input_arrays(
    variant: str,
    required: str,
    excluded: str,
) -> None:
    fields = set(_cache_fields(variant))

    assert required in fields
    if excluded != "never":
        assert excluded not in fields
