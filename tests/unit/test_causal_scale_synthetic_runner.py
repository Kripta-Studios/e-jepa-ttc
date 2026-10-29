from __future__ import annotations

import pytest

from scripts.train_causal_scale_v5_synthetic import _dataset_configs, _macro_metrics


def _raw_data() -> dict[str, object]:
    return {
        "data": {
            "common": {
                "canvas_size": 32,
                "endpoints": 3,
                "polarity_bins": 5,
                "delta_t_s": 0.1,
                "accumulation_s": 0.05,
                "micro_frames": 6,
            },
            "validation": {
                "groups": [
                    {"name": "val_801", "samples": 4, "seed": 801},
                    {"name": "val_802", "samples": 5, "seed": 802},
                ]
            },
        }
    }


def test_multigroup_dataset_config_preserves_preregistered_names_and_seeds() -> None:
    groups = _dataset_configs(_raw_data(), "validation")

    assert list(groups) == ["val_801", "val_802"]
    assert [config.seed for config in groups.values()] == [801, 802]
    assert [config.samples for config in groups.values()] == [4, 5]


def test_multigroup_dataset_config_rejects_duplicate_names() -> None:
    raw = _raw_data()
    validation = raw["data"]["validation"]  # type: ignore[index]
    validation["groups"][1]["name"] = "val_801"  # type: ignore[index]

    with pytest.raises(ValueError, match="repeats group"):
        _dataset_configs(raw, "validation")


def test_macro_metrics_do_not_pool_or_hide_missing_group_values() -> None:
    metrics = _macro_metrics(
        {
            "a": {"analytic_pearson": 0.9, "optional": None},
            "b": {"analytic_pearson": 1.0, "optional": 0.5},
        }
    )

    assert metrics == {"analytic_pearson": 0.95, "optional": None}
