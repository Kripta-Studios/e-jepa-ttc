import json
from pathlib import Path

import pytest

from e_jepa_ttc.evaluation.aggregate import aggregate_metric_files
from e_jepa_ttc.utils.io import write_structured


def _write_metrics(path: Path, *, seed: int, mae: float, relative: float) -> None:
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "downstream_seed": seed,
                "pretrained_encoder": {
                    "source_seed": 5,
                    "checkpoint_role": "best",
                    "checkpoint_selected_by": "validation_loss",
                },
                "checkpoint_epoch": seed + 1,
                "splits": {
                    "test": {
                        "count": 4,
                        "metrics": {
                            "mae_s": mae,
                            "mean_abs_relative_error_pct": relative,
                            "rmse_s": mae * 2.0,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_aggregate_metric_files(tmp_path: Path) -> None:
    first = tmp_path / "seed7.json"
    second = tmp_path / "seed13.json"
    _write_metrics(first, seed=7, mae=0.2, relative=5.0)
    _write_metrics(second, seed=13, mae=0.4, relative=7.0)

    payload = aggregate_metric_files(
        [first, second],
        split="test",
        metric_names=("mae_s", "mean_abs_relative_error_pct"),
    )

    assert payload["count"] == 2
    assert payload["rows"][0]["seed"] == 7
    assert payload["pretrain_seeds"] == [5]
    assert payload["downstream_seeds"] == [7, 13]
    assert payload["uncertainty_scope"] == "downstream_only_conditional_on_single_pretrain_seed"
    assert payload["summary"]["mae_s"]["mean"] == pytest.approx(0.3)
    assert payload["summary"]["mae_s"]["std"] == 0.0
    assert payload["summary"]["mae_s"]["pooled_std"] == pytest.approx(0.14142135623730951)
    assert payload["summary"]["mean_abs_relative_error_pct"]["mean"] == pytest.approx(6.0)


def test_aggregate_rejects_official_table_from_reused_test_split(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    split_protocol = tmp_path / "split.yaml"
    _write_metrics(metrics, seed=7, mae=0.2, relative=5.0)
    write_structured(
        split_protocol,
        {
            "status": "reused_test_diagnostic",
            "evaluation_role": "diagnostic",
            "allowed_claim_levels": ["development", "diagnostic"],
            "test_was_previously_inspected": True,
            "splits": {"test": ["fixture"]},
        },
    )

    with pytest.raises(ValueError, match="cannot produce"):
        aggregate_metric_files(
            [metrics],
            split="test",
            split_protocol_path=split_protocol,
            claim_level="official",
        )
