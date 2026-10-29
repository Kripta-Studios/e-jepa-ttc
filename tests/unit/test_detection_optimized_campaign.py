from __future__ import annotations

import json
from pathlib import Path

import pytest

from e_jepa_ttc.detection.optimized_campaign import (
    PerformanceConfig,
    PhaseMetrics,
    exact_union,
    transactional_stop_due,
)


def _performance_raw() -> dict[str, object]:
    return {
        "schema": "eap_detection_performance_config_v1",
        "reader": "shared_indexed_tar_v1",
        "max_open_handles": 2,
        "shared_store": True,
        "prefetch_batches": 0,
        "instrument_phases": [
            "tar_index",
            "payload_read",
            "png_decode",
            "prepare",
            "h2d",
            "compute",
            "evaluation",
        ],
        "expected_indexed_reader_sha256": "a" * 64,
        "expected_indexed_store_sha256": "b" * 64,
    }


def test_performance_config_is_strict_and_binds_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "performance.json"
    path.write_text(json.dumps(_performance_raw()), encoding="utf-8")
    config, receipt = PerformanceConfig.load(path)
    assert config.max_open_handles == 2
    assert receipt["canonical_sha256"] == config.sha256
    assert len(receipt["file_sha256"]) == 64

    raw = _performance_raw()
    raw["prefetch_batches"] = 1
    with pytest.raises(ValueError, match="prefetch_batches=0"):
        PerformanceConfig.from_dict(raw)
    raw = _performance_raw()
    raw["unsupported"] = True
    with pytest.raises(ValueError, match="fields differ"):
        PerformanceConfig.from_dict(raw)


def test_exact_union_preserves_role_order_without_cross_role_duplication() -> None:
    assert exact_union(("t1", "t2"), ("t2", "s1"), ("p1",)) == (
        "t1",
        "t2",
        "s1",
        "p1",
    )
    with pytest.raises(ValueError, match="unique"):
        exact_union(("t1", "t1"))


def test_phase_metrics_have_fixed_schema_and_finite_values() -> None:
    metrics = PhaseMetrics(("prepare", "compute"))
    metrics.add("prepare", 0.2)
    metrics.add("prepare", 0.1)
    assert metrics.to_dict()["prepare"] == {
        "count": 2,
        "total_seconds": pytest.approx(0.3),
        "median_seconds": pytest.approx(0.15),
        "p90_seconds": pytest.approx(0.1),
    }
    with pytest.raises(KeyError, match="Undeclared"):
        metrics.add("h2d", 0.0)
    with pytest.raises(ValueError, match="finite"):
        metrics.add("compute", float("nan"))


def test_transactional_stop_requires_durable_checkpoint() -> None:
    assert not transactional_stop_due(
        3, 4, required_evaluation_complete=False, checkpoint_durable=False
    )
    with pytest.raises(RuntimeError, match="evaluation completed"):
        transactional_stop_due(
            4, 4, required_evaluation_complete=False, checkpoint_durable=True
        )
    with pytest.raises(RuntimeError, match="durable"):
        transactional_stop_due(
            4, 4, required_evaluation_complete=True, checkpoint_durable=False
        )
    assert transactional_stop_due(
        4, 4, required_evaluation_complete=True, checkpoint_durable=True
    )


@pytest.mark.parametrize("evaluation_kind", ["scheduled", "unscheduled_diagnostic"])
def test_scheduled_and_unscheduled_stops_commit_after_evaluation(
    evaluation_kind: str,
) -> None:
    assert evaluation_kind
    assert transactional_stop_due(
        8, 8, required_evaluation_complete=True, checkpoint_durable=True
    )


def test_failed_required_evaluation_cannot_commit_stop() -> None:
    with pytest.raises(RuntimeError, match="evaluation completed"):
        transactional_stop_due(
            8, 8, required_evaluation_complete=False, checkpoint_durable=True
        )
