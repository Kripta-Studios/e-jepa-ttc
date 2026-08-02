from __future__ import annotations

import json
from pathlib import Path

from scripts.build_plan_execution_audit import (
    EXPECTED_BASELINE_SEEDS,
    EXPECTED_BASELINE_VARIANTS,
    _baseline_phase_state,
    _ssl_phase_state,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_baseline_phase_does_not_promote_max_batch_or_missing_metrics(tmp_path: Path) -> None:
    status, _, _ = _baseline_phase_state(tmp_path)
    assert status == "not_executed"

    runs = [
        {
            "variant": variant,
            "seed": seed,
            "status": "completed",
            "metrics_available": False,
            "validation_evaluation": {"status": "completed"},
        }
        for variant in EXPECTED_BASELINE_VARIANTS
        for seed in EXPECTED_BASELINE_SEEDS
    ]
    matrix = tmp_path / "artifacts/runs/garl_baseline_training_v1/matrix.json"
    _write_json(
        matrix,
        {
            "status": "completed",
            "max_batches": None,
            "full_matrix": True,
            "runs": runs,
        },
    )
    status, _, missing = _baseline_phase_state(tmp_path)
    assert status == "partial"
    assert any("signed validation metrics" in item for item in missing)

    _write_json(
        tmp_path / "artifacts/metrics/garl_baseline_training_v1_signed.json",
        {
            "artifact_type": "garl_baseline_metrics_v1",
            "status": "PASS",
            "expected_run_count": 9,
            "observed_metric_count": 9,
            "missing": [],
            "test_used_for_selection": False,
            "evttc_used_for_selection": False,
        },
    )
    status, _, missing = _baseline_phase_state(tmp_path)
    assert status == "verified"
    assert missing == []


def test_baseline_phase_recognizes_release_cache_matrix_v2(tmp_path: Path) -> None:
    runs = [
        {
            "variant": variant,
            "seed": seed,
            "status": "completed",
            "validation_metrics": {"signed_garl_metrics": {"paper_MiD_overall": 1.0}},
        }
        for variant in EXPECTED_BASELINE_VARIANTS
        for seed in EXPECTED_BASELINE_SEEDS
    ]
    _write_json(
        tmp_path / "artifacts/runs/garl_release_cache_training_v1/matrix.json",
        {"status": "failed", "full_matrix": False, "runs": []},
    )
    _write_json(
        tmp_path / "artifacts/runs/garl_release_cache_matrix_full_v2_workers2/matrix.json",
        {
            "status": "completed",
            "max_batches": None,
            "full_matrix": True,
            "bbox_protocol": "P0_oracle_bbox_roi",
            "runs": runs,
        },
    )
    _write_json(
        tmp_path / "artifacts/metrics/garl_release_cache_training_v1_signed.json",
        {
            "artifact_type": "garl_release_cache_training_metrics_v1",
            "status": "pass",
            "expected_run_count": 9,
            "observed_metric_count": 9,
            "missing": [],
            "test_used_for_selection": False,
            "evttc_used_for_selection": False,
        },
    )

    status, _, missing = _baseline_phase_state(tmp_path)
    assert status == "verified"
    assert missing == []


def test_ssl_phase_preserves_failure_and_requires_full_terminal_summary(tmp_path: Path) -> None:
    failure = tmp_path / "artifacts/runs/eap_ssl_train40_full_cuda_current/FAILURE.json"
    _write_json(failure, {"status": "failed", "negative_result_preserved": True})
    status, _, missing = _ssl_phase_state(tmp_path, {"ssl_full_train40_green": False})
    assert status == "partial"
    assert missing

    metrics = failure.parent / "metrics.json"
    _write_json(
        metrics,
        {
            "pretraining_regime": "eap_ssl",
            "epochs_completed": 1,
            "train_window_count": 16_384,
            "validation_window_count": 4_096,
            "trainer_config": {
                "max_train_samples": None,
                "max_validation_samples": None,
            },
        },
    )
    status, _, _ = _ssl_phase_state(tmp_path, {"ssl_full_train40_green": True})
    assert status == "verified"
