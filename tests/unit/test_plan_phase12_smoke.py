"""Tests for the bounded, fixture-only Phase 12 preparation smoke."""

from __future__ import annotations

import json
from pathlib import Path

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash


def test_phase12_smoke_generates_verified_local_artifact(tmp_path: Path) -> None:
    from scripts.run_plan_phase12_smoke import run_smoke

    payload = run_smoke(tmp_path / "phase12", seed=7)
    report = tmp_path / "phase12" / "phase12_smoke.json"

    assert payload["artifact_type"] == "plan_phase12_smoke_v1"
    assert payload["status"] == "passed"
    assert payload["metrics_are_not_real_dataset_results"] is True
    assert verify_artifact_hash(payload)
    assert (
        json.loads(report.read_text(encoding="utf-8"))["artifact_sha256"]
        == payload["artifact_sha256"]
    )

    scope = payload["scope"]
    assert scope["external_data"] is False
    assert scope["network"] is False
    assert scope["training"] is False
    assert scope["codabench"] is False

    checks = payload["checks"]
    assert checks["robustness"]["source_fixture_unchanged"] is True
    assert checks["robustness"]["all_predictions_finite"] is True
    assert checks["calibration"]["disjoint_ids"] is True
    assert checks["calibration"]["conformal"]["support_status"] == "supported"
    assert checks["calibration"]["temperature"]["support_status"] == "supported"
    assert checks["low_label"]["sequence_split_disjoint"] is True
    assert checks["low_label"]["selected_rows_are_train_only"] is True
    assert checks["export"]["metadata"]["verified_with_onnxruntime_cpu"] is True
    assert all(checks["no_leakage"].values())

    artifact_paths = [tmp_path / "phase12" / item["path"] for item in payload["artifacts"]]
    assert artifact_paths
    assert all(path.is_file() for path in artifact_paths)


def test_phase12_smoke_rejects_negative_seed(tmp_path: Path) -> None:
    from scripts.run_plan_phase12_smoke import run_smoke

    try:
        run_smoke(tmp_path, seed=-1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:  # pragma: no cover - assertion documents the contract
        raise AssertionError("Negative seeds must be rejected before creating a run.")
