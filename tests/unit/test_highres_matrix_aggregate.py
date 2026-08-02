from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.aggregate_highres_matrix_v1 import (
    _canonical_sha256,
    aggregate_highres_matrix,
)


def _write_artifact(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["artifact_sha256"] = _canonical_sha256(payload)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _source_payloads() -> dict[str, dict[str, Any]]:
    architecture_rows = [
        {"arm": arm, "global_attention_used": False, "causal_smoke": True}
        for arm in (
            "S3_R4_WINDOW_TEMPORAL",
            "S4_R4_WINDOW_MERGE_TEMPORAL",
            "S5_R4_WINDOW_MERGE_KDA",
        )
    ]
    temporal_rows = [
        {
            "arm": arm,
            "temporal_steps": steps,
            "global_attention_used": False,
            "selection_allowed": False,
        }
        for arm in (
            "S4_R4_WINDOW_MERGE_TEMPORAL",
            "S5_R4_WINDOW_MERGE_KDA",
        )
        for steps in (2, 5, 8, 16, 32)
    ]
    token_rows = [
        {
            "name": "R1",
            "theoretical_oom_guard_required": False,
            "theoretical_oom_guard_triggered": False,
        },
        {
            "name": "R2",
            "theoretical_oom_guard_required": False,
            "theoretical_oom_guard_triggered": False,
        },
        {
            "name": "R3",
            "theoretical_oom_guard_required": False,
            "theoretical_oom_guard_triggered": False,
        },
        {
            "name": "R4",
            "theoretical_oom_guard_required": True,
            "theoretical_oom_guard_triggered": True,
            "global_guard_error": "guarded before allocation",
        },
    ]

    def real_row(arm: str, mid: float, rte: float) -> dict[str, Any]:
        return {
            "arm": arm,
            "metrics_scope": "public_garl_eap_validation_short_screen",
            "global_attention_used": False,
            "validation_metrics": {
                "paper_MiD_overall": mid,
                "weighted_RTE_pct": rte,
                "failure_rate_pct": 0.0,
            },
        }

    return {
        "architecture": {
            "artifact_type": "highres_architecture_screen_v1",
            "schema_version": "v1",
            "status": "pass",
            "selection_allowed": False,
            "metrics_scope": "architecture_forward_backward_smoke",
            "results": architecture_rows,
        },
        "temporal": {
            "artifact_type": "highres_temporal_sweep_v1",
            "schema_version": "v1",
            "status": "pass",
            "selection_allowed": False,
            "metrics_scope": "forward_latency_and_theoretical_scaling_only",
            "results": temporal_rows,
        },
        "token": {
            "artifact_type": "highres_token_scaling_benchmark_v1",
            "schema_version": "v1",
            "status": "pass",
            "selection_allowed": False,
            "global_attention_over_r4_allocated": False,
            "results": token_rows,
        },
        "real": {
            "artifact_type": "highres_real_architecture_screen_v1",
            "schema_version": "v1",
            "status": "pass",
            "selection_allowed": False,
            "arms": [
                real_row("S3_R4_WINDOW_TEMPORAL", 210.0, 50.0),
                real_row("S4_R4_WINDOW_MERGE_TEMPORAL", 208.5853148794087, 54.442653582486756),
                real_row("S5_R4_WINDOW_MERGE_KDA", 205.08335208214112, 56.30085943441982),
            ],
            "s4_vs_s5": {
                "primary_metric": "paper_MiD_overall",
                "s4": 208.5853148794087,
                "s5": 205.08335208214112,
                "delta_s5_minus_s4": -3.501962797267576,
                "decision": "regression_in_short_screen",
            },
        },
    }


def _write_sources(root: Path, payloads: dict[str, dict[str, Any]]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, payload in payloads.items():
        path = root / f"{name}.json"
        _write_artifact(path, payload)
        paths[name] = path
    return paths


def test_aggregate_preserves_metrics_and_rejects_s5_kda(tmp_path: Path) -> None:
    paths = _write_sources(tmp_path, _source_payloads())
    result = aggregate_highres_matrix(
        architecture_screen_path=paths["architecture"],
        temporal_sweep_path=paths["temporal"],
        token_scaling_path=paths["token"],
        real_screen_path=paths["real"],
        repo_root=tmp_path,
        code_commit="test-commit",
        generated_at="2026-08-01T00:00:00+00:00",
    )

    assert result["status"] == "pass"
    assert result["training_executed"] is False
    assert result["decisions"]["s5_kda"]["status"] == "rejected"
    assert result["decisions"]["s5_kda"]["source_decision"] == "regression_in_short_screen"
    assert result["decisions"]["s5_kda"]["regressed_metrics"] == ["weighted_RTE_pct"]
    assert (
        result["matrix"]["real_screen"]["rows"][2]["validation_metrics"]["weighted_RTE_pct"]
        == 56.30085943441982
    )
    assert (
        result["safety_gates"]["r4_global_attention_not_allocated"][
            "r4_theoretical_oom_guard_triggered"
        ]
        is True
    )
    assert result["artifact_sha256"] == _canonical_sha256(result)
    assert len(result["matrix"]["temporal_sweep"]["rows"]) == 10


def test_historical_k1_is_never_mixed_into_matrix(tmp_path: Path) -> None:
    payloads = _source_payloads()
    payloads["architecture"]["results"].append(
        {"arm": "K1_OBJECT_KDA", "global_attention_used": False}
    )
    paths = _write_sources(tmp_path, payloads)
    result = aggregate_highres_matrix(
        architecture_screen_path=paths["architecture"],
        temporal_sweep_path=paths["temporal"],
        token_scaling_path=paths["token"],
        real_screen_path=paths["real"],
        repo_root=tmp_path,
        code_commit="test-commit",
        generated_at="2026-08-01T00:00:00+00:00",
    )

    matrix_text = json.dumps(result["matrix"], sort_keys=True)
    assert "K1_OBJECT_KDA" not in matrix_text
    assert result["decisions"]["historical_k1"]["included_in_matrix"] is False
    assert result["decisions"]["historical_k1"]["mixed_with_s3_s4_s5"] is False
    assert "K1_OBJECT_KDA" in result["excluded_arm_ids"]


def test_real_screen_without_regression_fails_closed(tmp_path: Path) -> None:
    payloads = _source_payloads()
    payloads["real"]["s4_vs_s5"]["decision"] = "no_regression_in_short_screen"
    paths = _write_sources(tmp_path, payloads)

    with pytest.raises(ValueError, match="does not report the required"):
        aggregate_highres_matrix(
            architecture_screen_path=paths["architecture"],
            temporal_sweep_path=paths["temporal"],
            token_scaling_path=paths["token"],
            real_screen_path=paths["real"],
            repo_root=tmp_path,
            code_commit="test-commit",
            generated_at="2026-08-01T00:00:00+00:00",
        )
