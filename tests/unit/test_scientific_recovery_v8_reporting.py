from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash
from scripts.build_scientific_recovery_v8_report import build_report, sha256_file
from scripts.package_scientific_recovery_v8_evidence import package_evidence


def _protocol(repo_root: Path) -> None:
    path = repo_root / "configs" / "protocol" / "scientific_recovery_v8_temporal.json"
    path.parent.mkdir(parents=True)
    payload = sign_artifact({"artifact_type": "scientific_recovery_v8_temporal_protocol_v1"})
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_report_is_blocked_without_v8_results_and_keeps_sealed_evaluations(tmp_path: Path) -> None:
    _protocol(tmp_path)

    report = build_report(tmp_path, tmp_path / "artifacts" / "scientific_recovery_v8" / "report")

    assert report["status"] == "blocked"
    assert report["evidence_counts"]["runs"] == 0
    assert report["sota_claim_allowed"] is False
    assert set(report["sealed_evaluations"].values()) == {"sealed"}
    assert any(item["phase"] == "C1" and "blocked" in item["status"] for item in report["phases"])
    assert verify_artifact_hash(report)


def test_report_accepts_only_signed_json_and_signed_csv_sources(tmp_path: Path) -> None:
    _protocol(tmp_path)
    root = tmp_path / "artifacts" / "scientific_recovery_v8" / "results"
    root.mkdir(parents=True)
    predictions = root / "timevol20_3_predictions.csv"
    predictions.write_text("token_id,prediction_ttc\na,1.0\n", encoding="utf-8")
    signed = sign_artifact(
        {
            "artifact_type": "scientific_recovery_v8_timevol20_3_aggregate_v1",
            "status": "completed_negative",
            "arm": "timevol20_3",
            "seed": 7,
            "sources": {
                "predictions": {
                    "path": str(predictions.relative_to(tmp_path)),
                    "sha256": sha256_file(predictions),
                }
            },
        }
    )
    (root / "aggregate.json").write_text(json.dumps(signed), encoding="utf-8")
    (root / "unsigned.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")

    report = build_report(tmp_path, tmp_path / "artifacts" / "scientific_recovery_v8" / "report")

    assert report["evidence_counts"]["validated_json"] == 1
    assert report["evidence_counts"]["validated_csv"] == 1
    assert report["runs"][0]["phase"] == "B1"
    assert any("signature" in item["error"] for item in report["validation_errors"])
    report_csv = tmp_path / "artifacts" / "scientific_recovery_v8" / "report"
    assert (report_csv / "scientific_recovery_v8_runs.csv").is_file()


def test_report_rejects_signed_result_without_result_schema(tmp_path: Path) -> None:
    _protocol(tmp_path)
    root = tmp_path / "artifacts" / "scientific_recovery_v8" / "results"
    root.mkdir(parents=True)
    malformed = sign_artifact({"artifact_type": "scientific_recovery_v8_b1"})
    (root / "malformed.json").write_text(json.dumps(malformed), encoding="utf-8")

    report = build_report(tmp_path, tmp_path / "artifacts" / "scientific_recovery_v8" / "report")

    assert report["evidence_counts"]["runs"] == 0
    assert any("non-empty status" in item["error"] for item in report["validation_errors"])


def test_package_is_deterministic_excludes_checkpoint_and_refuses_replacement(
    tmp_path: Path,
) -> None:
    _protocol(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "SCIENTIFIC_RECOVERY_V8_STATUS.md").write_text("status\n", encoding="utf-8")
    (docs / "SCIENTIFIC_RECOVERY_V8_REPORT.md").write_text("report\n", encoding="utf-8")
    config = tmp_path / "configs" / "experiment" / "scientific_recovery_v8_fold_chain"
    config.mkdir(parents=True)
    (config / "frozen_manifest.json").write_text("{}\n", encoding="utf-8")
    run = tmp_path / "artifacts" / "runs" / "scientific_recovery_v8_timevol20_3_fold0_seed7"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps(sign_artifact({"status": "failed"})), encoding="utf-8"
    )
    (run / "model_best.pt").write_bytes(b"checkpoint")
    output = tmp_path / "artifacts" / "scientific_recovery_v8" / "package" / "evidence.zip"

    manifest = package_evidence(tmp_path, output)

    assert output.is_file()
    assert verify_artifact_hash(manifest)
    assert manifest["checkpoints"][0]["path"].endswith("model_best.pt")
    with zipfile.ZipFile(output) as archive:
        checkpoint_name = (
            "artifacts/runs/scientific_recovery_v8_timevol20_3_fold0_seed7/model_best.pt"
        )
        assert checkpoint_name not in archive.namelist()
        assert "evidence_manifest.json" in archive.namelist()
    with pytest.raises(FileExistsError):
        package_evidence(tmp_path, output)
