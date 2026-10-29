"""Packaging and provenance guards for the independent official checkpoint runs."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts import run_garl_evttc_competition as runner


def _row(index: int, prediction: float | None) -> dict:
    return {
        "schema": runner.ROW_SCHEMA,
        "freeze_sha256": "fixed",
        "variant": "full",
        "identity": {
            "sequence_id": "fixture",
            "row_index": index,
            "index": index + 7,
            "timestamp_text": f"0.{index}000000001",
        },
        "status": "ok" if prediction is not None else "unavailable",
        "prediction_ttc_seconds": prediction,
        "cost_time_seconds": 0.25,
        "reason": "available" if prediction is not None else "missing_real_rgb_pair",
    }


def _expected() -> dict:
    return {"fixture": [(7, "0.0000000001"), (8, "0.1000000001")]}


def test_complete_archive_preserves_signed_values_and_timestamp_text(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "EXPECTED_ROWS", 2)
    summary = runner.package_variant(tmp_path, "full", [_row(0, -3.25), _row(1, 2.0)], _expected())
    assert summary["status"] == "COMPLETE"
    with zipfile.ZipFile(tmp_path / summary["results_zip"]) as archive:
        assert archive.namelist() == ["fixture/results.txt"]
        text = archive.read("fixture/results.txt").decode()
    assert "7\t0.0000000001\t-3.25\t0.25" in text


def test_unavailable_rows_produce_only_explicit_partial_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "EXPECTED_ROWS", 2)
    rows = [_row(0, -3.25), _row(1, None)]
    summary = runner.package_variant(tmp_path, "full", rows, _expected())
    assert summary["status"] == "PARTIAL" and summary["null_rows"] == 1
    assert not (tmp_path / "full/results.zip").exists()
    with zipfile.ZipFile(tmp_path / summary["partial_results_zip"]) as archive:
        assert "PARTIAL_NOT_SUBMISSION.txt" in archive.namelist()
        assert not any(name.endswith("results.txt") for name in archive.namelist())
        manifest = json.loads(archive.read("partial_manifest.json"))
    assert len(manifest["unavailable_rows"]) == 1


def test_missing_identity_and_mixed_checkpoint_package_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "EXPECTED_ROWS", 2)
    with pytest.raises(ValueError, match="identities"):
        runner.package_variant(tmp_path, "full", [_row(0, 2.0)], _expected())
    other = _row(1, 3.0)
    other["variant"] = "event"
    with pytest.raises(ValueError, match="Mixed checkpoints"):
        runner.package_variant(tmp_path, "full", [_row(0, 2.0), other], _expected())


def test_resume_rejects_changed_freeze_and_duplicate_rows(tmp_path):
    path: Path = tmp_path / "rows.jsonl"
    runner.append_row(path, _row(0, 1.0))
    with pytest.raises(ValueError, match="freeze/schema"):
        runner.load_rows(path, "changed", resume=True)
    assert len(runner.load_rows(path, "fixed", resume=True)) == 1
    runner.append_row(path, _row(0, 1.0))
    with pytest.raises(ValueError, match="Duplicate"):
        runner.load_rows(path, "fixed", resume=True)


def test_ground_truth_identity_receipt_rejects_wrong_population(tmp_path):
    path = tmp_path / "ground_truth_identity.json"
    audit = {
        "schema": "evttc_competition_gt_identity_inventory_v1",
        "status": "NO_VERIFIED_EXACT_QUERY_GROUND_TRUTH",
        "query_count": 434,
    }
    path.write_text(json.dumps(audit), encoding="utf-8")
    record = runner.ground_truth_record(path)
    assert record["sha256"] == runner.sha256_file(path)
    audit["query_count"] = 433
    path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="exact-query GT audit"):
        runner.ground_truth_record(path)
