from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

import scripts.run_evttc_competition as runner
from scripts.run_evttc_competition import (
    finalize,
    validate_signed_results,
    write_signed_results,
)


def _record(*, row_index: int, index: int, timestamp: str, prediction: float) -> dict:
    return {
        "identity": {
            "sequence_id": "CCRs1-low",
            "row_index": row_index,
            "index": index,
            "timestamp_text": timestamp,
        },
        "status": "ok",
        "prediction_ttc_seconds": prediction,
        "cost_time_seconds": 0.0125,
        "input_details": {"event_observations": []},
    }


def test_local_writer_preserves_signed_prediction_and_timestamp_text(tmp_path: Path) -> None:
    expected = {"CCRs1-low": [(7, "0.1000000000000000")]}
    report = write_signed_results(
        tmp_path,
        [_record(row_index=0, index=7, timestamp=expected["CCRs1-low"][0][1], prediction=-2.5)],
        expected,
    )

    assert report["valid"] is True
    lines = (tmp_path / "CCRs1-low" / "results.txt").read_text(encoding="utf-8").splitlines()
    assert lines[1].split("\t")[:3] == ["7", "0.1000000000000000", "-2.5"]


def test_local_validator_rejects_query_identity_change(tmp_path: Path) -> None:
    expected = {"CCRs1-low": [(7, "0.1000000000000000")]}
    write_signed_results(
        tmp_path,
        [_record(row_index=0, index=7, timestamp=expected["CCRs1-low"][0][1], prediction=2.5)],
        expected,
    )
    path = tmp_path / "CCRs1-low" / "results.txt"
    path.write_text(
        path.read_text(encoding="utf-8").replace("0.1000000000000000", "0.1"), encoding="utf-8"
    )

    report = validate_signed_results(tmp_path, expected)

    assert report["valid"] is False
    assert "Official query identity mismatch: CCRs1-low" in report["errors"]


def test_incomplete_population_writes_partial_marker_and_never_zip(tmp_path: Path) -> None:
    expected = {"CCRs1-low": [(7, "0.1"), (8, "0.2")]}
    summary = finalize(
        output=tmp_path,
        records=[_record(row_index=0, index=7, timestamp="0.1", prediction=2.5)],
        expected=expected,
        freeze_sha256="a" * 64,
        device="cpu",
        prepare_only=False,
        limit=None,
        started_utc="2026-10-28T00:00:00+00:00",
    )

    assert summary["status"] == "PARTIAL"
    assert (tmp_path / "results_available.txt").read_text(encoding="utf-8").startswith("PARTIAL")
    assert not (tmp_path / "results.zip").exists()
    assert not (tmp_path / "results").exists()


def test_postflight_rejects_changed_frozen_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code_hashes = {"scripts/run_evttc_competition.py": "a" * 64}
    monkeypatch.setattr(runner, "source_hashes", lambda: code_hashes)
    monkeypatch.setattr(runner, "verify_candidate", lambda candidate: ({}, {"seal": "same"}))
    monkeypatch.setattr(
        runner, "verify_frozen_runtime", lambda candidate: {"archive_sha256": "changed"}
    )

    with pytest.raises(RuntimeError, match="Frozen runtime archive"):
        runner.verify_frozen_inputs_unchanged(
            data_root=tmp_path,
            candidate=tmp_path,
            frozen_model={"seal": "same"},
            frozen_runtime={"archive_sha256": "frozen"},
            frozen_parent_attempt=None,
            frozen_sources={},
            frozen_code_hashes=code_hashes,
        )


def test_resume_with_error_ledger_fails_actionably(tmp_path: Path) -> None:
    (tmp_path / "errors.jsonl").write_text('{"status":"error"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="Cannot resume a run with errors.jsonl"):
        runner.run_rows(
            data_root=tmp_path,
            candidate=tmp_path,
            output=tmp_path,
            device="cpu",
            rows_by_sequence={},
            calibrations={},
            freeze_sha256="a" * 64,
            prepare_only=False,
            limit=None,
            resume=True,
        )


def test_parent_attempt_is_bound_by_hash_without_modification(tmp_path: Path) -> None:
    freeze = tmp_path / "RUN_FREEZE.json"
    errors = tmp_path / "errors.jsonl"
    rows = tmp_path / "all_rows.jsonl"
    freeze.write_text('{"schema":"old"}\n', encoding="utf-8")
    errors.write_text('{"status":"error"}\n', encoding="utf-8")
    rows.write_text('{"status":"ok"}\n', encoding="utf-8")
    before = (freeze.read_bytes(), rows.read_bytes(), errors.read_bytes())

    record = runner.parent_attempt_record(tmp_path)

    assert record["run_freeze_sha256"] == runner.sha256_file(freeze)
    assert record["all_rows_ledger_sha256"] == runner.sha256_file(rows)
    assert record["errors_ledger_sha256"] == runner.sha256_file(errors)
    assert record["read_only"] is True
    assert (freeze.read_bytes(), rows.read_bytes(), errors.read_bytes()) == before


def test_terminal_partial_exports_distinct_available_predictions(tmp_path: Path) -> None:
    expected = {"CCRs1-low": [(7, "0.1"), (8, "0.2")]}
    available = _record(row_index=0, index=7, timestamp="0.1", prediction=-2.5)
    unavailable = {
        **_record(row_index=1, index=8, timestamp="0.2", prediction=1.0),
        "status": "unavailable",
        "prediction_ttc_seconds": None,
        "reason": "no_causal_support",
    }

    summary = finalize(
        output=tmp_path,
        records=[available, unavailable],
        expected=expected,
        freeze_sha256="a" * 64,
        device="cpu",
        prepare_only=False,
        limit=None,
        started_utc="2026-10-28T00:00:00+00:00",
    )

    assert summary["status"] == "PARTIAL"
    assert summary["partial_available_rows"] == 1
    assert summary["partial_unavailable_rows"] == 1
    assert not (tmp_path / "results.zip").exists()
    assert not (tmp_path / "results").exists()
    with zipfile.ZipFile(tmp_path / "partial_results.zip") as archive:
        names = set(archive.namelist())
        assert "CCRs1-low/available_predictions.tsv" in names
        assert "PARTIAL_NOT_SUBMISSION.txt" in names
        assert all(not name.endswith("/results.txt") for name in names)
        content = archive.read("CCRs1-low/available_predictions.tsv").decode("utf-8")
        assert "7\t0.1\t-2.5\t" in content
        assert "8\t0.2" not in content
