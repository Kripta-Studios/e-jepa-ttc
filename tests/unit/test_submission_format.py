from __future__ import annotations

from pathlib import Path

from e_jepa_ttc.evaluation.submission_writer import (
    SUBMISSION_COLUMNS,
    validate_submission_root,
    write_sequence_results,
    write_submission_manifest,
)


def test_submission_writer_and_validator_preserve_query_identity(tmp_path: Path) -> None:
    sequence = write_sequence_results(
        tmp_path,
        sequence_id="CCRs1-low",
        indices=[0, 1],
        timestamps=[1000, 2000],
        ttc_seconds=[2.5, 2.0],
        cost_time_seconds=[0.01, 0.012517],
    )
    write_submission_manifest(
        tmp_path,
        candidate_name="SINGLE_REALTIME",
        sequence_files=[sequence],
        checkpoint_sha256="a" * 64,
        freeze_manifest_sha256="b" * 64,
        runtime_environment={"gpu": "test"},
    )
    report = validate_submission_root(
        tmp_path,
        expected_queries={"CCRs1-low": [(0, 1000.0), (1, 2000.0)]},
        require_sequences=1,
    )
    assert report["valid"] is True
    header = (tmp_path / "CCRs1-low" / "results.txt").read_text(encoding="utf-8").splitlines()[0]
    assert tuple(header.split("\t")) == SUBMISSION_COLUMNS


def test_submission_validator_rejects_tampering(tmp_path: Path) -> None:
    sequence = write_sequence_results(
        tmp_path,
        sequence_id="Slider-750",
        indices=[0],
        timestamps=[1000],
        ttc_seconds=[1.5],
        cost_time_seconds=[0.01],
    )
    write_submission_manifest(
        tmp_path,
        candidate_name="ENSEMBLE_ACCURACY",
        sequence_files=[sequence],
        checkpoint_sha256="a" * 64,
        freeze_manifest_sha256="b" * 64,
        runtime_environment={},
    )
    path = tmp_path / "Slider-750" / "results.txt"
    path.write_text(path.read_text(encoding="utf-8").replace("1.5", "-1"), encoding="utf-8")
    report = validate_submission_root(tmp_path)
    assert report["valid"] is False
    assert any("Hash mismatch" in error for error in report["errors"])
