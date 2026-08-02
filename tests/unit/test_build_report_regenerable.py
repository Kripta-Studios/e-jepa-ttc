from __future__ import annotations

import json
from pathlib import Path

from e_jepa_ttc.artifacts.hashing import sign_artifact
from scripts.build_report import build_report


def test_report_indexes_result_formats_and_does_not_invent_completion(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    metrics = artifacts / "metrics"
    metrics.mkdir(parents=True)
    signed = sign_artifact({"artifact_type": "unit_metrics_v1", "status": "completed", "mae": 1.5})
    (metrics / "signed.json").write_text(json.dumps(signed), encoding="utf-8")
    (metrics / "untyped.json").write_text(json.dumps({"mae": 2.0}), encoding="utf-8")
    (metrics / "table.csv").write_text("seed,mae\n7,1.5\n", encoding="utf-8")
    (metrics / "rows.jsonl").write_text('{"seed":7}\n', encoding="utf-8")
    (metrics / "table.parquet").write_bytes(b"fixture-parquet")

    report = build_report(tmp_path, artifacts / "tables" / "regenerable_report")

    assert report["artifact_count"] == 5
    rows = {row["path"]: row for row in report["artifacts"]}
    assert rows["artifacts/metrics/signed.json"]["declared_artifact_sha256_valid"] is True
    assert rows["artifacts/metrics/untyped.json"]["status"] == "unknown"
    assert rows["artifacts/metrics/table.csv"]["format"] == "csv"
    assert rows["artifacts/metrics/rows.jsonl"]["format"] == "jsonl"
    assert rows["artifacts/metrics/table.parquet"]["format"] == "parquet"
    assert report["status_counts"]["unknown"] == 4
