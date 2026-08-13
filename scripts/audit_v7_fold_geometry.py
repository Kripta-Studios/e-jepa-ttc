#!/usr/bin/env python
"""Audit V7 geometry retention against fold-local A4 and A8 references."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402

MEASURES = ("slope", "prediction_target_std_ratio")
RELATIONSHIPS = ("delta_log_height_vs_bbox", "delta_log_height_vs_physical")


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not verify_artifact_hash(payload):
        raise ValueError(f"invalid signed summary: {path}")
    return payload


def _extract(summary: dict[str, Any]) -> dict[str, float]:
    geometry = summary.get("dev_metrics", {}).get("geometry_diagnostics")
    if not isinstance(geometry, dict):
        raise ValueError("summary lacks dev geometry diagnostics")
    result: dict[str, float] = {}
    for relationship in RELATIONSHIPS:
        macro = geometry.get(relationship, {}).get("macro_by_sequence")
        if not isinstance(macro, dict):
            raise ValueError(f"summary lacks {relationship} macro diagnostics")
        for measure in MEASURES:
            value = float(macro[measure])
            if not np.isfinite(value):
                raise ValueError(f"non-finite geometry metric: {relationship}.{measure}")
            result[f"{relationship}.{measure}"] = value
    return result


def audit(
    *,
    candidate_summaries: list[Path],
    reference_summaries: list[Path],
    output: Path,
    minimum_retention: float,
) -> dict[str, Any]:
    """Require positive geometry and at least the frozen fraction of reference value."""

    if len(candidate_summaries) != 3 or len(reference_summaries) != 3:
        raise ValueError("geometry audit requires three candidate and three reference folds")
    candidate = [_read(path) for path in candidate_summaries]
    reference = [_read(path) for path in reference_summaries]
    candidate_values = [_extract(summary) for summary in candidate]
    reference_values = [_extract(summary) for summary in reference]
    keys = sorted(candidate_values[0])
    metrics: dict[str, Any] = {}
    passed = True
    for key in keys:
        candidate_mean = float(np.mean([fold[key] for fold in candidate_values]))
        reference_mean = float(np.mean([fold[key] for fold in reference_values]))
        retention = candidate_mean / reference_mean if reference_mean > 0.0 else float("nan")
        metric_passed = bool(
            candidate_mean > 0.0
            and reference_mean > 0.0
            and np.isfinite(retention)
            and retention >= minimum_retention
        )
        passed = passed and metric_passed
        metrics[key] = {
            "candidate_mean": candidate_mean,
            "reference_mean": reference_mean,
            "retention": retention,
            "positive": candidate_mean > 0.0,
            "passed": metric_passed,
        }
    report: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v7_geometry_retention_audit_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "completed",
        "minimum_retention": minimum_retention,
        "metrics": metrics,
        "geometry_positive": passed,
        "sources": {
            "candidate_summaries": [str(path.resolve()) for path in candidate_summaries],
            "reference_summaries": [str(path.resolve()) for path in reference_summaries],
        },
        "closed_evaluation": {
            "public_validation_used": False,
            "private_test_opened": False,
            "evttc_test_opened": False,
            "codabench_opened": False,
        },
    }
    sign_artifact(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-summaries", type=Path, nargs=3, required=True)
    parser.add_argument("--reference-summaries", type=Path, nargs=3, required=True)
    parser.add_argument("--minimum-retention", type=float, default=0.60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(
            candidate_summaries=[path.resolve(strict=True) for path in args.candidate_summaries],
            reference_summaries=[path.resolve(strict=True) for path in args.reference_summaries],
            output=args.output.resolve(),
            minimum_retention=args.minimum_retention,
        )
    except Exception as error:
        parser.exit(2, f"V7 geometry audit failed: {type(error).__name__}: {error}\n")
    print(
        json.dumps(
            {"output": str(args.output), "geometry_positive": report["geometry_positive"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
