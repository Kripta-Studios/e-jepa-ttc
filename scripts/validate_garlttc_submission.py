"""Validate a local GarlTTC submission without contacting the benchmark service."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_submission(
    submission_path: Path,
    sample_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    with submission_path.open("r", encoding="utf-8") as handle:
        submission = json.load(handle)
    with sample_path.open("r", encoding="utf-8") as handle:
        sample = json.load(handle)
    errors: list[str] = []
    if not isinstance(submission, dict):
        errors.append("submission root must be a JSON object")
    if not isinstance(sample, dict):
        errors.append("sample root must be a JSON object")
    submission_meta = submission.get("meta") if isinstance(submission, dict) else None
    results = submission.get("results") if isinstance(submission, dict) else None
    sample_results = sample.get("results") if isinstance(sample, dict) else None
    if (
        not isinstance(submission_meta, dict)
        or submission_meta.get("format") != "garlttc_prediction_v1"
    ):
        errors.append("meta.format must be garlttc_prediction_v1")
    if not isinstance(results, dict):
        errors.append("results must be an object")
        results = {}
    if not isinstance(sample_results, dict):
        errors.append("sample results must be an object")
        sample_results = {}
    sample_keys = set(sample_results)
    unknown_keys = sorted(set(results) - sample_keys)
    if unknown_keys:
        errors.append(f"unknown sample tokens: {unknown_keys[:3]}")
    for token, value in results.items():
        if not isinstance(value, dict) or set(value) != {"ttc"}:
            errors.append(f"{token}: value must contain only ttc")
            continue
        ttc = value.get("ttc")
        if (
            isinstance(ttc, bool)
            or not isinstance(ttc, (int, float))
            or not math.isfinite(float(ttc))
        ):
            errors.append(f"{token}: ttc must be finite numeric")
    result: dict[str, Any] = {
        "artifact_type": "garlttc_submission_validation_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "submission": submission_path.as_posix(),
        "submission_sha256": _sha256(submission_path),
        "sample_submission": sample_path.as_posix(),
        "sample_submission_sha256": _sha256(sample_path),
        "sample_token_count": len(sample_keys),
        "prediction_count": len(results),
        "token_coverage_fraction": len(set(results).intersection(sample_keys))
        / max(len(sample_keys), 1),
        "complete_token_coverage": set(results) == sample_keys,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "external_submission_sent": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_submission(args.submission, args.sample, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
