"""Build a compact signed comparison from causal-scale diagnostic summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_run(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("run must have LABEL=SUMMARY_JSON form")
    return label, Path(raw_path)


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def build(runs: list[tuple[str, Path]]) -> dict[str, Any]:
    """Load source summaries and retain only audit and selected-validation fields."""

    rows: list[dict[str, Any]] = []
    for label, path in runs:
        resolved = path.resolve()
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("stage") != "diagnostic":
            raise ValueError(f"{path} is not a diagnostic summary")
        metrics = raw.get("validation_metrics")
        selection = raw.get("selection")
        if not isinstance(metrics, dict) or not isinstance(selection, dict):
            raise ValueError(f"{path} lacks validation metrics or selection")
        source_identity = raw.get("artifact_identity")
        rows.append(
            {
                "label": label,
                "source_path": resolved.relative_to(ROOT).as_posix(),
                "source_file_sha256": _file_sha256(resolved),
                "source_artifact_identity": source_identity,
                "source_git_commit": raw.get("git_commit"),
                "source_git_dirty": raw.get("git_dirty"),
                "source_status": raw.get("status"),
                "best_epoch": int(selection["best_epoch"]),
                "elapsed_s": float(raw["elapsed_s"]),
                "analytic_pearson": _metric(metrics, "analytic_pearson"),
                "slope": _metric(metrics, "slope"),
                "sign_accuracy": _metric(metrics, "sign_accuracy"),
                "foreground_iou": _metric(metrics, "foreground_iou"),
                "log_ratio_mae": _metric(metrics, "log_ratio_mae"),
                "ttc_symmetric_relative_error": _metric(
                    metrics,
                    "ttc_symmetric_relative_error",
                ),
                "ratio_80_coverage": _metric(metrics, "ratio_80_coverage"),
                "translation_leakage_p95": _metric(
                    metrics,
                    "translation_leakage_p95",
                ),
            }
        )
    payload: dict[str, Any] = {
        "artifact_type": "causal_scale_v5_diagnostic_comparison_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "selectable": False,
        "evidence_scope": "synthetic_train_validation_diagnostics_only",
        "test_opened": False,
        "real_data_opened": False,
        "sota_claim_authorized": False,
        "rows": rows,
    }
    sign_artifact(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=_parse_run)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"output exists: {output}")
    payload = build(args.run)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with output.open("xb") as handle:
        handle.write(serialized.encode("utf-8"))
    print(json.dumps({"output": str(output), "rows": len(payload["rows"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
