"""Build a compact signed comparison from complete V8 diagnostic summaries."""

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", action="append", nargs=2, metavar=("NAME", "SUMMARY"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for name, raw_path in args.arm:
        path = Path(raw_path).resolve()
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["stage"] != "diagnostic" or summary["test_evaluation_count"] != 0:
            raise ValueError(f"arm {name} is not a sealed diagnostic")
        if summary["opened_splits"] != ["train", "validation"]:
            raise ValueError(f"arm {name} opened an unexpected split")
        rows.append(
            {
                "name": name,
                "summary_sha256": _sha256(path),
                "git_commit": summary["git_commit"],
                "git_dirty": summary["git_dirty"],
                "model_config": summary["model_config"],
                "loss_override": summary.get("loss_override"),
                "selection": summary["selection"],
                "validation_metrics": summary["validation_metrics"],
                "validation_group_metrics": summary["validation_group_metrics"],
                "opened_splits": summary["opened_splits"],
                "test_evaluation_count": summary["test_evaluation_count"],
            }
        )
    selected = min(rows, key=lambda row: float(row["selection"]["best_score"]))
    payload: dict[str, Any] = {
        "artifact_type": "causal_scale_v8_diagnostic_comparison_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "completed_validation_only",
        "selection_metric": "0.5_macro_plus_0.5_worst_group_score",
        "arms": rows,
        "selected_arm": selected["name"],
        "synthetic_gate_passed": False,
        "sealed_test_groups": [901, 902, 903],
        "real_data_opened": False,
        "sota_claim_authorized": False,
    }
    sign_artifact(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected": selected["name"],
                "artifact_sha256": payload["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
