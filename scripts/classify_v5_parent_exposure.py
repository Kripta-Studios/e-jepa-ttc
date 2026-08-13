"""Classify grouped runs initialized from the globally exposed A4 parent."""

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

GLOBAL_PARENT = ROOT / "artifacts/runs/scientific_recovery_a4_causal_left_seed7/model_best.pt"
RUN_NAMES = (
    "scientific_recovery_v5_a6_grouped_fold0_seed7",
    "scientific_recovery_v5_a6_grouped_fold1_seed7",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def build_report(
    run_root: Path,
    *,
    global_parent: Path = GLOBAL_PARENT,
    run_names: tuple[str, ...] = RUN_NAMES,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    """Build a fail-closed classification without modifying affected runs."""

    parent_sha = _sha256(global_parent)
    affected: list[dict[str, Any]] = []
    for name in run_names:
        path = run_root / name
        summary_path = path / "summary.json"
        progress_path = path / "state/progress.json"
        entry: dict[str, Any] = {
            "run_name": name,
            "path": _display_path(path, root=repository_root),
            "promotion_eligible": False,
            "status": "diagnostic_parent_exposed",
            "summary_present": summary_path.is_file(),
            "interrupted": not summary_path.is_file(),
        }
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            observed_parent = summary.get("initialization", {}).get("checkpoint_sha256")
            if observed_parent != parent_sha:
                raise ValueError(f"{name} does not bind the classified global parent")
            entry.update(
                {
                    "summary_sha256": _sha256(summary_path),
                    "summary_artifact_sha256": summary.get("artifact_sha256"),
                    "observed_parent_sha256": observed_parent,
                    "selection": summary.get("selection"),
                    "summary_left_unmodified": True,
                }
            )
        elif progress_path.is_file():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("status") != "running":
                raise ValueError(f"{name} incomplete progress has unexpected status")
            entry.update(
                {
                    "progress_sha256": _sha256(progress_path),
                    "last_completed_epoch": progress.get("epoch"),
                    "last_observed_best_selection": progress.get("best_selection"),
                    "termination_reason": "parent_exposure_detected",
                }
            )
        else:
            raise FileNotFoundError(f"affected run has no summary or progress: {path}")
        affected.append(entry)

    report: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v5_parent_exposure_classification_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "completed_promotion_blocked",
        "reason": "A4 parent was trained on all nine outer-fold sequences",
        "global_parent": {
            "path": _display_path(global_parent, root=repository_root),
            "sha256": parent_sha,
            "outer_dev_sequences_were_used_for_parent_gradients": True,
        },
        "affected_runs": affected,
        "required_replacement": (
            "fold-specific A4 parent trained only on each outer fold train partition"
        ),
        "contracts": {
            "affected_results_are_diagnostic_only": True,
            "affected_results_may_not_enter_a8_promotion": True,
            "affected_summaries_modified": False,
            "public_validation_opened": False,
            "private_test_opened": False,
        },
    }
    sign_artifact(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path, default=ROOT / "artifacts/runs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "artifacts/scientific_recovery_v5/audit/"
            "parent_exposure_classification.json"
        ),
    )
    args = parser.parse_args()
    try:
        report = build_report(args.run_root.resolve())
    except Exception as error:
        parser.exit(2, f"parent exposure classification failed: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
