"""Build the signed, label-free matched eAP subset manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.data.matched_eap_subset import (  # noqa: E402
    MatchedSubsetConfig,
    build_matched_eap_subset,
    validate_code_commit,
)


def _floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float.")
    return result


def _ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer.")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Project only the label-free GarlTTC data/train.parquet allow-list and "
            "write a signed nested whole-track-block manifest."
        )
    )
    parser.add_argument("--garlttc-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True, help="Frozen sequence split JSON.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--code-commit",
        help="Optional exact HEAD commit override; defaults to git rev-parse HEAD.",
    )
    parser.add_argument(
        "--diagnostic-override",
        action="store_true",
        help="Allow a dirty worktree and mark output diagnostic-only (not a claim).",
    )
    parser.add_argument("--horizons-s", type=_floats, default=(0.1, 0.2, 0.3))
    parser.add_argument("--horizon-tolerance-s", type=float, default=0.025)
    parser.add_argument("--exclusion-window-s", type=float, default=0.02)
    parser.add_argument("--stage-sizes", type=_ints, default=(256, 512, 1024, 2048))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--update-budget", type=int, default=1_000)
    parser.add_argument(
        "--allow-gate-failure",
        action="store_true",
        help="Write a diagnostic unsigned-for-promotion manifest when NCE gates fail.",
    )
    return parser


def _resolve_code_commit(explicit: str | None, *, diagnostic: bool) -> str:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Unable to resolve git HEAD/worktree status.") from exc
    commit = validate_code_commit(explicit or head)
    if explicit is not None and commit != validate_code_commit(head):
        raise ValueError("--code-commit must equal the repository HEAD commit.")
    if dirty and not diagnostic:
        raise ValueError(
            "Claim manifest build requires a clean worktree; use --diagnostic-override "
            "for a clearly non-claim artifact."
        )
    return commit


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code_commit = _resolve_code_commit(
            args.code_commit,
            diagnostic=args.diagnostic_override,
        )
        config = MatchedSubsetConfig(
            horizons_s=tuple(args.horizons_s),
            horizon_tolerance_s=args.horizon_tolerance_s,
            exclusion_window_s=args.exclusion_window_s,
            stage_sizes=tuple(args.stage_sizes),
            seed=args.seed,
            update_budget=args.update_budget,
        )
        manifest = build_matched_eap_subset(
            args.garlttc_root,
            args.split,
            config=config,
            output_path=args.output,
            code_commit=code_commit,
            diagnostic=args.diagnostic_override,
            strict_gate=not args.allow_gate_failure,
        )
        print(
            json.dumps(
                {
                    "status": "written",
                    "output": args.output.resolve().as_posix(),
                    "signature": manifest["signature"],
                    "stages": [
                        {
                            "stage": stage["stage"],
                            "nominal_row_count": stage["nominal_row_count"],
                            "actual_row_count": stage["actual_row_count"],
                        }
                        for stage in manifest["stages"]
                    ],
                    "nce_anchor_coverage": manifest["selection_report"]["nce_anchor_coverage"],
                    "minimum_negatives": manifest["selection_report"]["minimum_negatives"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(f"Matched eAP subset build rejected: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
