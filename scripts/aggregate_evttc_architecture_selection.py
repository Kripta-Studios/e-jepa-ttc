"""Aggregate EvTTC architecture runs by fold/seed without window-level pseudoreplication."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.data.benchmark10_guard import assert_no_sealed_benchmark_paths
from e_jepa_ttc.utils.io import write_structured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        help="Exact fold IDs to aggregate; other run directories are ignored.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Exact seed IDs to aggregate; other run directories are ignored.",
    )
    args = parser.parse_args()
    if args.folds is not None and len(set(args.folds)) != args.expected_folds:
        raise ValueError("--folds must contain exactly --expected-folds unique IDs.")
    if args.seeds is not None and len(set(args.seeds)) != args.expected_seeds:
        raise ValueError("--seeds must contain exactly --expected-seeds unique IDs.")
    expected_pairs = (
        {(fold, seed) for fold in args.folds for seed in args.seeds}
        if args.folds is not None and args.seeds is not None
        else None
    )
    assert_no_sealed_benchmark_paths((args.root, args.output))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary_path in sorted(args.root.glob("fold-*/*/seed-*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        variant = summary_path.parents[1].name
        fold = int(summary_path.parents[2].name.split("-")[-1])
        seed = int(summary_path.parent.name.split("-")[-1])
        if args.folds is not None and fold not in args.folds:
            continue
        if args.seeds is not None and seed not in args.seeds:
            continue
        grouped[variant].append(
            {
                "fold": fold,
                "seed": seed,
                "sequence_macro_selection_score": float(
                    payload["validation"]["sequence_macro_selection_score"]
                ),
                "sequence_macro_mean_relative_error": float(
                    payload["validation"]["sequence_macro_mean_relative_error"]
                ),
                "sequence_macro_mae_s": float(payload["validation"]["sequence_macro_mae_s"]),
                "worst_sequence_selection_score": float(
                    payload["validation"]["worst_sequence_selection_score"]
                ),
                "worst_sequence_mae_s": float(payload["validation"]["worst_sequence_mae_s"]),
                "milliseconds_per_window": payload["validation"].get("milliseconds_per_window"),
                "peak_vram_bytes": payload.get("peak_vram_bytes"),
                "summary": summary_path.as_posix(),
            }
        )
    rows: list[dict[str, Any]] = []
    required_runs = args.expected_folds * args.expected_seeds
    for variant, runs in sorted(grouped.items()):
        selection = np.asarray([run["sequence_macro_selection_score"] for run in runs])
        relative_error = np.asarray(
            [run["sequence_macro_mean_relative_error"] for run in runs]
        )
        mae = np.asarray([run["sequence_macro_mae_s"] for run in runs])
        run_pairs = {(run["fold"], run["seed"]) for run in runs}
        complete = (
            run_pairs == expected_pairs
            if expected_pairs is not None
            else len(run_pairs) == required_runs
        )
        rows.append(
            {
                "variant": variant,
                "run_count": len(runs),
                "required_run_count": required_runs,
                "complete_for_final_selection": complete,
                "folds": sorted({run["fold"] for run in runs}),
                "seeds": sorted({run["seed"] for run in runs}),
                "sequence_macro_selection_score_mean": float(selection.mean()),
                "sequence_macro_selection_score_std": (
                    float(selection.std(ddof=1)) if selection.shape[0] > 1 else 0.0
                ),
                "sequence_macro_mean_relative_error_mean": float(relative_error.mean()),
                "sequence_macro_mean_relative_error_std": (
                    float(relative_error.std(ddof=1))
                    if relative_error.shape[0] > 1
                    else 0.0
                ),
                "sequence_macro_mae_s_mean": float(mae.mean()),
                "sequence_macro_mae_s_std": (float(mae.std(ddof=1)) if mae.shape[0] > 1 else 0.0),
                "worst_run_sequence_selection_score": float(
                    max(run["worst_sequence_selection_score"] for run in runs)
                ),
                "worst_run_sequence_mae_s": float(
                    max(run["worst_sequence_mae_s"] for run in runs)
                ),
                "runs": runs,
            }
        )
    rows.sort(key=lambda row: row["sequence_macro_selection_score_mean"])
    payload = {
        "protocol": "evttc_architecture_grouped_cv_multiseed_aggregate_v2",
        "expected_folds": args.expected_folds,
        "expected_seeds": args.expected_seeds,
        "expected_fold_ids": args.folds,
        "expected_seed_ids": args.seeds,
        "required_runs_per_variant": required_runs,
        "all_variants_complete": bool(rows)
        and all(bool(row["complete_for_final_selection"]) for row in rows),
        "ranking": rows,
        "window_level_bootstrap_used": False,
        "benchmark10_opened": False,
    }
    write_structured(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
