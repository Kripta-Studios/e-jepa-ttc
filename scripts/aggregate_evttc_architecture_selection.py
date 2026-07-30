"""Aggregate EvTTC architecture runs by fold/seed without window-level pseudoreplication."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.data.benchmark10_guard import assert_no_sealed_benchmark_paths
from e_jepa_ttc.evaluation.bootstrap import paired_sequence_bootstrap_difference
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
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "peak_vram_bytes": payload.get("peak_vram_bytes"),
                "git_commit": payload.get("git_commit"),
                "source_tree_sha256": payload.get("source_tree_sha256"),
                "run_fingerprint": payload.get("run_fingerprint"),
                "cache_manifest_sha256": payload.get("cache_manifest_sha256"),
                "sample_selection_sha256": payload.get("sample_selection_sha256"),
                "initial_backbone_sha256": payload.get("initial_backbone_sha256"),
                "initial_common_head_sha256": payload.get("initial_common_head_sha256"),
                "train_samples": payload.get("train_samples"),
                "validation_samples": payload.get("validation_samples"),
                "trainer": payload.get("trainer"),
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
        per_seed: list[dict[str, Any]] = []
        for seed in sorted({int(run["seed"]) for run in runs}):
            seed_runs = [run for run in runs if int(run["seed"]) == seed]
            per_seed.append(
                {
                    "seed": seed,
                    "folds": sorted(int(run["fold"]) for run in seed_runs),
                    "run_count": len(seed_runs),
                    "sequence_macro_selection_score_mean": float(
                        np.mean(
                            [run["sequence_macro_selection_score"] for run in seed_runs]
                        )
                    ),
                    "sequence_macro_mean_relative_error_mean": float(
                        np.mean(
                            [run["sequence_macro_mean_relative_error"] for run in seed_runs]
                        )
                    ),
                    "sequence_macro_mae_s_mean": float(
                        np.mean([run["sequence_macro_mae_s"] for run in seed_runs])
                    ),
                    "elapsed_seconds_sum": float(
                        np.sum(
                            [
                                float(run.get("elapsed_seconds") or 0.0)
                                for run in seed_runs
                            ]
                        )
                    ),
                }
            )
        seed_selection = np.asarray(
            [row["sequence_macro_selection_score_mean"] for row in per_seed]
        )
        seed_relative = np.asarray(
            [row["sequence_macro_mean_relative_error_mean"] for row in per_seed]
        )
        seed_mae = np.asarray([row["sequence_macro_mae_s_mean"] for row in per_seed])
        latency = np.asarray(
            [
                float(run["milliseconds_per_window"])
                for run in runs
                if run["milliseconds_per_window"] is not None
            ]
        )
        elapsed = np.asarray(
            [float(run["elapsed_seconds"]) for run in runs if run["elapsed_seconds"] is not None]
        )
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
                    float(seed_selection.std(ddof=1))
                    if seed_selection.shape[0] > 1
                    else 0.0
                ),
                "fold_seed_selection_score_std": (
                    float(selection.std(ddof=1)) if selection.shape[0] > 1 else 0.0
                ),
                "sequence_macro_mean_relative_error_mean": float(relative_error.mean()),
                "sequence_macro_mean_relative_error_std": (
                    float(seed_relative.std(ddof=1))
                    if seed_relative.shape[0] > 1
                    else 0.0
                ),
                "fold_seed_mean_relative_error_std": (
                    float(relative_error.std(ddof=1))
                    if relative_error.shape[0] > 1
                    else 0.0
                ),
                "sequence_macro_mae_s_mean": float(mae.mean()),
                "sequence_macro_mae_s_std": (
                    float(seed_mae.std(ddof=1)) if seed_mae.shape[0] > 1 else 0.0
                ),
                "fold_seed_mae_s_std": (
                    float(mae.std(ddof=1)) if mae.shape[0] > 1 else 0.0
                ),
                "worst_run_sequence_selection_score": float(
                    max(run["worst_sequence_selection_score"] for run in runs)
                ),
                "worst_run_sequence_mae_s": float(
                    max(run["worst_sequence_mae_s"] for run in runs)
                ),
                "milliseconds_per_window_mean": (
                    float(latency.mean()) if latency.size else None
                ),
                "milliseconds_per_window_std": (
                    float(latency.std(ddof=1)) if latency.size > 1 else 0.0
                ),
                "elapsed_seconds_sum": float(elapsed.sum()) if elapsed.size else None,
                "peak_vram_bytes_max": max(
                    (int(run["peak_vram_bytes"]) for run in runs if run["peak_vram_bytes"]),
                    default=None,
                ),
                "runs": runs,
                "per_seed": per_seed,
            }
        )
    rows.sort(key=lambda row: row["sequence_macro_selection_score_mean"])
    matched_audits: list[dict[str, Any]] = []
    matched_variants = ("A0_MATCHED_GLOBAL", "A1_MATCHED_DENSE_BLOCK")
    if all(variant in grouped for variant in matched_variants):
        run_maps = {
            variant: {(run["fold"], run["seed"]): run for run in grouped[variant]}
            for variant in matched_variants
        }
        pairs = expected_pairs or set(run_maps[matched_variants[0]])
        control_fields = (
            "cache_manifest_sha256",
            "sample_selection_sha256",
            "initial_backbone_sha256",
            "initial_common_head_sha256",
            "train_samples",
            "validation_samples",
            "trainer",
        )
        for pair in sorted(pairs):
            baseline = run_maps[matched_variants[0]].get(pair)
            candidate = run_maps[matched_variants[1]].get(pair)
            missing_variants = [
                variant
                for variant, run in zip(
                    matched_variants, (baseline, candidate), strict=True
                )
                if run is None
            ]
            mismatches = (
                []
                if baseline is None or candidate is None
                else [field for field in control_fields if baseline[field] != candidate[field]]
            )
            passed = not missing_variants and not mismatches
            matched_audits.append(
                {
                    "fold": pair[0],
                    "seed": pair[1],
                    "passed": passed,
                    "missing_variants": missing_variants,
                    "mismatched_fields": mismatches,
                }
            )
        failed = [audit for audit in matched_audits if not audit["passed"]]
        if failed:
            raise ValueError(f"A0/A1 matched-control audit failed: {failed}")
    head_to_head: dict[str, Any] | None = None
    paired_oof_bootstrap: list[dict[str, Any]] = []
    row_map = {str(row["variant"]): row for row in rows}
    if all(variant in row_map for variant in matched_variants):
        baseline_row = row_map[matched_variants[0]]
        candidate_row = row_map[matched_variants[1]]
        metric_fields = {
            "selection_score": "sequence_macro_selection_score",
            "mean_relative_error": "sequence_macro_mean_relative_error",
            "mae_s": "sequence_macro_mae_s",
        }
        baseline_runs = {
            (run["fold"], run["seed"]): run for run in baseline_row["runs"]
        }
        candidate_runs = {
            (run["fold"], run["seed"]): run for run in candidate_row["runs"]
        }
        metric_comparisons: dict[str, Any] = {}
        for label, field in metric_fields.items():
            baseline_mean = float(baseline_row[f"{field}_mean"])
            candidate_mean = float(candidate_row[f"{field}_mean"])
            shared_pairs = sorted(set(baseline_runs) & set(candidate_runs))
            candidate_wins = sum(
                float(candidate_runs[pair][field]) < float(baseline_runs[pair][field])
                for pair in shared_pairs
            )
            metric_comparisons[label] = {
                "a0_mean": baseline_mean,
                "a1_mean": candidate_mean,
                "a1_relative_improvement_pct": 100.0
                * (baseline_mean - candidate_mean)
                / baseline_mean,
                "a1_pair_wins": candidate_wins,
                "a0_pair_wins": sum(
                    float(baseline_runs[pair][field]) < float(candidate_runs[pair][field])
                    for pair in shared_pairs
                ),
                "ties": sum(
                    float(baseline_runs[pair][field]) == float(candidate_runs[pair][field])
                    for pair in shared_pairs
                ),
                "pair_count": len(shared_pairs),
            }
        head_to_head = {
            "baseline": matched_variants[0],
            "candidate": matched_variants[1],
            "metrics": metric_comparisons,
            "a1_elapsed_time_ratio": float(candidate_row["elapsed_seconds_sum"])
            / float(baseline_row["elapsed_seconds_sum"]),
            "a1_latency_ratio": float(candidate_row["milliseconds_per_window_mean"])
            / float(baseline_row["milliseconds_per_window_mean"]),
        }
        bootstrap_seeds = args.seeds or sorted(
            set(baseline_row["seeds"]) & set(candidate_row["seeds"])
        )
        bootstrap_folds = args.folds or sorted(
            set(baseline_row["folds"]) & set(candidate_row["folds"])
        )
        for seed in bootstrap_seeds:
            true_parts: list[np.ndarray] = []
            baseline_parts: list[np.ndarray] = []
            candidate_parts: list[np.ndarray] = []
            sequence_parts: list[np.ndarray] = []
            missing_paths: list[str] = []
            for fold in bootstrap_folds:
                paths = {
                    variant: args.root
                    / f"fold-{fold}"
                    / variant
                    / f"seed-{seed}"
                    / "validation_predictions.npz"
                    for variant in matched_variants
                }
                missing_paths.extend(
                    path.as_posix() for path in paths.values() if not path.is_file()
                )
                if missing_paths:
                    continue
                with np.load(paths[matched_variants[0]]) as baseline_prediction:
                    baseline_true = baseline_prediction["ttc_true"]
                    baseline_tokens = (
                        baseline_prediction["sample_token"]
                        if "sample_token" in baseline_prediction.files
                        else None
                    )
                    baseline_sequences = baseline_prediction["sequence_id"]
                    baseline_values = baseline_prediction["ttc_pred"]
                with np.load(paths[matched_variants[1]]) as candidate_prediction:
                    candidate_tokens = (
                        candidate_prediction["sample_token"]
                        if "sample_token" in candidate_prediction.files
                        else None
                    )
                    if (baseline_tokens is None) != (candidate_tokens is None) or (
                        baseline_tokens is not None
                        and not np.array_equal(baseline_tokens, candidate_tokens)
                    ):
                        raise ValueError(
                            f"A0/A1 prediction tokens differ for fold={fold}, seed={seed}."
                        )
                    if not np.array_equal(
                        baseline_true, candidate_prediction["ttc_true"]
                    ):
                        raise ValueError(
                            f"A0/A1 targets differ for fold={fold}, seed={seed}."
                        )
                    if not np.array_equal(
                        baseline_sequences, candidate_prediction["sequence_id"]
                    ):
                        raise ValueError(
                            f"A0/A1 sequence order differs for fold={fold}, seed={seed}."
                        )
                    candidate_values = candidate_prediction["ttc_pred"]
                true_parts.append(baseline_true)
                baseline_parts.append(baseline_values)
                candidate_parts.append(candidate_values)
                sequence_parts.append(baseline_sequences)
            if missing_paths:
                paired_oof_bootstrap.append(
                    {
                        "seed": seed,
                        "status": "prediction_artifacts_missing",
                        "missing_paths": sorted(set(missing_paths)),
                    }
                )
                continue
            true = np.concatenate(true_parts)
            baseline_prediction = np.concatenate(baseline_parts)
            candidate_prediction = np.concatenate(candidate_parts)
            sequences = np.concatenate(sequence_parts)
            paired_oof_bootstrap.append(
                {
                    "seed": seed,
                    "status": "sequence_cluster_bootstrap",
                    "sample_count": int(true.shape[0]),
                    "sequence_count": int(np.unique(sequences).shape[0]),
                    "candidate_minus_baseline_mae_s": paired_sequence_bootstrap_difference(
                        true,
                        baseline_prediction,
                        candidate_prediction,
                        sequences,
                        iterations=2000,
                        confidence=0.95,
                        seed=seed,
                    ),
                    "candidate_minus_baseline_mean_abs_relative_error": (
                        paired_sequence_bootstrap_difference(
                            true,
                            baseline_prediction,
                            candidate_prediction,
                            sequences,
                            metric=lambda truth, estimate: float(
                                np.mean(
                                    np.abs(estimate - truth)
                                    / np.maximum(np.abs(truth), 1e-6)
                                )
                            ),
                            iterations=2000,
                            confidence=0.95,
                            seed=seed,
                        )
                    ),
                }
            )
    payload = {
        "protocol": "evttc_architecture_grouped_cv_multiseed_aggregate_v2",
        "expected_folds": args.expected_folds,
        "expected_seeds": args.expected_seeds,
        "expected_fold_ids": args.folds,
        "expected_seed_ids": args.seeds,
        "required_runs_per_variant": required_runs,
        "all_variants_complete": bool(rows)
        and all(bool(row["complete_for_final_selection"]) for row in rows),
        "matched_control_audit_passed": bool(matched_audits)
        and all(bool(audit["passed"]) for audit in matched_audits),
        "matched_control_audits": matched_audits,
        "a0_a1_head_to_head": head_to_head,
        "paired_oof_sequence_bootstrap_by_seed": paired_oof_bootstrap,
        "ranking": rows,
        "window_level_bootstrap_used": False,
        "benchmark10_opened": False,
    }
    write_structured(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
