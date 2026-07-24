"""Aggregate evaluation metrics JSON files across seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from e_jepa_ttc.evaluation.aggregate import DEFAULT_METRIC_NAMES, aggregate_metric_files
from e_jepa_ttc.utils.io import ensure_parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, help="Split key to aggregate, e.g. test.")
    parser.add_argument(
        "--split-protocol",
        type=Path,
        help="Split YAML/JSON whose claim metadata gates publication status.",
    )
    parser.add_argument(
        "--claim-level",
        choices=("development", "diagnostic", "official", "final"),
        default="diagnostic",
        help="Intended status of the generated aggregate table.",
    )
    parser.add_argument("--output", type=Path, help="Optional output JSON path.")
    parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        help="Metric name to include. Repeat for multiple metrics.",
    )
    parser.add_argument(
        "--baseline",
        nargs="+",
        type=Path,
        help="Optional baseline metrics JSON files for paired hierarchical bootstrap.",
    )
    parser.add_argument("metrics_json", nargs="+", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = tuple(args.metrics) if args.metrics else DEFAULT_METRIC_NAMES
    payload = aggregate_metric_files(
        args.metrics_json,
        split=args.split,
        metric_names=metrics,
        split_protocol_path=args.split_protocol,
        claim_level=args.claim_level,
    )
    
    if args.baseline:
        baseline_payload = aggregate_metric_files(
            args.baseline,
            split=args.split,
            metric_names=metrics,
            split_protocol_path=args.split_protocol,
            claim_level=args.claim_level,
        )
        
        # Paired hierarchical bootstrap
        import numpy as np
        
        iterations = 2000
        rng = np.random.default_rng(0)
        
        # Group experimental rows
        exp_groups = {}
        for row in payload["rows"]:
            pt_seed = row.get("pretrain_seed")
            if pt_seed is None:
                pt_seed = row.get("downstream_seed")
            exp_groups.setdefault(pt_seed, []).append(row)
            
        # Group baseline rows
        base_groups = {}
        for row in baseline_payload["rows"]:
            pt_seed = row.get("pretrain_seed")
            if pt_seed is None:
                pt_seed = row.get("downstream_seed")
            base_groups.setdefault(pt_seed, []).append(row)
            
        # Find common pretrain seeds
        common_pt = sorted(set(exp_groups.keys()) & set(base_groups.keys()))
        
        if common_pt:
            diff_summary = {}
            for metric in metrics:
                # Build matrices of values: [pt_seed][downstream_seed]
                # To pair them, we just align by downstream seed if possible, or just sample the same indices.
                # Actually, a strict paired bootstrap pairs by exact (pretrain_seed, downstream_seed).
                exp_vals = {pt: {} for pt in common_pt}
                base_vals = {pt: {} for pt in common_pt}
                for pt in common_pt:
                    for row in exp_groups[pt]:
                        ds = row.get("downstream_seed")
                        if metric in row["metrics"]:
                            exp_vals[pt][ds] = row["metrics"][metric]
                    for row in base_groups[pt]:
                        ds = row.get("downstream_seed")
                        if metric in row["metrics"]:
                            base_vals[pt][ds] = row["metrics"][metric]
                            
                valid_pt = []
                for pt in common_pt:
                    common_ds = sorted(set(exp_vals[pt].keys()) & set(base_vals[pt].keys()))
                    if common_ds:
                        # Keep only matched DS
                        exp_vals[pt] = [exp_vals[pt][ds] for ds in common_ds]
                        base_vals[pt] = [base_vals[pt][ds] for ds in common_ds]
                        valid_pt.append(pt)
                
                if not valid_pt:
                    continue
                    
                bootstrap_diffs = np.empty(iterations, dtype=np.float64)
                
                for i in range(iterations):
                    sampled_pt = rng.choice(valid_pt, size=len(valid_pt), replace=True)
                    sampled_means_exp = []
                    sampled_means_base = []
                    for pt in sampled_pt:
                        # For paired bootstrap within the cluster, sample the SAME indices for both exp and base
                        n_ds = len(exp_vals[pt])
                        ds_indices = rng.choice(n_ds, size=n_ds, replace=True)
                        sampled_means_exp.append(np.mean(np.array(exp_vals[pt])[ds_indices]))
                        sampled_means_base.append(np.mean(np.array(base_vals[pt])[ds_indices]))
                        
                    bootstrap_diffs[i] = np.mean(sampled_means_exp) - np.mean(sampled_means_base)
                    
                mean_diff = float(np.mean(bootstrap_diffs))
                lower = float(np.quantile(bootstrap_diffs, 0.025))
                upper = float(np.quantile(bootstrap_diffs, 0.975))
                
                diff_summary[metric] = {
                    "mean_difference": mean_diff,
                    "ci_lower_95": lower,
                    "ci_upper_95": upper,
                    "significant": (upper < 0) if mean_diff < 0 else (lower > 0),
                    "iterations": iterations,
                }
            
            payload["paired_bootstrap_vs_baseline"] = diff_summary

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        ensure_parent(args.output)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
