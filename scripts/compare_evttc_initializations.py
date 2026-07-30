"""Compare random and externally pretrained runs on identical EvTTC grouped folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e_jepa_ttc.data.benchmark10_guard import assert_no_sealed_benchmark_paths  # noqa: E402
from e_jepa_ttc.evaluation.bootstrap import (  # noqa: E402
    paired_sequence_bootstrap_difference,
)
from e_jepa_ttc.utils.io import write_structured  # noqa: E402

DEFAULT_VARIANT = "A0_MATCHED_GLOBAL"
METRICS = (
    "sequence_macro_selection_score",
    "sequence_macro_mean_relative_error",
    "sequence_macro_mae_s",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _variant_row(payload: dict[str, Any], source: Path, *, variant: str) -> dict[str, Any]:
    if payload.get("benchmark10_opened") is not False:
        raise ValueError(f"Aggregate is not sealed from Benchmark-10: {source}.")
    row = next(
        (item for item in payload.get("ranking", []) if item.get("variant") == variant),
        None,
    )
    if row is None or row.get("complete_for_final_selection") is not True:
        raise ValueError(f"{source} has no complete {variant} grouped-CV row.")
    return row


def _prediction_arrays(summary: str | Path) -> dict[str, np.ndarray]:
    path = Path(summary).parent / "validation_predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing OOF prediction artifact: {path}.")
    with np.load(path) as payload:
        return {name: payload[name].copy() for name in payload.files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--candidate-label", default="external_pretraining")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()
    assert_no_sealed_benchmark_paths((args.control, args.transfer, args.output))
    control_payload = json.loads(args.control.read_text(encoding="utf-8"))
    transfer_payload = json.loads(args.transfer.read_text(encoding="utf-8"))
    control = _variant_row(control_payload, args.control, variant=args.variant)
    transfer = _variant_row(transfer_payload, args.transfer, variant=args.variant)
    control_runs = {(run["fold"], run["seed"]): run for run in control["runs"]}
    transfer_runs = {(run["fold"], run["seed"]): run for run in transfer["runs"]}
    if set(control_runs) != set(transfer_runs):
        raise ValueError("Control and transfer do not contain identical fold/seed pairs.")
    control_fields = (
        "cache_manifest_sha256",
        "sample_selection_sha256",
        "initial_common_head_sha256",
        "train_samples",
        "validation_samples",
        "trainer",
    )
    mismatches = [
        {"fold": fold, "seed": seed, "field": field}
        for (fold, seed), baseline in sorted(control_runs.items())
        for field in control_fields
        if baseline.get(field) != transfer_runs[(fold, seed)].get(field)
    ]
    if mismatches:
        raise ValueError(f"Transfer matched-control audit failed: {mismatches[:10]}.")
    comparisons: dict[str, Any] = {}
    for metric in METRICS:
        control_mean = float(control[f"{metric}_mean"])
        transfer_mean = float(transfer[f"{metric}_mean"])
        comparisons[metric] = {
            "control_mean": control_mean,
            "candidate_transfer_mean": transfer_mean,
            "candidate_relative_improvement_pct": 100.0
            * (control_mean - transfer_mean)
            / control_mean,
            "candidate_pair_wins": sum(
                float(transfer_runs[pair][metric]) < float(control_runs[pair][metric])
                for pair in control_runs
            ),
            "control_pair_wins": sum(
                float(control_runs[pair][metric]) < float(transfer_runs[pair][metric])
                for pair in control_runs
            ),
            "pair_count": len(control_runs),
        }
    bootstrap: list[dict[str, Any]] = []
    for seed in sorted({int(pair[1]) for pair in control_runs}):
        true_parts: list[np.ndarray] = []
        control_parts: list[np.ndarray] = []
        transfer_parts: list[np.ndarray] = []
        sequence_parts: list[np.ndarray] = []
        for pair in sorted(pair for pair in control_runs if int(pair[1]) == seed):
            baseline = _prediction_arrays(control_runs[pair]["summary"])
            candidate = _prediction_arrays(transfer_runs[pair]["summary"])
            for field in ("ttc_true", "sequence_id", "sample_token"):
                if field in baseline or field in candidate:
                    if field not in baseline or field not in candidate:
                        raise ValueError(f"Prediction field {field} is asymmetric for {pair}.")
                    if not np.array_equal(baseline[field], candidate[field]):
                        raise ValueError(f"Prediction field {field} differs for {pair}.")
            true_parts.append(baseline["ttc_true"])
            control_parts.append(baseline["ttc_pred"])
            transfer_parts.append(candidate["ttc_pred"])
            sequence_parts.append(baseline["sequence_id"])
        truth = np.concatenate(true_parts)
        baseline_prediction = np.concatenate(control_parts)
        transfer_prediction = np.concatenate(transfer_parts)
        sequences = np.concatenate(sequence_parts)
        bootstrap.append(
            {
                "seed": seed,
                "sample_count": int(truth.shape[0]),
                "sequence_count": int(np.unique(sequences).shape[0]),
                "candidate_minus_control_mae_s": paired_sequence_bootstrap_difference(
                    truth,
                    baseline_prediction,
                    transfer_prediction,
                    sequences,
                    iterations=args.bootstrap_iterations,
                    confidence=0.95,
                    seed=seed,
                ),
                "candidate_minus_control_mean_abs_relative_error": (
                    paired_sequence_bootstrap_difference(
                        truth,
                        baseline_prediction,
                        transfer_prediction,
                        sequences,
                        metric=lambda target, estimate: float(
                            np.mean(np.abs(estimate - target) / np.maximum(np.abs(target), 1e-6))
                        ),
                        iterations=args.bootstrap_iterations,
                        confidence=0.95,
                        seed=seed,
                    )
                ),
            }
        )
    payload = {
        "artifact_type": "evttc_external_pretraining_transfer_comparison_v2",
        "variant": args.variant,
        "candidate_label": args.candidate_label,
        "control_aggregate": args.control.as_posix(),
        "control_aggregate_sha256": _sha256(args.control),
        "transfer_aggregate": args.transfer.as_posix(),
        "transfer_aggregate_sha256": _sha256(args.transfer),
        "matched_control_audit_passed": True,
        "fold_seed_pairs": [{"fold": fold, "seed": seed} for fold, seed in sorted(control_runs)],
        "comparisons": comparisons,
        "paired_oof_sequence_bootstrap_by_seed": bootstrap,
        "test_used_for_selection": False,
        "benchmark10_opened": False,
    }
    write_structured(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
