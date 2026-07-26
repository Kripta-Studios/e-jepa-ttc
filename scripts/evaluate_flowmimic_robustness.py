"""Evaluate paired E0/E1 checkpoints on validation-only raw-event corruptions."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact
from e_jepa_ttc.artifacts.protocol import get_current_protocol_identity
from e_jepa_ttc.data.ml_cache import build_voxel_cache
from e_jepa_ttc.representations.corruptions import (
    CORRUPTION_KINDS,
    EventCorruptionSpec,
)
from e_jepa_ttc.training.supervised import evaluate_supervised_checkpoint
from e_jepa_ttc.utils.io import read_structured, write_structured


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return payload


def parse_condition(value: str) -> tuple[str, float]:
    """Parse one CLI robustness condition as KIND=SEVERITY."""

    kind, separator, severity_text = value.partition("=")
    if not separator or kind not in CORRUPTION_KINDS or kind == "none":
        raise argparse.ArgumentTypeError(
            f"Expected non-clean KIND=SEVERITY with KIND in {CORRUPTION_KINDS}."
        )
    try:
        severity = float(severity_text)
        EventCorruptionSpec(kind=kind, severity=severity)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return kind, severity


def _configured_conditions(
    config: dict[str, Any],
    overrides: list[tuple[str, float]] | None,
) -> list[tuple[str, float]]:
    if overrides:
        return overrides
    conditions: list[tuple[str, float]] = []
    for kind, severities in config["robustness"]["conditions"].items():
        for severity in severities:
            conditions.append((str(kind), float(severity)))
    return conditions


def _run_paths(run_root: Path, variant: str, seed: int) -> tuple[Path, Path]:
    stem = variant.lower()
    return (
        run_root / f"flowmimic_full_{stem}_seed{seed}_ft30" / "metrics.json",
        run_root / f"flowmimic_full_{stem}_seed{seed}_ft30" / "tiny_cnn_best.pt",
    )


def _metric_subset(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "mae_s": float(metrics["mae_s"]),
        "mean_abs_relative_error_pct": float(metrics["mean_abs_relative_error_pct"]),
        "rmse_s": float(metrics["rmse_s"]),
        "median_abs_error_s": float(metrics["median_abs_error_s"]),
    }


def _condition_id(kind: str, severity: float) -> str:
    return f"{kind}={severity:g}"


def evaluate_flowmimic_robustness(
    *,
    config_path: Path,
    output_path: Path | None = None,
    condition_overrides: list[tuple[str, float]] | None = None,
) -> dict[str, Any]:
    """Build validation-only corrupt caches and evaluate all six paired checkpoints."""

    config = read_structured(config_path)
    data = config["data"]
    experiment = config["experiment"]
    robustness_config = config["robustness"]
    run_root = Path(config["outputs"]["run_root"])
    variants = tuple(str(value) for value in experiment["variants"])
    seeds = [int(value) for value in experiment["seeds"]]
    if variants != ("E0", "E1") or seeds != [7, 13, 21]:
        raise ValueError("Robustness gate requires the paired E0/E1 seeds 7/13/21.")
    evaluation_split = str(robustness_config["evaluation_split"])
    if evaluation_split == "test":
        raise ValueError("Robustness selection cannot evaluate the closed test split.")

    cache_path = Path(data["cache"])
    cache_sha256 = compute_file_hash(str(cache_path))
    if cache_sha256 != str(data["cache_sha256"]):
        raise ValueError("Clean cache SHA-256 differs from the frozen gate config.")
    with np.load(cache_path, allow_pickle=False) as cache:
        if "test" in set(cache["split"].astype(str).tolist()):
            raise ValueError("Clean gate cache physically contains test.")
        width = int(cache["width"])
        height = int(cache["height"])
        bins = int(cache["bins"])
        normalize = bool(cache["normalize"])
        metadata_channels = bool(cache["metadata_channels"])
        navigation_channels = bool(cache["navigation_channels"])

    model_records: list[dict[str, Any]] = []
    clean_metrics: dict[tuple[str, int], dict[str, float]] = {}
    for variant in variants:
        for seed in seeds:
            metrics_path, checkpoint_path = _run_paths(run_root, variant, seed)
            downstream = _read_json(metrics_path)
            if downstream.get("final_test_opened") is not False:
                raise ValueError(f"{variant} seed {seed} downstream opened final test.")
            if str(downstream["cache_sha256"]) != cache_sha256:
                raise ValueError(f"{variant} seed {seed} used a different cache.")
            clean = _metric_subset(downstream["splits"][evaluation_split]["metrics"])
            clean_metrics[(variant, seed)] = clean
            model_records.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "checkpoint_path": checkpoint_path.as_posix(),
                    "checkpoint_sha256": compute_file_hash(str(checkpoint_path)),
                    "model_name": str(downstream["model_name"]),
                    "clean_validation": clean,
                }
            )

    rows: list[dict[str, Any]] = []
    for model in model_records:
        rows.append(
            {
                "variant": model["variant"],
                "seed": model["seed"],
                "condition": "clean",
                "corruption_kind": "none",
                "corruption_severity": 0.0,
                "metrics": model["clean_validation"],
                "mae_degradation_absolute_s": 0.0,
                "mae_degradation_relative_pct": 0.0,
                "uncertainty_behavior_status": "unavailable_deterministic_ttc_head",
            }
        )

    conditions = _configured_conditions(config, condition_overrides)
    configured_full_conditions = _configured_conditions(config, None)
    full_matrix_complete = condition_overrides is None and conditions == configured_full_conditions
    corruption_seed = int(robustness_config["corruption_seed"])
    cache_records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="e_jepa_ttc_robustness_") as temporary:
        temporary_root = Path(temporary)
        for condition_index, (kind, severity) in enumerate(conditions):
            condition = _condition_id(kind, severity)
            print(
                f"[{datetime.now(UTC).isoformat()}] BUILD {condition} "
                f"({condition_index + 1}/{len(conditions)})",
                flush=True,
            )
            corrupted_cache = temporary_root / f"condition_{condition_index:02d}.npz"
            cache_summary = build_voxel_cache(
                manifest_path=Path(data["manifest"]),
                split_path=Path(data["split"]),
                index_path=Path(data["index"]),
                output_path=corrupted_cache,
                width=width,
                height=height,
                bins=bins,
                normalize=normalize,
                metadata_channels=metadata_channels,
                navigation_channels=navigation_channels,
                include_splits=[evaluation_split],
                corruption=EventCorruptionSpec(
                    kind=kind,
                    severity=severity,
                    seed=corruption_seed,
                ),
            )
            with np.load(corrupted_cache, allow_pickle=False) as corrupted:
                physical_splits = set(corrupted["split"].astype(str).tolist())
                sequence_ids = sorted(set(corrupted["sequence_id"].astype(str).tolist()))
            if physical_splits != {evaluation_split}:
                raise ValueError(f"{condition} cache has unexpected splits {physical_splits}.")
            if str(config["protocol"]["closed_sequence"]) in sequence_ids:
                raise ValueError(f"{condition} cache contains the closed sequence.")
            cache_records.append(
                {
                    "condition": condition,
                    "cache_sha256": cache_summary["cache_sha256"],
                    "window_count": cache_summary["window_count"],
                    "sequence_ids": sequence_ids,
                    "mean_source_events_per_window": cache_summary["mean_source_events_per_window"],
                    "mean_events_per_window": cache_summary["mean_events_per_window"],
                }
            )
            for model_index, model in enumerate(model_records):
                variant = str(model["variant"])
                seed = int(model["seed"])
                print(
                    f"[{datetime.now(UTC).isoformat()}] EVAL {condition} "
                    f"{variant} seed {seed} ({model_index + 1}/{len(model_records)})",
                    flush=True,
                )
                result = evaluate_supervised_checkpoint(
                    cache_path=corrupted_cache,
                    checkpoint_path=Path(model["checkpoint_path"]),
                    output_path=None,
                    batch_size=int(config["downstream"]["batch_size"]),
                    device_name=str(config["downstream"]["device"]),
                    evaluation_splits=(evaluation_split,),
                    model_name=str(model["model_name"]),
                )
                metrics = _metric_subset(result["splits"][evaluation_split]["metrics"])
                clean = clean_metrics[(variant, seed)]
                degradation = metrics["mae_s"] - clean["mae_s"]
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "condition": condition,
                        "corruption_kind": kind,
                        "corruption_severity": severity,
                        "metrics": metrics,
                        "mae_degradation_absolute_s": degradation,
                        "mae_degradation_relative_pct": (100.0 * degradation / clean["mae_s"]),
                        "sequence_metrics": result["splits"][evaluation_split].get("per_sequence"),
                        "uncertainty_behavior_status": ("unavailable_deterministic_ttc_head"),
                    }
                )
                gc.collect()

    aggregates: list[dict[str, Any]] = []
    all_conditions = ["clean", *[_condition_id(*condition) for condition in conditions]]
    for condition in all_conditions:
        for variant in variants:
            selected = [
                row for row in rows if row["condition"] == condition and row["variant"] == variant
            ]
            mae = np.asarray([row["metrics"]["mae_s"] for row in selected], dtype=np.float64)
            degradation = np.asarray(
                [row["mae_degradation_relative_pct"] for row in selected],
                dtype=np.float64,
            )
            aggregates.append(
                {
                    "condition": condition,
                    "variant": variant,
                    "seed_count": len(selected),
                    "mae_s_mean": float(np.mean(mae)),
                    "mae_s_std": float(np.std(mae, ddof=1)),
                    "mae_degradation_relative_pct_mean": float(np.mean(degradation)),
                    "mae_degradation_relative_pct_std": float(np.std(degradation, ddof=1)),
                }
            )

    comparisons: list[dict[str, Any]] = []
    for condition in all_conditions:
        e0 = next(
            row for row in aggregates if row["condition"] == condition and row["variant"] == "E0"
        )
        e1 = next(
            row for row in aggregates if row["condition"] == condition and row["variant"] == "E1"
        )
        comparisons.append(
            {
                "condition": condition,
                "e1_minus_e0_mae_s_mean": e1["mae_s_mean"] - e0["mae_s_mean"],
                "e1_minus_e0_relative_degradation_pct_points": (
                    e1["mae_degradation_relative_pct_mean"]
                    - e0["mae_degradation_relative_pct_mean"]
                ),
                "e1_lower_mean_mae": e1["mae_s_mean"] < e0["mae_s_mean"],
            }
        )

    nonclean_comparisons = [
        comparison for comparison in comparisons if comparison["condition"] != "clean"
    ]
    degradation_gap_limit = float(
        robustness_config["maximum_relative_degradation_gap_percentage_points"]
    )
    absolute_accuracy_passed = all(
        bool(comparison["e1_lower_mean_mae"]) for comparison in nonclean_comparisons
    )
    relative_fragility_passed = all(
        float(comparison["e1_minus_e0_relative_degradation_pct_points"]) <= degradation_gap_limit
        for comparison in nonclean_comparisons
    )
    protocol_version, protocol_sha256 = get_current_protocol_identity()
    payload: dict[str, Any] = {
        "artifact_type": "flowmimic_e0_e1_raw_event_robustness",
        "schema_version": "1.0",
        "evidence_type": "validation_robustness_multiseed",
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": _git_commit(),
        "config_path": config_path.as_posix(),
        "config_sha256": compute_file_hash(str(config_path)),
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
        "cache_sha256": cache_sha256,
        "evaluation_split": evaluation_split,
        "final_test_opened": False,
        "corruption_seed": corruption_seed,
        "model_records": model_records,
        "corrupted_cache_records": cache_records,
        "rows": rows,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "gate": {
            "e1_lower_mean_mae_in_every_nonclean_condition": absolute_accuracy_passed,
            "maximum_allowed_relative_degradation_gap_percentage_points": (degradation_gap_limit),
            "e1_relative_fragility_within_limit": relative_fragility_passed,
            "full_matrix_complete": full_matrix_complete,
            "passed": (
                full_matrix_complete and absolute_accuracy_passed and relative_fragility_passed
            ),
        },
        "uncertainty_evaluation_status": ("not_applicable_current_deterministic_supervised_head"),
        "limitations": [
            "validation contains only CCRs-side-high",
            "robustness corruptions are deterministic simulations, not new capture domains",
            "current downstream head has no predictive uncertainty output",
            "CPLA-high remains physically excluded and unopened",
        ],
    }
    sign_artifact(payload)
    destination = output_path or Path(config["outputs"]["robustness"])
    write_structured(destination, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all paired E0/E1 checkpoints on raw-event corruptions."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--condition",
        action="append",
        type=parse_condition,
        help="Override the full matrix with a repeatable KIND=SEVERITY condition.",
    )
    args = parser.parse_args()
    payload = evaluate_flowmimic_robustness(
        config_path=args.config,
        output_path=args.output,
        condition_overrides=args.condition,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
