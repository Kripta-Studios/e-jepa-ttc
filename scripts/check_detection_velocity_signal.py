"""Compare native velocity predictions with fixed baselines on identical box matches."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from e_jepa_ttc.detection import velocity_diagnostics  # noqa: E402
from e_jepa_ttc.detection.velocity_diagnostics import (  # noqa: E402
    _velocity,
    evaluate_velocity_labels,
)
from scripts.check_detection_image_signal import (  # noqa: E402
    METRIC_FLOAT_TOLERANCE,
    _validate_results,
    canonical_hash,
    load_results,
    metrics_equivalent,
    sha256_file,
)

METHOD = "native_velocity_shared_match_zero_train_mean_v1"
DECLARED_AT_UTC = "2026-10-29T13:49:51+00:00"


def fit_train_mean(
    truth: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[float], int]:
    """Fit a global component mean using only valid training-label velocities."""
    valid = [
        value
        for rows in truth.values()
        for row in rows
        if (value := _velocity(row, required=False)) is not None
    ]
    if not valid:
        raise ValueError("A train-mean velocity baseline requires valid training velocities")
    return [statistics.mean(row[index] for row in valid) for index in (0, 1)], len(valid)


def replace_velocities(
    predictions: Mapping[str, Sequence[Mapping[str, Any]]], velocity: Sequence[float]
) -> dict[str, list[dict[str, Any]]]:
    """Change velocities alone, preserving every input to the box matcher."""
    return {
        token: [{**copy.deepcopy(row), "velocity_mps": list(velocity)} for row in rows]
        for token, rows in predictions.items()
    }


def _assert_same_matches(reference: Mapping[str, Any], control: Mapping[str, Any]) -> None:
    for field in ("ground_truth_velocity_count", "box_matches", "valid_velocity_matches"):
        if reference[field] != control[field]:
            raise AssertionError(f"Velocity control changed matching population: {field}")
    for category, values in reference["per_class"].items():
        if (
            values["valid_velocity_matches"]
            != control["per_class"][category]["valid_velocity_matches"]
        ):
            raise AssertionError(f"Velocity control changed matched class: {category}")


def analyze_velocity_controls(
    train: Mapping[str, Any], selection: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate fixed baselines without changing box associations or fitting selection."""
    _, train_truth = _validate_results(train)
    predictions, truth = _validate_results(selection)
    if set(train_truth) & set(truth):
        raise ValueError("Train and selection tokens must be disjoint")
    # Validate original predictions first: missing values must not be filled by a control.
    observed = evaluate_velocity_labels(predictions, truth, threshold=2.0)
    if not metrics_equivalent(observed, selection["metrics"].get("native_velocity")):
        raise ValueError("Stored native_velocity metrics differ from recomputation")
    mean, train_velocity_count = fit_train_mean(train_truth)
    zero = evaluate_velocity_labels(
        replace_velocities(predictions, (0.0, 0.0)), truth, threshold=2.0
    )
    train_mean = evaluate_velocity_labels(
        replace_velocities(predictions, mean), truth, threshold=2.0
    )
    _assert_same_matches(observed, zero)
    _assert_same_matches(observed, train_mean)

    def delta(control: Mapping[str, Any]) -> float | None:
        head, baseline = observed["vector_l2_mean_mps"], control["vector_l2_mean_mps"]
        return head - baseline if head is not None and baseline is not None else None

    return {
        "method": METHOD,
        "declared_at_utc": DECLARED_AT_UTC,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "official_eap_score": False,
        "ttc_evaluated": False,
        "stored_metric_comparison": {
            "float_absolute_tolerance": METRIC_FLOAT_TOLERANCE,
            "float_relative_tolerance": METRIC_FLOAT_TOLERANCE,
            "structures_integers_booleans_strings_null": "exact",
        },
        "train_sample_count": len(train_truth),
        "selection_sample_count": len(truth),
        "train_tokens_sha256": canonical_hash(sorted(train_truth)),
        "selection_tokens_sha256": canonical_hash(sorted(truth)),
        "train_selection_tokens_disjoint": True,
        "train_mean_velocity_components_mps": mean,
        "train_valid_velocity_objects": train_velocity_count,
        "fit_method": "global arithmetic component means over valid training boxes only",
        "matching": "Same evaluator and unchanged boxes/classes/scores/order; strict ego XY < 2 m",
        "same_box_and_velocity_matching_counts": True,
        "supported_matched_classes": [
            category
            for category, values in observed["per_class"].items()
            if values["valid_velocity_matches"] > 0
        ],
        "observed_head": observed,
        "zero_velocity": zero,
        "train_mean_velocity": train_mean,
        "head_minus_zero_vector_l2_mps": delta(zero),
        "head_minus_train_mean_vector_l2_mps": delta(train_mean),
        "limitations": [
            "Label-component affinity only; physical frame and relative semantics unresolved",
            "No TTC, time-to-collision, closing speed or official AVE is computed",
            "Error comparison is conditional on detected box matches, not all ground-truth objects",
            "Missing GT velocity still consumes a box match but is excluded from velocity errors",
            "Missing predicted velocity fails even for an unmatched prediction",
            "Global training means, no per-class or selection fit, no model change",
        ],
    }


def main() -> int:
    """Read explicit pilot files and publish aggregate-only velocity controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-results", type=Path, required=True)
    parser.add_argument("--selection-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite control receipt: {args.output}")
    report = analyze_velocity_controls(
        load_results(args.train_results), load_results(args.selection_results)
    )
    report["source_sha256"] = {
        "train_results": sha256_file(args.train_results),
        "selection_results": sha256_file(args.selection_results),
        "script": sha256_file(Path(__file__)),
        "velocity_diagnostics": sha256_file(Path(velocity_diagnostics.__file__)),
        "input_validation": sha256_file(ROOT / "scripts/check_detection_image_signal.py"),
        "development_metrics": sha256_file(
            ROOT / "src/e_jepa_ttc/detection/development_metrics.py"
        ),
    }
    report["input_paths"] = {
        "train_results": str(args.train_results),
        "selection_results": str(args.selection_results),
    }
    report["python_version"] = sys.version
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key.startswith("head_minus_")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
