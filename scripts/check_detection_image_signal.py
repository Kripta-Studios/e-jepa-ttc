"""Evaluate predeclared train-only and frame-permutation controls, without Torch."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.detection import development_metrics  # noqa: E402
from e_jepa_ttc.detection.development_metrics import CLASSES, PROFILE, evaluate_detections  # noqa: E402

METHOD = "train_constant_prior_and_seed7_frame_derangement_v1"
DECLARED_AT_UTC = "2026-10-29T13:22:15+00:00"
SEED = 7
YAW_RESULTANT_EPSILON = 1e-12
METRIC_FLOAT_TOLERANCE = 1e-12
Box = dict[str, Any]
BoxesByToken = dict[str, list[Box]]


def sha256_file(path: Path) -> str:
    """Hash an explicit input file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    """Hash canonical JSON for deterministic token/permutation receipts."""
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_results(path: Path) -> dict[str, Any]:
    """Load only the explicitly named result file, refusing duplicate keys."""
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    if not isinstance(result, dict):
        raise ValueError("Result JSON must be an object")
    return result


def metrics_equivalent(expected: object, actual: object) -> bool:
    """Compare metric structures exactly, allowing only tiny float roundoff."""
    if isinstance(expected, float) and isinstance(actual, float):
        return math.isclose(
            expected,
            actual,
            rel_tol=METRIC_FLOAT_TOLERANCE,
            abs_tol=METRIC_FLOAT_TOLERANCE,
        )
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        return expected.keys() == actual.keys() and all(
            metrics_equivalent(value, actual[key]) for key, value in expected.items()
        )
    if isinstance(expected, list):
        assert isinstance(actual, list)
        return len(expected) == len(actual) and all(
            metrics_equivalent(left, right) for left, right in zip(expected, actual, strict=True)
        )
    return expected == actual


def _validate_results(payload: Mapping[str, Any]) -> tuple[BoxesByToken, BoxesByToken]:
    for key in ("predictions", "ground_truth", "metrics"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"Result must contain a {key} mapping")
    for key in ("predictions", "ground_truth"):
        if any(
            not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
            for rows in payload[key].values()
        ):
            raise ValueError(f"{key} values must be lists of boxes")
    if payload["metrics"].get("profile") != PROFILE:
        raise ValueError("Input metrics profile must match the current development profile")
    if payload["metrics"].get("official_eap_score") is not False:
        raise ValueError("Input must declare nonofficial development metrics")
    predictions = payload["predictions"]
    truth = payload["ground_truth"]
    # This checks token equality, fields, finite numbers, classes and score range.
    recomputed = evaluate_detections(predictions, truth)
    for key, value in recomputed.items():
        if not metrics_equivalent(value, payload["metrics"].get(key)):
            raise ValueError(f"Stored metric differs from recomputation: {key}")
    return predictions, truth


def fit_constant_prior(train_truth: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[Box]:
    """Fit one aggregate box per supported class from training labels alone."""
    evaluate_detections({token: [] for token in train_truth}, train_truth)
    priors = []
    for category in CLASSES:
        rows = [
            row for frame in train_truth.values() for row in frame if row["class_name"] == category
        ]
        if not rows:
            continue
        mean_sin = statistics.mean(math.sin(row["yaw_rad"]) for row in rows)
        mean_cos = statistics.mean(math.cos(row["yaw_rad"]) for row in rows)
        resultant = math.hypot(mean_sin, mean_cos)
        fallback = resultant <= YAW_RESULTANT_EPSILON
        frames_present = sum(
            any(row["class_name"] == category for row in frame) for frame in train_truth.values()
        )
        priors.append(
            {
                "class_name": category,
                "translation_m": [
                    statistics.median(row["translation_m"][i] for row in rows) for i in range(3)
                ],
                "size_lwh_m": [
                    statistics.median(row["size_lwh_m"][i] for row in rows) for i in range(3)
                ],
                "yaw_rad": 0.0 if fallback else math.atan2(mean_sin, mean_cos),
                "score": frames_present / len(train_truth),
                "train_object_count": len(rows),
                "train_frames_with_class": frames_present,
                "yaw_mean_resultant": resultant,
                "yaw_undefined_fallback_zero": fallback,
            }
        )
    return priors


def derange_predictions(
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[BoxesByToken, dict[str, str], int]:
    """Move complete frame predictions using the first seed-7 derangement."""
    tokens = sorted(predictions)
    if len(tokens) < 2:
        raise ValueError("A fixed-point-free permutation requires at least two selection frames")
    rng = random.Random(SEED)
    for attempt in range(1, 10001):
        source_tokens = list(tokens)
        rng.shuffle(source_tokens)
        if all(target != source for target, source in zip(tokens, source_tokens, strict=True)):
            mapping = dict(zip(tokens, source_tokens, strict=True))
            return (
                {
                    target: copy.deepcopy(list(predictions[source]))
                    for target, source in mapping.items()
                },
                mapping,
                attempt,
            )
    raise RuntimeError("No derangement in 10,000 draws; no alternative selected by AP")


def analyze_results(train: Mapping[str, Any], selection: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the fixed controls against held-out labels, without fitting on them."""
    _, train_truth = _validate_results(train)
    selection_predictions, selection_truth = _validate_results(selection)
    if set(train_truth) & set(selection_truth):
        raise ValueError("Train and selection sample tokens must be disjoint")
    priors = fit_constant_prior(train_truth)
    constant_predictions = {token: priors for token in selection_truth}
    permuted, assignment, attempts = derange_predictions(selection_predictions)
    observed = evaluate_detections(selection_predictions, selection_truth)
    constant = evaluate_detections(constant_predictions, selection_truth)
    shuffled = evaluate_detections(permuted, selection_truth)
    observed_ap = observed["dev_AP101_supported_classes"]

    def delta(control: Mapping[str, Any]) -> float | None:
        value = control["dev_AP101_supported_classes"]
        return observed_ap - value if observed_ap is not None and value is not None else None

    return {
        "method": METHOD,
        "declared_at_utc": DECLARED_AT_UTC,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "official_eap_score": False,
        "velocity_evaluated": False,
        "stored_metric_comparison": {
            "float_absolute_tolerance": METRIC_FLOAT_TOLERANCE,
            "float_relative_tolerance": METRIC_FLOAT_TOLERANCE,
            "structures_integers_booleans_strings_null": "exact",
        },
        "train_sample_count": len(train_truth),
        "selection_sample_count": len(selection_truth),
        "train_tokens_sha256": canonical_hash(sorted(train_truth)),
        "selection_tokens_sha256": canonical_hash(sorted(selection_truth)),
        "train_selection_tokens_disjoint": True,
        "train_fitted_priors": priors,
        "constant_prior": {
            "fit_source": "training ground_truth only",
            "center_and_size": "component medians over all boxes of each training class",
            "yaw": "circular mean; zero if mean resultant <= 1e-12, explicitly flagged",
            "score": "fraction of training frames containing that class",
            "boxes_per_frame": len(priors),
            "metrics": constant,
        },
        "frame_derangement": {
            "seed": SEED,
            "rng": "Python random.Random",
            "algorithm": "sorted tokens; first shuffle with no fixed points; maximum 10000 draws",
            "draws_until_first_derangement": attempts,
            "fixed_points": sum(target == source for target, source in assignment.items()),
            "is_bijection": set(assignment) == set(assignment.values()),
            "assignment_sha256": canonical_hash(assignment),
            "predictions_before_count": sum(map(len, selection_predictions.values())),
            "predictions_after_count": sum(map(len, permuted.values())),
            "metrics": shuffled,
        },
        "observed_model_metrics": observed,
        "dev_AP101_observed_minus_constant_prior": delta(constant),
        "dev_AP101_observed_minus_derangement": delta(shuffled),
        "limitations": [
            "These controls do not establish SOTA, official benchmark parity or causal proof",
            "Only one predeclared permutation is used; no permutation is selected by AP",
            "Similar road scenes can retain predictive priors after reassignment",
            "A single box per class is a deliberately limited training-only baseline",
            "Token disjointness alone does not certify independent acquisition groups",
            "No images, model weights, GPU execution or private test labels are used",
        ],
    }


def main() -> int:
    """Read two declared result files and write aggregate-only control results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-results", type=Path, required=True)
    parser.add_argument("--selection-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite control receipt: {args.output}")
    report = analyze_results(load_results(args.train_results), load_results(args.selection_results))
    report["source_sha256"] = {
        "train_results": sha256_file(args.train_results),
        "selection_results": sha256_file(args.selection_results),
        "script": sha256_file(Path(__file__)),
        "development_metrics": sha256_file(Path(development_metrics.__file__)),
    }
    report["input_paths"] = {
        "train_results": str(args.train_results),
        "selection_results": str(args.selection_results),
    }
    report["python_version"] = sys.version
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key.startswith("dev_AP101_")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
