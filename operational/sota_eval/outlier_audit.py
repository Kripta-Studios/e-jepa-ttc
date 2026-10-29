"""Read-only prediction audit; common output-bound sensitivity is post-hoc only."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METHODS = (
    "H8_seed7",
    "H8_seed13",
    "H8_seed23",
    "public_Garl_event_lhr",
    "public_Garl_rgb_event_full",
)


def digest(path: Path) -> str:
    """Hash each small source artifact, without accessing raw sensor containers."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read an already scored/published table, never a raw target asset."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def output_bound(path: Path) -> float:
    """Derive the registered H8 support from its reciprocal clamp in source AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "phase_to_ttc"
    )
    matches = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "clamp_min"
    ]
    if len(matches) != 1 or len(matches[0].args) != 1:
        raise ValueError("H8 output clamp must be re-audited")
    value = matches[0].args[0]
    if not (
        isinstance(value, ast.BinOp)
        and isinstance(value.op, ast.Div)
        and isinstance(value.left, ast.Constant)
        and value.left.value == 1.0
        and isinstance(value.right, ast.Constant)
        and isinstance(value.right.value, (float, int))
        and value.right.value > 0
    ):
        raise ValueError("H8 output clamp no longer has audited reciprocal form")
    return float(value.right.value)


def metrics(truth: np.ndarray, prediction: np.ndarray, sequence: np.ndarray) -> dict[str, float]:
    """Calculate signed-target errors without modifying targets or selecting rows."""
    if (
        len(truth) == 0
        or truth.shape != prediction.shape
        or sequence.shape != truth.shape
        or not np.isfinite(truth).all()
        or (truth == 0).any()
        or not np.isfinite(prediction).all()
    ):
        raise ValueError("Complete finite prediction cohort and finite nonzero GT required")
    absolute = np.abs(prediction - truth)
    relative = 100.0 * absolute / np.abs(truth)
    groups = np.unique(sequence)
    return {
        "micro_mae_seconds": float(absolute.mean()),
        "micro_median_absolute_error_seconds": float(np.median(absolute)),
        "micro_rmse_seconds": float(np.sqrt(np.mean(absolute**2))),
        "micro_rte_percent": float(relative.mean()),
        "macro_sequence_mae_seconds": float(
            np.mean([absolute[sequence == s].mean() for s in groups])
        ),
        "macro_sequence_rte_percent": float(
            np.mean([relative[sequence == s].mean() for s in groups])
        ),
    }


def sensitivity(
    truth: np.ndarray, predictions: dict[str, np.ndarray], sequence: np.ndarray, bound: float
) -> list[dict[str, Any]]:
    """Apply identical fixed output clipping to every model on the unchanged cohort."""
    rows = []
    for method, prediction in predictions.items():
        raw = metrics(truth, prediction, sequence)
        bounded = metrics(truth, np.clip(prediction, -bound, bound), sequence)
        for status, values in (
            ("ORIGINAL_UNALTERED", raw),
            ("POSTHOC_COMMON_OUTPUT_BOUND", bounded),
        ):
            rows.append(
                {
                    "method": method,
                    "analysis": status,
                    "rows": len(truth),
                    "sequences": len(np.unique(sequence)),
                    "common_bound_seconds": bound,
                    "predictions_outside_bound": int((np.abs(prediction) > bound).sum()),
                    "targets_outside_bound_retained": int((np.abs(truth) > bound).sum()),
                    **values,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write derived tables to the separate diagnostic output directory."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(campaign: Path, output: Path) -> dict[str, Any]:
    """Audit sealed predictions, retaining every eligible row and primary metric."""
    metrics_root = campaign / "expanded_metrics"
    rgb = campaign / "dev32_expanded_rgb"
    sources = {
        "scored_rows": metrics_root / "SCORED_ROWS.csv",
        "paired": metrics_root / "PAIRED.csv",
        "scoring_metadata": metrics_root / "METADATA.json",
        "manifest": rgb / "QUERY_MANIFEST.json",
        "prediction_seal": rgb / "PREDICTIONS_SEALED.json",
        "phase_source": ROOT / "src/e_jepa_ttc/simplex_t/phase.py",
        "model_caller": ROOT / "operational/evttc_transfer/models.py",
        "full_conversion": ROOT / "operational/evttc_rgb_transfer/model.py",
        "audit_source": Path(__file__),
    }
    hashes = {name: digest(path) for name, path in sources.items()}
    if output.resolve() == campaign.resolve() or output.resolve() in {
        metrics_root.resolve(),
        rgb.resolve(),
        (rgb / "predictions").resolve(),
    }:
        raise ValueError("A separate diagnostic output directory is required")
    all_rows = read_csv(sources["scored_rows"])
    if len({row["query_id"] for row in all_rows}) != len(all_rows):
        raise ValueError("Duplicate query identities")
    rows = [row for row in all_rows if row["scorer_gt_status"] == "ELIGIBLE"]
    truth = np.asarray([float(row["truth_ttc_seconds"]) for row in rows])
    sequence = np.asarray([row["sequence_id"] for row in rows])
    predictions = {name: np.asarray([float(row[name]) for row in rows]) for name in METHODS}
    bound = output_bound(sources["phase_source"])
    sensitivity_rows = sensitivity(truth, predictions, sequence, bound)
    full_error = np.abs(predictions[METHODS[-1]] - truth)
    order = np.argsort(-full_error, kind="stable")
    manifest = json.loads(sources["manifest"].read_text(encoding="utf-8"))
    ordinal = {row["query_id"]: index for index, row in enumerate(manifest["rows"])}
    outliers = []
    for index in order[:10]:
        row = rows[int(index)]
        path = rgb / "predictions" / f"query_{ordinal[row['query_id']]:05d}.json"
        sources[row["query_id"]] = path
        hashes[row["query_id"]] = digest(path)
        fragment = json.loads(path.read_text(encoding="utf-8"))
        if fragment["query_id"] != row["query_id"]:
            raise ValueError("Fragment identity mismatch")
        heights = np.asarray(fragment["heights"], dtype=np.float32)
        denominator = np.float32(1) - heights[0] / heights[1]
        denominator64 = 1.0 - float(heights[0]) / float(heights[1])
        with np.errstate(divide="ignore", invalid="ignore"):
            reconstructed = float(np.float32(0.1) / denominator)
        expected = float(row[METHODS[-1]])
        if not math.isfinite(reconstructed) or reconstructed != expected:
            raise ValueError("Stored TTC does not match native FP32 height equation")
        outliers.append(
            {
                "query_id": row["query_id"],
                "sequence_id": row["sequence_id"],
                "truth_ttc_seconds": float(truth[index]),
                "prediction_ttc_seconds": expected,
                "absolute_error_seconds": float(full_error[index]),
                "height_0": float(heights[0]),
                "height_1": float(heights[1]),
                "height_difference": float(heights[1] - heights[0]),
                "native_fp32_denominator": float(denominator),
                "diagnostic_fp64_denominator_same_stored_heights": denominator64,
                "native_fp32_ttc_reconstructed": reconstructed,
                "fraction_total_absolute_error": float(full_error[index] / full_error.sum()),
                "fraction_total_squared_error": float(
                    full_error[index] ** 2 / np.sum(full_error**2)
                ),
                "fragment_sha256": hashes[row["query_id"]],
            }
        )
    pairs = read_csv(sources["paired"])
    result = {
        "status": "COMPLETE_POSTHOC_DIAGNOSTIC_NOT_CONFIRMATORY",
        "total_rows": len(all_rows),
        "eligible_rows": len(rows),
        "gt_ineligible_rows_preserved_in_source": len(all_rows) - len(rows),
        "sequence_count": len(np.unique(sequence)),
        "bound_seconds_derived_from_h8_source": bound,
        "bound_derivation": (
            "phase_to_ttc: sign / abs(q).clamp_min(1/60); "
            "applied identically to all five output columns only"
        ),
        "primary_metrics_unchanged": True,
        "targets_changed": False,
        "prediction_based_row_filter": False,
        "optimizer_updates": 0,
        "gpu_work": False,
        "original_paired_confidence_intervals": pairs,
        "sensitivity": sensitivity_rows,
        "largest_full_errors": outliers,
        "interpretation": [
            "Primary MAE/RMSE retain all large finite predictions; "
            "median describes a different aspect of performance.",
            "Near-equal predicted heights cause an ill-conditioned native ratio denominator, "
            "not a missing-data failure.",
            "Post-hoc common clipping changes the emitted prediction rule; "
            "it is not an official Garl result or a replacement primary metric.",
            "Copied confidence intervals apply only to original predictions; "
            "no confirmatory interval or significance claim is made for clipped sensitivity.",
            "Existing CIs are paired sequence bootstrap on 32 sequence IDs; "
            "related recordings/scenarios may remain correlated.",
            "Local dev32 transfer is not Table VI or benchmark10 reproduction; "
            "public checkpoint training lineage and exact official cohort remain unresolved.",
            "Full Garl consumes RGB+events while H8 consumes events; "
            "H8 has bounded output support and a longer temporal history.",
        ],
        "source_hashes": {
            name: {"path": str(sources[name]), "sha256": sha} for name, sha in hashes.items()
        },
    }
    if any(digest(sources[name]) != sha for name, sha in hashes.items()):
        raise ValueError("Audit input changed while being read")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "COMMON_BOUND_SENSITIVITY_POSTHOC.csv", sensitivity_rows)
    write_csv(output / "FULL_TOP_ERRORS.csv", outliers)
    (output / "PAIRED_PRIMARY_UNALTERED.csv").write_bytes(sources["paired"].read_bytes())
    (output / "AUDIT.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "outlier_audit.py").write_bytes(Path(__file__).read_bytes())
    names = [
        "AUDIT.json",
        "COMMON_BOUND_SENSITIVITY_POSTHOC.csv",
        "FULL_TOP_ERRORS.csv",
        "PAIRED_PRIMARY_UNALTERED.csv",
        "outlier_audit.py",
    ]
    (output / "SHA256SUMS.txt").write_text(
        "".join(f"{digest(output / name)}  {name}\n" for name in names), encoding="utf-8"
    )
    return result


def main() -> int:
    """Run a deterministic CPU diagnostic on existing sealed predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.campaign, args.output)
    print(json.dumps({key: result[key] for key in ("status", "eligible_rows", "sequence_count")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
