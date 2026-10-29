"""Post-seal FCWD target join and explicit RTE scoring; never used by inference."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from operational.efficient_context.common import Lease, atomic_json, digest
from operational.sota_eval import scoring
from operational.sota_eval.fcwd_run import METHODS, ROOT, read, stamp, verify_seal

OFFICIAL_COMMIT = "79ff0842955304ec4f6164ec09baddc71386d225"
GT_CONTRACT = {
    "official_repository": "https://github.com/NAIL-HNU/event_aided_ttc",
    "official_commit": OFFICIAL_COMMIT,
    "sources": ["matlab_code/utils/plotSimpleResult.m", "matlab_code/utils/metricRee.m"],
    "header": False,
    "time_column_zero_based": 1,
    "time_unit": "relative_seconds",
    "ttc_column_zero_based": 4,
    "ttc_unit": "seconds",
    "primary_interpolation": "linear_only_inside_measured_GT_time_support; no extrapolation",
    "official_plotSimpleResult_interpolation": "linear_with_linear_extrapolation",
    "local_primary_differs_from_upstream_outside_GT_support": True,
    "rte": "100 * abs(prediction - truth) / abs(truth)",
    "selection_uses_gt": False,
    "other_gt_columns": "Ignored; never inputs to models",
}


def interpolate_truth(times: np.ndarray, targets: np.ndarray, query: float) -> tuple[float, str]:
    """Join at a fixed anchor inside measured support; never extrapolate primary GT."""
    if len(times) != len(targets) or len(times) < 2:
        raise ValueError("GT table requires at least two aligned timestamp/target rows")
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("GT timestamps must be finite and strictly increasing")
    if not math.isfinite(query):
        raise ValueError("Query time must be finite")
    if query < times[0] or query > times[-1]:
        return math.nan, "OUTSIDE_MEASURED_GT_SUPPORT"
    exact = np.flatnonzero(times == query)
    if len(exact):
        value = float(targets[int(exact[0])])
        return value, "EXACT" if math.isfinite(value) else "NONFINITE_GT"
    index = int(np.searchsorted(times, query))
    index = max(1, min(index, len(times) - 1))
    left, right = index - 1, index
    if not math.isfinite(float(targets[left])) or not math.isfinite(float(targets[right])):
        return math.nan, "NONFINITE_GT_ENDPOINT"
    weight = (query - float(times[left])) / (float(times[right]) - float(times[left]))
    value = float(targets[left]) + weight * (float(targets[right]) - float(targets[left]))
    status = "INTERPOLATED_LINEAR"
    return (value, status) if math.isfinite(value) else (math.nan, "NONFINITE_GT_ARITHMETIC")


def _target_assets(asset_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Restrict target reads to the three named public FCWD files, after sealing."""
    result = {}
    for sequence in range(1, 4):
        matching = [
            asset for asset in asset_manifest["assets"] if asset["name"] == f"fcwd_{sequence}_ttc"
        ]
        if len(matching) != 1:
            raise ValueError(f"Exactly one pinned FCWD{sequence} target asset is required")
        asset = matching[0]
        path = Path(asset["path"]).resolve()
        if (
            asset["role"] != "ttc_target"
            or path.name != "gt_ttc.csv"
            or path.parent.name != f"sequence_{sequence}"
            or path.parent.parent.name != "sealed_labels"
        ):
            raise ValueError("Target asset does not match the known public FCWD filename contract")
        if any(
            part.lower() in {"stage76", "test12", "private_test", "public_validation"}
            for part in path.parts
        ):
            raise ValueError("Protected non-FCWD target path rejected")
        result[f"FCWD{sequence}"] = asset
    return result


def score_sealed(
    manifest_path: Path,
    output: Path,
    asset_manifest_path: Path,
    *,
    bootstrap_draws: int = 10000,
    seed: int = 20261008,
) -> dict[str, Any]:
    """Require every committed prediction before opening any target CSV."""
    with Lease(output):
        # This is intentionally the first dependency access. Missing/corrupt seals
        # must fail before reading even the target asset manifest.
        manifest, predictions = verify_seal(output, manifest_path)
        source_freeze = read(output / "SOURCE_FREEZE.json")
        for relative, expected in source_freeze["sources"].items():
            if digest(ROOT / relative) != expected:
                raise ValueError(f"Frozen scorer/input/model source changed: {relative}")
        if digest(asset_manifest_path) != manifest["asset_manifest_sha256"]:
            raise ValueError("FCWD asset manifest differs from the label-free population binding")
        reference_path = Path(manifest["reference_contract_path"])
        if digest(reference_path) != manifest["reference_contract_sha256"]:
            raise ValueError("Official FCWD target-column evidence changed after population freeze")
        assets = _target_assets(read(asset_manifest_path))
        if set(row["sequence_id"] for row in manifest["rows"]) != set(assets):
            raise ValueError("FCWD scoring requires all three predeclared sequences")
        tables: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        gt_receipts = {}
        for sequence_id, asset in assets.items():
            path = Path(asset["path"])
            if digest(path) != asset["sha256"] or path.stat().st_size != asset["bytes"]:
                raise ValueError(f"FCWD GT bytes changed: {sequence_id}")
            table = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
            if table.shape[1] < 5:
                raise ValueError(
                    "Official FCWD GT CSV requires at least five columns without a header"
                )
            times, targets = table[:, 1].copy(), table[:, 4].copy()
            interpolate_truth(times, targets, float(times[0]))
            if digest(path) != asset["sha256"]:
                raise ValueError("FCWD target changed during scoring read")
            tables[sequence_id] = times, targets
            gt_receipts[sequence_id] = {
                "path": str(path.resolve()),
                "sha256": asset["sha256"],
                "bytes": asset["bytes"],
                "rows": len(table),
                "columns": table.shape[1],
                "time_min_s": float(times[0]),
                "time_max_s": float(times[-1]),
                "nonfinite_target_rows": int(np.count_nonzero(~np.isfinite(targets))),
            }
        result_rows = []
        counts: dict[str, int] = {}
        for row, prediction in zip(manifest["rows"], predictions, strict=True):
            relative = float(row["anchor_relative_seconds"])
            if not math.isclose(relative, row["anchor_us"] / 1_000_000, abs_tol=1e-12):
                raise ValueError("Query relative-time coordinate changed")
            truth, join_status = interpolate_truth(*tables[row["sequence_id"]], relative)
            counts[join_status] = counts.get(join_status, 0) + 1
            result_rows.append(
                {
                    "query_id": row["query_id"],
                    "sequence_id": row["sequence_id"],
                    "anchor_us": row["anchor_us"],
                    "anchor_relative_seconds": relative,
                    "truth_ttc_seconds": truth if math.isfinite(truth) else "",
                    "gt_join_status": join_status,
                    "prediction_status": prediction["status"],
                    **prediction["ttc"],
                }
            )
        score_dir = output / "scoring"
        score_dir.mkdir(parents=True, exist_ok=True)
        joined = score_dir / "SCORED_PREDICTIONS.csv"
        with joined.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
            writer.writeheader()
            writer.writerows(result_rows)
        full_unavailable = sum(
            prediction.get("unavailable_reasons", {}).get(METHODS[4])
            == "MISSING_AUDITED_EVENT_TO_RGB_SPATIAL_MAPPING"
            for prediction in predictions
        )
        admitted_methods = METHODS[:4] if full_unavailable == len(predictions) else METHODS
        report = scoring.run(
            joined, score_dir, admitted_methods, bootstrap_draws=bootstrap_draws, seed=seed
        )
        coverage = {
            "planned_methods": list(METHODS),
            "scored_methods": list(admitted_methods),
            "all_queries_retained": len(predictions),
            "sequence_count": 3,
            "head_selection_performed": False,
            "full_model": {
                "status": "DEPENDENCY_UNAVAILABLE"
                if full_unavailable == len(predictions)
                else "SCORED_WITH_EXPLICIT_COVERAGE",
                "queries_with_missing_spatial_mapping": full_unavailable,
                "not_a_negative_model_result": full_unavailable == len(predictions),
                "dependency": "Audited event-right to RGB-right spatial mapping without GT depth",
            },
        }
        atomic_json(score_dir / "MODEL_COVERAGE.json", coverage)
        protocol = {
            "status": "COMPLETE",
            "gt_contract": GT_CONTRACT,
            "gt_files": gt_receipts,
            "gt_join_counts": counts,
            "rows_retained": len(result_rows),
            "prediction_seal_sha256": digest(output / "PREDICTIONS_SEALED.json"),
            "source_freeze_sha256": digest(output / "SOURCE_FREEZE.json"),
            "asset_manifest_sha256": digest(asset_manifest_path),
            "query_manifest_sha256": digest(manifest_path),
            "reference_contract_sha256": digest(reference_path),
            "model_coverage_sha256": digest(score_dir / "MODEL_COVERAGE.json"),
            "joined_predictions_sha256": digest(joined),
            "scoring_manifest_sha256": digest(score_dir / "SHA256.json"),
            "scorer_module_sha256": digest(Path(__file__)),
            "population_selection_used_gt": False,
            "labels_opened_only_after_verified_prediction_seal": True,
            "bootstrap_warning": (
                "Only three sequences: descriptive cluster intervals are coarse and cannot "
                "certify broad generalization or independence."
            ),
            "comparison_warning": (
                "Common local causal oracle-event-ROI queries; no paper/table reproduction claim. "
                "Full model may be unavailable because event-to-RGB mapping is unresolved."
            ),
            "checkpoint_training_exclusion_certified": False,
            "optimizer_updates": 0,
            "model_predictions_rerun": False,
        }
        atomic_json(score_dir / "TARGET_JOIN_CONTRACT.json", protocol)
        receipt = {
            "status": "COMPLETE",
            "queries": len(result_rows),
            "methods": list(admitted_methods),
            "planned_methods": list(METHODS),
            "scoring_manifest_sha256": digest(score_dir / "SHA256.json"),
            "target_join_contract_sha256": digest(score_dir / "TARGET_JOIN_CONTRACT.json"),
            "prediction_seal_sha256": protocol["prediction_seal_sha256"],
            "cohort": report["cohort"],
            "checked_utc": stamp(),
            "optimizer_updates": 0,
        }
        atomic_json(output / "SCORING_COMPLETE.json", receipt)
        print(json.dumps(receipt, indent=2))
        return receipt
