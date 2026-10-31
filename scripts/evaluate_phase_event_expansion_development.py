"""Freeze label-blind causal event estimates, then score eligible Slider500 only."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from operational.evttc_competition.geometry import _distort_radtan, _undistort_radtan
from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_campaign.data import read
from operational.phase_campaign.evaluation import metrics
from operational.phase_event_expansion.model import (
    ExpansionResult,
    SolverConfig,
    estimate_expansion,
)
from operational.phase_event_expansion.validation import (
    require_confirmation_binding,
    validate_cache_manifest,
    validate_cached_window,
    validate_population_tokens,
)


def require_synthetic_gates(summary: dict) -> None:
    """Reject failed or empty confirmation before any real event inference."""
    total = summary.get("evaluation_total", 0)
    if not (
        summary.get("all_synthetic_gates_pass") is True
        and total > 0
        and summary.get("evaluation_passed") == total
    ):
        raise ValueError("Synthetic recovery/abstention gates have not passed")


def undistort_inputs(
    arrays: dict[str, np.ndarray], calibration: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Use recorded radtan calibration, preserving clocks and original event selection."""
    fx, fy, cx, cy = calibration["event_intrinsics"]
    focal = np.array([fx, fy], np.float64)
    principal = np.array([cx, cy], np.float64)
    distortion = np.asarray(calibration["event_distortion"], np.float64)
    raw = np.column_stack((arrays["x"], arrays["y"])).astype(np.float64)

    def undistort(pixels: np.ndarray) -> np.ndarray:
        normalized = (pixels - principal) / focal
        corrected = _undistort_radtan(normalized, distortion)
        residual = (_distort_radtan(corrected, distortion) - normalized) * focal
        # The audited projection helper uses12 iterations. At Slider raster
        # corners this can leave millipixel residuals; converge further rather
        # than weakening the declared1e-3pixel inversion tolerance.
        k1, k2, p1, p2 = distortion
        for _ in range(52):
            if not residual.size or np.max(np.abs(residual)) <= 1e-3:
                break
            x, y = corrected[:, 0], corrected[:, 1]
            radius = x * x + y * y
            radial = 1 + k1 * radius + k2 * radius * radius
            if np.any(np.abs(radial) < 1e-12):
                raise ValueError("Singular recorded distortion inversion")
            tx = 2 * p1 * x * y + p2 * (radius + 2 * x * x)
            ty = p1 * (radius + 2 * y * y) + 2 * p2 * x * y
            corrected = np.column_stack(
                ((normalized[:, 0] - tx) / radial, (normalized[:, 1] - ty) / radial)
            )
            residual = (_distort_radtan(corrected, distortion) - normalized) * focal
        if residual.size and np.max(np.abs(residual)) > 1e-3:
            raise ValueError("Recorded distortion inversion failed subpixel residual check")
        return corrected * focal + principal

    points = undistort(raw)
    x0, y0, x1, y1 = arrays["roi_xyxy"]
    alpha = np.linspace(0, 1, 65)
    xs, ys = x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0)
    perimeter = np.concatenate(
        (
            np.column_stack((xs, np.full_like(xs, y0))),
            np.column_stack((xs, np.full_like(xs, y1))),
            np.column_stack((np.full_like(ys, x0), ys)),
            np.column_stack((np.full_like(ys, x1), ys)),
        )
    )
    mapped = undistort(perimeter)
    envelope = np.concatenate((mapped, points))
    roi = np.concatenate((envelope.min(0), envelope.max(0)))
    events = np.column_stack((points, arrays["t_us"], np.where(arrays["p"] > 0, 1, -1)))
    if not np.isfinite(events).all() or not np.isfinite(roi).all():
        raise ValueError("Nonfinite calibrated event coordinates")
    return events, roi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cache", "synthetic", "population", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory")
    require_confirmation_binding(args.synthetic)
    require_synthetic_gates(read(args.synthetic / "SUMMARY.json"))
    synthetic = read(args.synthetic / "CONFIG.json")
    config = SolverConfig(**synthetic["solver"])
    # Bind the exact solver source tested on the frozen synthetic confirmation.
    source_path = Path("operational/phase_event_expansion/model.py")
    sources = synthetic.get("source_sha256", {})
    if isinstance(sources, dict):
        expected = [
            v
            for k, v in sources.items()
            if str(k).replace("\\", "/").endswith(str(source_path).replace("\\", "/"))
        ]
    else:
        expected = []
    if len(expected) != 1 or file_sha(source_path) != expected[0]:
        raise ValueError("Current solver differs from frozen synthetic confirmation")
    manifest = read(args.cache / "MANIFEST.json")
    if manifest["sequence_id"] != "Slider-500" or manifest["queries"] != 77:
        raise ValueError("Only the eligible Slider500 development recording is permitted")
    if file_sha(args.cache / "CACHE_FREEZE.json") != manifest["cache_freeze_sha256"]:
        raise ValueError("Changed sensor cache freeze")
    validate_cache_manifest(manifest, read(args.cache / "CACHE_FREEZE.json"))
    atomic_receipt(
        args.output / "CONFIG.json",
        {
            "schema": "phase_event_expansion_real_development_v1",
            "solver": asdict(config),
            "synthetic_config_sha256": file_sha(args.synthetic / "CONFIG.json"),
            "synthetic_summary_sha256": file_sha(args.synthetic / "SUMMARY.json"),
            "synthetic_binding_sha256": file_sha(args.synthetic / "CONFIRMATION_BINDING.json"),
            "cache_manifest_sha256": file_sha(args.cache / "MANIFEST.json"),
            "source_hashes": {
                str(p): file_sha(p)
                for p in (
                    Path(__file__),
                    source_path,
                    Path("operational/evttc_competition/geometry.py"),
                    Path("operational/phase_event_expansion/validation.py"),
                    Path("operational/phase_campaign/evaluation.py"),
                )
            },
            "preparation": (
                "radtan undistortion12to64iterations untilresidual<=1e-3pixels; "
                "65samples/ROI edge plus selected-event envelope"
            ),
            "scope": "Slider500 eligible development; not test, no comparison to official CMax",
            "target_access": "only after all77predictions are sealed",
        },
    )
    predictions, tokens, results, costs = [], [], [], []
    started = time.perf_counter()
    for i, row in enumerate(manifest["rows"]):
        path = args.cache / row["file"]
        if file_sha(path) != row["sha256"]:
            raise ValueError("Changed event window")
        query_start = time.perf_counter()
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        validate_cached_window(arrays, row)
        try:
            events, roi = undistort_inputs(arrays, manifest["calibration"])
        except ValueError as exc:
            result = ExpansionResult(
                None,
                None,
                None,
                False,
                "calibration_inversion_failure",
                time.perf_counter() - query_start,
                {"error": str(exc)},
            )
        else:
            result = estimate_expansion(events, roi, int(row["query_us"]), config)
        record = {
            "token": row["query_id"],
            "index": row["frame_index"],
            "query_us": row["query_us"],
            **asdict(result),
        }
        atomic_receipt(args.output / "queries" / f"{i:05d}.json", record)
        predictions.append(result.ttc_seconds if result.valid else np.nan)
        tokens.append(row["query_id"])
        results.append(record)
        costs.append(time.perf_counter() - query_start)
        elapsed = time.perf_counter() - started
        print(
            f"QUERY {i + 1}/77 valid={result.valid} reason={result.reason} "
            f"elapsed={elapsed:.1f}s eta={elapsed / (i + 1) * (76 - i):.1f}s",
            flush=True,
        )
    prediction = np.array(predictions)
    save_npz(
        args.output / "PREDICTIONS.npz",
        {"prediction": prediction, "token": np.array(tokens), "cost_seconds": np.array(costs)},
    )
    atomic_receipt(
        args.output / "PREDICTIONS_SEALED.json",
        {
            "sha256": file_sha(args.output / "PREDICTIONS.npz"),
            "rows": len(tokens),
            "targets_read": False,
            "config_sha256": file_sha(args.output / "CONFIG.json"),
        },
    )
    # Target access deliberately occurs below the prediction seal.
    with np.load(args.population, allow_pickle=False) as data:
        validate_population_tokens(data["token"])
        lookup = {str(token): i for i, token in enumerate(data["token"])}
        indices = np.array([lookup[token] for token in tokens])
        if not np.all(data["group"][indices] == "Slider-500"):
            raise ValueError("Target identity is not the eligible development recording")
        target, native = data["target"][indices], data["native"][indices, 0]
        geometry = data["geometry"][indices]
    valid = np.isfinite(prediction)
    report = {
        "scope": "eligible Slider500 development only; conditional and full coverage separated",
        "expert_conditional": metrics(target, prediction),
        "F0_full": metrics(target, native),
        "geometry_conditional": metrics(target, geometry),
        "expert_then_F0": metrics(target, np.where(valid, prediction, native)),
        "geometry_then_F0": metrics(target, np.where(np.isfinite(geometry), geometry, native)),
        "F0_on_expert_valid_rows": metrics(target[valid], native[valid]),
        "geometry_on_expert_valid_rows": metrics(target[valid], geometry[valid]),
        "reasons": dict(Counter(row["reason"] for row in results)),
        "timing": {
            "mean_seconds": float(np.mean(costs)),
            "p95_seconds": float(np.quantile(costs, 0.95)),
            "wall_seconds": time.perf_counter() - started,
            "scope": "cached NPZ read+undistort+solver+perqueryreceipt; excludes raw extraction",
        },
        "population_sha256": file_sha(args.population),
        "prediction_seal_sha256": file_sha(args.output / "PREDICTIONS_SEALED.json"),
    }
    atomic_receipt(args.output / "REPORT.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
