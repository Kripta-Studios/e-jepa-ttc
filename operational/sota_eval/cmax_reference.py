"""Reference-equation contrast maximization for fixed causal EvTTC queries.

Gallego et al. define contrast maximization by warping every event to a fixed
reference time and maximizing the variance of the resulting image of warped
events (IWE).  For an approaching affine expansion with endpoint inverse TTC
``q`` and observed moving center ``c(t)``, the fixed-reference warp is

``x_ref = c_ref + (1 - q * (t - t_ref)) * (x(t) - c(t))``.

This module corrects the older local exponential warp and its implicit
last-event reference.  The observed-box centers and positive TTC domain remain
declared EvTTC adaptations, so outputs are not paper-result reproductions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from operational.sota_eval import baselines

SCHEMA = "evttc_cmax_reference_equation_v1"
PAPER = "Gallego, Rebecq, and Scaramuzza, CVPR 2018"


@dataclass(frozen=True)
class ReferenceCMaxConfig:
    """Target-independent search and admission settings."""

    minimum_inverse_ttc_per_s: float = 0.01
    maximum_inverse_ttc_per_s: float = 10.0
    coarse_log_steps: int = 65
    optimizer_maximum_iterations: int = 64
    optimizer_absolute_tolerance: float = 1e-5
    minimum_events: int = 1_000
    maximum_events: int = 50_000
    minimum_survival_fraction: float = 0.8
    minimum_relative_contrast_gain: float = 0.01

    def validate(self) -> None:
        if not 0 < self.minimum_inverse_ttc_per_s < self.maximum_inverse_ttc_per_s:
            raise ValueError("inverse-TTC bounds must be finite, positive, and increasing")
        if self.coarse_log_steps < 9 or self.optimizer_maximum_iterations <= 0:
            raise ValueError("optimizer budgets are too small")
        if self.minimum_events <= 0 or self.maximum_events < self.minimum_events:
            raise ValueError("event budgets are invalid")
        if not 0 < self.minimum_survival_fraction <= 1:
            raise ValueError("minimum survival fraction must lie in (0,1]")


def warp_affine_expansion_to_reference(
    event_xy: np.ndarray,
    event_time_s: np.ndarray,
    *,
    inverse_ttc_per_s: float,
    reference_time_s: float,
    reference_center_xy: tuple[float, float],
    event_centers_xy: np.ndarray,
) -> np.ndarray:
    """Warp events with the paper's affine expansion equation."""

    points = np.asarray(event_xy, dtype=np.float64)
    times = np.asarray(event_time_s, dtype=np.float64).reshape(-1)
    centers = np.asarray(event_centers_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("event_xy must have shape [N,2]")
    if times.shape != (len(points),) or centers.shape != points.shape:
        raise ValueError("times and event centers must match event_xy")
    if not np.isfinite(points).all() or not np.isfinite(times).all():
        raise ValueError("events must be finite")
    if not np.isfinite(centers).all() or inverse_ttc_per_s < 0:
        raise ValueError("centers must be finite and inverse TTC non-negative")
    delta = times - float(reference_time_s)
    scale = 1.0 - float(inverse_ttc_per_s) * delta
    reference_center = np.asarray(reference_center_xy, dtype=np.float64)
    return reference_center + scale[:, None] * (points - centers)


def signed_iwe(
    warped_xy: np.ndarray,
    polarities: np.ndarray,
    *,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, float]:
    """Bilinearly accumulate the signed IWE used by the contrast objective."""

    height, width = image_shape
    points = np.asarray(warped_xy, dtype=np.float64)
    polarity = np.where(np.asarray(polarities).reshape(-1) > 0, 1.0, -1.0)
    if height <= 1 or width <= 1 or points.shape != (len(polarity), 2):
        raise ValueError("IWE inputs or image shape are invalid")
    x, y = points[:, 0], points[:, 1]
    inside = (
        np.isfinite(points).all(axis=1)
        & (x >= 0.0)
        & (x < width - 1.0)
        & (y >= 0.0)
        & (y < height - 1.0)
    )
    image = np.zeros((height, width), dtype=np.float64)
    if not np.any(inside):
        return image, 0.0
    x, y, polarity = x[inside], y[inside], polarity[inside]
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    dx, dy = x - x0, y - y0
    for offset_x, offset_y, weight in (
        (0, 0, (1.0 - dx) * (1.0 - dy)),
        (1, 0, dx * (1.0 - dy)),
        (0, 1, (1.0 - dx) * dy),
        (1, 1, dx * dy),
    ):
        np.add.at(image, (y0 + offset_y, x0 + offset_x), polarity * weight)
    return image, float(np.mean(inside))


def _score(
    rate: float,
    *,
    event_xy: np.ndarray,
    event_time_s: np.ndarray,
    polarities: np.ndarray,
    reference_time_s: float,
    reference_center_xy: tuple[float, float],
    event_centers_xy: np.ndarray,
    image_shape: tuple[int, int],
    minimum_survival_fraction: float,
) -> tuple[float, float]:
    warped = warp_affine_expansion_to_reference(
        event_xy,
        event_time_s,
        inverse_ttc_per_s=rate,
        reference_time_s=reference_time_s,
        reference_center_xy=reference_center_xy,
        event_centers_xy=event_centers_xy,
    )
    image, survival = signed_iwe(warped, polarities, image_shape=image_shape)
    if survival < minimum_survival_fraction:
        return float("-inf"), survival
    return float(np.var(image)), survival


def maximize_reference_cmax(
    event_xy: np.ndarray,
    event_time_s: np.ndarray,
    polarities: np.ndarray,
    *,
    reference_time_s: float,
    reference_center_xy: tuple[float, float],
    event_centers_xy: np.ndarray,
    image_shape: tuple[int, int],
    config: ReferenceCMaxConfig | None = None,
) -> dict[str, Any]:
    """Maximize positive-domain signed-IWE variance with fixed budgets."""

    resolved = config or ReferenceCMaxConfig()
    resolved.validate()
    points = np.asarray(event_xy, dtype=np.float64)
    times = np.asarray(event_time_s, dtype=np.float64).reshape(-1)
    polarity = np.asarray(polarities).reshape(-1)
    centers = np.asarray(event_centers_xy, dtype=np.float64)
    if len(points) < resolved.minimum_events:
        return {"valid": False, "reason": "insufficient_events", "event_count": len(points)}
    if len(points) > resolved.maximum_events:
        indices = np.linspace(0, len(points) - 1, resolved.maximum_events, dtype=np.int64)
        points, times, polarity, centers = (
            points[indices],
            times[indices],
            polarity[indices],
            centers[indices],
        )
    evaluations = 0

    def objective(rate: float) -> float:
        nonlocal evaluations
        evaluations += 1
        contrast, _ = _score(
            rate,
            event_xy=points,
            event_time_s=times,
            polarities=polarity,
            reference_time_s=reference_time_s,
            reference_center_xy=reference_center_xy,
            event_centers_xy=centers,
            image_shape=image_shape,
            minimum_survival_fraction=resolved.minimum_survival_fraction,
        )
        return contrast

    grid = np.geomspace(
        resolved.minimum_inverse_ttc_per_s,
        resolved.maximum_inverse_ttc_per_s,
        resolved.coarse_log_steps,
    )
    contrasts = np.asarray([objective(float(rate)) for rate in grid])
    best = int(np.argmax(contrasts))
    lower = float(grid[max(0, best - 1)])
    upper = float(grid[min(len(grid) - 1, best + 1)])
    if lower < upper and np.isfinite(contrasts[best]):
        optimized: Any = minimize_scalar(
            lambda rate: -objective(float(rate)),
            bounds=(lower, upper),
            method="bounded",
            options={
                "xatol": resolved.optimizer_absolute_tolerance,
                "maxiter": resolved.optimizer_maximum_iterations,
            },
        )
        rate = float(optimized.x)
    else:
        rate = float(grid[best])
    contrast, survival = _score(
        rate,
        event_xy=points,
        event_time_s=times,
        polarities=polarity,
        reference_time_s=reference_time_s,
        reference_center_xy=reference_center_xy,
        event_centers_xy=centers,
        image_shape=image_shape,
        minimum_survival_fraction=resolved.minimum_survival_fraction,
    )
    null_contrast, null_survival = _score(
        0.0,
        event_xy=points,
        event_time_s=times,
        polarities=polarity,
        reference_time_s=reference_time_s,
        reference_center_xy=reference_center_xy,
        event_centers_xy=centers,
        image_shape=image_shape,
        minimum_survival_fraction=resolved.minimum_survival_fraction,
    )
    gain = (contrast - null_contrast) / max(abs(null_contrast), 1e-12)
    valid = bool(
        np.isfinite(contrast)
        and gain >= resolved.minimum_relative_contrast_gain
        and survival >= resolved.minimum_survival_fraction
    )
    return {
        "valid": valid,
        "reason": "ok" if valid else "insufficient_positive_contrast_gain",
        "inverse_ttc_per_s": rate,
        "prediction_ttc_s": 1.0 / rate if valid else None,
        "contrast": contrast if np.isfinite(contrast) else None,
        "null_contrast": null_contrast if np.isfinite(null_contrast) else None,
        "relative_contrast_gain": gain if np.isfinite(gain) else None,
        "survival_fraction": survival,
        "null_survival_fraction": null_survival,
        "event_count": int(len(points)),
        "evaluations": evaluations + 2,
    }


def _predict_row(
    row: Mapping[str, Any],
    input_config: baselines.BaselineConfig,
    config: ReferenceCMaxConfig,
) -> dict[str, Any]:
    events, roi, box_times, boxes = baselines._load_query_events(row, input_config)
    if len(events) < config.minimum_events:
        return {"valid": False, "reason": f"insufficient_roi_events:{len(events)}"}
    interpolated = np.column_stack(
        [np.interp(events[:, 0], box_times, boxes[:, coordinate]) for coordinate in range(4)]
    )
    widths = interpolated[:, 2] - interpolated[:, 0]
    heights = interpolated[:, 3] - interpolated[:, 1]
    margin_x = np.maximum(widths * input_config.object_margin_fraction, 1.0)
    margin_y = np.maximum(heights * input_config.object_margin_fraction, 1.0)
    keep = (
        (events[:, 1] >= interpolated[:, 0] - margin_x)
        & (events[:, 1] <= interpolated[:, 2] + margin_x)
        & (events[:, 2] >= interpolated[:, 1] - margin_y)
        & (events[:, 2] <= interpolated[:, 3] + margin_y)
    )
    selected = events[keep]
    centers = np.column_stack(
        (
            0.5 * (interpolated[:, 0] + interpolated[:, 2]),
            0.5 * (interpolated[:, 1] + interpolated[:, 3]),
        )
    )[keep]
    current = boxes[-1]
    result = maximize_reference_cmax(
        selected[:, 1:3],
        selected[:, 0],
        selected[:, 3],
        reference_time_s=int(row["anchor_us"]) * 1e-6,
        reference_center_xy=(
            float(0.5 * (current[0] + current[2])),
            float(0.5 * (current[1] + current[3])),
        ),
        event_centers_xy=centers,
        image_shape=(roi[3] - roi[1], roi[2] - roi[0]),
        config=config,
    )
    result["roi_xyxy"] = list(roi)
    result["roi_event_count"] = int(len(events))
    result["object_event_count"] = int(len(selected))
    return result


def _fragment_path(output: Path, query_id: str) -> Path:
    return output / "fragments" / f"{hashlib.sha256(query_id.encode()).hexdigest()[:16]}.json"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_reference_cmax(
    *,
    query_manifest: str | Path,
    output_dir: str | Path,
    selection: str = "first_per_sequence",
    maximum_queries: int | None = 32,
    config: ReferenceCMaxConfig | None = None,
) -> dict[str, Any]:
    """Run or resume the corrected fixed-query CMax variant."""

    resolved = config or ReferenceCMaxConfig()
    resolved.validate()
    manifest_path = Path(query_manifest).resolve(strict=True)
    _, rows = baselines._validate_manifest(manifest_path, maximum_queries, selection)
    output = Path(output_dir)
    source_sha = baselines._file_sha256(__file__)
    manifest_sha = baselines._file_sha256(manifest_path)
    input_config = baselines.BaselineConfig()
    fragments: list[dict[str, Any]] = []
    for row in rows:
        path = _fragment_path(output, str(row["query_id"]))
        binding = {
            "schema": SCHEMA,
            "query_id": str(row["query_id"]),
            "query_metadata_sha256": str(row["metadata_sha256"]),
            "query_manifest_sha256": manifest_sha,
            "source_sha256": source_sha,
            "config": asdict(resolved),
        }
        if path.is_file():
            fragment = json.loads(path.read_text(encoding="utf-8"))
            baselines._verify_signed(fragment)
            if any(fragment.get(key) != value for key, value in binding.items()):
                raise ValueError(f"resume binding mismatch for {row['query_id']}")
        else:
            started = time.perf_counter()
            try:
                result = _predict_row(row, input_config, resolved)
                failure = None if result.get("valid") is True else str(result.get("reason"))
            except (RuntimeError, ValueError, OSError, np.linalg.LinAlgError) as error:
                result = {"valid": False, "reason": f"{type(error).__name__}:{error}"}
                failure = str(result["reason"])
            fragment = baselines._signed(
                {
                    **binding,
                    "status": "SUCCESS" if failure is None else "FAILED_RETAINED",
                    "sequence_id": str(row["sequence_id"]),
                    "anchor_us": int(row["anchor_us"]),
                    "prediction_ttc_s": result.get("prediction_ttc_s"),
                    "failure": failure,
                    "diagnostics": result,
                    "runtime_s": time.perf_counter() - started,
                    "target_fields_opened": [],
                    "scientific_optimizer_updates": 0,
                }
            )
            _write_json(path, fragment)
        fragments.append(fragment)
    success = [fragment for fragment in fragments if fragment["status"] == "SUCCESS"]
    summary = baselines._signed(
        {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "query_manifest": manifest_path.as_posix(),
            "query_manifest_sha256": manifest_sha,
            "query_selection": selection,
            "query_count": len(rows),
            "successful": len(success),
            "failed_retained": len(rows) - len(success),
            "success_fraction": len(success) / len(rows),
            "config": asdict(resolved),
            "source_sha256": source_sha,
            "fragment_sha256": {
                str(fragment["query_id"]): str(fragment["artifact_sha256"])
                for fragment in fragments
            },
            "equation": "x_ref = c_ref + (1 - q*(t-t_ref))*(x(t)-c(t))",
            "objective": "variance of bilinearly accumulated signed IWE",
            "reference": PAPER,
            "reproduction_of_paper_result": False,
            "adaptations": [
                "positive inverse-TTC domain fixed before target access",
                "causal observed-box ROI and moving center",
                "fixed log-spaced coarse search plus bounded scalar refinement",
                "survival admission prevents out-of-image collapse",
            ],
            "target_fields_opened": [],
            "success_only_metrics_computed": False,
            "scientific_optimizer_updates": 0,
        }
    )
    _write_json(output / "SUMMARY.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selection",
        choices=("all", "first_per_sequence"),
        default="first_per_sequence",
    )
    parser.add_argument("--maximum-queries", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_reference_cmax(
        query_manifest=args.queries,
        output_dir=args.output_dir,
        selection=args.selection,
        maximum_queries=args.maximum_queries,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ReferenceCMaxConfig",
    "maximize_reference_cmax",
    "run_reference_cmax",
    "signed_iwe",
    "warp_affine_expansion_to_reference",
]
