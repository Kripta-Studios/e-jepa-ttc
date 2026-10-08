"""Run fixed-query CMax and STRTTC adaptations without opening TTC targets.

The implementations called here are source-traceable local ports.  They are
deliberately reported as EvTTC adaptations rather than reproductions of the
papers: the fixed query manifest supplies causal observed boxes, and STRTTC is
run independently per query instead of carrying the MATLAB sequence state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

OFFICIAL_STRTTC_COMMIT = "79ff0842955304ec4f6164ec09baddc71386d225"
OFFICIAL_STRTTC_REPOSITORY = "https://github.com/NAIL-HNU/event_aided_ttc"
CMAX_PAPER = "Gallego, Rebecq, and Scaramuzza, CVPR 2018"
METHODS = ("cmax", "strttc")
SENSOR_WIDTH = 1280
SENSOR_HEIGHT = 720
SCHEMA = "evttc_fixed_query_geometric_baselines_v1"


@dataclass(frozen=True)
class BaselineConfig:
    """Fixed, target-independent resource and geometry settings."""

    lookback_us: int = 200_000
    roi_margin_fraction: float = 0.10
    object_margin_fraction: float = 0.10
    cmax_maximum_events: int = 50_000
    strttc_maximum_events: int = 250_000
    cmax_minimum_events: int = 1_000
    strttc_minimum_events: int = 2_000
    cmax_minimum_ttc_s: float = 0.10
    cmax_maximum_ttc_s: float = 100.0
    cmax_coarse_steps: int = 65
    cmax_minimum_relative_contrast_gain: float = 0.01
    strttc_nonlinear_refinement: bool = True
    strttc_nonlinear_maximum_function_evaluations: int = 40

    def validate(self) -> None:
        if self.lookback_us <= 0 or self.roi_margin_fraction < 0:
            raise ValueError("lookback and ROI margin must be non-negative/positive")
        if min(self.cmax_maximum_events, self.strttc_maximum_events) <= 0:
            raise ValueError("event budgets must be positive")
        if self.cmax_coarse_steps < 9:
            raise ValueError("CMax requires at least nine coarse steps")
        if self.cmax_maximum_ttc_s <= self.cmax_minimum_ttc_s:
            raise ValueError("CMax TTC bounds are invalid")
        if not self.strttc_nonlinear_refinement:
            raise ValueError("this protocol requires the published nonlinear STRTTC stage")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return payload


def _verify_signed(value: Mapping[str, Any]) -> None:
    payload = dict(value)
    expected = payload.pop("artifact_sha256", None)
    if not isinstance(expected, str) or _canonical_sha256(payload) != expected:
        raise ValueError("artifact canonical SHA-256 mismatch")


def _validate_manifest(
    path: Path, maximum_queries: int | None, selection: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("status") != "LABEL_FREE_MANIFEST":
        raise ValueError("query manifest must be a LABEL_FREE_MANIFEST")
    forbidden = {str(value).lower() for value in document.get("forbidden_assets", [])}
    required_forbidden = {"ttc.csv", "gt.hdf5", "distance", "depth", "navigation"}
    if not required_forbidden.issubset(forbidden):
        raise ValueError("manifest does not explicitly forbid every target-bearing asset")
    raw_rows = document.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("query manifest rows must be a non-empty list")
    metadata_hashes = [
        raw.get("metadata_sha256") if isinstance(raw, dict) else None for raw in raw_rows
    ]
    if _canonical_sha256(metadata_hashes) != document.get("rows_metadata_sha256"):
        raise ValueError("query manifest rows_metadata_sha256 mismatch")
    if selection == "first_per_sequence":
        selected_rows: list[object] = []
        selected_sequences: set[str] = set()
        for raw in raw_rows:
            if not isinstance(raw, dict):
                raise ValueError("every query row must be an object")
            sequence_id = str(raw.get("sequence_id"))
            if sequence_id not in selected_sequences:
                selected_sequences.add(sequence_id)
                selected_rows.append(raw)
        raw_rows = selected_rows
    elif selection != "all":
        raise ValueError("selection must be 'all' or 'first_per_sequence'")
    if maximum_queries is not None:
        if maximum_queries <= 0:
            raise ValueError("maximum_queries must be positive")
        raw_rows = raw_rows[:maximum_queries]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    forbidden_row_fields = {"ttc", "ttc_seconds", "target_ttc", "distance", "depth", "gt"}
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise ValueError("every query row must be an object")
        required = {
            "query_id",
            "sequence_id",
            "anchor_us",
            "raw_path",
            "raw_stat",
            "windows_us",
            "boxes_xyxy3",
            "metadata_sha256",
        }
        missing = sorted(required.difference(raw))
        if missing:
            raise ValueError(f"query row lacks required fields: {missing}")
        unexpected = forbidden_row_fields.intersection(str(key).lower() for key in raw)
        if unexpected:
            raise ValueError(f"query row contains forbidden target fields: {sorted(unexpected)}")
        query_id = str(raw["query_id"])
        if query_id in seen:
            raise ValueError(f"duplicate query_id: {query_id}")
        seen.add(query_id)
        metadata = dict(raw)
        metadata_sha = str(metadata.pop("metadata_sha256"))
        if _canonical_sha256(metadata) != metadata_sha:
            raise ValueError(f"{query_id}: metadata SHA-256 mismatch")
        windows = np.asarray(raw["windows_us"], dtype=np.int64)
        boxes = np.asarray(raw["boxes_xyxy3"], dtype=np.float64)
        if windows.shape != (3, 2) or boxes.shape != (3, 4):
            raise ValueError(f"{query_id}: expected three windows and three xyxy boxes")
        if not np.array_equal(windows[:, 1] - windows[:, 0], np.full(3, 100_000)):
            raise ValueError(f"{query_id}: windows are not exact 100 ms intervals")
        if int(windows[-1, 1]) != int(raw["anchor_us"]):
            raise ValueError(f"{query_id}: final window does not end at anchor_us")
        if np.any(~np.isfinite(boxes)) or np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError(f"{query_id}: boxes are non-finite or degenerate")
        raw_path = Path(str(raw["raw_path"]))
        stat = raw_path.stat()
        pin = raw["raw_stat"]
        if stat.st_size != int(pin["size_bytes"]) or stat.st_mtime_ns != int(pin["mtime_ns"]):
            raise ValueError(f"{query_id}: raw HDF5 stat differs from frozen manifest")
        rows.append(dict(raw))
    return document, rows


def _roi(boxes: np.ndarray, margin_fraction: float) -> tuple[int, int, int, int]:
    x0, y0 = boxes[:, :2].min(axis=0)
    x1, y1 = boxes[:, 2:].max(axis=0)
    margin_x = max(float(x1 - x0) * margin_fraction, 2.0)
    margin_y = max(float(y1 - y0) * margin_fraction, 2.0)
    result = (
        max(0, int(math.floor(x0 - margin_x))),
        max(0, int(math.floor(y0 - margin_y))),
        min(SENSOR_WIDTH, int(math.ceil(x1 + margin_x))),
        min(SENSOR_HEIGHT, int(math.ceil(y1 + margin_y))),
    )
    if result[2] - result[0] <= 1 or result[3] - result[1] <= 1:
        raise ValueError("causal observed-box ROI is degenerate")
    return result


def _uniform_bound(values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, maximum, dtype=np.int64)
    return values[indices]


def _load_query_events(
    row: Mapping[str, Any], config: BaselineConfig
) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray, np.ndarray]:
    """Load only the final fixed 200 ms and return ROI and box timeline."""

    from operational.evttc_transfer.inputs import EvTTCEventReader

    windows = np.asarray(row["windows_us"], dtype=np.int64)
    boxes = np.asarray(row["boxes_xyxy3"], dtype=np.float64)
    roi = _roi(boxes, config.roi_margin_fraction)
    parts: list[np.ndarray] = []
    with EvTTCEventReader(Path(str(row["raw_path"]))) as reader:
        for start_us, end_us in windows[-2:]:
            for piece in reader.iter_window_chunks(int(start_us), int(end_us)):
                selected = (
                    (piece["x"] >= roi[0])
                    & (piece["x"] < roi[2])
                    & (piece["y"] >= roi[1])
                    & (piece["y"] < roi[3])
                )
                if np.any(selected):
                    parts.append(
                        np.column_stack(
                            (
                                piece["t"][selected].astype(np.float64) * 1e-6,
                                piece["x"][selected].astype(np.float64) - roi[0],
                                piece["y"][selected].astype(np.float64) - roi[1],
                                np.where(piece["p"][selected] > 0, 1.0, -1.0),
                            )
                        )
                    )
    events = np.concatenate(parts) if parts else np.empty((0, 4), dtype=np.float64)
    box_times_s = windows[:, 1].astype(np.float64) * 1e-6
    local_boxes = boxes - np.asarray((roi[0], roi[1], roi[0], roi[1]), dtype=np.float64)
    return events, roi, box_times_s, local_boxes


def _event_intrinsics(
    raw_path: Path, roi: tuple[int, int, int, int]
) -> tuple[float, float, float, float]:
    import h5py

    name = "prophesee/event_cam_left/calib/intrinsics"
    with h5py.File(raw_path, "r") as handle:
        if name not in handle:
            raise ValueError("event-left intrinsics are missing")
        values = np.asarray(handle[name], dtype=np.float64).reshape(-1)
    if values.shape[0] < 4 or values[0] <= 0 or values[1] <= 0:
        raise ValueError("event-left intrinsics are invalid")
    return float(values[0]), float(values[1]), float(values[2] - roi[0]), float(values[3] - roi[1])


def _predict_cmax(
    events: np.ndarray,
    roi: tuple[int, int, int, int],
    box_times_s: np.ndarray,
    boxes: np.ndarray,
    config: BaselineConfig,
) -> tuple[float, dict[str, Any]]:
    from e_jepa_ttc.geometry.cmax import maximize_radial_event_contrast

    if len(events) < config.cmax_minimum_events:
        raise RuntimeError(f"insufficient_roi_events:{len(events)}")
    interpolated = np.column_stack(
        [np.interp(events[:, 0], box_times_s, boxes[:, coordinate]) for coordinate in range(4)]
    )
    widths = interpolated[:, 2] - interpolated[:, 0]
    heights = interpolated[:, 3] - interpolated[:, 1]
    margin_x = np.maximum(widths * config.object_margin_fraction, 1.0)
    margin_y = np.maximum(heights * config.object_margin_fraction, 1.0)
    keep = (
        (events[:, 1] >= interpolated[:, 0] - margin_x)
        & (events[:, 1] <= interpolated[:, 2] + margin_x)
        & (events[:, 2] >= interpolated[:, 1] - margin_y)
        & (events[:, 2] <= interpolated[:, 3] + margin_y)
    )
    object_events = events[keep]
    centers = np.column_stack(
        (
            0.5 * (interpolated[:, 0] + interpolated[:, 2]),
            0.5 * (interpolated[:, 1] + interpolated[:, 3]),
        )
    )[keep]
    if len(object_events) < config.cmax_minimum_events:
        raise RuntimeError(f"insufficient_object_events:{len(object_events)}")
    if len(object_events) > config.cmax_maximum_events:
        indices = np.linspace(0, len(object_events) - 1, config.cmax_maximum_events, dtype=np.int64)
        object_events = object_events[indices]
        centers = centers[indices]
    current = boxes[-1]
    endpoint_center = (
        float(0.5 * (current[0] + current[2])),
        float(0.5 * (current[1] + current[3])),
    )
    result = maximize_radial_event_contrast(
        object_events[:, 1:3],
        object_events[:, 0],
        object_events[:, 3],
        image_shape=(roi[3] - roi[1], roi[2] - roi[0]),
        center_xy=endpoint_center,
        event_centers_xy=centers,
        minimum_ttc_s=config.cmax_minimum_ttc_s,
        maximum_ttc_s=config.cmax_maximum_ttc_s,
        coarse_steps=config.cmax_coarse_steps,
        minimum_events=config.cmax_minimum_events,
        minimum_relative_contrast_gain=config.cmax_minimum_relative_contrast_gain,
    )
    diagnostics = {
        "roi_event_count": int(len(events)),
        "object_event_count": int(keep.sum()),
        "optimized_event_count": int(len(object_events)),
        "reason": result.reason,
        "inverse_ttc_per_s": (
            result.inverse_ttc_per_s if np.isfinite(result.inverse_ttc_per_s) else None
        ),
        "contrast": result.contrast if np.isfinite(result.contrast) else None,
        "null_contrast": result.null_contrast if np.isfinite(result.null_contrast) else None,
        "relative_contrast_gain": (
            result.relative_contrast_gain
            if np.isfinite(result.relative_contrast_gain)
            else None
        ),
        "survival_fraction": result.survival_fraction,
        "confidence": result.confidence,
        "evaluations": result.evaluations,
    }
    if not result.valid:
        raise RuntimeError(f"cmax_invalid:{result.reason}")
    return float(result.ttc_seconds), diagnostics


def _predict_strttc(
    query_id: str,
    raw_path: Path,
    events: np.ndarray,
    roi: tuple[int, int, int, int],
    endpoint_time_s: float,
    config: BaselineConfig,
) -> tuple[float, dict[str, Any]]:
    from e_jepa_ttc.geometry.strttc import inverse_ttc_at_endpoint, refine_strttc_on_time_surface
    from e_jepa_ttc.geometry.strttc_frontend import STRTTCFrontendConfig, run_strttc_linear_frontend

    if len(events) < config.strttc_minimum_events:
        raise RuntimeError(f"insufficient_roi_events:{len(events)}")
    bounded = _uniform_bound(events, config.strttc_maximum_events)
    seed = int.from_bytes(hashlib.sha256(query_id.encode("utf-8")).digest()[:4], "little")
    frontend = STRTTCFrontendConfig(seed=seed)
    intrinsics = _event_intrinsics(raw_path, roi)
    result = run_strttc_linear_frontend(
        bounded,
        width=roi[2] - roi[0],
        height=roi[3] - roi[1],
        intrinsics=intrinsics,
        config=frontend,
    )
    parameters = refine_strttc_on_time_surface(
        result.linear.parameters,
        result.contour_txy,
        result.reference_time_s,
        result.nearest_linear_time_surface,
        intrinsics,
        maximum_function_evaluations=config.strttc_nonlinear_maximum_function_evaluations,
    )
    q_endpoint = inverse_ttc_at_endpoint(
        float(parameters[0]), endpoint_time_s - result.absolute_reference_time_s
    )
    if q_endpoint <= 0.0 or not np.isfinite(q_endpoint):
        raise RuntimeError("non_positive_or_non_finite_endpoint_inverse_ttc")
    ttc = float(1.0 / q_endpoint)
    if not np.isfinite(ttc):
        raise RuntimeError("non_finite_ttc")
    return ttc, {
        "roi_event_count": int(len(events)),
        "optimized_event_count": int(len(bounded)),
        "normal_flow_count": int(len(result.normal_flow_xy)),
        "linear_inlier_ratio": result.linear.inlier_ratio,
        "linear_inverse_ttc_at_reference": result.linear.inverse_ttc,
        "nonlinear_inverse_ttc_at_reference": float(parameters[0]),
        "inverse_ttc_at_endpoint": q_endpoint,
        "nonlinear_refinement": True,
    }


def _method_claim(method: str) -> dict[str, Any]:
    if method == "cmax":
        return {
            "label": "local_causal_radial_cmax_adaptation",
            "reference": CMAX_PAPER,
            "reproduction_of_paper_result": False,
            "differences": [
                "paper framework, but a custom radial looming warp for TTC",
                "causal observed-box ROI and interpolated observed-box centers",
                "fixed broad positive TTC search and deterministic event cap",
                "independent fixed EvTTC queries rather than a paper benchmark protocol",
            ],
        }
    return {
        "label": "local_causal_strttc_source_port_adaptation",
        "reference": OFFICIAL_STRTTC_REPOSITORY,
        "official_commit": OFFICIAL_STRTTC_COMMIT,
        "reproduction_of_paper_result": False,
        "differences": [
            "causal 200 ms observed-box ROI instead of the official FCWD epoch construction",
            "deterministic bounded RANSAC and contour budgets",
            "median plus Gaussian approximation; MATLAB bilateral filtering omitted",
            "independent queries without official lastOptimizedResult sequence fallback",
            "no TTC ground truth is opened or printed",
            "published nonlinear least-squares stage is enabled",
        ],
    }


def _fragment_path(output: Path, method: str, query_id: str) -> Path:
    token = hashlib.sha256(query_id.encode("utf-8")).hexdigest()[:16]
    return output / method / "fragments" / f"{token}.json"


def _run_fragment(
    *,
    row: Mapping[str, Any],
    method: str,
    config: BaselineConfig,
    output: Path,
    manifest_sha256: str,
    source_sha256: str,
) -> dict[str, Any]:
    path = _fragment_path(output, method, str(row["query_id"]))
    binding = {
        "schema": SCHEMA,
        "method": method,
        "query_id": str(row["query_id"]),
        "query_metadata_sha256": str(row["metadata_sha256"]),
        "query_manifest_sha256": manifest_sha256,
        "runner_source_sha256": source_sha256,
        "config": asdict(config),
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        _verify_signed(existing)
        for key, value in binding.items():
            if existing.get(key) != value:
                raise ValueError(f"resume fragment binding mismatch for {row['query_id']}: {key}")
        return existing
    started = time.perf_counter()
    prediction: float | None = None
    diagnostics: dict[str, Any] = {}
    failure: str | None = None
    try:
        events, roi, box_times, boxes = _load_query_events(row, config)
        if method == "cmax":
            prediction, diagnostics = _predict_cmax(events, roi, box_times, boxes, config)
        elif method == "strttc":
            prediction, diagnostics = _predict_strttc(
                str(row["query_id"]),
                Path(str(row["raw_path"])),
                events,
                roi,
                int(row["anchor_us"]) * 1e-6,
                config,
            )
        else:
            raise ValueError(f"unsupported method: {method}")
        diagnostics["roi_xyxy"] = list(roi)
    except (RuntimeError, ValueError, OSError, np.linalg.LinAlgError) as error:
        failure = f"{type(error).__name__}:{error}"
    elapsed = time.perf_counter() - started
    fragment = _signed(
        {
            **binding,
            "status": "SUCCESS" if failure is None else "FAILED_RETAINED",
            "sequence_id": str(row["sequence_id"]),
            "anchor_us": int(row["anchor_us"]),
            "prediction_ttc_s": prediction,
            "failure": failure,
            "diagnostics": diagnostics,
            "runtime_s": elapsed,
            "target_fields_opened": [],
            "scientific_optimizer_updates": 0,
            "claim": _method_claim(method),
        }
    )
    _atomic_json(path, fragment)
    return fragment


def run_baselines(
    *,
    query_manifest: str | Path,
    output_dir: str | Path,
    methods: Sequence[str] = METHODS,
    maximum_queries: int | None = None,
    selection: str = "all",
    config: BaselineConfig | None = None,
) -> dict[str, Any]:
    """Run or resume every requested method/query fragment."""

    resolved = config or BaselineConfig()
    resolved.validate()
    selected_methods = tuple(dict.fromkeys(str(method).lower() for method in methods))
    if not selected_methods or any(method not in METHODS for method in selected_methods):
        raise ValueError(f"methods must be a non-empty subset of {METHODS}")
    manifest_path = Path(query_manifest).resolve(strict=True)
    manifest, rows = _validate_manifest(manifest_path, maximum_queries, selection)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_sha = _file_sha256(manifest_path)
    source_sha = _file_sha256(__file__)
    fragments: list[dict[str, Any]] = []
    for row in rows:
        for method in selected_methods:
            fragments.append(
                _run_fragment(
                    row=row,
                    method=method,
                    config=resolved,
                    output=output,
                    manifest_sha256=manifest_sha,
                    source_sha256=source_sha,
                )
            )
    by_method: dict[str, Any] = {}
    for method in selected_methods:
        selected = [fragment for fragment in fragments if fragment["method"] == method]
        success = [fragment for fragment in selected if fragment["status"] == "SUCCESS"]
        by_method[method] = {
            "requested": len(selected),
            "successful": len(success),
            "failed_retained": len(selected) - len(success),
            "complete_fragment_coverage": len(selected) == len(rows),
            "success_fraction": len(success) / len(selected),
            "runtime_s": sum(float(fragment["runtime_s"]) for fragment in selected),
            "claim": _method_claim(method),
        }
    summary = _signed(
        {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "query_manifest": manifest_path.as_posix(),
            "query_manifest_sha256": manifest_sha,
            "query_manifest_protocol": manifest.get("protocol"),
            "query_count": len(rows),
            "query_selection": selection,
            "methods": list(selected_methods),
            "config": asdict(resolved),
            "runner_source_sha256": source_sha,
            "results": by_method,
            "fragment_sha256": {
                f"{fragment['method']}:{fragment['query_id']}": fragment["artifact_sha256"]
                for fragment in fragments
            },
            "prediction_inputs": ["raw events", "camera intrinsics", "causal observed boxes"],
            "target_fields_opened": [],
            "success_only_metrics_computed": False,
            "scientific_optimizer_updates": 0,
            "claim_boundary": (
                "Fixed historical development queries with oracle observed boxes. Failures are "
                "retained. These are local adapted baselines, not paper-result reproductions or "
                "official blind-benchmark results."
            ),
        }
    )
    _atomic_json(output / "SUMMARY.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--maximum-queries", type=int)
    parser.add_argument("--selection", choices=("all", "first_per_sequence"), default="all")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_baselines(
        query_manifest=args.queries,
        output_dir=args.output_dir,
        methods=args.methods,
        maximum_queries=args.maximum_queries,
        selection=args.selection,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["BaselineConfig", "main", "run_baselines"]
