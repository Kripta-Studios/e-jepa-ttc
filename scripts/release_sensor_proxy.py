"""Exploratory S1 reconstruction from available RGB geometry and signed RAW events."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import h5py
import numpy as np
import psutil

from e_jepa_ttc.data.evttc_object_cache import _distort_radtan, _undistort_radtan
from operational.evttc_transfer.inputs import EvTTCEventReader, _calibration, _calibration_sha256
from operational.release_research.physics import raw_phase_to_q
from operational.release_research.protocol import ResearchRun, write_json
from scripts.release_observability import check_stat
from scripts.release_research import read, sources, verify_complete


def project_event_points(points: np.ndarray, calibration: dict) -> tuple[np.ndarray, np.ndarray]:
    """Use inverse calibrated rotation; translation/depth remain an explicit approximation."""
    event, rgb = calibration["event_intrinsics"], calibration["rgb_intrinsics"]
    normalized = np.column_stack(
        ((points[:, 0] - event[2]) / event[0], (points[:, 1] - event[3]) / event[1])
    )
    normalized = _undistort_radtan(normalized, calibration["event_distortion"])
    rays = np.column_stack((normalized, np.ones(len(points))))
    rotated = rays @ calibration["event_from_rgb"][:3, :3]
    valid = rotated[:, 2] > 0
    planar = rotated[:, :2] / np.where(valid, rotated[:, 2], 1)[:, None]
    distorted = _distort_radtan(planar, calibration["rgb_distortion"])
    return np.column_stack(
        (distorted[:, 0] * rgb[0] + rgb[2], distorted[:, 1] * rgb[1] + rgb[3])
    ), valid


def sample_bilinear(frame: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate luminance, retaining an explicit in-image mask."""
    height, width = frame.shape
    finite = np.isfinite(points).all(axis=1)
    safe = np.nan_to_num(points, nan=-1.0, posinf=-1.0, neginf=-1.0)
    valid = (
        finite
        & (safe[:, 0] >= 0)
        & (safe[:, 1] >= 0)
        & (safe[:, 0] < width - 1)
        & (safe[:, 1] < height - 1)
    )
    x, y = np.clip(safe[:, 0], 0, width - 2), np.clip(safe[:, 1], 0, height - 2)
    ix, iy = x.astype(int), y.astype(int)
    fx, fy = x - ix, y - iy
    values = (
        (1 - fx) * (1 - fy) * frame[iy, ix]
        + fx * (1 - fy) * frame[iy, ix + 1]
        + (1 - fx) * fy * frame[iy + 1, ix]
        + fx * fy * frame[iy + 1, ix + 1]
    )
    return values, valid


def texture_motion(
    frame: np.ndarray, calibration: dict, xedges: np.ndarray, yedges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Average radial log-luminance gradients in fixed cells with a reference-only mask."""
    gray = frame.astype(float).mean(axis=2) if frame.ndim == 3 else frame.astype(float)
    patches = []
    for y0, y1 in zip(yedges[:-1], yedges[1:], strict=True):
        for x0, x1 in zip(xedges[:-1], xedges[1:], strict=True):
            xx, yy = np.meshgrid(np.linspace(x0, x1 - 1, 5), np.linspace(y0, y1 - 1, 5))
            patches.append(np.column_stack((xx.ravel(), yy.ravel())))
    points = np.concatenate(patches)
    offsets = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
    mapped, ray_valid = project_event_points(
        (points[None] + offsets[:, None]).reshape(-1, 2), calibration
    )
    sampled, image_valid = sample_bilinear(gray, mapped)
    sampled = sampled.reshape(5, -1)
    valid = (ray_valid & image_valid).reshape(5, -1).all(axis=0)
    valid &= ((sampled >= 5) & (sampled <= 250)).all(axis=0)
    logs = np.log(np.maximum(sampled, 1) / 255)
    gradient = np.column_stack(((logs[1] - logs[2]) / 2, (logs[3] - logs[4]) / 2))
    center = np.array([(xedges[0] + xedges[-1] - 1) / 2, (yedges[0] + yedges[-1] - 1) / 2])
    radial = -(gradient * (points - center)).sum(axis=1).reshape(-1, 25)
    valid = valid.reshape(-1, 25)
    coverage = valid.mean(axis=1)
    average = np.where(valid, radial, 0).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    return average, coverage


def count_cells(
    events: np.ndarray, xedges: np.ndarray, yedges: np.ndarray, start_us: int, end_us: int
) -> tuple[np.ndarray, np.ndarray]:
    """Count both polarities without dropping time boundaries or changing spatial units."""
    if end_us <= start_us or events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("Invalid event interval")
    if len(events) and (
        np.any(np.diff(events[:, 2]) < 0)
        or events[:, 2].min() < start_us
        or events[:, 2].max() >= end_us
        or not np.isin(events[:, 3], [0, 1]).all()
    ):
        raise ValueError("Noncausal or noncanonical events")
    counts = []
    for polarity in (0, 1):
        selected = events[events[:, 3] == polarity]
        # Histogram includes its right edge; explicitly enforce the half-open ROI.
        selected = selected[(selected[:, 0] < xedges[-1]) & (selected[:, 1] < yedges[-1])]
        counts.append(
            np.histogram2d(selected[:, 1], selected[:, 0], bins=(yedges, xedges))[0].ravel()
        )
    return counts[1], counts[0]


def fit_nonnegative_line(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    """Weighted two-parameter calibration; fit scale only on the training acquisitions."""
    if x.shape != y.shape or x.shape != weight.shape or not len(x):
        raise ValueError("Mismatched or empty calibration data")
    if not all(np.isfinite(v).all() for v in (x, y, weight)) or np.any(weight <= 0):
        raise ValueError("Invalid calibration observations/weights")
    weight = weight / weight.sum()
    mx, my = float(weight @ x), float(weight @ y)
    variance = float(weight @ ((x - mx) ** 2))
    slope = max(0.0, float(weight @ ((x - mx) * (y - my))) / variance) if variance > 1e-20 else 0.0
    return slope, my - slope * mx


def extract(
    screen: Path,
    expanded: Path,
    predictions: Path,
    output: Path,
    profile_queries: int | None = None,
) -> None:
    """Build a target-free reconstruction cache on the exact existing screen inventory."""
    for folder in (screen, expanded, predictions):
        verify_complete(folder)
    records = read(screen / "PREDICTIONS.json")["records"]
    r0 = {r["token"]: r for r in read(predictions / "PREDICTIONS.json")["records"]}
    if len(records) != 149 or set(r0) != {r["token"] for r in records}:
        raise ValueError("Expected frozen 149-query cohort")
    if profile_queries is not None:
        if not 1 <= profile_queries <= 3:
            raise ValueError("CPU profiling is limited to three queries")
        records = records[:profile_queries]
    inputs = {
        **sources(),
        "source:scripts/release_sensor_proxy.py": Path(__file__),
        "screen": screen / "COMPLETE.json",
        "expanded": expanded / "COMPLETE.json",
        "predictions": predictions / "COMPLETE.json",
        "states": predictions / "PREDICTIONS.json",
        "preregistration": Path(
            "docs/release_research_implementation_20261031/SENSOR_PROXY_PREREGISTRATION.md"
        ),
        "resource_precheck": Path(
            "docs/release_research_implementation_20261031/evidence/SENSOR_PROXY_RESOURCES_BEFORE.json"
        ),
    }
    for name in (
        "operational/evttc_transfer/inputs.py",
        "src/e_jepa_ttc/data/evttc_object_cache.py",
        "scripts/release_observability.py",
    ):
        inputs["source:" + name] = Path(name)
    for row in records:
        identity = hashlib.sha256(row["token"].encode()).hexdigest()[:24]
        inputs[identity + ":calls"] = screen / "queries" / identity / "CALLS.json"
        inputs[identity + ":source"] = expanded / "queries" / identity / "QUERY.json"
    run = ResearchRun(
        output,
        inputs,
        {
            "experiment_id": "S1-RGB-proxy-extract-v1",
            "targets_read": False,
            "optimizer_updates": 0,
            "queries": len(records),
            "profiling_only": profile_queries is not None,
            "exposure": "development",
            "scope": "approximate RGB/ray-alignment proxy; no physical calibration",
            "thread_limits": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
        },
    )
    rows = []
    started = time.perf_counter()
    try:
        (output / "queries").mkdir()
        for number, row in enumerate(records):
            result = {k: row[k] for k in ("token", "group", "query_us")}
            result["status"] = "slider_alignment_not_admitted"
            identity = hashlib.sha256(row["token"].encode()).hexdigest()[:24]
            if row["group"] != "Slider-500":
                calls = read(screen / "queries" / identity / "CALLS.json")["calls"]
                items = [item for c in calls if c["prefix"] == "E" for item in c["observations"]]
                if not items or r0[row["token"]]["raw_r0"] is None:
                    result["status"] = "no_event_or_R0_state"
                else:
                    item = max(items, key=lambda value: value["anchor_us"])
                    windows = item["windows_us"][-2:]
                    if (
                        len(windows) != 2
                        or windows[0][0] < row["query_us"] - 650000
                        or windows[-1][1] > row["query_us"]
                    ):
                        raise ValueError("Event history not admissible")
                    source = read(expanded / "queries" / identity / "QUERY.json")
                    path = Path(source["raw_path"])
                    check_stat(path, source["raw_stat"])
                    with h5py.File(path, "r") as handle:
                        calibration = _calibration(handle)
                        if _calibration_sha256(calibration) != source["calibration_sha256"]:
                            raise ValueError("Camera calibration changed")
                        times = np.asarray(handle["blackflys/left/ts"])
                        index = int(np.searchsorted(times, windows[-1][0], side="right")) - 1
                        rgb_data = handle["blackflys/left/data"]
                        if not isinstance(rgb_data, h5py.Dataset):
                            raise ValueError("RGB frames must be a dataset")
                        frame = np.asarray(rgb_data[index]) if index >= 0 else None
                    result.update(
                        windows_us=windows,
                        reference_us=int(times[index]) if index >= 0 else None,
                        translation_ignored=calibration["event_from_rgb"][:3, 3].tolist(),
                    )
                    if frame is None or int(times[index]) < row["query_us"] - 650000:
                        result["status"] = "no_causal_reference_within_budget"
                    else:
                        raw_q = raw_phase_to_q(r0[row["token"]]["raw_r0"])
                        denominator = 1 + (row["query_us"] - windows[-1][0]) * 1e-6 * raw_q
                        roi = np.array(source["square_xyxy"], float)
                        left, top = max(0, int(np.ceil(roi[0]))), max(0, int(np.ceil(roi[1])))
                        right, bottom = (
                            min(1280, int(np.ceil(roi[2]))),
                            min(720, int(np.ceil(roi[3]))),
                        )
                        if denominator <= 1e-12 or right - left < 16 or bottom - top < 16:
                            result["status"] = "invalid_transport_or_grid"
                        else:
                            q = raw_q / denominator
                            xedges, yedges = (
                                np.linspace(left, right, 17).astype(int),
                                np.linspace(top, bottom, 17).astype(int),
                            )
                            motion, coverage = texture_motion(frame, calibration, xedges, yedges)
                            area = (np.diff(yedges)[:, None] * np.diff(xedges)[None, :]).ravel()
                            rates, totals, hashes = [], [], []
                            with EvTTCEventReader(path) as reader:
                                for a, b in windows:
                                    raw = reader.read_window(int(a), int(b))
                                    events = np.column_stack(
                                        [raw[k] for k in ("x", "y", "t", "p")]
                                    ).astype(np.int64)
                                    positive, negative = count_cells(events, xedges, yedges, a, b)
                                    rates.append((positive - negative) / area / ((b - a) * 1e-6))
                                    totals.append(positive + negative)
                                    hashes.append(
                                        hashlib.sha256(
                                            np.ascontiguousarray(events, dtype="<i8").tobytes()
                                        ).hexdigest()
                                    )
                            mask = (coverage >= 0.8) & (totals[-1] > 0)
                            result.update(
                                status="measured",
                                q_at_window_start=q,
                                q_at_query=raw_q,
                                cells=256,
                                admitted_cells=int(mask.sum()),
                                silent_cells=int((totals[-1] == 0).sum()),
                                mapped_cells=int((coverage >= 0.8).sum()),
                                roi=roi.tolist(),
                                event_sha256=hashes,
                                reference_sha256=hashlib.sha256(
                                    np.ascontiguousarray(frame).tobytes()
                                ).hexdigest(),
                                arrays="queries/" + identity + ".npz",
                            )
                            np.savez_compressed(
                                output / result["arrays"],
                                target_rate=rates[-1],
                                previous_rate=rates[0],
                                texture_motion=motion,
                                valid=mask,
                                coverage=coverage,
                                cell_area=area,
                                event_count=totals[-1],
                                allow_pickle=False,
                            )
                    check_stat(path, source["raw_stat"])
            rows.append(result)
            print(f"{number + 1}/{len(records)} {row['group']} {result['status']}", flush=True)
        write_json(output / "FEATURES.json", {"records": rows, "targets_read": False})
        write_json(
            output / "RUNTIME.json",
            {
                "elapsed_seconds": time.perf_counter() - started,
                "scope": "CPU extraction only",
                "memory_info_bytes": psutil.Process().memory_info()._asdict(),
            },
        )
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def analyze(capture: Path, output: Path, extra_control: str | None = None) -> None:
    """Fit equal-size nuisance models with acquisition exclusion and report sensor residuals."""
    verify_complete(capture)
    records = read(capture / "FEATURES.json")["records"]
    config = read(capture / "FREEZE.json")["config"]
    if extra_control not in (None, "rgb_observed"):
        raise ValueError("Only the registered observed-RGB control is permitted")
    expected = "S1-RGB-proxy-extract-v1" if extra_control is None else "S1-reference-extract-v1"
    if (
        config.get("experiment_id") != expected
        or config.get("targets_read") is not False
        or config.get("profiling_only", False)
        or len(records) != 149
    ):
        raise ValueError("Expected complete target-free S1 proxy extraction")
    inputs = {
        **sources(),
        "source:scripts/release_sensor_proxy.py": Path(__file__),
        "capture": capture / "COMPLETE.json",
        "features": capture / "FEATURES.json",
        "preregistration": Path(
            "docs/release_research_implementation_20261031/SENSOR_PROXY_PREREGISTRATION.md"
        ),
    }
    for row in records:
        if "arrays" in row:
            inputs["arrays:" + row["token"]] = capture / row["arrays"]
    if extra_control is not None:
        inputs["reference_control_plan"] = Path(
            "docs/release_research_implementation_20261031/SENSOR_REFERENCE_CONTROL.md"
        )
    run = ResearchRun(
        output,
        inputs,
        {
            "experiment_id": "S1-RGB-proxy-analysis-v1",
            "targets_read": False,
            "sensor_counts_used_for_fitting": True,
            "optimizer_updates": 0,
            "exposure": "development; R0 parents exposed",
            "bootstrap_seed": 7,
            "extra_control": extra_control,
        },
    )
    try:
        retained = [r for r in records if r["status"] == "measured" and r["admitted_cells"] > 0]
        groups = sorted({r["group"] for r in retained})
        if len(groups) < 3:
            raise ValueError("Too few acquisitions for excluded-acquisition calibration")
        data = {}
        permutations = []
        for group in groups:
            ordered = sorted(
                [r for r in retained if r["group"] == group], key=lambda r: r["query_us"]
            )
            for index, row in enumerate(ordered):
                donor = ordered[(index - 1) % len(ordered)]
                permutations.append(
                    {
                        "token": row["token"],
                        "donor": donor["token"],
                        "q_changed": row["q_at_window_start"] != donor["q_at_window_start"],
                    }
                )
                with np.load(capture / row["arrays"], allow_pickle=False) as archive:
                    valid = archive["valid"]
                    texture = archive["texture_motion"][valid]
                    data[row["token"]] = {
                        "target": archive["target_rate"][valid],
                        "activity": archive["previous_rate"][valid],
                        "texture": texture,
                        "physical": texture * row["q_at_window_start"],
                        "permuted": texture * donor["q_at_window_start"],
                    }
                    if extra_control is not None:
                        data[row["token"]][extra_control] = archive[extra_control][valid]
        arms = ("activity", "texture", "physical", "permuted") + (
            (extra_control,) if extra_control is not None else ()
        )
        fits, scored = [], []
        for heldout in groups:
            training = [r for r in retained if r["group"] != heldout]
            for arm in arms:
                x, y, weights = [], [], []
                for row in training:
                    part = data[row["token"]]
                    count = sum(r["group"] == row["group"] for r in training)
                    x.extend(part[arm])
                    y.extend(part["target"])
                    weights.extend([1 / count / len(part["target"])] * len(part["target"]))
                slope, intercept = fit_nonnegative_line(np.array(x), np.array(y), np.array(weights))
                fits.append(
                    {
                        "held_out_group": heldout,
                        "arm": arm,
                        "slope": slope,
                        "intercept": intercept,
                        "training_groups": sorted({r["group"] for r in training}),
                    }
                )
                for row in retained:
                    if row["group"] != heldout:
                        continue
                    part = data[row["token"]]
                    residual = slope * part[arm] + intercept - part["target"]
                    scored.append(
                        {
                            "token": row["token"],
                            "group": heldout,
                            "arm": arm,
                            "mse": float(np.mean(residual**2)),
                            "mae": float(np.mean(np.abs(residual))),
                        }
                    )
        summary = {}
        for arm in arms:
            by_group = {
                group: {
                    metric: float(
                        np.mean(
                            [r[metric] for r in scored if r["arm"] == arm and r["group"] == group]
                        )
                    )
                    for metric in ("mse", "mae")
                }
                for group in groups
            }
            summary[arm] = {
                "macro_mse": float(np.mean([v["mse"] for v in by_group.values()])),
                "macro_mae": float(np.mean([v["mae"] for v in by_group.values()])),
                "by_acquisition": by_group,
            }
        indices = np.random.default_rng(7).integers(0, len(groups), size=(2000, len(groups)))
        differences = {}
        for arm in (a for a in arms if a != "physical"):
            delta = np.array(
                [
                    summary["physical"]["by_acquisition"][g]["mse"]
                    - summary[arm]["by_acquisition"][g]["mse"]
                    for g in groups
                ]
            )
            differences[arm] = {
                "mean": float(delta.mean()),
                "ci95": np.quantile(delta[indices].mean(axis=1), [0.025, 0.975]).tolist(),
            }
        metrics = {
            "population": len(records),
            "queries_with_cells": len(retained),
            "acquisitions": len(groups),
            "admitted_cells": sum(r["admitted_cells"] for r in retained),
            "arms": summary,
            "physical_minus_control_mse": differences,
            "fits": len(fits),
            "permutations_changing_q": sum(p["q_changed"] for p in permutations),
            "S2_admitted": False,
            "reason": "exploratory response/alignment proxy; "
            "residual is not TTC or identified sensor calibration",
        }
        write_json(output / "METRICS.json", metrics)
        write_json(output / "FITS.json", fits)
        write_json(output / "SCORED.json", scored)
        write_json(output / "PERMUTATIONS.json", permutations)
        lines = [
            "# S1 exploratorio con referencia RGB aproximada",
            "",
            f"{len(retained)}/149 consultas con celdas admitidas; "
            f"{len(groups)} adquisiciones; {len(fits)} ajustes.",
            "Sin targets TTC. R0 tiene exposición histórica; la calibración se ajusta "
            "excluyendo la adquisición evaluada.",
            "",
            "| Control | MSE macro de tasa firmada | MAE macro de tasa firmada |",
            "|---|---:|---:|",
        ]
        lines += [
            f"| {arm} | {value['macro_mse']:.6f} | {value['macro_mae']:.6f} |"
            for arm, value in summary.items()
        ]
        lines += [
            "",
            "| Físico menos control | Delta MSE | CI95 por adquisición |",
            "|---|---:|---|",
        ]
        lines += [
            f"| {arm} | {value['mean']:.6f} | {value['ci95']} |"
            for arm, value in differences.items()
        ]
        lines += [
            "",
            "La escala ajustada no equivale al umbral físico del sensor.",
            "RGB/ISP, umbrales asimétricos, paralaje y textura de referencia "
            "pueden dominar el residuo.",
            "Slider se conserva como no admitido por alineación provisional; "
            "ninguna consulta desaparece del inventario.",
            "Las celdas silenciosas no imponen q=0 ni una pérdida de silencio.",
            "Este resultado no mide mejora TTC ni abre S2 automáticamente.",
        ]
        (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    """Extract first; calibrate only from a complete sealed cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("extract", "analyze"))
    parser.add_argument("--screen", type=Path, default=Path("output/phase_timewarp/screen149_v1"))
    parser.add_argument(
        "--expanded", type=Path, default=Path("output/phase_observation_cache/expanded149")
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("output/release_research_20261031/diagnostics149_v4"),
    )
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--profile-queries", type=int)
    parser.add_argument("--extra-control", choices=("rgb_observed",))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "extract":
        extract(args.screen, args.expanded, args.predictions, args.output, args.profile_queries)
    else:
        if args.capture is None:
            parser.error("analyze requires --capture")
        analyze(args.capture, args.output, args.extra_control)


if __name__ == "__main__":
    main()
