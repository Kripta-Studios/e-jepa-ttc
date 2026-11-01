"""Check observed RGB brightness changes against the already-sealed event count cache."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import h5py
import numpy as np
import psutil

from operational.evttc_transfer.inputs import _calibration, _calibration_sha256
from operational.release_research.protocol import ResearchRun, write_json
from scripts.release_observability import check_stat
from scripts.release_research import read, sources, verify_complete
from scripts.release_sensor_proxy import project_event_points, sample_bilinear


def observed_change(
    reference: np.ndarray,
    endpoint: np.ndarray,
    calibration: dict,
    xedges: np.ndarray,
    yedges: np.ndarray,
    duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure both reference coverage and a common endpoint mask in sensor coordinates."""
    if not np.isfinite(duration) or duration <= 0 or reference.shape != endpoint.shape:
        raise ValueError("Paired frames and a positive observed duration are required")
    gray = [
        a.astype(float).mean(axis=2) if a.ndim == 3 else a.astype(float)
        for a in (reference, endpoint)
    ]
    patches = []
    for y0, y1 in zip(yedges[:-1], yedges[1:], strict=True):
        for x0, x1 in zip(xedges[:-1], xedges[1:], strict=True):
            xx, yy = np.meshgrid(np.linspace(x0, x1 - 1, 5), np.linspace(y0, y1 - 1, 5))
            patches.append(np.column_stack((xx.ravel(), yy.ravel())))
    points = np.concatenate(patches)
    offsets = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
    mapped, rays = project_event_points(
        (points[None] + offsets[:, None]).reshape(-1, 2), calibration
    )
    initial, initial_valid = sample_bilinear(gray[0], mapped)
    initial = initial.reshape(5, -1)
    valid = (rays & initial_valid).reshape(5, -1).all(axis=0)
    valid &= ((initial >= 5) & (initial <= 250)).all(axis=0)
    reference_coverage = valid.reshape(-1, 25).mean(axis=1)
    final, final_valid = sample_bilinear(gray[1], mapped[: len(points)])
    valid &= final_valid & (final >= 5) & (final <= 250)
    rates = (np.log(np.maximum(final, 1)) - np.log(np.maximum(initial[0], 1))) / duration
    valid = valid.reshape(-1, 25)
    mean_rate = np.where(valid, rates.reshape(-1, 25), 0).sum(axis=1) / np.maximum(
        valid.sum(axis=1), 1
    )
    return mean_rate, reference_coverage, valid.mean(axis=1)


def extract(capture: Path, expanded: Path, output: Path) -> None:
    """Read RGB only; preserve the original signed-count targets and physical states."""
    for folder in (capture, expanded):
        verify_complete(folder)
    records = read(capture / "FEATURES.json")["records"]
    if len(records) != 149 or read(capture / "FREEZE.json")["config"]["profiling_only"]:
        raise ValueError("Require the full original sensor proxy cache")
    inputs = {
        **sources(),
        "source:scripts/release_sensor_reference.py": Path(__file__),
        "source:scripts/release_sensor_proxy.py": Path("scripts/release_sensor_proxy.py"),
        "source:scripts/release_observability.py": Path("scripts/release_observability.py"),
        "source:operational/evttc_transfer/inputs.py": Path("operational/evttc_transfer/inputs.py"),
        "source:src/e_jepa_ttc/data/evttc_object_cache.py": Path(
            "src/e_jepa_ttc/data/evttc_object_cache.py"
        ),
        "capture": capture / "COMPLETE.json",
        "features": capture / "FEATURES.json",
        "expanded": expanded / "COMPLETE.json",
        "preregistration": Path(
            "docs/release_research_implementation_20261031/SENSOR_REFERENCE_CONTROL.md"
        ),
    }
    for row in records:
        if row["status"] == "measured":
            identity = hashlib.sha256(row["token"].encode()).hexdigest()[:24]
            inputs["source_query:" + identity] = expanded / "queries" / identity / "QUERY.json"
            inputs["original_arrays:" + identity] = capture / row["arrays"]
    run = ResearchRun(
        output,
        inputs,
        {
            "experiment_id": "S1-reference-extract-v1",
            "targets_read": False,
            "profiling_only": False,
            "optimizer_updates": 0,
            "scope": "observed RGB reconstruction control; not TTC or calibrated sensor",
        },
    )
    began = time.perf_counter()
    try:
        (output / "queries").mkdir()
        result = []
        for number, original in enumerate(records):
            row = dict(original)
            if row["status"] == "measured":
                identity = hashlib.sha256(row["token"].encode()).hexdigest()[:24]
                source = read(expanded / "queries" / identity / "QUERY.json")
                path = Path(source["raw_path"])
                check_stat(path, source["raw_stat"])
                with h5py.File(path, "r") as handle:
                    calibration = _calibration(handle)
                    if _calibration_sha256(calibration) != source["calibration_sha256"]:
                        raise ValueError("Calibration changed")
                    times = np.asarray(handle["blackflys/left/ts"])
                    first = int(np.searchsorted(times, row["reference_us"]))
                    last = int(np.searchsorted(times, row["query_us"], side="right")) - 1
                    if first >= len(times) or int(times[first]) != row["reference_us"]:
                        raise ValueError("Reference timestamp changed")
                    data = handle["blackflys/left/data"]
                    if not isinstance(data, h5py.Dataset):
                        raise ValueError("RGB frames must be a dataset")
                    frames = [np.asarray(data[i]) for i in (first, last)]
                if (
                    hashlib.sha256(np.ascontiguousarray(frames[0]).tobytes()).hexdigest()
                    != row["reference_sha256"]
                ):
                    raise ValueError("Reference RGB bytes changed")
                if last <= first:
                    row["status"] = "no_later_causal_rgb_endpoint"
                    row["admitted_cells"] = 0
                    row.pop("arrays")
                else:
                    roi = np.array(row["roi"])
                    left, top = max(0, int(np.ceil(roi[0]))), max(0, int(np.ceil(roi[1])))
                    right, bottom = min(1280, int(np.ceil(roi[2]))), min(720, int(np.ceil(roi[3])))
                    xedges, yedges = (
                        np.linspace(left, right, 17).astype(int),
                        np.linspace(top, bottom, 17).astype(int),
                    )
                    rate, original_coverage, joint_coverage = observed_change(
                        frames[0],
                        frames[1],
                        calibration,
                        xedges,
                        yedges,
                        (int(times[last]) - int(times[first])) * 1e-6,
                    )
                    with np.load(capture / original["arrays"], allow_pickle=False) as archive:
                        arrays = {k: archive[k] for k in archive.files}
                    np.testing.assert_array_equal(original_coverage, arrays["coverage"])
                    arrays["valid"] &= joint_coverage >= 0.8
                    arrays["rgb_observed"] = rate
                    arrays["joint_coverage"] = joint_coverage
                    row.update(
                        admitted_cells=int(arrays["valid"].sum()),
                        original_admitted_cells=original["admitted_cells"],
                        rgb_endpoint_us=int(times[last]),
                        rgb_start_offset_us=int(times[first]) - row["windows_us"][-1][0],
                        rgb_end_offset_us=int(times[last]) - row["windows_us"][-1][1],
                        endpoint_sha256=hashlib.sha256(
                            np.ascontiguousarray(frames[1]).tobytes()
                        ).hexdigest(),
                    )
                    np.savez_compressed(output / row["arrays"], allow_pickle=False, **arrays)
                check_stat(path, source["raw_stat"])
                print(f"{number + 1}/149 {row['group']} {row['status']}", flush=True)
            result.append(row)
        write_json(output / "FEATURES.json", {"records": result, "targets_read": False})
        write_json(
            output / "RUNTIME.json",
            {
                "elapsed_seconds": time.perf_counter() - began,
                "memory_info_bytes": psutil.Process().memory_info()._asdict(),
                "scope": "CPU RGB-only reference control",
            },
        )
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    """Create the cache for the registered observed-RGB sensor control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture", type=Path, default=Path("output/release_research_20261031/sensor_proxy149_v1")
    )
    parser.add_argument(
        "--expanded", type=Path, default=Path("output/phase_observation_cache/expanded149")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract(args.capture, args.expanded, args.output)


if __name__ == "__main__":
    main()
