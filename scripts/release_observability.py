"""Extract label-blind RAW observability in the exact frozen screen149 query geometry."""

from __future__ import annotations

import argparse
import hashlib
import time
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np

from operational.evttc_competition.readers import SensorSequence
from operational.evttc_transfer.inputs import (
    EvTTCEventReader,
    _calibration,
    _calibration_sha256,
)
from operational.release_research.normal_quality import (
    PHYSICAL_FIELDS,
    SUPPORT_FIELDS,
    fit_surface,
    make_surface,
    physical_vector,
)
from operational.release_research.protocol import ResearchRun, sha256, write_json
from scripts.release_research import read, sources, verify_complete


def source_paths() -> dict[str, Path]:
    """Bind the original readers and plane fitting implementations, without changing them."""
    result = sources()
    for name in (
        "scripts/release_observability.py",
        "operational/evttc_transfer/inputs.py",
        "operational/evttc_competition/readers.py",
        "src/e_jepa_ttc/data/evttc.py",
        "src/e_jepa_ttc/models/normal_flow_routing_v10.py",
    ):
        result["source:" + name] = Path(name)
    return result


def check_stat(path: Path, expected: dict) -> None:
    """Retain the original road file binding; extracted event bytes get a new digest."""
    stat = path.stat()
    if stat.st_size != expected["size_bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
        raise ValueError(f"RAW stat differs from frozen release query: {path.name}")


def extract(
    screen: Path, expanded: Path, slider_root: Path, output: Path, limit: int | None = None
) -> None:
    """One bounded CPU extraction; does not run producers, update weights or access TTC."""
    verified = verify_complete(screen)
    verify_complete(expanded)
    freeze = read(screen / "FREEZE.json")
    if freeze["schema"] != "phase_canonical_screen149_v1" or not verified["baseline_exact"]:
        raise ValueError("Require the exact baseline-verified 149-query screen")
    records = read(screen / "PREDICTIONS.json")["records"]
    if len(records) != 149 or [r["token"] for r in records] != freeze["tokens"]:
        raise ValueError("Unexpected screen population")
    if limit is not None:
        if limit < 1 or limit > len(records):
            raise ValueError("Invalid profile limit")
        records = records[:limit]
    inputs = {
        **source_paths(),
        "parent_complete": screen / "COMPLETE.json",
        "expanded_complete": expanded / "COMPLETE.json",
        "preregistration": Path(
            "docs/release_research_implementation_20261031/OBSERVABILITY_PREREGISTRATION.md"
        ),
    }
    for record in records:
        identity = hashlib.sha256(record["token"].encode()).hexdigest()[:24]
        for filename in ("CALLS.json", "PAIR_INPUTS.npz", "EXTRACTION.json"):
            inputs[identity + ":" + filename] = screen / "queries" / identity / filename
        inputs[identity + ":QUERY.json"] = expanded / "queries" / identity / "QUERY.json"
    sensor_receipt = Path("output/phase_campaign/Slider500/SENSORS.json")
    sensors = read(sensor_receipt)
    inputs["slider_sensor_receipt"] = sensor_receipt
    if any(r["group"] == "Slider-500" for r in records):
        for name, spec in sensors["files"].items():
            path = slider_root / "Slider-500" / name
            if sha256(path) != spec["sha256"]:
                raise ValueError("Slider sensor differs from original audited source")
            inputs["slider:" + name] = path
    run = ResearchRun(
        output,
        inputs,
        {
            "experiment_id": "O0-RAW-release-roi-v1",
            "exposure": "development",
            "targets_read": False,
            "optimizer_updates": 0,
            "seed": 7,
            "queries": len(records),
            "profiling_only": limit is not None,
            "physical_fields": list(PHYSICAL_FIELDS),
            "support_fields": list(SUPPORT_FIELDS),
            "roi": "exact existing common release square, unresized sensor raster",
            "windows": "last two actual windows of last requested event observation",
            "plane_policy": "16x16 centers, separate polarities, 9x9 patches, latest 10ms",
            "normalization": "one isotropic scale for position, gradients and normal speed",
            "road_identity": "original stat/calibration plus SHA256 of consumed event windows",
            "new_raw_full_file_hash": False,
            "missing": "retain query, no new prediction or uncertainty calibration",
        },
    )
    rows = []
    began = time.perf_counter()
    try:
        with ExitStack() as stack:
            slider = None
            for number, record in enumerate(records):
                started = time.perf_counter()
                identity = hashlib.sha256(record["token"].encode()).hexdigest()[:24]
                folder = screen / "queries" / identity
                row = read(expanded / "queries" / identity / "QUERY.json")
                extraction = read(folder / "EXTRACTION.json")
                calls = [c for c in read(folder / "CALLS.json")["calls"] if c["prefix"] == "E"]
                candidates = [(call, item) for call in calls for item in call["observations"]]
                result = {k: record[k] for k in ("token", "group", "query_us")}
                result.update(
                    physical=[None] * 6, support=[None] * 6, status="no_event_observation"
                )
                if candidates:
                    call, item = max(candidates, key=lambda x: x[1]["anchor_us"])
                    windows = item["windows_us"][-2:]
                    if (
                        windows[-1][1] > record["query_us"]
                        or windows[0][0] < record["query_us"] - 650000
                    ):
                        raise ValueError("Causal input span differs")
                    if record["group"] == "Slider-500":
                        if slider is None:
                            slider = stack.enter_context(
                                SensorSequence(
                                    slider_root, "Slider-500", slider_profile="event_aided_ttc"
                                )
                            )
                        roi = np.asarray(extraction["preparation"]["event_square"], np.float64)
                        box = np.asarray(
                            extraction["preparation"]["bbox_sources"][-1]["event_box"], np.float64
                        )
                        resolution = (slider.event_width, slider.event_height)
                        raw_windows = [slider.read_events(int(a), int(b)) for a, b in windows]
                        raw_path = slider_root / "Slider-500/slider_500.bag"
                        camera = "event_aided_ttc:/dvs/events"
                    else:
                        raw_path = Path(row["raw_path"])
                        check_stat(raw_path, row["raw_stat"])
                        with h5py.File(raw_path, "r") as handle:
                            if (
                                _calibration_sha256(_calibration(handle))
                                != row["calibration_sha256"]
                            ):
                                raise ValueError("Road calibration differs")
                        roi = np.asarray(row["square_xyxy"], np.float64)
                        box = np.asarray(row["boxes_xyxy3"][-1], np.float64)
                        resolution = (1280, 720)
                        raw_windows = []
                        with EvTTCEventReader(raw_path) as reader:
                            for a, b in windows:
                                raw = reader.read_window(int(a), int(b))
                                raw_windows.append(
                                    np.column_stack([raw[k] for k in ("x", "y", "t", "p")]).astype(
                                        np.int64
                                    )
                                )
                        check_stat(raw_path, row["raw_stat"])
                        camera = "prophesee/event_cam_left"
                    fits, identities, roi_counts = [], [], []
                    for raw, (a, b) in zip(raw_windows, windows, strict=True):
                        surface, origin, count = make_surface(raw, roi, resolution, int(a), int(b))
                        fits.append(fit_surface(surface, origin, roi, box, int(b)))
                        roi_counts.append(count)
                        identities.append(
                            {
                                "start_us": int(a),
                                "end_us": int(b),
                                "events": len(raw),
                                "roi_events": count,
                                "int64_xy_tp_sha256": hashlib.sha256(
                                    np.ascontiguousarray(raw, dtype="<i8").tobytes()
                                ).hexdigest(),
                            }
                        )
                    support = None
                    disagreement = None
                    with np.load(folder / "PAIR_INPUTS.npz", allow_pickle=False) as archive:
                        key = f"call{call['call_id']}"
                        pair = key + ".PAIR_E_MATCHED.input133"
                        if pair in archive:
                            support = float(np.mean(archive[pair][item["batch_index"], 131:133]))
                        if key + ".expert3" in archive:
                            disagreement = float(
                                np.ptp(archive[key + ".expert3"][item["batch_index"]])
                            )
                    full_area = (roi[2] - roi[0]) * (roi[3] - roi[1])
                    retained = (
                        max(0.0, min(resolution[0], roi[2]) - max(0.0, roi[0]))
                        * max(0.0, min(resolution[1], roi[3]) - max(0.0, roi[1]))
                        / full_area
                    )
                    result.update(
                        status="measured" if fits[-1]["all"] else "no_accepted_planes",
                        physical=physical_vector(fits[0], fits[1]),
                        support=[
                            float(np.log1p(roi_counts[-1])),
                            fits[-1]["plane_fraction"],
                            support,
                            disagreement,
                            len(candidates),
                            retained,
                        ],
                        raw_path=str(raw_path),
                        camera=camera,
                        roi=roi.tolist(),
                        object_box=box.tolist(),
                        raw_windows=identities,
                        snapshots=fits,
                        source_call_status=call["status"],
                        available_us=item["available_us"],
                    )
                result["elapsed_seconds"] = time.perf_counter() - started
                destination = output / "queries"
                destination.mkdir(exist_ok=True)
                write_json(destination / (identity + ".json"), result)
                rows.append({k: v for k, v in result.items() if k != "snapshots"})
                print(
                    f"{number + 1}/{len(records)} {record['token']} {result['status']} "
                    f"{result['elapsed_seconds']:.2f}s",
                    flush=True,
                )
        write_json(
            output / "FEATURES.json",
            {
                "records": rows,
                "physical_fields": list(PHYSICAL_FIELDS),
                "support_fields": list(SUPPORT_FIELDS),
                "targets_read": False,
            },
        )
        write_json(
            output / "RUNTIME.json",
            {
                "elapsed_seconds": time.perf_counter() - began,
                "queries": len(rows),
                "scope": "CPU RAW read and diagnostic only; not end-to-end latency",
            },
        )
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    """Extract a full screen or a bounded first-query performance profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, default=Path("output/phase_timewarp/screen149_v1"))
    parser.add_argument(
        "--expanded", type=Path, default=Path("output/phase_observation_cache/expanded149")
    )
    parser.add_argument(
        "--slider-root", type=Path, default=Path("output/phase_campaign/Slider500/inputs")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-limit", type=int)
    args = parser.parse_args()
    extract(args.screen, args.expanded, args.slider_root, args.output, args.profile_limit)


if __name__ == "__main__":
    main()
