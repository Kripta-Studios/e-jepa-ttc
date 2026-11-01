"""V0/V1 label-blind RGB surface and coordinate audit of the frozen release screen."""

from __future__ import annotations

import argparse
import hashlib
import time
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np

from e_jepa_ttc.rgb_port.data import _effective_square, crop_uint8
from operational.evttc_competition.readers import SensorSequence
from operational.release_research.physics import transport
from operational.release_research.protocol import ResearchRun, sha256, write_json
from operational.release_research.surface_correspondence import persistent_surface
from scripts.release_observability import check_stat
from scripts.release_research import read, sources, verify_complete


def extract(
    screen: Path, expanded: Path, slider_root: Path, output: Path, limit: int | None = None
) -> None:
    """Read only the exact original RGB timestamps and crop; retain missing observations."""
    receipt = verify_complete(screen)
    verify_complete(expanded)
    if not receipt["baseline_exact"]:
        raise ValueError("Require an exact verified baseline")
    records = read(screen / "PREDICTIONS.json")["records"]
    if (
        len(records) != 149
        or [r["token"] for r in records] != read(screen / "FREEZE.json")["tokens"]
    ):
        raise ValueError("Require the frozen 149-query population")
    if limit is not None:
        if not 1 <= limit <= 149:
            raise ValueError("Invalid profile limit")
        records = records[:limit]
    inputs = {
        **sources(),
        "parent": screen / "COMPLETE.json",
        "expanded": expanded / "COMPLETE.json",
        "preregistration": Path(
            "docs/release_research_implementation_20261031/SURFACE_PREREGISTRATION.md"
        ),
    }
    for name in (
        "scripts/release_surface.py",
        "scripts/release_observability.py",
        "src/e_jepa_ttc/rgb_port/data.py",
        "operational/evttc_competition/readers.py",
    ):
        inputs["source:" + name] = Path(name)
    for row in records:
        identity = hashlib.sha256(row["token"].encode()).hexdigest()[:24]
        for name in ("CALLS.json", "EXTRACTION.json"):
            inputs[identity + ":" + name] = screen / "queries" / identity / name
        inputs[identity + ":QUERY.json"] = expanded / "queries" / identity / "QUERY.json"
    if any(r["group"] == "Slider-500" for r in records):
        receipt_path = Path("output/phase_campaign/Slider500/SENSORS.json")
        inputs["slider_receipt"] = receipt_path
        for name, spec in read(receipt_path)["files"].items():
            path = slider_root / "Slider-500" / name
            if sha256(path) != spec["sha256"]:
                raise ValueError("Slider original sensor identity differs")
            inputs["slider:" + name] = path
    run = ResearchRun(
        output,
        inputs,
        {
            "experiment_id": "V0-V1-persistent-RGB-v1",
            "exposure": "development",
            "targets_read": False,
            "optimizer_updates": 0,
            "queries": len(records),
            "profiling_only": limit is not None,
            "seed": 7,
            "road_identity": "original stat binding and SHA256 of exact consumed RGB frames",
            "scope": "geometry diagnostic; not a trained gate or replacement release",
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
                source = read(expanded / "queries" / identity / "QUERY.json")
                extraction = read(folder / "EXTRACTION.json")
                items = [
                    item
                    for call in read(folder / "CALLS.json")["calls"]
                    if call["prefix"] == "R"
                    for item in call["observations"]
                ]
                result = {k: record[k] for k in ("token", "group", "query_us")}
                result.update(status="no_rgb_observation", q_at_query=None)
                if items:
                    item = max(items, key=lambda x: x["anchor_us"])
                    times = np.asarray(item["timestamps_us"], dtype=np.int64)
                    if (
                        item["available_us"] > record["query_us"]
                        or times[0] < record["query_us"] - 650000
                        or times[-1] > record["query_us"]
                    ):
                        raise ValueError("RGB context violates causal clock or 650ms budget")
                    result["timestamps_us"] = times.tolist()
                    result["source_accepted"] = item["source_accepted"]
                    if len(times) != 3:
                        result["status"] = "fewer_than_three_real_frames"
                    else:
                        if record["group"] == "Slider-500":
                            if slider is None:
                                slider = stack.enter_context(
                                    SensorSequence(
                                        slider_root, "Slider-500", slider_profile="event_aided_ttc"
                                    )
                                )
                            preparation = extraction["preparation"]
                            square = preparation["rgb_square"]
                            original = next(
                                o
                                for o in preparation["rgb_observations"]
                                if o["anchor_us"] == item["anchor_us"]
                            )
                            indices = np.asarray(original["frame_indices"], dtype=np.int64)
                            if not np.array_equal(slider.rgb_timestamps_us[indices], times):
                                raise ValueError("Slider RGB clock differs")
                            frames = [slider.read_rgb(int(i)) for i in indices]
                            historical_squares = [square for _ in items]
                        else:
                            path = Path(source["raw_path"])
                            check_stat(path, source["raw_stat"])
                            square = item["square_xyxy"]
                            with h5py.File(path, "r") as handle:
                                ts = np.asarray(handle["blackflys/left/ts"])
                                indices = np.searchsorted(ts, times)
                                if np.any(indices >= len(ts)) or not np.array_equal(
                                    ts[indices], times
                                ):
                                    raise ValueError(
                                        "Road RGB clock differs; no nearest-frame substitution"
                                    )
                                data = handle["blackflys/left/data"]
                                if not isinstance(data, h5py.Dataset):
                                    raise ValueError("RGB frames must be a dataset")
                                frames = [np.asarray(data[int(i)]) for i in indices]
                            check_stat(path, source["raw_stat"])
                            historical_squares = [i["square_xyxy"] for i in items]
                        crops = np.stack([crop_uint8(frame, square) for frame in frames])
                        geometry = [_effective_square(frame, square) for frame in frames]
                        measurement = persistent_surface(crops.mean(axis=1) / 255.0, times)
                        q = measurement["q_at_last_rgb"]
                        query_q = None
                        transport_reason = "no_surface_measurement"
                        if q is not None:
                            try:
                                query_q = transport(
                                    q, (record["query_us"] - int(times[-1])) * 1e-6
                                )[0]
                                transport_reason = "ok"
                            except ValueError as error:
                                transport_reason = str(error)
                        result.update(
                            status="measured"
                            if query_q is not None
                            else "insufficient_or_invalid_surface",
                            q_at_query=query_q,
                            transport_reason=transport_reason,
                            measurement=measurement,
                            square_requested=square,
                            effective_crops=geometry,
                            history_squares=historical_squares,
                            within_triplet_same_crop=len(set(geometry)) == 1,
                            history_same_requested_crop=all(
                                s == historical_squares[0] for s in historical_squares
                            ),
                            anisotropic_resize=any(
                                (g[2] - g[0]) != (g[3] - g[1]) for g in geometry
                            ),
                            sensor_rgb_sha256=[
                                hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest()
                                for f in frames
                            ],
                            cropped_rgb_sha256=hashlib.sha256(crops.tobytes()).hexdigest(),
                        )
                result["elapsed_seconds"] = time.perf_counter() - started
                (output / "queries").mkdir(exist_ok=True)
                write_json(output / "queries" / (identity + ".json"), result)
                rows.append(
                    {k: v for k, v in result.items() if k != "measurement"}
                    | {
                        "surface": {
                            k: v for k, v in result.get("measurement", {}).items() if k != "matches"
                        }
                    }
                )
                print(
                    f"{number + 1}/{len(records)} {record['token']} {result['status']} "
                    f"{result['elapsed_seconds']:.2f}s",
                    flush=True,
                )
        write_json(output / "FEATURES.json", {"records": rows, "targets_read": False})
        write_json(
            output / "RUNTIME.json",
            {
                "elapsed_seconds": time.perf_counter() - began,
                "queries": len(rows),
                "scope": "CPU geometry including RAW RGB read; not end-to-end release latency",
            },
        )
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    """Run the fixed full population or an explicitly marked performance profile."""
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
