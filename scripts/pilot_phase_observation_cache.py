"""Capture two eligible native queries and replay clean/shortened frozen contexts."""

from __future__ import annotations

import argparse
import dataclasses
import shutil
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from operational.evttc_competition.geometry import load_calibration
from operational.evttc_competition.prepare import prepare_query
from operational.evttc_competition.readers import SensorSequence
from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_adapt_training import configure_cuda
from operational.phase_campaign.data import read
from operational.phase_observation_cache.capture import (
    CapturingCompetitionRuntime,
    capture_road,
    instrument_producers,
)
from operational.phase_observation_cache.replay import capture_context, replay
from operational.phase_robust.bundle import verified_upstream_paths
from operational.phase_robust.cache import SensorCache
from scripts.final_train40_transfer import HEAD, FinalTransferPredictor, reject_targets


def sensor_row_receipt(row: dict) -> dict:
    """Preserve exact reader Decimal clocks as decimal text in JSON receipts."""
    return {k: str(v) if isinstance(v, Decimal) else v for k, v in row.items()}


def serialize_observations(observations: dict) -> tuple[dict, dict]:
    """Separate compact raw arrays from per-observation identity and physical clocks."""
    arrays, metadata = {}, {}
    for modality, items in observations.items():
        arrays[f"{modality}.raw17"] = np.array([x["raw17"] for x in items], np.float32).reshape(
            -1, 17
        )
        arrays[f"{modality}.expert3"] = np.array([x["expert3"] for x in items], np.float32).reshape(
            -1, 3
        )
        metadata[modality] = [
            {k: v for k, v in x.items() if k not in {"raw17", "expert3"}} for x in items
        ]
    return arrays, metadata


def load_observations(destination: Path) -> dict:
    """Reload owned native observations from the actual NPZ and clock JSON artifacts."""
    metadata = read(destination / "OBSERVATIONS.json")
    result = {}
    with np.load(destination / "OBSERVATIONS.npz", allow_pickle=False) as stored:
        for modality in ("event", "rgb"):
            raw, experts = stored[f"{modality}.raw17"], stored[f"{modality}.expert3"]
            if raw.shape != (len(metadata[modality]), 17) or experts.shape != (len(raw), 3):
                raise ValueError("Stored observation metadata/array alignment differs")
            result[modality] = [
                {**item, "raw17": raw[i].copy(), "expert3": experts[i].copy()}
                for i, item in enumerate(metadata[modality])
            ]
    return result


def verify_replays(
    destination: Path,
    models: dict,
    normalizers: dict,
    observations: dict,
    query_us: int,
    original: dict[str, np.ndarray],
    device: str,
) -> dict:
    """Prove clean parity first, then replay only context prefixes of length 1/2/3."""
    loaded = load_observations(destination)
    expected_arrays, expected_metadata = serialize_observations(observations)
    restored_arrays, restored_metadata = serialize_observations(loaded)
    if expected_metadata != restored_metadata:
        raise AssertionError("Stored observation clocks/metadata differ")
    for key in expected_arrays:
        np.testing.assert_array_equal(expected_arrays[key], restored_arrays[key])
    observations = loaded
    start = time.perf_counter()
    prediction, clean = replay(models, normalizers, observations, query_us, device)
    if set(clean) != set(original):
        raise AssertionError("Replay changed captured context/head inventory")
    for key in clean:
        np.testing.assert_array_equal(clean[key], original[key], err_msg=f"clean replay: {key}")
    status = "fused" if observations["rgb"] else "event_only"
    if not observations["event"]:
        status = "unavailable"
        if prediction is not None or clean:
            raise AssertionError("Missing events must remain unavailable")
    elif status == "fused":
        if f"{HEAD}.input_0" not in clean or clean[f"{HEAD}.input_0"].shape != (1, 340):
            raise AssertionError("Fused observations require the actual phase340/head path")
    elif not clean or any(not key.startswith("E_CTX_MATCHED.") for key in clean):
        raise AssertionError("Event-only replay must execute only the event context")
    save_npz(destination / "CLEAN_REPLAY.npz", clean)
    result: dict[str, Any] = {
        "status": status,
        "disk_roundtrip_exact": True,
        "clean_exact": True,
        "clean_compared_arrays": len(clean),
        "clean_ttc": prediction,
        "clean_replay_seconds": time.perf_counter() - start,
        "truncations": {},
    }
    if status == "unavailable":
        result["semantics"] = "No event support; no context or head prediction fabricated"
        return result
    for keep in (1, 2, 3):
        start = time.perf_counter()
        value, arrays = replay(models, normalizers, observations, query_us, device, keep=keep)
        for modality, name in (("event", "E_CTX_MATCHED"), ("rgb", "R_CTX")):
            if not observations[modality]:
                continue
            valid = arrays[f"{name}.input_2"][0]
            timing = arrays[f"{name}.input_1"][0, valid]
            assert timing[0, 2] == 0
            assert len(timing) == min(keep, len(observations[modality]))
        save_npz(destination / f"LAST_{keep}.npz", arrays)
        result["truncations"][str(keep)] = {
            "ttc_seconds": value,
            "seconds": time.perf_counter() - start,
            "counts": {m: min(keep, len(items)) for m, items in observations.items()},
            "first_retained_interarrival_zero": True,
            "earliest_retained_sensor_us": {
                m: min(x["first_sensor_ts_us"] for x in items[-keep:])
                for m, items in observations.items()
                if items
            },
        }
    result["semantics"] = (
        "Context restart only; original retained producer sensor windows unchanged"
    )
    return result


def main() -> None:
    """Select by group/time identity, never targets or prediction errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "upstream",
        "population",
        "original-config",
        "road-metadata",
        "slider-root",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--resume-road", type=Path, help="Reuse strictly sealed completed road pilot"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new immutable pilot output")
    configure_cuda()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(7)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    started = time.perf_counter()
    paths = verified_upstream_paths(args.upstream)
    with np.load(args.population, allow_pickle=False) as stored:
        # Deliberately do not load target arrays from the development population.
        groups, tokens, historical = stored["group"], stored["token"], stored["native"]
    config = read(args.original_config)
    excluded = set(config["excluded_road"] + config["excluded_slider"])
    if set(groups) & excluded or len(tokens) != 2608 or len(set(groups)) != 25:
        raise ValueError("Eligible development population changed")
    metadata = {"evttc:" + r["query_id"]: r for r in read(args.road_metadata)["rows"]}
    group = sorted(g for g in set(groups) if not g.startswith("Slider"))[0]
    candidates = sorted(
        np.flatnonzero(groups == group), key=lambda i: metadata[str(tokens[i])]["anchor_us"]
    )
    road_index = candidates[len(candidates) // 2]
    row = {k: v for k, v in metadata[str(tokens[road_index])].items() if k != "ttc_csv_path"}
    reject_targets(row)
    slider_indices = sorted(
        np.flatnonzero(groups == "Slider-500"), key=lambda i: int(str(tokens[i]).split(":")[-1])
    )
    slider_index = slider_indices[len(slider_indices) // 2]
    slider_ordinal = int(str(tokens[slider_index]).split(":")[-1])
    sources = list(Path("operational/phase_observation_cache").glob("*.py")) + [
        Path(__file__),
        Path("scripts/final_train40_transfer.py"),
        Path("operational/final_refit/test12.py"),
        Path("operational/final_refit/features.py"),
        Path("operational/evttc_competition/runtime.py"),
        Path("operational/evttc_competition/prepare.py"),
        Path("operational/evttc_competition/readers.py"),
        Path("operational/rgb_port/transfer.py"),
        Path("operational/rgb_port_revision/transfer_inputs.py"),
        Path("operational/rgb_port/infer_experts.py"),
        Path("docs/superpowers/plans/2026-10-31-phase-native-observation-cache.md"),
    ]
    source_hashes = {str(p): file_sha(p) for p in sources}
    freeze = {
        "schema": "phase_native_observation_pilot_v1",
        "upstream": read(args.upstream / "BUNDLE.json")["parent"],
        "inputs": {
            str(p): file_sha(p) for p in (args.population, args.original_config, args.road_metadata)
        },
        "selected_tokens": [str(tokens[road_index]), str(tokens[slider_index])],
        "selection": (
            "First eligible road group alphabetically, temporal median; Slider500 temporal median"
        ),
        "source_hashes": source_hashes,
        "device": args.device,
        "threads": 2,
        "seed": 7,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tf32": False,
        "patch_scope": "Dedicated process, one inference owner, scoped restored patches only",
        "targets_read": False,
        "optimizer_updates": 0,
        "sensor_restart_claim": False,
        "slider_sensor_sha256": {
            name: file_sha(args.slider_root / "Slider-500" / name)
            for name in ("bbox.csv", "calibParams.mat", "slider_500.bag")
        },
        "road_raw_binding": "Frozen raw_stat checked by original reader; no new full HDF5 hash",
    }
    atomic_receipt(args.output / "FREEZE.json", freeze)
    for source in sources:
        snapshot = args.output / "executed_source" / source.resolve().relative_to(Path.cwd())
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, snapshot)
    atomic_receipt(args.output / "ROAD_QUERY.json", row)
    load_start = time.perf_counter()
    runtime = FinalTransferPredictor(paths, args.device)
    load_seconds = time.perf_counter() - load_start
    results = {}
    destination = args.output / "road"
    if args.resume_road is not None:
        previous = read(args.resume_road / "FREEZE.json")
        partial = read(args.resume_road / "PARTIAL_ROAD_SEAL.json")
        if (
            previous["selected_tokens"] != freeze["selected_tokens"]
            or previous["upstream"] != freeze["upstream"]
            or previous["inputs"] != freeze["inputs"]
            or previous["device"] != freeze["device"]
            or previous["tf32"] != freeze["tf32"]
            or file_sha(args.resume_road / "FREEZE.json") != partial["freeze_sha256"]
        ):
            raise ValueError("Resume road identity or execution precision changed")
        for relative, digest in partial["files"].items():
            source_path = (args.resume_road / relative).resolve()
            if (
                not source_path.is_relative_to(args.resume_road.resolve())
                or file_sha(source_path) != digest
            ):
                raise ValueError("Changed or escaping completed road shard")
        for name, digest in previous["source_hashes"].items():
            if Path(name).name != Path(__file__).name and file_sha(Path(name)) != digest:
                raise ValueError("Frozen sensor/context implementation changed")
        shutil.copytree(args.resume_road / "road", destination)
        results["road"] = read(destination / "PARITY.json")
        if results["road"]["clean_exact"] is not True:
            raise ValueError("Road replay was not verified")
        atomic_receipt(
            args.output / "ROAD_RESUME.json",
            {
                "source": str(args.resume_road.resolve()),
                "partial_seal_sha256": file_sha(args.resume_road / "PARTIAL_ROAD_SEAL.json"),
                "old_source_hashes": previous["source_hashes"],
                "new_source_hashes": source_hashes,
                "reason": "Fix Decimal JSON receipt; verified road extraction not repeated",
                "repeated_road_producer_calls": 0,
            },
        )
        with np.load(destination / "ORIGINAL_CONTEXT.npz", allow_pickle=False) as stored:
            original_context = {k: stored[k] for k in stored.files}
        disk_result = verify_replays(
            destination,
            runtime.models,
            runtime.normalizers,
            load_observations(destination),
            row["anchor_us"],
            original_context,
            args.device,
        )
        atomic_receipt(destination / "DISK_PARITY.json", disk_result)
        results["road"]["disk_roundtrip_exact"] = disk_result["disk_roundtrip_exact"]
        print("ROAD reused exact verified captured observations and replays", flush=True)
    else:
        capture_road_query(
            runtime, row, destination, historical[road_index, 0], args.device, results
        )
    slider = CapturingCompetitionRuntime(runtime.models, runtime.normalizers, device=args.device)
    preparation_start = time.perf_counter()
    camera = load_calibration(args.slider_root / "Slider-500", "slider_candidate")
    with SensorSequence(args.slider_root, "Slider-500", slider_profile="event_aided_ttc") as source:
        rows = source.bbox_rows()
        slider_row = rows[slider_ordinal]
        cached = SensorCache(source)
        prepared = prepare_query(cast(SensorSequence, cached), rows, slider_row, camera)
        preparation_seconds = time.perf_counter() - preparation_start
        atomic_receipt(args.output / "SLIDER_QUERY.json", sensor_row_receipt(dict(slider_row)))
        print(
            f"SLIDER selected {tokens[slider_index]} anchor={slider_row['anchor_us']}", flush=True
        )
        with (
            instrument_producers(runtime.models, args.device) as (calls, producer_arrays),
            capture_context(runtime.models) as original,
        ):
            slider_result = slider.predict(prepared.events, prepared.rgb, slider_row["anchor_us"])
    destination = args.output / "slider500"
    arrays, observation_metadata = serialize_observations(slider.observations)
    save_npz(destination / "OBSERVATIONS.npz", arrays)
    save_npz(destination / "PRODUCER_DIAGNOSTICS.npz", producer_arrays)
    save_npz(destination / "ORIGINAL_CONTEXT.npz", original)
    atomic_receipt(destination / "OBSERVATIONS.json", observation_metadata)
    atomic_receipt(
        destination / "EXTRACTION.json",
        {
            "result": dataclasses.asdict(slider_result),
            "preparation": prepared.details,
            "calibration_metadata": camera.metadata,
            "preparation_seconds": preparation_seconds,
            "requested": slider.requested,
            "producer_calls": calls,
        },
    )
    result = verify_replays(
        destination,
        runtime.models,
        runtime.normalizers,
        slider.observations,
        slider_row["anchor_us"],
        original,
        args.device,
    )
    result["historical_native_F0_seconds"] = float(historical[slider_index, 0])
    result["historical_absolute_difference_seconds"] = abs(
        result["clean_ttc"] - float(historical[slider_index, 0])
    )
    result["historical_comparison_scope"] = (
        "Descriptive precision comparison only; clean parity refers to same extraction"
    )
    atomic_receipt(destination / "PARITY.json", result)
    results["slider500"] = result
    if any(file_sha(Path(p)) != digest for p, digest in source_hashes.items()):
        raise ValueError("Instrumented implementation changed during pilot")
    atomic_receipt(
        args.output / "COMPLETE.json",
        {
            "status": "COMPLETE",
            "queries": 2,
            "clean_exact": all(r["clean_exact"] for r in results.values()),
            "replays": results,
            "model_load_seconds": load_seconds,
            "total_seconds": time.perf_counter() - started,
            "timing_scope": "Instrumented GPU/host timings; not a production latency benchmark",
            "freeze_sha256": file_sha(args.output / "FREEZE.json"),
            "targets_read": False,
            "optimizer_updates": 0,
            "artifacts": {
                str(p.relative_to(args.output)): file_sha(p)
                for p in args.output.rglob("*")
                if p.is_file()
            },
        },
    )
    print(
        "COMPLETE two native observations pilot queries; clean exact and causal context replay",
        flush=True,
    )


def capture_road_query(
    runtime: FinalTransferPredictor,
    row: dict,
    destination: Path,
    historical_f0: float,
    device: str,
    results: dict,
) -> None:
    """Extract and verify one road query, separately resumable after metadata failures."""
    print(f"ROAD selected {row['query_id']} anchor={row['anchor_us']}", flush=True)
    with (
        instrument_producers(runtime.models, device) as (calls, producer_arrays),
        capture_context(runtime.models) as original,
    ):
        road_result, observations, details = capture_road(runtime, row)
    arrays, observation_metadata = serialize_observations(observations)
    save_npz(destination / "OBSERVATIONS.npz", arrays)
    save_npz(destination / "PRODUCER_DIAGNOSTICS.npz", producer_arrays)
    save_npz(destination / "ORIGINAL_CONTEXT.npz", original)
    atomic_receipt(destination / "OBSERVATIONS.json", observation_metadata)
    atomic_receipt(
        destination / "EXTRACTION.json",
        {"result": road_result, "details": details, "producer_calls": calls},
    )
    result = verify_replays(
        destination,
        runtime.models,
        runtime.normalizers,
        observations,
        row["anchor_us"],
        original,
        device,
    )
    result["historical_native_F0_seconds"] = float(historical_f0)
    result["historical_absolute_difference_seconds"] = abs(
        result["clean_ttc"] - float(historical_f0)
    )
    result["historical_comparison_scope"] = (
        "Descriptive precision comparison only; clean parity refers to same extraction"
    )
    atomic_receipt(destination / "PARITY.json", result)
    results["road"] = result
    print("ROAD clean replay exact; last1/2/3 completed", flush=True)


if __name__ == "__main__":
    main()
