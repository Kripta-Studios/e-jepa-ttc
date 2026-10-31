"""Seal 149 identity-selected native caches and 447 context restarts before labels."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import cast

import numpy as np
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
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
from operational.phase_observation_cache.integrity import (
    load_bound_observations,
    seal_observations,
)
from operational.phase_observation_cache.replay import capture_context
from operational.phase_robust.bundle import verified_upstream_paths
from operational.phase_robust.cache import SensorCache
from scripts.final_train40_transfer import HEAD, FinalTransferPredictor, reject_targets
from scripts.pilot_phase_observation_cache import (
    sensor_row_receipt,
    serialize_observations,
    verify_replays,
)


def canonical_sha(value: object) -> str:
    """Hash strict canonical JSON, retaining exact identity and clock types."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def select_indices(groups: np.ndarray, tokens: np.ndarray, metadata: dict) -> list[int]:
    """Choose floor(n*q) for q=.25,.5,.75 in each road group and all Slider500."""
    roads = sorted(set(groups) - {"Slider-500"})
    if len(tokens) != 2608 or len(set(tokens)) != len(tokens) or len(roads) != 24:
        raise ValueError("Frozen 2608-row, 24-road-group population required")
    selected = []
    for group in roads:
        indices = sorted(
            np.flatnonzero(groups == group),
            key=lambda i: (metadata[str(tokens[i])]["anchor_us"], str(tokens[i])),
        )
        picks = [int(indices[len(indices) * numerator // 4]) for numerator in (1, 2, 3)]
        if len(set(picks)) != 3:
            raise ValueError("Road group lacks three distinct temporal quantiles")
        selected.extend(picks)
    slider = sorted(
        np.flatnonzero(groups == "Slider-500"), key=lambda i: int(str(tokens[i]).split(":")[-1])
    )
    if [int(str(tokens[i]).split(":")[-1]) for i in slider] != list(range(77)):
        raise ValueError("Need exactly original Slider500 ordinals 0..76")
    return selected + [int(i) for i in slider]


def native_vector(arrays: dict, device: str) -> np.ndarray:
    """Extract F0/RGB/event TTC with the frozen conversion on the original device."""
    vector = np.full(3, np.nan, np.float32)
    for name, index in (("R_CTX", 1), ("E_CTX_MATCHED", 2)):
        key = f"{name}.output_point_phase"
        if key in arrays:
            vector[index] = float(phase_to_ttc(torch.from_numpy(arrays[key]).to(device))[0].cpu())
    if f"{HEAD}.output_ttc" in arrays:
        vector[0] = arrays[f"{HEAD}.output_ttc"][0]
    elif np.isfinite(vector[2]):
        vector[0] = vector[2]
    return vector


def verify_artifacts(root: Path, artifacts: dict) -> None:
    """Reject changed or escaping artifact paths on resume."""
    for relative, digest in artifacts.items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root.resolve()) or file_sha(path) != digest:
            raise ValueError(f"Changed or escaping artifact: {relative}")


def source_paths() -> list[Path]:
    """Bind instrumentation and each original producer/preparation/context dependency."""
    return sorted(
        set(Path("operational/phase_observation_cache").glob("*.py"))
        | {
            Path(__file__).resolve(),
            Path("scripts/pilot_phase_observation_cache.py"),
            Path("scripts/final_train40_transfer.py"),
            Path("operational/final_refit/test12.py"),
            Path("operational/final_refit/features.py"),
            Path("operational/evttc_competition/runtime.py"),
            Path("operational/evttc_competition/prepare.py"),
            Path("operational/evttc_competition/readers.py"),
            Path("operational/evttc_competition/geometry.py"),
            Path("operational/phase_robust/cache.py"),
            Path("operational/rgb_port/transfer.py"),
            Path("operational/rgb_port_revision/transfer_inputs.py"),
            Path("operational/rgb_port/infer_experts.py"),
            Path("src/e_jepa_ttc/simplex_t/model.py"),
            Path("src/e_jepa_ttc/simplex_t/phase.py"),
            Path("docs/superpowers/plans/2026-10-31-phase-native-observation-cache.md"),
            Path("docs/superpowers/plans/2026-10-31-phase-native-context-augmentation.md"),
        }
    )


def try_reuse_pilot(pilot: Path, destination: Path, binding: dict, freeze: dict) -> bool:
    """Reuse only an exact query and unchanged producer pipeline; verify all prior bytes."""
    if not (pilot / "COMPLETE.json").is_file():
        return False
    previous = read(pilot / "FREEZE.json")
    if binding["token"] not in previous["selected_tokens"]:
        return False
    if previous["upstream"] != freeze["upstream"] or previous["inputs"] != freeze["inputs"]:
        return False
    modality = "slider500" if binding["group"] == "Slider-500" else "road"
    query_file = "SLIDER_QUERY.json" if modality == "slider500" else "ROAD_QUERY.json"
    if canonical_sha(read(pilot / query_file)) != binding["query_sha256"]:
        return False
    # New wrapper validation does not change original execution; original modules must match.
    for name, digest in previous["source_hashes"].items():
        if (
            "phase_observation_cache" in name
            or "pilot_phase_observation_cache" in name
            or name.startswith("docs")
        ):
            continue
        if modality == "road" and "evttc_competition" in name:
            continue
        if file_sha(Path(name)) != digest:
            return False
    if (
        modality == "slider500"
        and previous["slider_sensor_sha256"] != freeze["slider_sensor_sha256"]
    ):
        return False
    completed = read(pilot / "COMPLETE.json")
    verify_artifacts(pilot, completed["artifacts"])
    for name in (
        "OBSERVATIONS.npz",
        "ORIGINAL_CONTEXT.npz",
        "PRODUCER_DIAGNOSTICS.npz",
        "EXTRACTION.json",
    ):
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pilot / modality / name, destination / name)
    metadata = read(pilot / modality / "OBSERVATIONS.json")
    if modality == "slider500":
        attach_event_windows(metadata, read(destination / "EXTRACTION.json")["preparation"])
    atomic_receipt(destination / "OBSERVATIONS.json", metadata)
    atomic_receipt(
        destination / "PILOT_REUSE.json",
        {
            "pilot_complete_sha256": file_sha(pilot / "COMPLETE.json"),
            "pilot_freeze_sha256": file_sha(pilot / "FREEZE.json"),
            "source": str(pilot / modality),
            "old_source_hashes": previous["source_hashes"],
            "repeated_producer_calls": 0,
        },
    )
    return True


def attach_event_windows(observations: dict, preparation: dict) -> None:
    """Join exact recorded event windows by accepted anchor within this query only."""
    windows = {
        item["anchor_us"]: item["windows_us"]
        for item in preparation["event_observations"]
        if "windows_us" in item
    }
    for item in observations["event"]:
        item["windows_us"] = windows[item["anchor_us"]]


def store_capture(
    destination: Path, observations: dict, original: dict, producer_arrays: dict, extraction: dict
) -> None:
    arrays, metadata = serialize_observations(observations)
    save_npz(destination / "OBSERVATIONS.npz", arrays)
    atomic_receipt(destination / "OBSERVATIONS.json", metadata)
    save_npz(destination / "ORIGINAL_CONTEXT.npz", original)
    save_npz(destination / "PRODUCER_DIAGNOSTICS.npz", producer_arrays)
    atomic_receipt(destination / "EXTRACTION.json", extraction)


def finish_query(
    destination: Path, binding: dict, runtime: FinalTransferPredictor, device: str
) -> dict:
    """Validate sealed disk data, prove clean parity, then preserve each restart."""
    seal_observations(destination, binding)
    observations = load_bound_observations(destination, binding)
    with np.load(destination / "ORIGINAL_CONTEXT.npz", allow_pickle=False) as stored:
        original = {k: stored[k] for k in stored.files}
    parity = verify_replays(
        destination,
        runtime.models,
        runtime.normalizers,
        observations,
        binding["query_us"],
        original,
        device,
    )
    atomic_receipt(destination / "PARITY.json", parity)
    native = [native_vector(original, device)]
    for keep in (1, 2, 3):
        if parity["status"] == "unavailable":
            native.append(np.full(3, np.nan, np.float32))
        else:
            with np.load(destination / f"LAST_{keep}.npz", allow_pickle=False) as stored:
                native.append(native_vector({k: stored[k] for k in stored.files}, device))
    if parity["clean_ttc"] is not None and native[0][0] != parity["clean_ttc"]:
        raise AssertionError("Extracted native F0 differs from exact clean replay")
    save_npz(destination / "NATIVE.npz", {"native": np.stack(native)})
    extraction = read(destination / "EXTRACTION.json")
    result = {
        "binding": binding,
        "clean_exact": True,
        "disk_roundtrip_exact": True,
        "status": parity["status"],
        "reason": extraction["result"].get("model_reason", parity["status"]),
        "artifacts": {
            p.name: file_sha(p)
            for p in destination.iterdir()
            if p.is_file() and p.name != "COMPLETE.json"
        },
    }
    atomic_receipt(destination / "COMPLETE.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {
        "upstream": "output/phase_submission/FINAL_MODEL/upstream",
        "population": "output/phase_submission/training/POPULATION.npz",
        "original-config": "output/phase_submission/training/CONFIG.json",
        "road-metadata": "output/phase_adapt_final/data_v3/POPULATION.json",
        "slider-root": "output/phase_campaign/Slider500/inputs",
        "pilot": "output/phase_observation_cache/pilot/completed",
        "output": "output/phase_observation_cache/expanded149",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()
    configure_cuda()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    paths = verified_upstream_paths(args.upstream)
    with np.load(args.population, allow_pickle=False) as stored:
        groups, tokens = stored["group"], stored["token"]  # No target array opened.
    config = read(args.original_config)
    if set(groups) & set(config["excluded_road"] + config["excluded_slider"]):
        raise ValueError("Excluded group in population")
    metadata = {"evttc:" + row["query_id"]: row for row in read(args.road_metadata)["rows"]}
    selected = select_indices(groups, tokens, metadata)
    sources = {str(p.resolve().relative_to(Path.cwd())): file_sha(p) for p in source_paths()}
    freeze = {
        "schema": "phase_native_context_augmentations_v1",
        "targets_read": False,
        "optimizer_updates": 0,
        "selection": "floor(n*q),q=.25,.5,.75 per road; Slider500 0..76",
        "selected_tokens": [str(tokens[i]) for i in selected],
        "selected_groups": [str(groups[i]) for i in selected],
        "inputs": {
            str(p): file_sha(p) for p in (args.population, args.original_config, args.road_metadata)
        },
        "upstream": read(args.upstream / "BUNDLE.json")["parent"],
        "source_hashes": sources,
        "device": args.device,
        "threads": 2,
        "seed": 7,
        "tf32": False,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "slider_sensor_sha256": {
            name: file_sha(args.slider_root / "Slider-500" / name)
            for name in ("bbox.csv", "calibParams.mat", "slider_500.bag")
        },
        "road_raw_binding": "Original raw_stat checked by unchanged reader; no new full HDF5 hash",
        "semantics": "Context restart only; retained producer sensor windows unchanged",
    }
    atomic_receipt(args.output / "FREEZE.json", freeze)
    for relative, digest in sources.items():
        destination = args.output / "executed_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if file_sha(destination) != digest:
                raise ValueError("Executed source snapshot differs")
        else:
            shutil.copyfile(relative, destination)
    runtime = FinalTransferPredictor(paths, args.device)
    slider = CapturingCompetitionRuntime(runtime.models, runtime.normalizers, device=args.device)
    camera = load_calibration(args.slider_root / "Slider-500", "slider_candidate")
    records = []
    with SensorSequence(args.slider_root, "Slider-500", slider_profile="event_aided_ttc") as source:
        slider_rows = source.bbox_rows()
        cached = SensorCache(source)
        for index, population_index in enumerate(selected):
            token, group = str(tokens[population_index]), str(groups[population_index])
            is_slider = group == "Slider-500"
            row = (
                dict(slider_rows[int(token.split(":")[-1])])
                if is_slider
                else {k: v for k, v in metadata[token].items() if k != "ttc_csv_path"}
            )
            reject_targets(row)
            query_receipt = sensor_row_receipt(row)
            binding = {
                "token": token,
                "group": group,
                "query_us": int(row["anchor_us"]),
                "query_sha256": canonical_sha(query_receipt),
                "calibration_sha256": (
                    canonical_sha(camera.metadata)
                    if is_slider
                    else canonical_sha({"road_query": query_receipt})
                ),
                "model_sha256": canonical_sha(freeze["upstream"]),
                "source_sha256": canonical_sha(sources),
            }
            destination = args.output / "queries" / hashlib.sha256(token.encode()).hexdigest()[:24]
            receipt_path = destination / "COMPLETE.json"
            if receipt_path.exists():
                receipt = read(receipt_path)
                if (
                    receipt["binding"] != binding
                    or not receipt["clean_exact"]
                    or not receipt["disk_roundtrip_exact"]
                ):
                    raise ValueError("Completed query identity/parity differs")
                verify_artifacts(destination, receipt["artifacts"])
                load_bound_observations(destination, binding)
                records.append((token, group, destination, receipt))
                print(f"{index + 1}/149 verified resume {token}", flush=True)
                continue
            if destination.exists():
                raise ValueError(
                    f"Incomplete immutable query requires explicit recovery: {destination}"
                )
            query_start = time.perf_counter()
            try:
                reused = try_reuse_pilot(args.pilot, destination, binding, freeze)
                atomic_receipt(destination / "QUERY.json", query_receipt)
                if not reused:
                    with (
                        instrument_producers(runtime.models, args.device) as (
                            calls,
                            producer_arrays,
                        ),
                        capture_context(runtime.models) as original,
                    ):
                        if is_slider:
                            preparation_start = time.perf_counter()
                            prepared = prepare_query(
                                cast(SensorSequence, cached), slider_rows, row, camera
                            )
                            preparation_seconds = time.perf_counter() - preparation_start
                            result = slider.predict(prepared.events, prepared.rgb, row["anchor_us"])
                            observations = slider.observations
                            attach_event_windows(observations, prepared.details)
                            extraction = {
                                "result": dataclasses.asdict(result),
                                "preparation": prepared.details,
                                "calibration_metadata": camera.metadata,
                                "preparation_seconds": preparation_seconds,
                                "requested": slider.requested,
                                "producer_calls": calls,
                            }
                        else:
                            result, observations, details = capture_road(runtime, row)
                            extraction = {
                                "result": result,
                                "details": details,
                                "producer_calls": calls,
                            }
                    store_capture(destination, observations, original, producer_arrays, extraction)
                receipt = finish_query(destination, binding, runtime, args.device)
                records.append((token, group, destination, receipt))
                print(
                    f"{index + 1}/149 {receipt['status']} {token} "
                    f"{time.perf_counter() - query_start:.2f}s reuse={reused}",
                    flush=True,
                )
            except Exception as error:
                atomic_receipt(
                    destination / "FAILED.json",
                    {
                        "binding": binding,
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                        "targets_read": False,
                    },
                )
                raise
    augment = {k: [] for k in ("token", "group", "keep", "native", "status", "reason")}
    clean = {k: [] for k in ("token", "group", "native", "status", "reason")}
    for token, group, destination, receipt in records:
        with np.load(destination / "NATIVE.npz", allow_pickle=False) as stored:
            values = stored["native"]
        for key, value in (
            ("token", token),
            ("group", group),
            ("native", values[0]),
            ("status", receipt["status"]),
            ("reason", receipt["reason"]),
        ):
            clean[key].append(value)
        for keep in (1, 2, 3):
            for key, value in (
                ("token", token),
                ("group", group),
                ("keep", keep),
                ("native", values[keep]),
                ("status", receipt["status"]),
                ("reason", receipt["reason"]),
            ):
                augment[key].append(value)
    save_npz(args.output / "AUGMENTATIONS.npz", {k: np.asarray(v) for k, v in augment.items()})
    save_npz(args.output / "CLEAN.npz", {k: np.asarray(v) for k, v in clean.items()})
    if any(file_sha(Path(p)) != digest for p, digest in sources.items()):
        raise ValueError("Source changed during extraction; refuse final seal")
    atomic_receipt(
        args.output / "COMPLETE.json",
        {
            "schema": freeze["schema"],
            "status": "COMPLETE",
            "queries": 149,
            "augmentations": 447,
            "clean_exact": True,
            "disk_roundtrip_exact": True,
            "targets_read": False,
            "optimizer_updates": 0,
            "freeze_sha256": file_sha(args.output / "FREEZE.json"),
            "inputs": freeze["inputs"],
            "native_order": ["F0", "RGB", "event"],
            "status_counts": {
                s: sum(r[3]["status"] == s for r in records)
                for s in sorted({r[3]["status"] for r in records})
            },
            "total_seconds": time.perf_counter() - started,
            "timing_scope": "Instrumented extraction/replay; not production latency",
            "artifacts": {
                str(p.relative_to(args.output)): file_sha(p)
                for p in args.output.rglob("*")
                if p.is_file() and p != args.output / "COMPLETE.json"
            },
        },
    )
    print("COMPLETE sealed 149 clean disk replays and 447 context restarts; no targets", flush=True)


if __name__ == "__main__":
    main()
