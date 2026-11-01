"""Seal target-blind canonical PAIR predictions on the existing fixed 149 queries."""

import argparse
import dataclasses
import hashlib
import platform
import shutil
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch

from operational.evttc_competition.geometry import Calibration, load_calibration
from operational.evttc_competition.prepare import prepare_query
from operational.evttc_competition.readers import SensorSequence
from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_adapt_training import configure_cuda
from operational.phase_campaign.data import read
from operational.phase_observation_cache.capture import CapturingCompetitionRuntime, capture_road
from operational.phase_observation_cache.replay import capture_context, replay
from operational.phase_robust.bundle import verified_upstream_paths
from operational.phase_robust.cache import SensorCache
from operational.phase_startup.runtime import RouterEnsemble
from operational.phase_submission.runtime import load_submission_model
from operational.phase_timewarp.canonical_pair import CanonicalPair
from operational.phase_timewarp.capture import PAIRS, bind_calls, capture_calls
from operational.phase_timewarp.screen import replay_pair_batches
from scripts.audit_phase_native_source_closure import DEPENDENCIES
from scripts.build_phase_native_context_augmentations import (
    attach_event_windows,
    native_vector,
    verify_artifacts,
)
from scripts.final_train40_transfer import FinalTransferPredictor, reject_targets
from scripts.pilot_phase_observation_cache import serialize_observations
from scripts.pilot_phase_producer_timewarp import exact_arrays, read_arrays, verify_sensor_identity
from scripts.train_phase_native_context_fusion import validate_source_closure

PLAN = Path("docs/superpowers/plans/2026-10-31-phase-canonical-screen149.md")


def capture_one(
    runtime: FinalTransferPredictor,
    source: Path,
    row: dict,
    slider_rows: list,
    cached: SensorCache,
    camera: Calibration,
    device: str,
) -> tuple[list, dict, dict, dict]:
    """Extract once using unchanged preparers; pixel tensors live only for this query."""
    with (
        capture_calls(runtime.models, device) as (calls, arrays),
        capture_context(runtime.models) as context,
    ):
        if row.get("query_id"):
            result, observations, details = capture_road(runtime, row)
            candidates = {"event": observations["event"], "rgb": details["rgb_prepared"]}
            extraction = {"result": result, "details": details}
            mode = "float64_seconds"
        else:
            prepared = prepare_query(cast(SensorSequence, cached), slider_rows, row, camera)
            slider = CapturingCompetitionRuntime(runtime.models, runtime.normalizers, device=device)
            slider_result = slider.predict(prepared.events, prepared.rgb, row["anchor_us"])
            observations = slider.observations
            candidates = slider.requested or {"event": [], "rgb": []}
            attach_event_windows(observations, prepared.details)
            attach_event_windows(candidates, prepared.details)
            extraction = {
                "result": dataclasses.asdict(slider_result),
                "preparation": prepared.details,
            }
            mode = "float32_microseconds"
    bind_calls(calls, arrays, candidates, observations, dt_mode=mode)
    exact_arrays(context, read_arrays(source / "ORIGINAL_CONTEXT.npz"), "historical native context")
    raw, _ = serialize_observations(observations)
    exact_arrays(raw, read_arrays(source / "OBSERVATIONS.npz"), "historical observations")
    compact = {key: value for key, value in arrays.items() if not key.endswith(".pixels")}
    return calls, compact, context, extraction


def predict_query(
    runtime: FinalTransferPredictor,
    router: RouterEnsemble,
    calls: list,
    compact: dict,
    original_context: dict,
    geometry: float,
    quality: np.ndarray,
    query_us: int,
    device: str,
    destination: Path,
) -> dict:
    """Require exact original compact replay before executing the fixed intervention."""
    originals = runtime.models
    variants = {
        "original": originals,
        "canonical": {**originals, **{k: CanonicalPair(originals[k]) for k in PAIRS}},
    }
    results = {}
    for name, models in variants.items():
        result = replay_pair_batches(models, calls, compact, device)
        _, context = replay(models, runtime.normalizers, result["observations"], query_us, device)
        if name == "original":
            for call in calls:
                if call["status"] != "accepted":
                    continue
                key = f"call{call['call_id']}"
                pair = "PAIR_E_MATCHED" if call["prefix"] == "E" else "PAIR_R"
                for field in ("raw17", "expert3"):
                    np.testing.assert_array_equal(
                        result["arrays"][key + "." + field], compact[key + "." + field]
                    )
                np.testing.assert_array_equal(
                    result["arrays"][key + ".raw_phase"], compact[f"{key}.{pair}.raw_phase"]
                )
            exact_arrays(context, original_context, "compact original context/head")
        native = native_vector(context, device)
        with torch.inference_mode():
            adapted = router(
                torch.from_numpy(native)[None],
                torch.tensor([geometry], dtype=torch.float32),
                torch.from_numpy(quality)[None],
            )
        value = float(adapted[0])
        arrays, metadata = serialize_observations(result["observations"])
        folder = destination / name
        save_npz(folder / "PAIR_REPLAY.npz", result["arrays"])
        save_npz(folder / "CONTEXT.npz", context)
        save_npz(folder / "OBSERVATIONS.npz", arrays)
        atomic_receipt(folder / "OBSERVATIONS.json", metadata)
        atomic_receipt(folder / "CALLS.json", {"calls": result["calls"]})
        results[name] = {
            "native": [float(v) if np.isfinite(v) else None for v in native],
            "router": value if np.isfinite(value) else None,
            "accepted_observations": {m: len(v) for m, v in result["observations"].items()},
            "rejected_calls": [r for r in result["calls"] if r["status"] != "accepted"],
        }
    return results


def main() -> None:
    """Freeze sources and parent receipts, execute and seal before any target join."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in {
        "expanded": "output/phase_observation_cache/expanded149",
        "bundle": "output/phase_submission/FINAL_MODEL",
        "population": "output/phase_submission/training/POPULATION.npz",
        "source-closure": (
            "output/phase_observation_cache/expanded149_sourceclosure/SOURCE_CLOSURE.json"
        ),
        "slider-root": "output/phase_campaign/Slider500/inputs",
        "output": "output/phase_timewarp/screen149_v1",
    }.items():
        parser.add_argument("--" + name, type=Path, default=Path(default))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh output; never overwrite a sealed or failed run")
    parent = read(args.expanded / "FREEZE.json")
    verify_artifacts(args.expanded, read(args.expanded / "COMPLETE.json")["artifacts"])
    validate_source_closure(args.source_closure, args.expanded)
    if parent["inputs"].get(str(args.population)) != file_sha(args.population):
        raise ValueError("Population differs from selected cache")
    tokens, groups = parent["selected_tokens"], parent["selected_groups"]
    if len(tokens) != 149 or len(set(tokens)) != 149 or groups.count("Slider-500") != 77:
        raise ValueError("Require fixed 72-road/77-Slider population")
    # Only inference fields; target is deliberately not read in this process.
    with np.load(args.population, allow_pickle=False) as stored:
        population = {k: stored[k] for k in ("token", "group", "geometry", "quality")}
    index = {str(token): i for i, token in enumerate(population["token"])}
    if len(index) != len(population["token"]):
        raise ValueError("Duplicate population token")
    paths = verified_upstream_paths(args.bundle / "upstream")
    if read(args.bundle / "upstream/BUNDLE.json")["parent"] != parent["upstream"]:
        raise ValueError("Upstream parent differs")
    sources = (
        set(DEPENDENCIES)
        | set(parent["source_hashes"])
        | {str(p) for p in Path("operational/phase_timewarp").glob("*.py")}
        | {str(p) for p in Path("operational/phase_startup").glob("*.py")}
        | {
            str(Path(__file__).resolve().relative_to(Path.cwd())),
            str(PLAN),
            "operational/phase_submission/runtime.py",
            "scripts/pilot_phase_producer_timewarp.py",
            "tests/unit/test_phase_canonical_screen.py",
        }
    )
    hashes = {str(Path(p)): file_sha(Path(p)) for p in sorted(sources)}
    inputs = {
        str(p): file_sha(p)
        for p in (
            args.expanded / "COMPLETE.json",
            args.source_closure,
            args.population,
            args.bundle / "MODEL.json",
            args.bundle / "router/FINAL_FREEZE.json",
        )
    }
    for filename, digest in parent["slider_sensor_sha256"].items():
        if file_sha(args.slider_root / "Slider-500" / filename) != digest:
            raise ValueError("Slider sensor source changed")
    freeze = {
        "schema": "phase_canonical_screen149_v1",
        "tokens": tokens,
        "groups": groups,
        "canonical_dt_seconds": 0.1,
        "targets_read": False,
        "optimizer_updates": 0,
        "source_hashes": hashes,
        "inputs": inputs,
        "upstream": parent["upstream"],
        "device": args.device,
        "seed": 7,
        "threads": 2,
        "tf32": False,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "start_time": datetime.now(UTC).isoformat(),
        "scope": "exposed development screen; original F0 and delivered startup router separately",
        "geometry_quality": "unchanged inference fields from hash-bound eligible population",
        "criteria": (
            "fixed in source-bound plan; no lost coverage and "
            "four nonworsening macro metrics per route"
        ),
    }
    atomic_receipt(args.output / "FREEZE.json", freeze)
    for relative in hashes:
        destination = args.output / "executed_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative, destination)
    configure_cuda()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    try:
        router = load_submission_model(args.bundle)
        runtime = FinalTransferPredictor(paths, args.device)
        camera = load_calibration(args.slider_root / "Slider-500", "slider_candidate")
        records = []
        with SensorSequence(
            args.slider_root, "Slider-500", slider_profile="event_aided_ttc"
        ) as reader:
            slider_rows, cached = reader.bbox_rows(), SensorCache(reader)
            for number, (token, group) in enumerate(zip(tokens, groups, strict=True)):
                query_start = time.perf_counter()
                identity = hashlib.sha256(token.encode()).hexdigest()[:24]
                source, destination = (
                    args.expanded / "queries" / identity,
                    args.output / "queries" / identity,
                )
                old = read(source / "COMPLETE.json")
                if old["binding"]["token"] != token or old["binding"]["group"] != group:
                    raise ValueError("Query identity changed")
                row = read(source / "QUERY.json")
                if group == "Slider-500":
                    row = dict(slider_rows[int(token.split(":")[-1])])
                    verify_sensor_identity(row, read(source / "QUERY.json"))
                reject_targets(row)
                calls, compact, context, extraction = capture_one(
                    runtime, source, row, slider_rows, cached, camera, args.device
                )
                save_npz(destination / "PAIR_INPUTS.npz", compact)
                atomic_receipt(destination / "CALLS.json", {"calls": calls})
                atomic_receipt(destination / "EXTRACTION.json", extraction)
                # Read the saved compact cache, not live pixel outputs, for both variants.
                compact = read_arrays(destination / "PAIR_INPUTS.npz")
                i = index[token]
                if population["group"][i] != group:
                    raise ValueError("Population group binding differs")
                results = predict_query(
                    runtime,
                    router,
                    calls,
                    compact,
                    context,
                    population["geometry"][i],
                    population["quality"][i],
                    row["anchor_us"],
                    args.device,
                    destination,
                )
                record = {
                    "token": token,
                    "group": group,
                    "query_us": int(row["anchor_us"]),
                    "baseline_exact": True,
                    "results": results,
                }
                atomic_receipt(destination / "RESULT.json", record)
                records.append(record)
                elapsed = time.perf_counter() - start
                print(
                    f"{number + 1}/149 {token} {time.perf_counter() - query_start:.2f}s "
                    f"elapsed={elapsed:.1f}s ETA={elapsed / (number + 1) * (148 - number):.1f}s",
                    flush=True,
                )
        verify_artifacts(Path.cwd(), hashes)
        for path, digest in inputs.items():
            if file_sha(Path(path)) != digest:
                raise ValueError("Input changed during execution")
        atomic_receipt(
            args.output / "PREDICTIONS.json", {"records": records, "targets_read": False}
        )
        atomic_receipt(
            args.output / "COMPLETE.json",
            {
                "status": "COMPLETE",
                "queries": 149,
                "baseline_exact": True,
                "targets_read": False,
                "optimizer_updates": 0,
                "freeze_sha256": file_sha(args.output / "FREEZE.json"),
                "elapsed_seconds": time.perf_counter() - start,
                "end_time": datetime.now(UTC).isoformat(),
                "artifacts": {
                    str(p.relative_to(args.output)): file_sha(p)
                    for p in args.output.rglob("*")
                    if p.is_file()
                },
            },
        )
    except Exception as error:
        atomic_receipt(
            args.output / "FAILED.json",
            {"error": repr(error), "traceback": traceback.format_exc(), "targets_read": False},
        )
        raise


if __name__ == "__main__":
    main()
