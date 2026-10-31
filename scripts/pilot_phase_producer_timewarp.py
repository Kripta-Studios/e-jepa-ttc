"""Two-query, zero-update producer timewarp diagnostic with exact same-batch clean replay."""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import cast

import numpy as np
import torch

from operational.evttc_competition.geometry import load_calibration
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
from operational.phase_timewarp.capture import bind_calls, capture_calls
from operational.phase_timewarp.pilot import compare_stages, finite_ratio, replay_calls
from scripts.audit_phase_native_source_closure import DEPENDENCIES
from scripts.build_phase_native_context_augmentations import (
    attach_event_windows,
    native_vector,
    verify_artifacts,
)
from scripts.final_train40_transfer import FinalTransferPredictor, reject_targets
from scripts.pilot_phase_observation_cache import sensor_row_receipt, serialize_observations
from scripts.train_phase_native_context_fusion import validate_source_closure

TOKENS = ["evttc:EXTENDED:CCRm-low-0-overlap-0:17000441", "Slider-500:38"]


def verify_sensor_identity(row: dict, saved: dict) -> None:
    """Compare exact JSON identities, including tuple-to-list serialization."""
    if json.loads(json.dumps(sensor_row_receipt(row), allow_nan=False)) != saved:
        raise ValueError("Selected Slider identity differs from original pilot")


def read_arrays(path: Path) -> dict:
    """Read only locally produced sensor/trace NPZ files; never training populations."""
    with np.load(path, allow_pickle=False) as stored:
        return {k: stored[k] for k in stored.files}


def save_execution(
    destination: Path, result: dict, runtime: FinalTransferPredictor, query_us: int, device: str
) -> tuple[float | None, dict]:
    """Preserve each producer attempt before context/head replay, including domain failures."""
    save_npz(destination / "PRODUCERS.npz", result["arrays"])
    atomic_receipt(destination / "CALLS.json", {"calls": result["calls"]})
    arrays, metadata = serialize_observations(result["observations"])
    save_npz(destination / "OBSERVATIONS.npz", arrays)
    atomic_receipt(destination / "OBSERVATIONS.json", metadata)
    atomic_receipt(
        destination / "SELECTION.json",
        {
            "retained_physical_ids": result["selected_ids"],
            "accepted_producer_ids": result["accepted_ids"],
            "discarded_by_clock_budget": result["discarded"],
            "producer_rejected": [r for r in result["calls"] if r["status"] != "accepted"],
        },
    )
    value, context = replay(
        runtime.models, runtime.normalizers, result["observations"], query_us, device
    )
    save_npz(destination / "CONTEXT.npz", context)
    value = value if value is not None and np.isfinite(value) else None
    atomic_receipt(
        destination / "RESULT.json",
        {
            "native_frozen_F0_seconds": value,
            "available": value is not None,
            "unavailable_reason": "missing_or_nonfinite_frozen_head" if value is None else None,
            "scope": "Original nine-component F0; no adapted geometry router, no accuracy claim",
        },
    )
    return value, context


def exact_arrays(actual: dict, expected: dict, scope: str) -> None:
    """Require exact inventories, shapes, dtypes and values for the identity gate."""
    if set(actual) != set(expected):
        raise AssertionError(f"{scope}: array inventory differs")
    for key in actual:
        if actual[key].dtype != expected[key].dtype:
            raise AssertionError(f"{scope}: dtype differs for{key}")
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=f"{scope}:{key}")


def capture_query(
    name: str, runtime: FinalTransferPredictor, args: argparse.Namespace, destination: Path
) -> tuple[int, dict]:
    """Invoke original preparers/runtime once and map every actual producer batch member."""
    query_start = time.perf_counter()
    if name == "road":
        row = read(args.old_pilot / "ROAD_QUERY.json")
        reject_targets(row)
        with (
            capture_calls(runtime.models, args.device) as (calls, arrays),
            capture_context(runtime.models) as original,
        ):
            result, observations, details = capture_road(runtime, row)
        candidates = {"event": observations["event"], "rgb": details["rgb_prepared"]}
        extraction = {"result": result, "details": details}
        query_us = row["anchor_us"]
    else:
        preparation_start = time.perf_counter()
        camera = load_calibration(args.slider_root / "Slider-500", "slider_candidate")
        with SensorSequence(
            args.slider_root, "Slider-500", slider_profile="event_aided_ttc"
        ) as reader:
            rows = reader.bbox_rows()
            row = dict(rows[38])
            query_us = row["anchor_us"]
            verify_sensor_identity(row, read(args.old_pilot / "SLIDER_QUERY.json"))
            prepared = prepare_query(cast(SensorSequence, SensorCache(reader)), rows, row, camera)
        preparation_seconds = time.perf_counter() - preparation_start
        slider = CapturingCompetitionRuntime(
            runtime.models, runtime.normalizers, device=args.device
        )
        with (
            capture_calls(runtime.models, args.device) as (calls, arrays),
            capture_context(runtime.models) as original,
        ):
            native = slider.predict(prepared.events, prepared.rgb, query_us)
        observations = slider.observations
        attach_event_windows(observations, prepared.details)
        candidates = slider.requested
        attach_event_windows(candidates, prepared.details)
        extraction = {
            "result": dataclasses.asdict(native),
            "preparation": prepared.details,
            "preparation_seconds": preparation_seconds,
            "calibration": camera.metadata,
        }
    bind_calls(
        calls,
        arrays,
        candidates,
        observations,
        dt_mode="float64_seconds" if name == "road" else "float32_microseconds",
    )
    accepted = {
        f"{m}:{item['anchor_us']}": item for m, items in observations.items() for item in items
    }
    for call in calls:
        for item in call["observations"]:
            if item["source_accepted"]:
                source = accepted[item["observation_id"]]
                for field in ("raw17", "expert3"):
                    np.testing.assert_array_equal(
                        source[field], arrays[f"call{call['call_id']}.{field}"][item["batch_index"]]
                    )
    save_npz(destination / "CAPTURE_ARRAYS.npz", arrays)
    save_npz(destination / "ORIGINAL_CONTEXT.npz", original)
    atomic_receipt(destination / "CAPTURE_CALLS.json", {"calls": calls})
    atomic_receipt(destination / "QUERY.json", sensor_row_receipt(row))
    atomic_receipt(
        destination / "EXTRACTION.json",
        {**extraction, "elapsed_seconds": time.perf_counter() - query_start},
    )
    atomic_receipt(
        destination / "CAPTURE_SEAL.json",
        {
            "query_us": query_us,
            "token": TOKENS[0 if name == "road" else 1],
            "artifacts": {p.name: file_sha(p) for p in destination.iterdir() if p.is_file()},
        },
    )
    return query_us, original


def run_query(name: str, runtime: FinalTransferPredictor, args: argparse.Namespace) -> dict:
    destination = args.output / name
    query_us, original = capture_query(name, runtime, args, destination)
    seal = read(destination / "CAPTURE_SEAL.json")
    verify_artifacts(destination, seal["artifacts"])
    calls = read(destination / "CAPTURE_CALLS.json")["calls"]
    if any(not item["source_accepted"] for call in calls for item in call["observations"]):
        raise ValueError("Original requested cohort contains rejected rows; capture preserved")
    arrays = read_arrays(destination / "CAPTURE_ARRAYS.npz")
    original_disk = read_arrays(destination / "ORIGINAL_CONTEXT.npz")
    exact_arrays(original_disk, original, "original capture disk roundtrip")
    clean = replay_calls(runtime.models, calls, arrays, query_us, 1.0, args.device)
    clean_ttc, clean_context = save_execution(
        destination / "a1", clean, runtime, query_us, args.device
    )
    exact_arrays(clean["arrays"], arrays, "same-batch speed1 producers")
    exact_arrays(clean_context, original_disk, "same-batch speed1 contexts/head")
    requested = [o["observation_id"] for call in calls for o in call["observations"]]
    clean_native = native_vector(clean_context, args.device)
    report = {
        "identity_exact": True,
        "compared_producer_arrays": len(arrays),
        "compared_context_arrays": len(clean_context),
        "clean_frozen_F0_seconds": clean_ttc,
        "requested_ids": requested,
        "factors": {},
    }
    atomic_receipt(
        destination / "IDENTITY_PARITY.json", {k: v for k, v in report.items() if k != "factors"}
    )
    print(f"{name}: a1 same-batch producer and context/head EXACT", flush=True)
    for speed, label in ((0.5, "a0p5"), (2.0, "a2")):
        warped = replay_calls(runtime.models, calls, arrays, query_us, speed, args.device)
        value, context = save_execution(destination / label, warped, runtime, query_us, args.device)
        changed = set(warped["selected_ids"]) != set(clean["selected_ids"])
        if changed:
            control = replay_calls(
                runtime.models,
                calls,
                arrays,
                query_us,
                1.0,
                args.device,
                only_ids=set(warped["selected_ids"]),
            )
            control_value, control_context = save_execution(
                destination / (label + "_matched_clean"), control, runtime, query_us, args.device
            )
        else:
            control, control_value, control_context = clean, clean_ttc, clean_context
        controls = [
            (r["original_call_id"], [o["observation_id"] for o in r["observations"]])
            for r in control["calls"]
        ]
        warped_groups = [
            (r["original_call_id"], [o["observation_id"] for o in r["observations"]])
            for r in warped["calls"]
        ]
        if controls != warped_groups:
            raise AssertionError("Matched clean control changed retained producer batch grouping")
        warped_native = native_vector(context, args.device)
        control_native = native_vector(control_context, args.device)
        result = {
            "speed": speed,
            "stage_by_observation": compare_stages(warped, control, speed),
            "control": "matched_original_clock_subset_rerun" if changed else "a1_full_same_batch",
            "matched_batch_groups_exact": True,
            "retained_physical_ids": warped["selected_ids"],
            "accepted_producer_ids": warped["accepted_ids"],
            "control_accepted_ids": control["accepted_ids"],
            "head_comparison_isolating": set(warped["accepted_ids"])
            == set(control["accepted_ids"]),
            "native_order": ["F0_frozen", "RGB_context", "event_context"],
            "native_ttc_ratio_against_matched_control": finite_ratio(
                speed * warped_native, control_native
            ),
            "native_ttc_ratio_against_full_clean_descriptive": finite_ratio(
                speed * warped_native, clean_native
            ),
            "frozen_F0_seconds": value,
            "control_frozen_F0_seconds": control_value,
            "discarded_by_clock_budget": warped["discarded"],
            "scope": (
                "Equivariance diagnostic, no accuracy or promotion; "
                "source tensors transformed, not raw-stream re-encoding"
            ),
        }
        atomic_receipt(destination / label / "DIAGNOSTIC.json", result)
        report["factors"][label] = result
        print(
            f"{name}: {label} retained={len(warped['selected_ids'])} "
            f"accepted={len(warped['accepted_ids'])}",
            flush=True,
        )
    atomic_receipt(destination / "COMPLETE.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {
        "upstream": "output/phase_submission/FINAL_MODEL/upstream",
        "old-pilot": "output/phase_observation_cache/pilot/completed",
        "expanded": "output/phase_observation_cache/expanded149",
        "source-closure": (
            "output/phase_observation_cache/expanded149_sourceclosure/SOURCE_CLOSURE.json"
        ),
        "slider-root": "output/phase_campaign/Slider500/inputs",
        "output": "output/phase_timewarp/pilot_v1",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(
        p.name not in {"FREEZE.json", "PREFLIGHT.json", "executed_source"}
        for p in args.output.iterdir()
    ):
        raise ValueError("Existing execution/failure evidence is immutable; use a new output")
    old = read(args.old_pilot / "FREEZE.json")
    if old["selected_tokens"] != TOKENS:
        raise ValueError("Original two-query identity changed")
    verify_artifacts(args.old_pilot, read(args.old_pilot / "COMPLETE.json")["artifacts"])
    expanded = read(args.expanded / "COMPLETE.json")
    verify_artifacts(args.expanded, expanded["artifacts"])
    validate_source_closure(args.source_closure, args.expanded)
    paths = verified_upstream_paths(args.upstream)
    upstream = read(args.upstream / "BUNDLE.json")["parent"]
    if upstream != old["upstream"]:
        raise ValueError("Frozen nine-component model differs")
    sources = (
        set(map(Path, DEPENDENCIES))
        | set(Path("operational/phase_timewarp").glob("*.py"))
        | {
            Path(__file__).resolve(),
            Path("tests/unit/test_phase_timewarp_capture_pilot.py"),
            Path("tests/unit/test_phase_timewarp.py"),
            Path("scripts/final_train40_transfer.py"),
            Path("scripts/build_phase_native_context_augmentations.py"),
            Path("scripts/pilot_phase_observation_cache.py"),
            Path("operational/phase_observation_cache/capture.py"),
            Path("operational/phase_observation_cache/replay.py"),
            Path("operational/evttc_competition/runtime.py"),
            Path("operational/evttc_competition/prepare.py"),
            Path("operational/evttc_competition/geometry.py"),
            Path("operational/evttc_competition/readers.py"),
            Path("operational/phase_robust/cache.py"),
            Path("src/e_jepa_ttc/data/eap_representation.py"),
            Path("src/e_jepa_ttc/efficient_context/mapped_union.py"),
            Path("src/e_jepa_ttc/representations/voxel_grid.py"),
            Path("src/e_jepa_ttc/data/types.py"),
            Path("docs/superpowers/plans/2026-10-31-phase-producer-timewarp-pilot.md"),
        }
    )
    hashes = {str(p.resolve().relative_to(Path.cwd())): file_sha(p) for p in sorted(sources)}
    input_paths = [
        args.old_pilot / "COMPLETE.json",
        args.old_pilot / "FREEZE.json",
        args.expanded / "COMPLETE.json",
        args.source_closure,
        args.source_closure.parent / "SNAPSHOT.json",
    ]
    freeze = {
        "schema": "phase_producer_timewarp_pilot_v1",
        "tokens": TOKENS,
        "speeds": [0.5, 1, 2],
        "device": args.device,
        "seed": 7,
        "threads": 2,
        "tf32": False,
        "torch": torch.__version__,
        "upstream": upstream,
        "source_hashes": hashes,
        "inputs": {str(p): file_sha(p) for p in input_paths},
        "sensor_hashes": old["slider_sensor_sha256"],
        "phase_delta_seconds": 0.1,
        "targets_read": False,
        "optimizer_updates": 0,
        "scope": (
            "Fixed captured requested cohort variable cadence; "
            "original frozen nine-component F0, not adapted geometry router"
        ),
        "domain_rejection": (
            "Preserve original retained batch; record any batch rejection explicitly"
        ),
        "raw_representation": (
            "Events0..10 fixed,11 duration-rate transform; no raw re-encoding claim"
        ),
    }
    for name, digest in freeze["sensor_hashes"].items():
        if file_sha(args.slider_root / "Slider-500" / name) != digest:
            raise ValueError("Slider source changed")
    atomic_receipt(args.output / "FREEZE.json", freeze)
    for relative, digest in hashes.items():
        destination = args.output / "executed_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if file_sha(destination) != digest:
                raise ValueError("Archived execution source changed")
        else:
            shutil.copyfile(relative, destination)
    atomic_receipt(
        args.output / "PREFLIGHT.json",
        {
            "status": "PASS",
            "targets_read": False,
            "producer_calls": 0,
            "freeze_sha256": file_sha(args.output / "FREEZE.json"),
        },
    )
    if args.preflight_only:
        print(
            "PREFLIGHT PASS original pilot/149/sourceclosure/model/sensor bindings; "
            "no GPU/model forward",
            flush=True,
        )
        return
    if (args.output / "COMPLETE.json").exists() or (args.output / "road").exists():
        raise ValueError("Pilot execution already exists; preserve it and use a new output")
    configure_cuda()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    try:
        runtime = FinalTransferPredictor(paths, args.device)
        reports = {name: run_query(name, runtime, args) for name in ("road", "slider500")}
        for path, digest in {**hashes, **freeze["inputs"]}.items():
            if file_sha(Path(path)) != digest:
                raise ValueError("Frozen source or input changed")
        atomic_receipt(
            args.output / "COMPLETE.json",
            {
                "status": "COMPLETE",
                "queries": 2,
                "identity_exact": True,
                "targets_read": False,
                "optimizer_updates": 0,
                "freeze_sha256": file_sha(args.output / "FREEZE.json"),
                "reports": reports,
                "elapsed_seconds": time.perf_counter() - started,
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
            {
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "targets_read": False,
                "optimizer_updates": 0,
            },
        )
        raise
    print("COMPLETE two-query producer timewarp; no training or targets", flush=True)


if __name__ == "__main__":
    main()
