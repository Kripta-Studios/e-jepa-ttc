"""Paced raw replay of one sensor sequence through the complete TRAIN40 service.

This measures an artificial distinct-ROI capacity workload, not twenty tracked
objects or physical sensor arrival latency. Disk acquisition stays inside the
deadline-to-host-TTC clock. The optional original-oracle parity pass runs later.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes  # noqa: E402
from operational.evttc_rgb_transfer.inputs import _rgb_box  # noqa: E402
from operational.evttc_transfer.inputs import (  # noqa: E402
    EvTTCEventReader,
    _calibration,
    _calibration_sha256,
    _canonical_sha256,
    project_rgb_box_rotation_only,
)
from operational.rgb_port import transfer as engine  # noqa: E402
from operational.rgb_port_revision.transfer_inputs import sensor_triplets  # noqa: E402
from operational.train40_runtime.measurement import (  # noqa: E402
    PREDICTION_COMPONENTS,
    prediction_parity,
    snapshot_files,
    verify_snapshot,
)
from operational.train40_runtime.multiobject import (  # noqa: E402
    MultiObjectTrain40Service,
    ObjectRegion,
    RawRGBHistory,
)
from operational.train40_runtime.paced import (  # noqa: E402
    Arrival,
    pacing_clock_metadata,
    run_paced_replay,
)
from operational.train40_runtime.replay_inputs import SCHEMA  # noqa: E402
from scripts.benchmark_train40_multiobject import (  # noqa: E402
    artificial_regions,
    prime,
    reference_receipt,
    reference_rgb,
)
from scripts.benchmark_train40_runtime import (  # noqa: E402
    _candidate_inventory,
    _source_inventory,
    _validate_output,
    _validate_reference,
    _write,
)
from scripts.final_train40_transfer import (  # noqa: E402
    FinalTransferPredictor,
    load_models,
    read,
    reject_targets,
    sha,
    verify_candidate,
    verify_frozen_runtime,
)

STEP_US = 20_000
EVENT_DTYPES = {"x": np.int32, "y": np.int32, "t": np.int64, "p": np.int8}
SCOPE = "PACED_DISK_RAW_REPLAY_ARTIFICIAL_DISTINCT_ROIS_ALL_HOST_TTC"
MILESTONE_CHECKS = (
    "all_scheduled_arrivals_recorded",
    "no_drops_or_source_errors",
    "complete_object_coverage",
    "actual_availability_before_service",
    "every_deadline_response_under_20ms",
    "no_prior_work_at_scheduled_arrivals",
    "finished_before_end_of_last_period",
    "offered_50hz_for_at_least_60s",
)


def replay_schedule(
    manifest: Mapping[str, Any], sequence_id: str, updates: int | None = None
) -> list[dict[str, Any]]:
    """Retain excluded slots at their real clock positions; never close gaps."""
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("targets_read") is not False
        or manifest.get("optimizer_updates") != 0
        or manifest.get("provenance", {}).get("step_us") != STEP_US
    ):
        raise ValueError("A label-free causal replay manifest with a 20000us step is required")
    inventory = [
        item for item in manifest["inventory"] if item["physical_sequence_id"] == sequence_id
    ]
    if len(inventory) != 1:
        raise ValueError("Select exactly one registered physical sequence")
    registered = inventory[0]
    accepted = [row for row in manifest["rows"] if row["physical_sequence_id"] == sequence_id]
    excluded = [row for row in manifest["excluded"] if row["physical_sequence_id"] == sequence_id]
    count = int(registered["scheduled_updates"])
    if count <= 0 or len(accepted) + len(excluded) != count:
        raise ValueError("Every scheduled slot must be accepted or explicitly excluded")
    first = int(registered["requested_bounds_us"][0])
    slots: dict[int, dict[str, Any]] = {}
    raw_path = Path(registered["raw_path"]).resolve()
    for row in [*accepted, *excluded]:
        anchor = int(row["anchor_us"])
        if anchor in slots:
            raise ValueError("Duplicate accepted/excluded sensor-clock slot")
        if "reason" not in row:
            unsigned = {key: value for key, value in row.items() if key != "metadata_sha256"}
            if row.get("metadata_sha256") != _canonical_sha256(unsigned):
                raise ValueError("Replay row metadata digest differs")
            if (
                Path(row["raw_path"]).resolve() != raw_path
                or row["raw_stat"] != registered["raw_stat"]
            ):
                raise ValueError("Replay rows must identify one unchanged physical raw source")
            if row.get("replay_step_us") != STEP_US:
                raise ValueError("Replay row cadence differs")
            reject_targets(row)
        slots[anchor] = dict(row)
    expected = [first + index * STEP_US for index in range(count)]
    if set(slots) != set(expected):
        raise ValueError("Replay has an unaccounted clock gap, rollback or out-of-range row")
    if updates is not None and (
        isinstance(updates, bool) or not isinstance(updates, int) or not 1 <= updates <= count
    ):
        raise ValueError("updates must select a positive prefix of the declared schedule")
    return [slots[anchor] for anchor in expected[:updates]]


def _own_readonly(value: Any, dtype: Any = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def bounded_events(
    reader: EvTTCEventReader, start_us: int, end_us: int, max_bytes: int
) -> Mapping[str, np.ndarray]:
    """Reject capacity before allocating output; preserve every event or fail."""
    first, last = reader._bounds(start_us, end_us)
    count = last - first
    if count * 17 > max_bytes:
        raise ValueError("RAW_EVENT_BYTE_LIMIT: packet/history cannot be truncated")
    result = {key: np.empty(count, dtype=dtype) for key, dtype in EVENT_DTYPES.items()}
    cursor = 0
    for chunk in reader.iter_window_chunks(start_us, end_us):
        length = len(chunk["t"])
        if cursor + length > count:
            raise ValueError("Raw source changed while reading the bounded interval")
        for key in result:
            result[key][cursor : cursor + length] = chunk[key]
        cursor += length
    if cursor != count:
        raise ValueError("Raw source changed while reading the bounded interval")
    for value in result.values():
        value.setflags(write=False)
    return MappingProxyType(result)


@dataclass(frozen=True)
class RawUpdate:
    """Owned immutable CPU bytes; no open reader, ROI transform or representation."""

    index: int
    row_json: bytes
    events: Mapping[str, np.ndarray]
    rgb: RawRGBHistory
    boxes: tuple[tuple[float, float, float, float], ...]
    calibration: Mapping[str, np.ndarray]
    byte_count: int
    acquisition_thread: int


class DiskRawSource:
    """Open/close every HDF5 handle on the acquiring thread, including failures."""

    def __init__(self, schedule: Sequence[Mapping[str, Any]], max_payload_bytes: int) -> None:
        if max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")
        self.schedule = schedule
        self.max_payload_bytes = max_payload_bytes

    def acquire_raw(self, index: int) -> RawUpdate:
        """Read only the current raw packet, causal RGB frames and bbox metadata."""
        row = self.schedule[index]
        if "reason" in row:
            raise ValueError(f"MANIFEST_EXCLUSION: {row['reason']}")
        query = int(row["anchor_us"])
        sources = row["bbox_sources"]
        if len(sources) != 3 or any(
            not 0 <= int(item["timestamp_us"]) <= int(item["available_us"]) <= query
            for item in sources
        ):
            raise ValueError("Future or invalid bbox metadata is unavailable at this query")
        row_json = json.dumps(dict(row), sort_keys=True, allow_nan=False).encode()
        if len(row_json) > min(self.max_payload_bytes, 1024**2):
            raise ValueError("RAW_METADATA_BYTE_LIMIT")
        if any(Path(item["path"]).stat().st_size > 1024**2 for item in sources):
            raise ValueError("RAW_ANNOTATION_BYTE_LIMIT")
        # Validation reads metadata only; projection and square construction are
        # deliberately postponed until the consuming thread's inference clock.
        boxes = tuple(_rgb_box(item) for item in sources)
        path = Path(row["raw_path"])
        before = path.stat()
        if any(
            row["raw_stat"][key] != value
            for key, value in (("size_bytes", before.st_size), ("mtime_ns", before.st_mtime_ns))
        ):
            raise ValueError("Raw source changed since replay manifest generation")
        with EvTTCEventReader(path) as reader:
            handle = reader._require_handle()
            calibration = {key: _own_readonly(value) for key, value in _calibration(handle).items()}
            if _calibration_sha256(calibration) != row["calibration_sha256"]:
                raise ValueError("Raw calibration differs from replay selection")
            used = len(row_json) + sum(value.nbytes for value in calibration.values())
            events = bounded_events(reader, query - STEP_US, query, self.max_payload_bytes - used)
            used += sum(value.nbytes for value in events.values())
            data, stamps = handle.get("blackflys/left/data"), handle.get("blackflys/left/ts")
            frames: dict[int, np.ndarray] = {}
            timestamps: dict[int, int] = {}
            groups: list[tuple[int, ...]] = []
            reason = "RGB_DATASET_MISSING"
            if isinstance(data, h5py.Dataset) and isinstance(stamps, h5py.Dataset):
                if stamps.ndim != 1 or len(stamps) * 8 > self.max_payload_bytes:
                    raise ValueError("RAW_CLOCK_BYTE_LIMIT")
                clock = np.asarray(stamps, dtype=np.int64)
                groups = [
                    tuple(int(item) for item in group) for group in sensor_triplets(clock, query)
                ]
                ids = sorted({item for group in groups for item in group})
                if (
                    len(ids) > 10
                    or data.dtype != np.uint8
                    or len(data.shape) != 4
                    or data.shape[-1] != 3
                ):
                    raise ValueError("Raw RGB must contain bounded uint8 HWC frames")
                frame_bytes = int(np.prod(data.shape[1:]))
                if used + frame_bytes * len(ids) > self.max_payload_bytes:
                    raise ValueError("RAW_PAYLOAD_BYTE_LIMIT: RGB frames cannot be truncated")
                for frame_id in ids:
                    frames[frame_id] = _own_readonly(data[frame_id], np.uint8)
                    timestamps[frame_id] = int(clock[frame_id])
                used += sum(value.nbytes for value in frames.values())
                reason = "" if groups else "INSUFFICIENT_CAUSAL_RGB_HISTORY"
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Raw source changed during acquisition")
        if used > self.max_payload_bytes:
            raise ValueError("RAW_PAYLOAD_BYTE_LIMIT")
        return RawUpdate(
            index,
            row_json,
            events,
            RawRGBHistory(
                query, MappingProxyType(frames), MappingProxyType(timestamps), tuple(groups), reason
            ),
            boxes,
            MappingProxyType(calibration),
            used,
            threading.get_ident(),
        )

    def initial_history(self, max_bytes: int) -> Mapping[str, np.ndarray]:
        """Read the one startup 600ms interval, before any measured arrival."""
        row = self.schedule[0]
        if "reason" in row:
            raise ValueError("Cannot initialize from an excluded first replay slot")
        end = int(row["anchor_us"]) - STEP_US
        with EvTTCEventReader(Path(row["raw_path"])) as reader:
            handle = reader._require_handle()
            clock = handle[reader.layout.t]
            if not isinstance(clock, h5py.Dataset) or not len(clock):
                raise ValueError("Raw event clock must be a nonempty dataset")
            if int(clock[0]) > end - 600_000 or int(clock[-1]) < end:
                raise ValueError("Startup requires the full preceding 600ms of sensor support")
            return bounded_events(reader, end - 600_000, end, max_bytes)


def current_regions(payload: RawUpdate, count: int) -> list[ObjectRegion]:
    """Compute current calibrated geometry inside infer_all, not in acquisition."""
    projected = [project_rgb_box_rotation_only(box, payload.calibration) for box in payload.boxes]
    event = common_square_from_boxes(projected, (1, 2), margin_fraction=0.25)
    rgb = common_square_from_boxes(payload.boxes, (1, 2), margin_fraction=0.25)
    return artificial_regions(event, rgb, count)


class ContinuousConsumer:
    """Latch gaps or failures: no hidden refilling, skipping, or stale-history TTC."""

    def __init__(self, service: Any, schedule: Sequence[Mapping[str, Any]], count: int) -> None:
        self.service, self.schedule, self.count = service, schedule, count
        self.next_index = 0
        self.failure: str | None = None
        self.service_receipts: list[dict[str, Any]] = []

    def infer_all(self, arrival: Arrival[RawUpdate]) -> dict[str, dict[str, Any]]:
        """Return all completed host outputs or fail this entire update explicitly."""
        payload = arrival.payload
        if self.failure is not None:
            raise RuntimeError(f"HISTORY_INVALID_REINITIALIZATION_REQUIRED: {self.failure}")
        try:
            if payload.index != self.next_index or arrival.update_id != f"update-{payload.index}":
                raise ValueError(
                    f"RAW_GAP: expected index {self.next_index}, received {payload.index}"
                )
            row = json.loads(payload.row_json)
            query = int(self.schedule[payload.index]["anchor_us"])
            if (
                row["query_id"] != self.schedule[payload.index]["query_id"]
                or int(row["anchor_us"]) != query
            ):
                raise ValueError("Payload identity differs from the scheduled sensor query")
            regions = current_regions(payload, self.count)
            result = self.service.update(
                payload.events, query - STEP_US, query, regions, payload.rgb
            )
            outputs = result["objects"]
            ids = [item["object_id"] for item in outputs]
            if len(ids) != self.count or set(ids) != {item.object_id for item in regions}:
                raise ValueError("Service returned duplicate, missing or unexpected object IDs")
            for item in outputs:
                if item.get("success", True) is not True:
                    raise ValueError("Service explicitly marked an object prediction unsuccessful")
                for name in PREDICTION_COMPONENTS:
                    value = item["predictions"][name]
                    if (
                        name == "R_CTX_TRAIN40"
                        and value is None
                        and item.get("rgb_available") is False
                    ):
                        continue
                    if (
                        isinstance(value, (bool, str))
                        or not isinstance(value, (float, int))
                        or not np.isfinite(value)
                    ):
                        raise ValueError(
                            "Every TTC must be a finite scalar already available on host"
                        )
            self.service_receipts.append(
                {
                    **result,
                    "update_id": arrival.update_id,
                    "query_id": row["query_id"],
                    "anchor_us": query,
                    "raw_payload_bytes": payload.byte_count,
                    "acquisition_thread": payload.acquisition_thread,
                    "inference_thread": threading.get_ident(),
                }
            )
            self.next_index += 1
            return {item["object_id"]: {**item, "success": True} for item in outputs}
        except Exception as exc:
            self.failure = f"{type(exc).__name__}: {exc}"
            raise


def separate_parity(
    source: DiskRawSource, consumer: ContinuousConsumer, models: Any, norms: Any
) -> dict[str, Any]:
    """Re-read each raw query for an independent original producer/context pass."""
    oracle = FinalTransferPredictor.__new__(FinalTransferPredictor)
    oracle.device, oracle.models, oracle.normalizers = torch.device("cuda"), models, norms
    measured = {item["update_id"]: item for item in consumer.service_receipts}
    checks = []
    for index, row in enumerate(source.schedule):
        update_id = f"update-{index}"
        try:
            payload = source.acquire_raw(index)
            regions = current_regions(payload, consumer.count)
            expected, receipts = [], []
            for region in regions:
                modified = {**row, "square_xyxy": list(region.event_square)}
                raw = engine.prepare_event(modified)
                rgb = reference_rgb(payload.rgb, region)
                with (
                    patch.object(engine, "prepare_event", return_value=raw),
                    patch.object(
                        engine,
                        "raw_rgb_observations",
                        return_value=(rgb, payload.rgb.unavailable_reason),
                    ),
                ):
                    prediction = oracle.predict(modified)
                expected.append({**prediction, "query_id": region.object_id})
                receipts.append(reference_receipt(modified, region, rgb))
            outputs = measured.get(update_id, {}).get("objects", [])
            parity = prediction_parity(
                expected,
                [{**item, "query_id": item["object_id"]} for item in outputs],
                [item.object_id for item in regions],
            )
            receipt_equal = len(outputs) == len(receipts) and all(
                item.get("input_receipt") == receipt
                for item, receipt in zip(outputs, receipts, strict=True)
            )
            checks.append(
                {
                    "update_id": update_id,
                    "parity": parity,
                    "input_receipts_exact": receipt_equal,
                    "passed": parity["passed"] and receipt_equal,
                }
            )
        except Exception as exc:
            checks.append(
                {"update_id": update_id, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
            )
    return {
        "performed": True,
        "passed": bool(checks) and all(item["passed"] for item in checks),
        "scope": "SEPARATE_ORIGINAL_ORACLE_ALL_SCHEDULED_UPDATES_AND_OBJECTS",
        "updates": checks,
    }


def capacity_admission(
    paced: Mapping[str, Any] | None,
    parity: Mapping[str, Any],
    originals_passed: bool,
    *,
    target_object_count: int = 20,
) -> dict[str, Any]:
    """Version the 1/2-object milestones without changing the legacy 20-object gate."""
    if (
        isinstance(target_object_count, bool)
        or not isinstance(target_object_count, int)
        or not 1 <= target_object_count <= 20
    ):
        raise ValueError("target_object_count must be one to twenty")
    complete = bool(
        paced
        and paced["grouped"]["passed"]
        and paced["dropped_updates"] == 0
        and paced["source_errors"] == 0
        and paced["capacity_checks"]["all_scheduled_arrivals_recorded"]
    )
    checks = {
        name: bool(paced and paced.get("capacity_checks", {}).get(name) is True)
        for name in MILESTONE_CHECKS
    }
    ids = paced.get("object_ids", []) if paced else []
    records = paced.get("object_records", []) if paced else []
    checks["required_object_count"] = len(ids) == len(set(ids)) == target_object_count
    checks["all_three_host_ttc_per_required_object"] = bool(records) and all(
        record.get("success") is True
        and all(
            isinstance(record.get("result", {}).get("predictions", {}).get(name), (int, float))
            and not isinstance(record["result"]["predictions"][name], bool)
            and np.isfinite(record["result"]["predictions"][name])
            for name in PREDICTION_COMPONENTS
        )
        for record in records
    )
    admitted = bool(
        complete
        and all(checks.values())
        and parity.get("performed") is True
        and parity.get("passed") is True
        and originals_passed
    )
    return {
        "schema": "train40_paced_milestone_gate_v2",
        "target_object_count": target_object_count,
        "milestone_checks": checks,
        "complete_measurement_coverage": complete,
        "milestone_with_parity_passed": admitted,
        "capacity_with_parity_passed": admitted and target_object_count == 20,
        "legacy_twenty_object_capacity_checks_passed": bool(
            paced and paced.get("capacity_checks_passed")
        ),
    }


def run(args: argparse.Namespace) -> None:
    """Write partial evidence and original-file preservation even on failure."""
    _validate_output(
        args.output,
        [args.candidate, args.reference, args.manifest.parent, args.base_manifest.parent],
    )
    args.output.mkdir(parents=True, exist_ok=False)
    original: dict[str, Any] = {}
    service = None
    caught: BaseException | None = None
    paced: dict[str, Any] | None = None
    parity: dict[str, Any] = {
        "performed": False,
        "passed": False,
        "reason": "SEPARATE_PARITY_NOT_RUN",
    }
    consumer = None
    schedule: list[dict[str, Any]] = []
    startup: dict[str, Any] = {}
    try:
        if args.queue_capacity < 1 or args.max_initial_bytes < 1:
            raise ValueError("Queue capacity and initial raw byte budget must be positive")
        if args.backend != "prefix" and (
            args.event_query_graphs or args.event_normalization != "sort"
        ):
            raise ValueError("Event query graphs/topk require the prefix backend")
        paths, candidate_binding = verify_candidate(args.candidate)
        frozen = verify_frozen_runtime(args.candidate)
        base = read(args.base_manifest)
        _, reference_files, reference_binding = _validate_reference(
            args.reference, args.base_manifest, base["rows"], candidate_binding
        )
        manifest = read(args.manifest)
        if manifest["provenance"]["base_manifest_canonical_sha256"] != _canonical_sha256(base):
            raise ValueError("Replay provenance does not bind the preserved base manifest")
        schedule = replay_schedule(manifest, args.sequence, args.updates)
        source = DiskRawSource(schedule, args.max_payload_bytes)
        if args.backend == "prefix":
            from operational.train40_runtime.temporal_prefix import TemporalPrefixEventHistory

            history_class = TemporalPrefixEventHistory
        else:
            from operational.train40_runtime.integral import IntegralEventHistory

            history_class = IntegralEventHistory
        inventory = _source_inventory(frozen, reference_binding)
        sources = [
            path
            for path in inventory
            if path.parent != ROOT / "operational/train40_runtime"
            or f"operational.train40_runtime.{path.stem}" in sys.modules
        ]
        sources.extend(
            [Path(__file__).resolve(), ROOT / "scripts/benchmark_train40_multiobject.py"]
        )
        annotation_files = [
            Path(item["path"]) for row in schedule for item in row.get("bbox_sources", [])
        ]
        original = snapshot_files(
            [
                *_candidate_inventory(args.candidate),
                *reference_files,
                args.manifest,
                args.base_manifest,
                *sources,
                *annotation_files,
            ]
        )
        _write(args.output / "original_snapshot.json", original)
        source_copies = args.output / "sources"
        source_copies.mkdir()
        for path in sorted(set(sources)):
            target = source_copies / (
                str(path.relative_to(ROOT)).replace("\\", "__").replace("/", "__") + ".txt"
            )
            target.write_bytes(path.read_bytes())
            if sha(target) != sha(path):
                raise ValueError("Source changed while preserving exact bytes")
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(7)
        binding = {
            "schema": "train40_paced_service_binding_v1",
            "scope": SCOPE,
            "candidate": candidate_binding,
            "manifest_sha256": sha(args.manifest),
            "base_manifest_sha256": sha(args.base_manifest),
            "sources": {str(path.relative_to(ROOT)): sha(path) for path in sorted(set(sources))},
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "precision": "FP32_TF32_OFF",
            "scheduler": pacing_clock_metadata(args.wait_strategy),
            "seed": 7,
            "sequence": args.sequence,
            "objects": args.objects,
            "scheduled_updates": len(schedule),
            "excluded_slots_retained": sum("reason" in row for row in schedule),
            "configuration": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "initialization_excluded": "preceding 600ms, model loading, synthetic shape priming",
            "raw_io_included": True,
            "roi_geometry_in_inference": True,
            "precomputed_representations": False,
            "queue_byte_upper_bound": args.queue_capacity * args.max_payload_bytes,
            "live_payload_byte_upper_bound": (args.queue_capacity + 3) * args.max_payload_bytes,
            "memory_scope": (
                "Owned queue/consumer/producer previous/producer assembling payload bytes; "
                "excludes model, ring, Python receipts, HDF5 and bounded read scratch"
            ),
            "raw_event_timestamp_scratch_limit_bytes": 256 * 1024**2,
            "actual_peak_queue_bytes_measured": False,
            "twenty_real_tracks": False,
            "annotation_availability_measured": False,
            "physical_goal_demonstrated": False,
            "raw_source_integrity_scope": "size/mtime plus calibration digest; no full RAW hash",
        }
        _write(args.output / "binding.json", binding)
        begin = time.perf_counter()
        models, norms = load_models(paths, "cuda")
        options: dict[str, Any] = {
            "device": "cuda",
            "max_objects": args.objects,
            "roi_batch_size": min(4, args.objects),
        }
        if args.backend == "prefix":
            options.update(graphs=args.event_query_graphs, normalization=args.event_normalization)
        elif args.event_query_graphs or args.event_normalization != "sort":
            raise ValueError("Event query graphs/topk require the prefix backend")
        history: Any = history_class(**options)
        if args.event_query_graphs:
            history.prime_query_graphs([args.objects])
        service = MultiObjectTrain40Service(
            models,
            norms,
            history,
            max_objects=args.objects,
            reuse_event_encodings=False,
            rgb_object_chunk_size=1,
            shared_rgb_images=True,
            parallel_branches=True,
            rgb_workers=args.rgb_workers,
        )
        startup["model_and_service_seconds"] = time.perf_counter() - begin
        startup["synthetic_shape_priming_seconds"] = prime(service, args.objects)
        startup["synthetic_primed_rgb_triplets"] = 5
        startup["unprimed_shapes_remain_inside_measurement"] = True
        begin = time.perf_counter()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="train40-startup-raw") as pool:
            raw = pool.submit(source.initial_history, args.max_initial_bytes).result()
        end = int(schedule[0]["anchor_us"]) - STEP_US
        history.initialize(raw, end - 600_000, end)
        torch.cuda.synchronize()
        startup.update(
            history_initialization_seconds=time.perf_counter() - begin,
            initial_interval_us=[end - 600_000, end],
            initial_raw_bytes=sum(value.nbytes for value in raw.values()),
        )
        del raw
        _write(args.output / "startup.json", startup)
        consumer = ContinuousConsumer(service, schedule, args.objects)
        paced = run_paced_replay(
            source.acquire_raw,
            consumer.infer_all,
            [f"roi_{index:02d}" for index in range(args.objects)],
            updates=len(schedule),
            rate_hz=50.0,
            queue_capacity=args.queue_capacity,
            wait_strategy=args.wait_strategy,
        )
        _write(args.output / "paced.json", paced)
        _write(args.output / "service_updates.json", consumer.service_receipts)
        if args.verify_parity:
            parity = separate_parity(source, consumer, models, norms)
            _write(args.output / "parity.json", parity)
    except BaseException as exc:
        caught = exc
    finally:
        cleanup_errors = []
        if service is not None:
            try:
                service.close()
            except BaseException as exc:
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        intact = (
            verify_snapshot(original)
            if original
            else {"passed": False, "reason": "SNAPSHOT_NOT_ESTABLISHED"}
        )
        raw_checks = []
        for path, identity in {
            row["raw_path"]: row["raw_stat"] for row in schedule if "raw_path" in row
        }.items():
            try:
                stat = Path(path).stat()
                passed = (
                    identity["size_bytes"] == stat.st_size
                    and identity["mtime_ns"] == stat.st_mtime_ns
                )
                raw_checks.append({"path": path, "passed": passed})
            except OSError as exc:
                raw_checks.append({"path": path, "passed": False, "error": str(exc)})
        raw_intact = bool(raw_checks) and all(item["passed"] for item in raw_checks)
        admission = capacity_admission(
            paced, parity, intact["passed"], target_object_count=args.objects
        )
        completed = admission["complete_measurement_coverage"]
        valid = (
            completed and caught is None and not cleanup_errors and intact["passed"] and raw_intact
        )
        capacity = bool(valid and admission["capacity_with_parity_passed"])
        summary = {
            "schema": "train40_paced_service_summary_v1",
            "scope": SCOPE,
            "status": "COMPLETE"
            if valid and (not args.verify_parity or parity["passed"])
            else "FAILED",
            "paced": paced,
            "separate_parity": parity,
            "originals": intact,
            "raw_source_stat_checks": raw_checks,
            "startup": startup,
            "history_failure": consumer.failure if consumer is not None else None,
            "milestone": {
                **admission,
                "milestone_with_parity_passed": bool(
                    valid and admission["milestone_with_parity_passed"]
                ),
            },
            "capacity_with_parity_passed": capacity,
            "physical_goal_demonstrated": False,
            "target_20_objects_50hz_demonstrated": False,
            "goal_note": (
                "Versioned 1/2-object milestones require the same 60s, all <20ms, full TTC and "
                "separate parity gates; they do not imply the 20-object or physical-sensor goal"
            ),
            "cleanup_errors": cleanup_errors,
            "error": f"{type(caught).__name__}: {caught}" if caught else None,
            "targets_read": False,
            "optimizer_updates": 0,
        }
        if consumer is not None:
            _write(args.output / "service_updates.json", consumer.service_receipts)
        _write(args.output / "summary.json", summary)
    if caught is not None:
        raise caught
    if not valid or (args.verify_parity and not parity["passed"]):
        raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    """Build the explicit bounded replay CLI without initializing CUDA."""
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "reference", "base-manifest", "manifest", "output"):
        result.add_argument(f"--{name}", type=Path, required=True)
    result.add_argument(
        "--sequence", required=True, help="One physical sequence ID from the causal replay manifest"
    )
    result.add_argument("--objects", type=int, choices=range(1, 21), default=20)
    result.add_argument("--updates", type=int, help="Declared prefix only; no sampling or looping")
    result.add_argument("--queue-capacity", type=int, default=2)
    result.add_argument(
        "--wait-strategy",
        choices=("event", "sleep"),
        default="sleep",
        help="Absolute-deadline waiting; event preserves the legacy scheduler for A/B comparisons",
    )
    result.add_argument("--max-payload-bytes", type=int, default=128 * 1024**2)
    result.add_argument("--max-initial-bytes", type=int, default=512 * 1024**2)
    result.add_argument("--backend", choices=("integral", "prefix"), default="prefix")
    result.add_argument("--event-query-graphs", action="store_true")
    result.add_argument("--event-normalization", choices=("sort", "topk"), default="sort")
    result.add_argument("--rgb-workers", type=int, choices=range(1, 9), default=8)
    result.add_argument(
        "--verify-parity",
        action="store_true",
        help="After timing, run original oracle for ALL scheduled queries/ROIs (potentially slow)",
    )
    return result


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
