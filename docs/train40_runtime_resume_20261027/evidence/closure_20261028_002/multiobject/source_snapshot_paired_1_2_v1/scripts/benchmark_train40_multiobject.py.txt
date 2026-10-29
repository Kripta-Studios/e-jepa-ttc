"""Complete last-packet TRAIN40 pilot using distinct artificial regions.

Each population shares one real raw sensor history. Object zero is the original
annotated ROI; the others are deterministic translated regions, not annotated
tracks. Every declared region receives its own original-model parity check.
History rebuilding, raw disk loading and synthetic shape priming are excluded.
Current ROI work, the new raw packet, all transfers/models and all TTC outputs
are included. This pilot does not establish sustained 50 Hz operation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes  # noqa: E402
from e_jepa_ttc.rgb_port.data import crop_uint8  # noqa: E402
from operational.evttc_transfer.inputs import EvTTCEventReader  # noqa: E402
from operational.rgb_port import transfer as engine  # noqa: E402
from operational.train40_runtime.inputs import PersistentSensorInputs  # noqa: E402
from operational.train40_runtime.integral import IntegralEventHistory  # noqa: E402
from operational.train40_runtime.measurement import (  # noqa: E402
    latency_distribution,
    prediction_parity,
    snapshot_files,
    verify_snapshot,
)
from operational.train40_runtime.multiobject import (  # noqa: E402
    MultiObjectTrain40Service,
    ObjectRegion,
    RawRGBHistory,
    Square,
    read_raw_rgb_history,
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

SCOPE = "ARTIFICIAL_DISTINCT_ROIS_SHARED_REAL_RAW_LAST_PACKET_COMPLETE_UPDATE"


def imported_runtime_paths() -> set[Path]:
    """Snapshot actual runtime module files, excluding unrelated experiments."""
    runtime_root = ROOT / "operational/train40_runtime"
    paths = set()
    for name, module in tuple(sys.modules.items()):
        if name == "operational.train40_runtime" or name.startswith("operational.train40_runtime."):
            filename = getattr(module, "__file__", None)
            if filename is not None:
                path = Path(filename).resolve()
                if path.parent == runtime_root:
                    paths.add(path)
    return paths


def artificial_regions(event_square: Any, rgb_square: Any, count: int) -> list[ObjectRegion]:
    """Keep the original ROI first; translate every other region deterministically."""
    if not 1 <= count <= 20:
        raise ValueError("The registered capacity population is one to twenty regions")
    result = []
    for index in range(count):
        dx, dy = (index % 5) * 17, (index // 5) * 13
        event = tuple(
            float(value) + (dx if axis % 2 == 0 else dy) for axis, value in enumerate(event_square)
        )
        rgb = tuple(
            float(value) + (dx if axis % 2 == 0 else dy) for axis, value in enumerate(rgb_square)
        )
        result.append(ObjectRegion(f"roi_{index:02d}", cast(Square, event), cast(Square, rgb)))
    if len({item.event_square for item in result}) != count:
        raise ValueError("Artificial load must contain distinct regions")
    return result


def reference_rgb(
    history: RawRGBHistory, region: ObjectRegion
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """Use the original PIL crop adapter independently of the optimized workers."""
    if region.rgb_square is None:
        return []
    frames = {
        index: crop_uint8(frame, region.rgb_square) for index, frame in history.frames.items()
    }
    result = []
    for group in history.triplets:
        times = [history.timestamps_us[index] for index in group]
        pixels = np.ascontiguousarray(
            np.stack([frames[index] for index in group]).astype(np.float32) / 255
        )
        result.append(
            (
                pixels,
                {
                    "first_sensor_ts_us": times[0],
                    "anchor_us": times[-1],
                    "available_us": history.query_us,
                    "timestamps_us": times,
                    "frame_indices": list(group),
                    "delta_t_s": (np.diff(times) / 1e6).tolist(),
                    "square_xyxy": list(region.rgb_square),
                },
            )
        )
    return result


def reference_receipt(row: dict[str, Any], region: ObjectRegion, rgb: list[Any]) -> dict[str, Any]:
    """Bind every ROI, clock, frame and window, beyond history-length parity."""
    anchor = int(row["anchor_us"])
    windows = np.asarray(row["windows_us"], dtype=np.int64)
    return {
        "query_us": anchor,
        "event_square": list(region.event_square),
        "rgb_square": list(region.rgb_square) if region.rgb_square is not None else None,
        "event_windows_us": (windows[None] - engine.EVENT_LAGS_US[:, None, None]).tolist(),
        "rgb_observations": [dict(metadata) for _, metadata in rgb],
    }


def prime(service: MultiObjectTrain40Service, count: int) -> float:
    """Capture standard shapes using synthetic zero data before sensor queries."""
    query = 1_000_000
    objects = artificial_regions((0, 0, 128, 128), (0, 0, 128, 128), count)
    rgb = [
        [
            (
                np.zeros((3, 3, 128, 128), dtype=np.float32),
                {
                    "delta_t_s": [0.1, 0.1],
                    "first_sensor_ts_us": query - (6 - i) * 100_000,
                    "anchor_us": query - (4 - i) * 100_000,
                    "available_us": query,
                },
            )
            for i in range(5)
        ]
        for _ in range(count)
    ]
    begin = time.perf_counter()
    service.infer_prepared(objects, query, torch.zeros((count, 4, 3, 12, 128, 128)), rgb)
    torch.cuda.synchronize()
    return time.perf_counter() - begin


def input_digests(service: MultiObjectTrain40Service) -> list[dict[str, Any]]:
    """Hash the actual consumed tensors after the timed update, not reconstructions."""
    assert service.last_event_inputs is not None
    event = service.last_event_inputs.cpu().numpy()
    receipt = service.last_rgb_cache_receipt or {}
    prepared = service.last_rgb_prepared_inputs
    pixel_hashes: dict[int, list[str]] = {}
    if prepared is not None and prepared.pixels is not None:
        prepared_pixels = prepared.pixels.cpu().numpy()
        position = 0
        for index, metadata in enumerate(prepared.metadata):
            pixel_hashes[index] = [
                hashlib.sha256(np.ascontiguousarray(prepared_pixels[position + offset])).hexdigest()
                for offset in range(len(metadata))
            ]
            position += len(metadata)
    keys = dict(
        zip(receipt.get("present_indices", []), receipt.get("key_digests", []), strict=True)
    )
    features = service.last_rgb_producer_features
    present = (
        receipt.get("present_indices", [])
        if receipt
        else [
            index
            for index, rows in enumerate(
                prepared.metadata if prepared is not None else service.last_rgb_inputs or []
            )
            if rows
        ]
    )
    feature_hashes = {}
    delta_hashes = {}
    if service.last_rgb_input_delta is not None:
        deltas = service.last_rgb_input_delta.cpu().numpy()
        groups = len(deltas) // len(present)
        delta_hashes = {
            index: hashlib.sha256(
                np.ascontiguousarray(deltas[position * groups : (position + 1) * groups])
            ).hexdigest()
            for position, index in enumerate(present)
        }
    if features is not None:
        raw, experts = (tensor.cpu().numpy() for tensor in features)
        groups = len(raw) // len(present)
        for position, index in enumerate(present):
            begin, end = position * groups, (position + 1) * groups
            feature_hashes[index] = {
                "raw": hashlib.sha256(np.ascontiguousarray(raw[begin:end])).hexdigest(),
                "experts": hashlib.sha256(np.ascontiguousarray(experts[begin:end])).hexdigest(),
            }
    return [
        {
            "event_sha256": hashlib.sha256(np.ascontiguousarray(event[index])).hexdigest(),
            "rgb_sha256": [
                hashlib.sha256(np.ascontiguousarray(pixels)).hexdigest()
                for pixels, _ in service.last_rgb_inputs[index]
            ]
            if service.last_rgb_inputs is not None
            else pixel_hashes.get(index),
            "rgb_delta_sha256": delta_hashes.get(index),
            "rgb_producer_features": feature_hashes.get(index),
            "rgb_cache_key": keys.get(index),
        }
        for index in range(len(event))
    ]


def attest_cached_inputs(timings: list[dict[str, Any]], expected_rgb: list[list[str]]) -> None:
    """Require every cache hit to match an independently checked miss in this case.

    Hits have no RGB pixel tensor and never claim one. Their actual producer
    feature hashes and full cache key must match a preceding pixel-checked miss.
    All event tensors must match the repeat independently compared numerically.
    """
    ledger: dict[str, dict[str, str]] = {}
    checked_event = [item["event_sha256"] for item in timings[-1]["input_digests"]]
    for result in timings:
        checks = []
        for position, digest in enumerate(result["input_digests"]):
            pixels, key = digest["rgb_sha256"], digest["rgb_cache_key"]
            features = digest["rgb_producer_features"]
            if pixels is not None:
                valid = pixels == expected_rgb[position]
                if valid and key is not None and features is not None:
                    ledger[key] = features
            elif not expected_rgb[position]:
                valid = features is None and key is None
            else:
                valid = key is not None and key in ledger and ledger[key] == features
            checks.append(valid)
        result["rgb_input_provenance_passed"] = checks
        result["same_inputs_as_tensor_checked_repeat"] = (
            all(checks)
            and [item["event_sha256"] for item in result["input_digests"]] == checked_event
        )


def cache_exercise_receipt(timings: list[dict[str, Any]]) -> dict[str, Any]:
    """A cache experiment must exercise stored-miss to actual-feature-hit reuse."""
    receipts = [row.get("rgb_feature_cache") or {} for row in timings]
    statuses = [receipt.get("status") for receipt in receipts]
    hits = statuses.count("HIT")
    eligible = any(receipt.get("present_objects", 0) > 0 for receipt in receipts)
    passed = bool(receipts) and (
        (
            statuses[0] == "MISS"
            and receipts[0].get("stored") is True
            and hits > 0
            and all(
                all(digest["rgb_sha256"] is None for digest in row["input_digests"])
                for row, status in zip(timings, statuses, strict=True)
                if status == "HIT"
            )
        )
        if eligible
        else all(status == "NO_RGB" for status in statuses)
    )
    return {
        "passed": passed,
        "eligible": eligible,
        "statuses": statuses,
        "hit_updates": hits,
        "miss_updates": statuses.count("MISS"),
        "no_rgb_updates": statuses.count("NO_RGB"),
    }


def run(args: argparse.Namespace) -> None:
    """Persist failures, partial results and immutable-candidate checks."""
    _validate_output(args.output, [args.candidate, args.reference, args.manifest.parent])
    args.output.mkdir(parents=True, exist_ok=False)
    paths, binding = verify_candidate(args.candidate)
    frozen = verify_frozen_runtime(args.candidate)
    rows = read(args.manifest)["rows"]
    reject_targets(rows)
    original_reference, reference_files, reference_binding = _validate_reference(
        args.reference, args.manifest, rows, binding
    )
    if (
        not args.indices
        or len(set(args.indices)) != len(args.indices)
        or any(not 0 <= index < len(rows) for index in args.indices)
        or not args.counts
        or len(set(args.counts)) != len(args.counts)
        or any(not 1 <= count <= 20 for count in args.counts)
        or args.repeats < 1
        or (args.rgb_feature_cache and args.repeats < 2)
    ):
        raise ValueError(
            "Unique valid indices/counts and positive repeats are required; cache needs >=2 repeats"
        )
    if args.rgb_feature_cache:
        # Import enabled dependencies before source inventory/exclusion decisions.
        from operational.train40_runtime.multiobject_cache import RGBBatchCache
    if args.rgb_uint8:
        importlib.import_module("operational.train40_runtime.rgb_uint8")
    if args.backend == "prefix":
        importlib.import_module("operational.train40_runtime.temporal_prefix")
    inventory = _source_inventory(frozen, reference_binding)
    # Load optional paths above, then bind their actual import closure. A new
    # runtime import later in this run invalidates admission, even if that file
    # did not exist when the initial inventory was built.
    loaded_runtime = imported_runtime_paths()
    excluded = [
        path
        for path in inventory
        if path.parent == ROOT / "operational/train40_runtime" and path not in loaded_runtime
    ]
    sources = [*[path for path in inventory if path not in excluded], Path(__file__).resolve()]
    originals = snapshot_files(
        [*_candidate_inventory(args.candidate), *reference_files, args.manifest, *sources]
    )
    _write(args.output / "original_snapshot.json", originals)
    records: list[dict[str, Any]] = []
    startup = []
    caught: BaseException | None = None
    service = None
    inputs = PersistentSensorInputs(rgb_workers=1)
    finished = False
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(7)
        binding.update(
            scope=SCOPE,
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            sources={str(path.relative_to(ROOT)): sha(path) for path in sources},
            indices=args.indices,
            counts=args.counts,
            repeats=args.repeats,
            representation_backend=args.backend,
            manifest_sha256=sha(args.manifest),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            python=platform.python_version(),
            gpu=torch.cuda.get_device_name(),
            precision="FP32_TF32_OFF",
            seed=7,
            twenty_real_annotated_tracks=False,
            physical_50hz_goal_demonstrated=False,
            last_packet_timing=True,
            disk_and_initial_history_excluded=True,
            future_frames_or_representations_precomputed=False,
            target_20_objects_50hz_demonstrated=False,
            reuse_event_encodings_override=False if args.no_event_reuse else None,
            shared_rgb_images=args.shared_rgb_images,
            rgb_object_chunk_size=args.rgb_object_chunk_size,
            parallel_preprocessing=not args.serial_preprocessing,
            event_query_graphs=args.event_query_graphs,
            event_normalization=args.event_normalization,
            rgb_feature_cache=args.rgb_feature_cache,
            rgb_uint8=args.rgb_uint8,
            cache_scope="PER_CASE_COLD_MISS_THEN_IDENTICAL_RAW_REPEATS"
            if args.rgb_feature_cache
            else None,
            unrelated_unimported_source_exclusions=[
                str(path.relative_to(ROOT)) for path in excluded
            ],
        )
        _write(args.output / "binding.json", binding)
        models, norms = load_models(paths, "cuda")
        cache_identity = (
            {
                "model_sha256": hashlib.sha256(
                    json.dumps(binding["components"], sort_keys=True).encode()
                ).hexdigest(),
                "runtime_sha256": hashlib.sha256(
                    json.dumps(
                        {
                            "sources": binding["sources"],
                            "torch": torch.__version__,
                            "cuda": torch.version.cuda,
                            "cudnn": torch.backends.cudnn.version(),
                            "precision": "FP32_TF32_OFF_NO_AUTOCAST",
                            "layout": "CONTIGUOUS",
                            "rgb_object_chunk_size": args.rgb_object_chunk_size,
                            "shared_rgb_images": args.shared_rgb_images,
                            "parallel": args.parallel,
                            "rgb_endpoint_reuse": True,
                            "rgb_uint8": args.rgb_uint8,
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
            }
            if args.rgb_feature_cache
            else None
        )
        if cache_identity is not None:
            binding["rgb_cache_identity"] = cache_identity
            _write(args.output / "binding.json", binding)
        oracle = FinalTransferPredictor.__new__(FinalTransferPredictor)
        oracle.device = torch.device("cuda")
        oracle.models, oracle.normalizers = models, norms
        for count in args.counts:
            if args.backend == "prefix":
                from operational.train40_runtime.temporal_prefix import TemporalPrefixEventHistory

                history = TemporalPrefixEventHistory(
                    device="cuda",
                    max_objects=count,
                    roi_batch_size=min(4, count),
                    graphs=args.event_query_graphs,
                    normalization=args.event_normalization,
                )
                if args.event_query_graphs:
                    history.prime_query_graphs([count])
            else:
                if args.event_query_graphs or args.event_normalization != "sort":
                    raise ValueError("Event query graphs/topk require the prefix backend")
                history = IntegralEventHistory(
                    device="cuda", max_objects=count, roi_batch_size=min(4, count)
                )
            feature_cache = (
                RGBBatchCache(
                    model_sha256=cache_identity["model_sha256"],
                    runtime_sha256=cache_identity["runtime_sha256"],
                    rgb_object_chunk_size=args.rgb_object_chunk_size,
                )
                if cache_identity is not None
                else None
            )
            service = MultiObjectTrain40Service(
                models,
                norms,
                history,
                max_objects=count,
                parallel_branches=args.parallel,
                rgb_workers=args.rgb_workers,
                retain_input_diagnostics=True,
                reuse_event_encodings=False if args.no_event_reuse else None,
                shared_rgb_images=args.shared_rgb_images,
                rgb_object_chunk_size=args.rgb_object_chunk_size,
                parallel_preprocessing=not args.serial_preprocessing,
                rgb_feature_cache=feature_cache,
                rgb_uint8=args.rgb_uint8,
            )
            startup.append(
                {
                    "objects": count,
                    "synthetic_priming_seconds": prime(service, count),
                    "rgb_uint8_allocation": (
                        service._rgb_uint8.allocation_receipt
                        if service._rgb_uint8 is not None
                        else None
                    ),
                }
            )
            _write(args.output / "startup.json", startup)
            for index in args.indices:
                if feature_cache is not None:
                    feature_cache.clear()
                row = rows[index]
                anchor = int(row["anchor_us"])
                boxes = [inputs._box(source) for source in list(row["bbox_sources"])[-2:]]
                rgb_square = common_square_from_boxes(boxes, (0, 1), margin_fraction=0.25)
                objects = artificial_regions(row["square_xyxy"], rgb_square, count)
                raw_rgb = read_raw_rgb_history(inputs, row)
                with EvTTCEventReader(Path(row["raw_path"])) as reader:
                    old = reader.read_window(anchor - 620_000, anchor - 20_000)
                    packet = reader.read_window(anchor - 20_000, anchor)
                timings = []
                for repetition in range(args.repeats):
                    history.initialize(old, anchor - 620_000, anchor - 20_000)
                    torch.cuda.synchronize()
                    result = service.update(packet, anchor - 20_000, anchor, objects, raw_rgb)
                    result.update(
                        index=index,
                        repetition=repetition,
                        regions=[asdict(x) for x in objects],
                        scope=SCOPE,
                        packet_events=len(packet["t"]),
                        ring=history.stats,
                        diagnostic_tensor_hashes_outside_clock=True,
                    )
                    result["input_digests"] = input_digests(service)
                    # Recorded immediately: later parity failures do not erase measured cases.
                    timings.append(result)
                    with (args.output / "updates.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(result, allow_nan=False) + "\n")
                    print(
                        json.dumps(
                            {
                                "objects": count,
                                "index": index,
                                "repeat": repetition,
                                "ms": result["complete_wall_ms"],
                            }
                        ),
                        flush=True,
                    )
                assert service.last_event_inputs is not None
                actual_event = service.last_event_inputs.cpu().numpy()
                expected_outputs = []
                input_checks = []
                receipts = []
                expected_rgb_digests = []
                for position, obj in enumerate(objects):
                    modified = {**row, "square_xyxy": list(obj.event_square)}
                    raw = engine.prepare_event(modified)
                    reference_pixels = reference_rgb(raw_rgb, obj)
                    expected_pixels = [
                        hashlib.sha256(np.ascontiguousarray(pixels)).hexdigest()
                        for pixels, _ in reference_pixels
                    ]
                    expected_rgb_digests.append(expected_pixels)
                    actual_pixel_digests = [
                        result["input_digests"][position]["rgb_sha256"]
                        for result in timings
                        if result["input_digests"][position]["rgb_sha256"] is not None
                    ]
                    expected_delta_digest = hashlib.sha256(
                        np.asarray(
                            [meta["delta_t_s"] for _, meta in reference_pixels], dtype=np.float32
                        )
                    ).hexdigest()
                    actual_delta_digests = [
                        result["input_digests"][position]["rgb_delta_sha256"]
                        for result in timings
                        if result["input_digests"][position]["rgb_delta_sha256"] is not None
                    ]
                    input_checks.append(
                        {
                            "object_id": obj.object_id,
                            "event_exact": bool(
                                np.array_equal(actual_event[position], raw["own_events"])
                            ),
                            "event_passed": bool(
                                np.allclose(
                                    actual_event[position], raw["own_events"], rtol=1e-5, atol=1e-5
                                )
                            ),
                            "event_max_abs": float(
                                np.abs(actual_event[position] - raw["own_events"]).max()
                            ),
                            "rgb_exact": (bool(actual_pixel_digests) or not expected_pixels)
                            and all(actual == expected_pixels for actual in actual_pixel_digests),
                            "rgb_validation": "SHA256_OF_ACTUAL_MISSES_AND_CANONICAL_PIXELS",
                            "rgb_delta_exact": (bool(actual_delta_digests) or not expected_pixels)
                            and all(
                                actual == expected_delta_digest for actual in actual_delta_digests
                            ),
                        }
                    )
                    with (
                        patch.object(engine, "prepare_event", return_value=raw),
                        patch.object(
                            engine,
                            "raw_rgb_observations",
                            return_value=(reference_pixels, raw_rgb.unavailable_reason),
                        ),
                    ):
                        expected = oracle.predict(modified)
                    expected["query_id"] = obj.object_id
                    expected_outputs.append(expected)
                    receipts.append(reference_receipt(row, obj, reference_pixels))
                attest_cached_inputs(timings, expected_rgb_digests)
                cache_exercise = cache_exercise_receipt(timings) if args.rgb_feature_cache else None
                for result in timings:
                    result["cache_exercise"] = cache_exercise
                    candidates = [
                        {**output, "query_id": output["object_id"]} for output in result["objects"]
                    ]
                    result["parity"] = prediction_parity(
                        expected_outputs, candidates, [obj.object_id for obj in objects]
                    )
                    result["input_parity"] = input_checks
                    result["history_receipts_exact"] = [
                        output["input_receipt"] == expected_receipt
                        for output, expected_receipt in zip(
                            result["objects"], receipts, strict=True
                        )
                    ]
                    primary = {**result["objects"][0], "query_id": row["query_id"]}
                    result["preserved_primary_parity"] = prediction_parity(
                        [original_reference[index]], [primary], [row["query_id"]]
                    )
                    records.append(result)
                _write(
                    args.output / f"case_{count:02d}_{index:05d}.json",
                    {"reference": expected_outputs, "updates": timings},
                )
                del actual_event, old, packet
            service.close()
            service = None
            del history
            gc.collect()
            torch.cuda.empty_cache()
        finished = True
    except BaseException as exc:
        caught = exc
    finally:
        cleanup_errors = []
        for resource in (service, inputs):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as exc:
                    cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        intact = verify_snapshot(originals)
        unexpected_imports = sorted(
            str(path.relative_to(ROOT)) for path in imported_runtime_paths() - loaded_runtime
        )
        admitted = bool(records) and all(
            record["parity"]["passed"]
            and record["preserved_primary_parity"]["passed"]
            and all(
                row["event_passed"] and row["rgb_exact"] and row["rgb_delta_exact"]
                for row in record["input_parity"]
            )
            and record["same_inputs_as_tensor_checked_repeat"]
            and all(record["history_receipts_exact"])
            and (not args.rgb_feature_cache or record["cache_exercise"]["passed"])
            for record in records
        )
        cache_hit_updates = sum(
            (record.get("rgb_feature_cache") or {}).get("status") == "HIT" for record in records
        )
        if args.rgb_feature_cache:
            admitted = admitted and cache_hit_updates > 0
        complete_count = len(args.counts) * len(args.indices) * args.repeats
        passed = (
            finished
            and admitted
            and len(records) == complete_count
            and intact["passed"]
            and not cleanup_errors
            and not unexpected_imports
        )
        summary = {
            "status": "COMPLETE" if passed else "FAILED",
            "passed": passed,
            "scope": SCOPE,
            "expected_updates": complete_count,
            "validated_updates": len(records),
            "originals": intact,
            "cleanup_errors": cleanup_errors,
            "unexpected_excluded_source_imports": unexpected_imports,
            "error": f"{type(caught).__name__}: {caught}" if caught else None,
            "target_20_objects_50hz_demonstrated": False,
            "rgb_cache_hit_updates": cache_hit_updates,
            "rgb_cache_miss_updates": sum(
                (record.get("rgb_feature_cache") or {}).get("status") == "MISS"
                for record in records
            ),
            "latency_by_count_ms": {
                str(count): latency_distribution(
                    [row["complete_wall_ms"] for row in records if row["object_count"] == count]
                )
                for count in args.counts
                if any(row["object_count"] == count for row in records)
            },
        }
        _write(args.output / "summary.json", summary)
    if caught is not None:
        raise caught
    if not passed:
        raise RuntimeError("Multiobject pilot failed; inspect preserved summary and case files")


def main() -> None:
    """Parse an explicitly artificial population and fresh output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", nargs="+", type=int, default=[1, 2, 5, 10, 20])
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 200, 553])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rgb-workers", type=int, default=8)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument(
        "--no-event-reuse",
        action="store_true",
        help="Preserve original event encoder batch; retain RGB endpoint reuse",
    )
    parser.add_argument("--backend", choices=["integral", "prefix"], default="integral")
    parser.add_argument(
        "--shared-rgb-images",
        action="store_true",
        help="Convert each shared raw RGB frame to PIL once per update",
    )
    parser.add_argument(
        "--rgb-object-chunk-size",
        type=int,
        help="Run RGB producers on this many objects per call; one preserves B7 encoders",
    )
    parser.add_argument(
        "--serial-preprocessing",
        action="store_true",
        help="Prepare RGB before event dispatch, with both inside the update clock",
    )
    parser.add_argument(
        "--event-query-graphs",
        action="store_true",
        help="Prime prefix ROI query graphs using synthetic data at startup",
    )
    parser.add_argument("--event-normalization", choices=["sort", "topk"], default="sort")
    parser.add_argument(
        "--rgb-feature-cache",
        action="store_true",
        help="Measure a cold miss then identical-history feature hits; context always recomputed",
    )
    parser.add_argument(
        "--rgb-uint8",
        action="store_true",
        help="Transfer unique uint8 crops with exact FP32 LUT and bounded buffer leases",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
