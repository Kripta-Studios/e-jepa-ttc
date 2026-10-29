"""Explicit synthetic startup coverage for the native TRAIN40 sensor ABI.

The 650 ms history and 100 ms +/- 1 ms frame spacing allow at most seven
frames, hence five overlapping T3 observations. At stream startup the native
selector also permits one T2 observation. This fixed specification never reads
the replay schedule, future pixels, event packets or future annotations.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from operational.train40_runtime.multiobject import MultiObjectTrain40Service, ObjectRegion

NATIVE_RGB_LAYOUTS = ((0, 0), (1, 2), *((history, 3) for history in range(1, 6)))


def _memory(device: torch.device) -> dict[str, int | str | None]:
    if device.type != "cuda":
        return {"scope": "CPU_NO_CUDA_ALLOCATION", "allocated_bytes": None, "reserved_bytes": None}
    return {
        "scope": "PYTORCH_PROCESS_CURRENT_ALLOCATOR_NOT_EXCLUSIVE_OR_PEAK",
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
    }


def _graphs(service: MultiObjectTrain40Service) -> list[dict[str, Any]]:
    return [
        {
            "producer": row["producer"],
            "shape": row["shape"],
            "endpoint_reuse": row["endpoint_reuse"],
        }
        for row in service.producers.diagnostics()
    ]


def prime_supported_histories(
    service: MultiObjectTrain40Service,
    *,
    object_counts: Sequence[int] | None = None,
    receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prime fixed native histories before any sensor history is installed.

    ``object_counts`` defaults to the service's declared capacity. Supply all
    intended counts explicitly for a service that changes object count. The
    conservative RGB chunk of one also covers mixed RGB availability at each
    primed count; other chunk policies leave such unlisted shapes unprimed.
    Unexpected shapes still execute their normal, timed cold setup in service.

    An optional empty ``receipt`` is populated incrementally so the caller can
    preserve completed layouts and a failed capture without hiding startup OOMs.
    Synthetic results are discarded, never entered into a feature cache, and do
    not count as numerical parity or sustained-service evidence.
    """
    counts = [service.max_objects] if object_counts is None else list(object_counts)
    if (
        not counts
        or any(type(count) is not int or not 1 <= count <= service.max_objects for count in counts)
        or len(set(counts)) != len(counts)
    ):
        raise ValueError("Priming object counts must be unique integers within service capacity")
    if receipt is not None and receipt:
        raise ValueError("Synthetic priming requires an empty receipt")
    state = service.history.stats
    if (
        service._closed
        or service._event_overlap_inflight
        or service.history.ready
        or state.get("end_us") is not None
        or state.get("ingested_packets", 0)
        or state.get("queries", 0)
    ):
        raise ValueError("Synthetic priming must precede reset, ingestion and sensor queries")
    report = {} if receipt is None else receipt
    started = time.perf_counter()
    report.update(
        schema="train40_synthetic_abi_priming_v1",
        status="RUNNING",
        scope="EXPLICIT_SYNTHETIC_STARTUP_ONLY_NO_RAW_OR_FUTURE_DATA",
        object_counts=counts,
        layouts=[
            {"rgb_observations": history, "rgb_steps": steps}
            for history, steps in NATIVE_RGB_LAYOUTS
        ],
        event_observations=4,
        event_steps=3,
        rgb_object_chunk_size=service.rgb_object_chunk_size,
        endpoint_reuse_config=dict(service.producers.reuse_by_modality),
        mixed_rgb_availability_covered=service.rgb_object_chunk_size == 1,
        nonoverlapping_multi_observation_rgb_primed=False,
        nonoverlap_reason=(
            "Native sensor_triplets structurally shares endpoints; guards remain active"
        ),
        unexpected_shapes_remain_inside_update_clock=True,
        event_early_launch_path_exercised=False,
        numerical_parity_admitted=False,
        targets_read=False,
        optimizer_updates=0,
        records=[],
    )
    query = 1_000_000
    square = (0.0, 0.0, 128.0, 128.0)
    try:
        if service.device.type == "cuda":
            torch.cuda.synchronize(service.device)
        report["memory_before"] = _memory(service.device)
        report["producer_graphs_before"] = _graphs(service)
        for count in counts:
            event = torch.zeros((count, 4, 3, 12, 128, 128), dtype=torch.float32)
            for history, steps in NATIVE_RGB_LAYOUTS:
                record: dict[str, Any] = {
                    "objects": count,
                    "rgb_observations": history,
                    "rgb_steps": steps,
                    "event_producer_shape": [count * 4, 3, 12, 128, 128],
                    "rgb_producer_shapes": [
                        [
                            min(service.rgb_object_chunk_size or count, count - start) * history,
                            steps,
                            3,
                            128,
                            128,
                        ]
                        for start in range(0, count, service.rgb_object_chunk_size or count)
                    ]
                    if history
                    else [],
                    "rgb_endpoint_reuse_active": bool(
                        history > 1 and service.producers.reuse_by_modality["R"]
                    ),
                    "layout_reason": (
                        "NO_RGB_FALLBACK"
                        if not history
                        else "NATIVE_STARTUP_T2"
                        if steps == 2
                        else "NATIVE_OVERLAPPING_T3_CHAIN"
                    ),
                    "status": "RUNNING",
                }
                report["records"].append(record)
                before = _graphs(service)
                record["memory_before"] = _memory(service.device)
                begin = time.perf_counter()
                try:
                    objects = [
                        ObjectRegion(f"synthetic_{i:02d}", square, square if history else None)
                        for i in range(count)
                    ]
                    pixels = np.zeros((steps, 3, 128, 128), dtype=np.float32)
                    observations = []
                    for i in range(history):
                        timestamps = [
                            query - (history + steps - 2 - i - step) * 100_000
                            for step in range(steps)
                        ]
                        observations.append(
                            (
                                pixels,
                                {
                                    "delta_t_s": [0.1] * (steps - 1),
                                    "first_sensor_ts_us": timestamps[0],
                                    "anchor_us": timestamps[-1],
                                    "available_us": query,
                                    "timestamps_us": timestamps,
                                    "frame_indices": list(range(i, i + steps)),
                                    "square_xyxy": list(square),
                                },
                            )
                        )
                    service.infer_prepared(
                        objects, query, event, [list(observations) for _ in objects]
                    )
                    if service.device.type == "cuda":
                        torch.cuda.synchronize(service.device)
                    record["status"] = "COMPLETE"
                except BaseException as error:
                    record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
                    raise
                finally:
                    record["seconds"] = time.perf_counter() - begin
                    record["memory_after"] = _memory(service.device)
                    record["new_producer_graphs"] = [
                        row for row in _graphs(service) if row not in before
                    ]
        report["status"] = "COMPLETE"
    except BaseException as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["total_seconds"] = time.perf_counter() - started
        report["memory_after"] = _memory(service.device)
        report["producer_graphs_after"] = _graphs(service)
    return report


__all__ = ["NATIVE_RGB_LAYOUTS", "prime_supported_histories"]
