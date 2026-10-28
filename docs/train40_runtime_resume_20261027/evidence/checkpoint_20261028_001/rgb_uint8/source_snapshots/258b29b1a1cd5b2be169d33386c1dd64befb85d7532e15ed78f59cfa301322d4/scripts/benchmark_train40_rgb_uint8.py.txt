"""Synthetic exactness and RGB-preparation timing for unique uint8 transfers.

This measures only RGB crops, transfer, normalization and triplet assembly. Raw
sensor generation, runtime allocation and reference comparisons are excluded
from timed samples. No models, targets or frozen predictions are accessed. Even
a CUDA run is not a complete TTC update or evidence of sustained 50 Hz service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402
import PIL  # noqa: E402
import torch  # noqa: E402

from e_jepa_ttc.rgb_port.data import crop_uint8  # noqa: E402
from operational.train40_runtime.inputs import _normalized_crop  # noqa: E402
from operational.train40_runtime.measurement import latency_distribution  # noqa: E402
from operational.train40_runtime.multiobject import (  # noqa: E402
    ObjectRegion,
    RawRGBHistory,
    Square,
)
from operational.train40_runtime.rgb_uint8 import RGBUInt8Runtime  # noqa: E402

SCOPE = "SYNTHETIC_RGB_PREPARATION_ONLY_NOT_TTC_OR_SUSTAINED_SERVICE"
SOURCES = (
    "scripts/benchmark_train40_rgb_uint8.py",
    "operational/train40_runtime/rgb_uint8.py",
    "operational/train40_runtime/encoding.py",
    "operational/train40_runtime/inputs.py",
    "operational/train40_runtime/multiobject.py",
    "operational/train40_runtime/measurement.py",
    "src/e_jepa_ttc/rgb_port/data.py",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_snapshot() -> dict[str, str]:
    return {relative: _sha(ROOT / relative) for relative in SOURCES}


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    return value


def _write(path: Path, result: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def synthetic_case(
    count: int, history_count: int, steps: int, height: int, width: int, seed: int
) -> tuple[list[ObjectRegion], RawRGBHistory]:
    """Create causal immutable RAW frames and distinct fractional/clipped ROIs."""
    rng = np.random.default_rng(seed)
    frame_count = history_count + steps - 1
    frames = {
        index: rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        for index in range(frame_count)
    }
    for frame in frames.values():
        frame.flags.writeable = False
    timestamps = {index: 1_000_000 + index * 50_000 for index in frames}
    groups = tuple(tuple(range(index, index + steps)) for index in range(history_count))
    history = RawRGBHistory(timestamps[frame_count - 1] + 10_000, frames, timestamps, groups)
    regions = []
    for index in range(count):
        x0, y0 = (index % 5) * width / 25 - 0.25, (index // 5) * height / 25 - 0.75
        square = cast(Square, (x0, y0, x0 + 0.82 * width + 0.5, y0 + 0.9 * height + 0.5))
        regions.append(ObjectRegion(f"synthetic_{index:02d}", square, square))
    if len({region.rgb_square for region in regions}) != count:
        raise ValueError("Synthetic workload must use distinct object regions")
    return regions, history


def reference_prepare(
    objects: Sequence[ObjectRegion], history: RawRGBHistory, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[list[dict[str, Any]]]]:
    """Rebuild the current ROI-first normalized/expanded host path."""
    sensors, deltas, metadata = [], [], []
    for region in objects:
        assert region.rgb_square is not None
        frames = {
            index: _normalized_crop(frame, region.rgb_square)
            for index, frame in history.frames.items()
        }
        current = []
        for group in history.triplets:
            times = [history.timestamps_us[index] for index in group]
            sensor = np.ascontiguousarray(np.stack([frames[index] for index in group]))
            delta = (np.diff(times) / 1e6).tolist()
            sensors.append(torch.from_numpy(sensor))
            deltas.append(delta)
            current.append(
                {
                    "first_sensor_ts_us": times[0],
                    "anchor_us": times[-1],
                    "available_us": history.query_us,
                    "timestamps_us": times,
                    "frame_indices": list(group),
                    "delta_t_s": delta,
                    "square_xyxy": list(region.rgb_square),
                }
            )
        metadata.append(current)
    return (
        torch.stack(sensors).to(device),
        torch.tensor(deltas, dtype=torch.float32, device=device),
        metadata,
    )


def _roi_first_parity(objects: Sequence[ObjectRegion], history: RawRGBHistory) -> bool:
    for region in objects:
        assert region.rgb_square is not None
        for frame in history.frames.values():
            original = np.ascontiguousarray(
                crop_uint8(frame, region.rgb_square).astype(np.float32) / 255
            )
            if not np.array_equal(original, _normalized_crop(frame, region.rgb_square)):
                return False
    return True


def _input_digest(objects: Sequence[ObjectRegion], history: RawRGBHistory) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "query_us": history.query_us,
                "timestamps_us": history.timestamps_us,
                "triplets": history.triplets,
                "objects": [(item.object_id, item.rgb_square) for item in objects],
                "frame_shapes": [list(frame.shape) for frame in history.frames.values()],
            },
            sort_keys=True,
        ).encode()
    )
    for index in sorted(history.frames):
        digest.update(memoryview(history.frames[index]).cast("B"))
    return digest.hexdigest()


def _tensor_parity(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    old, new = expected.detach().cpu().numpy(), actual.detach().cpu().numpy()
    same_shape = old.shape == new.shape
    exact = same_shape and old.dtype == new.dtype and np.array_equal(old, new)
    return {
        "exact": bool(exact),
        "reference_shape": list(old.shape),
        "candidate_shape": list(new.shape),
        "reference_dtype": str(old.dtype),
        "candidate_dtype": str(new.dtype),
        "different_values": int(np.count_nonzero(old != new)) if same_shape else None,
        "max_abs_error": (
            None if not same_shape else 0.0 if exact else float(np.max(np.abs(old - new)))
        ),
    }


def _latency(milliseconds: Sequence[float]) -> dict[str, Any]:
    summary = latency_distribution(milliseconds)
    return {
        key: summary[key]
        for key in (
            "count",
            "mean_ms",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "min_ms",
            "max_ms",
            "quantile_method",
        )
    }


def _byte_contract(
    receipt: Mapping[str, Any], allocation: Mapping[str, Any], count: int, groups: int, steps: int
) -> dict[str, Any]:
    crop_pixels, rows = 3 * 128 * 128, count * groups
    unique_bytes = count * (groups + steps - 1) * crop_pixels
    metadata_capacity = rows * (3 * 8 + 2 * 4)
    expected = {
        "unique_crop_bytes": unique_bytes,
        "baseline_expanded_fp32_bytes": rows * steps * crop_pixels * 4,
        "candidate_h2d_bytes": unique_bytes + rows * steps * 8 + rows * (steps - 1) * 4,
        "host_buffer_bytes": unique_bytes + metadata_capacity,
        "device_buffer_bytes": unique_bytes * 9
        + rows * 3 * crop_pixels * 4
        + metadata_capacity
        + 1024,
    }
    observed = {
        key: allocation.get(key) if key.endswith("buffer_bytes") else receipt.get(key)
        for key in expected
    }
    return {"passed": expected == observed, "expected": expected, "observed": observed}


def _case(args: argparse.Namespace, count: int, groups: int, steps: int) -> dict[str, Any]:
    device = torch.device(args.device)
    objects, history = synthetic_case(
        count, groups, steps, args.height, args.width, args.seed + groups * 10 + steps
    )
    allocation_start = time.perf_counter_ns()
    runtime = RGBUInt8Runtime(
        device=device,
        max_objects=count,
        max_history=groups,
        max_frames=groups + steps - 1,
        crop_workers=args.crop_workers,
        shared_images=not args.roi_first_images,
    )
    _sync(device)
    allocation_ms = (time.perf_counter_ns() - allocation_start) / 1e6
    result: dict[str, Any] = {
        "objects": count,
        "history_observations": groups,
        "steps": steps,
        "distinct_raw_frames": len(history.frames),
        "distinct_roi_frame_crops": count * len(history.frames),
        "raw_frame_shape": [args.height, args.width, 3],
        "synthetic_input_sha256": _input_digest(objects, history),
        "runtime_allocation_ms_excluded": allocation_ms,
        "runtime_allocation_receipt": _jsonable(runtime.allocation_receipt),
        "calculated_bytes": {
            "unique_uint8_crops": count * len(history.frames) * 3 * 128 * 128,
            "expanded_fp32_sensor": count * groups * steps * 3 * 128 * 128 * 4,
            "expanded_delta_fp32": count * groups * (steps - 1) * 4,
            "shared_raw_frames_uint8": sum(frame.nbytes for frame in history.frames.values()),
        },
    }
    try:
        baseline, delta, metadata = reference_prepare(objects, history, device)
        batch = runtime.prepare(objects, history)
        _sync(device)
        parity = {
            "sensor": _tensor_parity(batch.sensor, baseline),
            "delta": _tensor_parity(batch.delta, delta),
            "metadata_exact": _jsonable(batch.metadata) == metadata,
            "present_indices_exact": tuple(batch.present_indices) == tuple(range(count)),
            "original_vs_roi_first_exact": _roi_first_parity(objects, history),
            "byte_contract": _byte_contract(
                batch.receipt, runtime.allocation_receipt, count, groups, steps
            ),
        }
        parity["passed"] = all(
            (
                parity["sensor"]["exact"],
                parity["delta"]["exact"],
                parity["metadata_exact"],
                parity["present_indices_exact"],
                parity["original_vs_roi_first_exact"],
                parity["byte_contract"]["passed"],
            )
        )
        result["parity"] = parity
        result["initial_receipt"] = _jsonable(batch.receipt)
        runtime.release(batch)
        del baseline, delta
        if not parity["passed"]:
            return result
        durations, receipts = [], []
        for iteration in range(args.warmups + args.repeats):
            _sync(device)
            start = time.perf_counter_ns()
            batch = runtime.prepare(objects, history)
            _sync(device)
            runtime.release(batch)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            if iteration >= args.warmups:
                durations.append(elapsed_ms)
                receipts.append(_jsonable(batch.receipt))
        result["rgb_preparation_wall_ms"] = durations
        result["rgb_preparation_latency"] = _latency(durations)
        result["measured_receipts"] = receipts
        if args.measure_reference:
            result["reference_preparation_options"] = {
                "crop_path": "ROI_FIRST_NORMALIZED_CROP",
                "crop_workers": 1,
                "shared_images": False,
                "candidate_matches_crop_options": args.roi_first_images and args.crop_workers == 1,
                "comparison_scope": "SAME_INPUTS_AND_OUTPUTS_NOT_TRANSPORT_ISOLATION",
            }
            reference_times = []
            for iteration in range(args.warmups + args.repeats):
                _sync(device)
                start = time.perf_counter_ns()
                reference = reference_prepare(objects, history, device)
                _sync(device)
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                del reference
                if iteration >= args.warmups:
                    reference_times.append(elapsed_ms)
            result["original_host_expanded_preparation_wall_ms"] = reference_times
            result["original_host_expanded_preparation_latency"] = _latency(reference_times)
        return result
    finally:
        runtime.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--objects", type=int, nargs="+", default=[1, 2, 20])
    parser.add_argument("--history-counts", type=int, nargs="+", default=[5])
    parser.add_argument("--steps", type=int, nargs="+", choices=(2, 3), default=[3])
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--crop-workers", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--measure-reference", action="store_true")
    parser.add_argument(
        "--roi-first-images", action="store_true", help="Disable the shared PIL image option"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "artifacts/train40_runtime_resume_20261027/rgb_uint8/cpu_synthetic_v1/summary.json",
        help="New JSON destination; an existing result is never overwritten",
    )
    return parser


def main() -> int:
    """Run the declared synthetic workload and persist exactness/timing evidence."""
    parser = _parser()
    args = parser.parse_args()
    if (
        any(not 1 <= value <= 20 for value in args.objects)
        or any(not 1 <= value <= 8 for value in args.history_counts)
        or min(args.height, args.width) < 2
        or args.warmups < 0
        or min(args.repeats, args.crop_workers, args.torch_threads) < 1
    ):
        parser.error("Invalid population, history, image size, worker count or repetition count")
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error(f"Refusing to overwrite previous evidence: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    before = _source_snapshot()
    result: dict[str, Any] = {
        "status": "RUNNING",
        "scope": SCOPE,
        "start_time": _now(),
        "device": args.device,
        "gpu_used": args.device == "cuda",
        "models_loaded": False,
        "targets_read": False,
        "frozen_candidate_accessed": False,
        "complete_ttc_latency_claimed": False,
        "sustained_hz_claimed": False,
        "configuration": _jsonable(vars(args)),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "pillow_version": PIL.__version__,
        "sources_before": before,
        "timing_boundary": {
            "included": [
                "current ROI Pillow crop and resize",
                "unique crop host packing",
                "host-to-device copies for the selected path",
                "normalization LUT and triplet assembly",
                "device completion synchronization and batch release",
            ],
            "excluded": [
                "synthetic raw frame generation",
                "startup/preallocation",
                "reference and parity comparison",
                "events, RGB producers, context, TTC and output transport",
            ],
        },
        "cases": [],
    }
    _write(args.output, result)
    try:
        for count in args.objects:
            for groups in args.history_counts:
                for steps in args.steps:
                    record = _case(args, count, groups, steps)
                    result["cases"].append(record)
                    _write(args.output, result)
                    if not record["parity"]["passed"]:
                        raise RuntimeError("Exact RGB sensor/metadata parity failed")
        result["sources_after"] = _source_snapshot()
        result["sources_unchanged"] = result["sources_after"] == before
        if not result["sources_unchanged"]:
            raise RuntimeError("Inspected source files changed during the benchmark")
        result["status"] = "COMPLETE"
        result["all_cases_exact"] = True
        result["end_time"] = _now()
        _write(args.output, result)
        print(json.dumps({"status": result["status"], "cases": len(result["cases"])}))
        return 0
    except BaseException as error:
        result["status"] = "FAILED"
        result["error"] = f"{type(error).__name__}: {error}"
        result["end_time"] = _now()
        result["sources_after"] = _source_snapshot()
        result["sources_unchanged"] = result["sources_after"] == before
        _write(args.output, result)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
