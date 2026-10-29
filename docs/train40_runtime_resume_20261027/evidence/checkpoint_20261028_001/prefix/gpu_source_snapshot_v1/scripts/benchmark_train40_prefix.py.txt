"""Compare bucket and temporal-prefix representations under the same packet clock.

This measures RAW CPU packet -> all ROI event tensors, without models or TTC.
Real-data ROIs after the primary ROI are synthetic capacity regions, not claimed
annotated objects. Disk replay and initial history fill are outside the clock.
Backends run sequentially to avoid simultaneous GPU memory/work contention.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from e_jepa_ttc.efficient_context.mapped_union import encode_mapped, map_roi  # noqa: E402
from operational.evttc_transfer.inputs import EvTTCEventReader  # noqa: E402
from operational.train40_runtime.integral import (  # noqa: E402
    HISTORY_US,
    UPDATE_US,
    IntegralEventHistory,
)
from operational.train40_runtime.measurement import latency_distribution  # noqa: E402
from operational.train40_runtime.temporal_prefix import TemporalPrefixEventHistory  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sync(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _slice(raw: dict[str, np.ndarray], start: int, end: int) -> dict[str, np.ndarray]:
    lo, hi = np.searchsorted(raw["t"], [start, end])
    return {key: value[lo:hi] for key, value in raw.items()}


def _rois(primary: list[float], count: int) -> list[tuple[float, float, float, float]]:
    x0, y0, x1, y1 = primary
    width, height = x1 - x0, y1 - y0
    result = [(x0, y0, x1, y1)]
    for index in range(1, count):
        scale = 0.75 + 0.025 * index
        cx = (x0 + x1) * 0.5 + width * 0.06 * ((index % 5) - 2)
        cy = (y0 + y1) * 0.5 + height * 0.06 * ((index // 5) - 2)
        result.append(
            (
                cx - width * scale / 2,
                cy - height * scale / 2,
                cx + width * scale / 2,
                cy + height * scale / 2,
            )
        )
    return result


def _canonical(raw: dict[str, np.ndarray], end: int, roi: tuple[float, ...], size: int) -> Tensor:
    mapped = map_roi(raw, roi, size, 0.0)
    six = torch.stack(
        [
            encode_mapped(mapped, left, left + 100_000, size, "prefix-benchmark")
            for left in range(end - HISTORY_US, end, 100_000)
        ]
    )
    return torch.stack([six[index : index + 3] for index in range(4)])


def run(args: argparse.Namespace) -> None:
    """Run both exact representations and preserve success or failure receipts."""
    if args.steps < 1 or args.warmup < 0 or any(n < 1 or n > 20 for n in args.objects):
        raise ValueError("positive steps, nonnegative warmup, and 1..20 objects required")
    if len(set(args.objects)) != len(args.objects):
        raise ValueError("object-count experiments must be unique")
    if not args.output.resolve().is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("benchmark output must be a fresh directory inside this repo's artifacts")
    reference_bytes = (
        (args.steps + args.warmup) * sum(args.objects) * 4 * 3 * 12 * args.roi_size**2 * 4
    )
    if reference_bytes > args.max_reference_bytes:
        raise ValueError(
            "requested parity reference tensors exceed the explicit host memory budget"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    total_updates = args.warmup + args.steps
    sources = [
        Path(__file__),
        ROOT / "operational/train40_runtime/integral.py",
        ROOT / "operational/train40_runtime/temporal_prefix.py",
        ROOT / "operational/train40_runtime/normalization.py",
    ]
    source_hashes = {str(path.relative_to(ROOT)): _sha(path) for path in sources}
    if args.synthetic:
        start = 738
        anchor = start + HISTORY_US + total_updates * UPDATE_US
        rng = np.random.default_rng(7)
        event_count = int((anchor - start) / UPDATE_US) * args.packet_events
        raw = {
            "x": rng.integers(0, args.width, event_count, dtype=np.int32),
            "y": rng.integers(0, args.height, event_count, dtype=np.int32),
            "t": np.sort(rng.integers(start, anchor, event_count, dtype=np.int64)),
            "p": rng.choice(np.array([-1, 1], dtype=np.int8), event_count),
        }
        primary = [0.0, 0.0, float(args.width), float(args.height)]
        data_binding: dict[str, Any] = {"synthetic": True, "seed": 7}
    else:
        if args.manifest is None:
            raise ValueError("--manifest required without --synthetic")
        row = json.loads(args.manifest.read_text(encoding="utf-8"))["rows"][args.index]
        anchor = int(row["anchor_us"])
        start = anchor - HISTORY_US - total_updates * UPDATE_US
        if start < 0:
            raise ValueError("not enough positive-clock history for requested steps")
        reader = EvTTCEventReader(Path(row["raw_path"])).open()
        try:
            raw = reader.read_window(start, anchor)
        finally:
            reader.close()
        primary = row["square_xyxy"]
        data_binding = {
            "synthetic": False,
            "manifest_sha256": _sha(args.manifest),
            "query_id": row["query_id"],
            "index": args.index,
            "raw_path": row["raw_path"],
            "additional_rois": "SYNTHETIC_CAPACITY_REGIONS_NOT_ANNOTATED_TRACKS",
        }
    initial_end = start + HISTORY_US
    initial = _slice(raw, start, initial_end)
    packets = [
        _slice(raw, initial_end + i * UPDATE_US, initial_end + (i + 1) * UPDATE_US)
        for i in range(total_updates)
    ]
    binding = {
        "scope": "RAW_CPU_PACKET_TO_EVENT_REPRESENTATIONS_NO_MODELS_NO_TTC",
        "data": data_binding,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "sources": source_hashes,
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if torch.device(args.device).type == "cuda" else None,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "start_us": start,
        "anchor_us": anchor,
        "warmup_updates": args.warmup,
        "measured_updates": args.steps,
        "objects": args.objects,
        "roi_size": args.roi_size,
        "normalization": args.normalization,
        "input_hashes": {
            key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in raw.items()
        },
        "packet_events": [len(packet["t"]) for packet in packets],
        "target_50hz_demonstrated": False,
    }
    _write(args.output / "binding.json", binding)
    results: list[dict[str, Any]] = []
    expected_outputs: dict[int, list[Tensor]] = {}
    errors: list[str] = []
    caught: BaseException | None = None
    try:
        for backend_type in (IntegralEventHistory, TemporalPrefixEventHistory):
            for object_count in args.objects:
                rois = _rois(primary, object_count)
                options: dict[str, Any] = {}
                if backend_type is TemporalPrefixEventHistory:
                    options["normalization"] = args.normalization
                backend = backend_type(
                    device=args.device,
                    width=args.width,
                    height=args.height,
                    roi_size=args.roi_size,
                    max_packet_events=args.max_packet_events,
                    max_objects=object_count,
                    roi_batch_size=min(args.roi_batch_size, object_count),
                    **options,
                )
                begin = time.perf_counter()
                backend.initialize(initial, start, initial_end)
                _sync(args.device)
                initialization_ms = (time.perf_counter() - begin) * 1000
                times = []
                outputs: list[Tensor] = []
                exact_all = True
                max_abs = 0.0
                for step, packet in enumerate(packets):
                    left = initial_end + step * UPDATE_US
                    _sync(args.device)
                    begin_ns = time.perf_counter_ns()
                    backend.ingest(packet, left, left + UPDATE_US)
                    output = backend.query_many(rois)
                    _sync(args.device)
                    elapsed_ms = (time.perf_counter_ns() - begin_ns) / 1e6
                    # Host transfer and validation happen outside the declared
                    # device-tensor output boundary, equally for both backends.
                    actual = output.cpu().clone()
                    if backend_type is IntegralEventHistory:
                        outputs.append(actual)
                    else:
                        expected = expected_outputs[object_count][step]
                        exact_all &= torch.equal(actual, expected)
                        max_abs = max(max_abs, float((actual - expected).abs().max()))
                    if step >= args.warmup:
                        times.append(elapsed_ms)
                    print(
                        json.dumps(
                            {
                                "backend": backend_type.__name__,
                                "objects": object_count,
                                "step": step,
                                "ms": elapsed_ms,
                            }
                        ),
                        flush=True,
                    )
                if backend_type is IntegralEventHistory:
                    expected_outputs[object_count] = outputs
                canonical = _canonical(raw, anchor, tuple(rois[0]), args.roi_size)
                canonical_diff = float((actual[0] - canonical).abs().max())
                canonical_passed = bool(torch.allclose(actual[0], canonical, rtol=1e-5, atol=1e-5))
                record = {
                    "backend": backend_type.__name__,
                    "objects": object_count,
                    "initialization_ms_excluded": initialization_ms,
                    "latency_ms": latency_distribution(times),
                    "samples_ms": times,
                    "prefix_vs_bucket_bit_exact": exact_all
                    if backend_type is TemporalPrefixEventHistory
                    else None,
                    "prefix_vs_bucket_max_abs": max_abs
                    if backend_type is TemporalPrefixEventHistory
                    else None,
                    "canonical_primary_final_max_abs": canonical_diff,
                    "canonical_primary_final_passed": canonical_passed,
                    "distinct_final_tensor_count": len(
                        {hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in actual}
                    ),
                    "stats": backend.stats,
                    "rois": rois,
                }
                results.append(record)
                _write(args.output / "results.json", results)
                if not exact_all or not canonical_passed:
                    errors.append(
                        f"{backend_type.__name__}/{object_count}: representation parity failed"
                    )
                del backend, output, actual
                gc.collect()
                if torch.device(args.device).type == "cuda":
                    torch.cuda.empty_cache()
    except BaseException as exc:
        caught = exc
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        unchanged = all(
            _sha(path) == source_hashes[str(path.relative_to(ROOT))] for path in sources
        )
        if not unchanged:
            errors.append("runtime source changed during benchmark")
        _write(
            args.output / "summary.json",
            {
                "status": "FAILED" if errors else "COMPLETE",
                "errors": errors,
                "sources_unchanged": unchanged,
                "results": results,
                "target_50hz_demonstrated": False,
            },
        )
    if caught:
        raise caught
    if errors:
        raise RuntimeError("Prefix benchmark failed; inspect summary.json")


def main() -> None:
    """Parse a standalone, scope-labelled before/after representation benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--index", type=int, default=553)
    parser.add_argument("--objects", type=int, nargs="+", default=[1, 20])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--roi-batch-size", type=int, default=4)
    parser.add_argument("--normalization", choices=("sort", "topk"), default="sort")
    parser.add_argument("--max-packet-events", type=int, default=2_000_000)
    parser.add_argument("--max-reference-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--packet-events", type=int, default=200)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
