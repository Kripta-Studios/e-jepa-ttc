"""Separate CPU copy, payload SHA-256 and descriptor costs on one real RGB history.

This isolated microbenchmark is not a causal replay or a runtime identity change.
Both paths independently copy and hash the same already-available RAW frames.
Validation, source inventory and logging are outside the measured intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from operational.train40_runtime.rgb_raw_ring import FrozenRGBFrame, RGBSourceIdentity

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
SCOPE = "CPU_ONLY_ORIGINAL_FRAME_FREEZE_VS_COPY_SHA_DESCRIPTOR_NO_RUNTIME_CHANGE"


@dataclass(frozen=True)
class SplitSealing:
    """Diagnostic bytes and identity; this is deliberately not a runtime token."""

    payload: bytes = field(repr=False)
    checksum: str
    identity: str
    shape: tuple[int, int, int]
    copy_ns: int
    payload_sha256_ns: int
    descriptor_ns: int
    total_ns: int


def split_sealing(
    frame: np.ndarray, frame_id: int, timestamp_us: int, source: RGBSourceIdentity
) -> SplitSealing:
    """Time independent C-order copying, payload SHA and original descriptor SHA.

    These intervals exclude the factory's validation, private record allocation,
    weak registry and token publication. They retain the exact identity formula.
    Their sum is not an implemented or admitted replacement for the factory.
    """
    import numpy as np

    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected original nonempty uint8 HWC RGB data")
    if min(frame.shape) < 1:
        raise ValueError("Expected original nonempty uint8 HWC RGB data")
    started = time.perf_counter_ns()
    payload = frame.tobytes(order="C")
    copied = time.perf_counter_ns()
    checksum = hashlib.sha256(payload).hexdigest()
    hashed = time.perf_counter_ns()
    shape = (int(frame.shape[0]), int(frame.shape[1]), 3)
    descriptor = (
        source.name,
        source.revision_sha256,
        frame_id,
        timestamp_us,
        shape,
        "uint8_HWC_RGB",
        checksum,
    )
    identity = hashlib.sha256(json.dumps(descriptor, separators=(",", ":")).encode()).hexdigest()
    finished = time.perf_counter_ns()
    return SplitSealing(
        payload,
        checksum,
        identity,
        shape,
        copied - started,
        hashed - copied,
        finished - hashed,
        finished - started,
    )


def measure_original(raw: Any, source: RGBSourceIdentity) -> tuple[list[FrozenRGBFrame], int]:
    """Time the unchanged factory over every frame in the original history."""
    from operational.train40_runtime.rgb_raw_ring import freeze_rgb_frame

    ids = sorted(raw.frames)
    started = time.perf_counter_ns()
    tokens = [
        freeze_rgb_frame(raw.frames[index], index, raw.timestamps_us[index], source)
        for index in ids
    ]
    return tokens, time.perf_counter_ns() - started


def measure_split(raw: Any, source: RGBSourceIdentity) -> tuple[list[SplitSealing], int]:
    """Time the independently instrumented stages without using any original digest."""
    ids = sorted(raw.frames)
    started = time.perf_counter_ns()
    values = [
        split_sealing(raw.frames[index], index, raw.timestamps_us[index], source) for index in ids
    ]
    return values, time.perf_counter_ns() - started


def compare_pair(
    raw: Any, tokens: list[FrozenRGBFrame], split: list[SplitSealing]
) -> list[dict[str, Any]]:
    """Check identity and all pixel bytes after both paths' clocks have stopped."""
    import numpy as np

    ids = sorted(raw.frames)
    if len(tokens) != len(ids) or len(split) != len(ids):
        raise ValueError("Both routes must cover every original frame exactly once")
    checks = []
    for index, token, value in zip(ids, tokens, split, strict=True):
        pixels = np.frombuffer(value.payload, dtype=np.uint8).reshape(value.shape)
        checks.append(
            {
                "frame_id": index,
                "timestamp_us": raw.timestamps_us[index],
                "shape": list(raw.frames[index].shape),
                "strides": list(raw.frames[index].strides),
                "dtype": str(raw.frames[index].dtype),
                "c_contiguous": bool(raw.frames[index].flags.c_contiguous),
                "bytes": raw.frames[index].nbytes,
                "frame_id_and_clock_exact": token.frame_id == index
                and token.timestamp_us == raw.timestamps_us[index],
                "payload_bytes_exact": token.payload == value.payload,
                "pixels_exact": bool(np.array_equal(token.array, raw.frames[index]))
                and bool(np.array_equal(pixels, raw.frames[index])),
                "payload_sha256_exact": token.content_sha256 == value.checksum,
                "descriptor_sha256_exact": token.identity_sha256 == value.identity,
                "original_sha256": token.content_sha256,
                "split_sha256": value.checksum,
                "original_identity_sha256": token.identity_sha256,
                "split_identity_sha256": value.identity,
                "original_factory_reported_ms": token.freeze_seconds * 1000,
                "copy_ms": value.copy_ns / 1e6,
                "payload_sha256_ms": value.payload_sha256_ns / 1e6,
                "descriptor_ms": value.descriptor_ns / 1e6,
                "split_frame_ms": value.total_ns / 1e6,
            }
        )
    return checks


def make_pair(raw: Any, source: RGBSourceIdentity, *, order: list[str]) -> dict[str, Any]:
    """Execute an AB or BA pair, retaining all timings and validation outcomes."""
    if order not in (["original", "split"], ["split", "original"]):
        raise ValueError("Specify exactly one original and one independently split route")
    results = {}
    for name in order:
        results[name] = (
            measure_original(raw, source) if name == "original" else measure_split(raw, source)
        )
    tokens, original_ns = results["original"]
    split, split_ns = results["split"]
    checks = compare_pair(raw, tokens, split)
    fields = (
        "frame_id_and_clock_exact",
        "payload_bytes_exact",
        "pixels_exact",
        "payload_sha256_exact",
        "descriptor_sha256_exact",
    )
    copy_ns = sum(value.copy_ns for value in split)
    sha_ns = sum(value.payload_sha256_ns for value in split)
    descriptor_ns = sum(value.descriptor_ns for value in split)
    return {
        "order": order,
        "passed": all(all(check[name] for name in fields) for check in checks),
        "frames": len(checks),
        "bytes": sum(check["bytes"] for check in checks),
        "original_ms": original_ns / 1e6,
        "split_total_ms": split_ns / 1e6,
        "copy_ms": copy_ns / 1e6,
        "payload_sha256_ms": sha_ns / 1e6,
        "descriptor_ms": descriptor_ns / 1e6,
        "split_wrapper_and_instrumentation_ms": (split_ns - copy_ns - sha_ns - descriptor_ns) / 1e6,
        "frame_checks": checks,
        "parity_outside_both_clocks": True,
        "no_digest_reused_between_routes": True,
    }


def distribution(values: list[float]) -> dict[str, float | int]:
    """Describe the declared measured population without dropping slow samples."""
    import numpy as np

    data = np.asarray(values, dtype=np.float64)
    if not len(data) or not np.isfinite(data).all():
        raise ValueError("A nonempty finite CPU timing population is required")
    return {
        "count": len(data),
        "mean_ms": float(data.mean()),
        "median_ms": float(np.median(data)),
        "p95_ms": float(np.quantile(data, 0.95)),
        "p99_ms": float(np.quantile(data, 0.99)),
        "min_ms": float(data.min()),
        "max_ms": float(data.max()),
    }


def run(args: argparse.Namespace) -> None:
    """Read one available history and preserve the complete paired CPU diagnostic."""
    if args.repeats < 1 or args.warmups < 0:
        raise ValueError("Positive repeats and nonnegative warmups are required")
    output, manifest = args.output.resolve(), args.manifest.resolve()
    if output == manifest.parent or output.is_relative_to(manifest.parent):
        raise ValueError("Diagnostic output must be separate from preserved manifest inputs")
    output.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "schema": "train40_frame_sealing_cpu_diagnostic_v1",
        "scope": SCOPE,
        "status": "FAILED",
        "gpu_used": False,
        "runtime_modified": False,
        "weights_or_normalizers_loaded": False,
        "targets_read": False,
        "optimizer_updates": 0,
        "sustained_or_causal_claim": False,
        "records": [],
    }
    original: dict[str, Any] = {}
    raw_path: Path | None = None
    raw_stat: tuple[int, int] | None = None
    caught: BaseException | None = None
    try:
        import numpy as np
        import torch

        from operational.train40_runtime.inputs import PersistentSensorInputs
        from operational.train40_runtime.measurement import snapshot_files, verify_snapshot
        from operational.train40_runtime.multiobject import read_raw_rgb_history
        from operational.train40_runtime.rgb_raw_ring import RGBSourceIdentity
        from scripts.benchmark_train40_rgb_pil_exact import project_source_paths
        from scripts.final_train40_transfer import reject_targets

        torch.set_num_threads(2)
        rows = json.loads(manifest.read_text(encoding="utf-8"))["rows"]
        reject_targets(rows)
        if not 0 <= args.index < len(rows):
            raise ValueError("Index lies outside the preserved query manifest")
        row = rows[args.index]
        raw_path = Path(row["raw_path"]).resolve()
        stat = raw_path.stat()
        raw_stat = (stat.st_size, stat.st_mtime_ns)
        source = RGBSourceIdentity(
            str(raw_path),
            hashlib.sha256(
                json.dumps((str(raw_path), *raw_stat), separators=(",", ":")).encode()
            ).hexdigest(),
        )
        sources = {Path(__file__).resolve(), *project_source_paths(sys.modules, ROOT)}
        original = snapshot_files([manifest, *sources])
        (output / "original_snapshot.json").write_text(
            json.dumps(original, indent=2), encoding="utf-8"
        )
        (output / "sources").mkdir()
        source_records = []
        for index, path in enumerate(sorted(sources)):
            payload = path.read_bytes()
            saved = Path("sources") / f"{index:03d}_{path.name}.txt"
            (output / saved).write_bytes(payload)
            source_records.append(
                {
                    "source": str(path.relative_to(ROOT)),
                    "snapshot": saved.as_posix(),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        (output / "sources.json").write_text(json.dumps(source_records, indent=2), encoding="utf-8")
        summary["binding"] = {
            "command_argv": sys.argv,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status": subprocess.check_output(
                ["git", "status", "--short"], text=True
            ).splitlines(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "clock": vars(time.get_clock_info("perf_counter")),
            "index": args.index,
            "query_id": row["query_id"],
            "query_us": int(row["anchor_us"]),
            "raw_path": str(raw_path),
            "raw_size_bytes": raw_stat[0],
            "raw_mtime_ns": raw_stat[1],
            "source_revision_sha256": source.revision_sha256,
            "warmup_pairs": args.warmups,
            "measured_pairs": args.repeats,
            "quantile_method": "linear",
            "paired_order": "AB then BA, alternating within warmup and measured populations",
            "clock_excludes": "HDF5 read, verification, source snapshots, reporting",
            "split_limitations": (
                "No validation/record/registry/token publication; "
                "instrumented stages are diagnostic, not runtime V2"
            ),
            "future_frames_prehashed": False,
        }
        with PersistentSensorInputs() as inputs:
            raw = read_raw_rgb_history(inputs, row)
            if len(raw.frames) != 7:
                raise ValueError("This protocol expects the seven-frame index-553 history")
            if any(stamp > raw.query_us for stamp in raw.timestamps_us.values()):
                raise ValueError("Future RGB timestamps cannot enter this diagnostic")
            frame_digests: dict[int, str] | None = None
            for phase, pairs in (("warmup", args.warmups), ("measured", args.repeats)):
                for repeat in range(pairs):
                    order = ["original", "split"] if repeat % 2 == 0 else ["split", "original"]
                    record = make_pair(raw, source, order=order)
                    record.update(phase=phase, repeat=repeat)
                    digests = {
                        item["frame_id"]: item["original_sha256"] for item in record["frame_checks"]
                    }
                    if frame_digests is None:
                        frame_digests = digests
                    record["same_loaded_source_pixels_as_first_pair"] = digests == frame_digests
                    record["passed"] &= record["same_loaded_source_pixels_as_first_pair"]
                    summary["records"].append(record)
                    with (output / "pairs.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
            fields = (
                "original_ms",
                "split_total_ms",
                "copy_ms",
                "payload_sha256_ms",
                "descriptor_ms",
                "split_wrapper_and_instrumentation_ms",
            )
            summary["distributions"] = {
                phase: {
                    name: distribution(
                        [record[name] for record in summary["records"] if record["phase"] == phase]
                    )
                    for name in fields
                }
                for phase in ("warmup", "measured")
                if any(record["phase"] == phase for record in summary["records"])
            }
            summary["coverage"] = {
                "expected_pairs": args.warmups + args.repeats,
                "recorded_pairs": len(summary["records"]),
                "frames_per_pair": len(raw.frames),
                "passed": len(summary["records"]) == args.warmups + args.repeats,
            }
        summary["all_parity_passed"] = all(record["passed"] for record in summary["records"])
        summary["status"] = (
            "COMPLETE"
            if summary["coverage"]["passed"] and summary["all_parity_passed"]
            else "PARITY_FAILED"
        )
    except BaseException as error:
        caught = error
        summary.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        if original:
            from operational.train40_runtime.measurement import verify_snapshot

            summary["preservation"] = verify_snapshot(original)
            if not summary["preservation"]["passed"]:
                summary["status"] = "FAILED"
        if raw_path is not None and raw_stat is not None:
            try:
                after = raw_path.stat()
                summary["raw_stat_preserved"] = (after.st_size, after.st_mtime_ns) == raw_stat
            except OSError as error:
                summary["raw_stat_preserved"] = False
                summary["raw_stat_error"] = f"{type(error).__name__}: {error}"
            if not summary["raw_stat_preserved"]:
                summary["status"] = "FAILED"
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
    if caught is not None:
        raise caught
    if summary["status"] != "COMPLETE":
        raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    """Expose a reproducible CPU-only diagnostic with the declared default population."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--index", type=int, default=553)
    result.add_argument("--warmups", type=int, default=2)
    result.add_argument("--repeats", type=int, default=20)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
