"""Audit exact integer RGB crop on real RAW histories, without models or TTC.

Every measured clock starts with required RAW bytes already available on CPU.
It includes immutable sealing of newly required frames, RAW H2D, current ROI
coefficient construction/upload, integer resizing, assembly and device sync.
Disk reads, prefix rebuilding and independent parity checks are separate. This
is a single-update preprocessing experiment, never a sustained service test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
SCOPE = "CPU_RAW_READY_TO_EXACT_RGB_DEVICE_READY_NO_MODELS_NO_TTC_NO_DISK"


def project_source_paths(modules: Mapping[str, Any], checkout: Path) -> set[Path]:
    """Bind only owned namespaces and reject missing or foreign owned source files.

    Vendor modules may expose synthetic relative ``__file__`` values such as
    Torch's ``_classes.py``. Those are not project dependencies merely because
    resolving their names against the working directory lands inside checkout.
    Namespace packages have no file; executable owned modules must have one.
    """
    root = checkout.resolve()
    paths: set[Path] = set()
    for name, module in tuple(modules.items()):
        if name.partition(".")[0] not in {"operational", "e_jepa_ttc", "scripts"}:
            continue
        filename = getattr(module, "__file__", None)
        if filename is None and getattr(module, "__path__", None) is not None:
            continue
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"Owned runtime source is missing for module {name}")
        path = Path(filename).resolve()
        if not path.is_relative_to(root):
            raise ValueError(
                f"Owned runtime module {name} was loaded outside this checkout: {path}"
            )
        if path.suffix != ".py" or not path.is_file():
            raise ValueError(f"Owned runtime source is missing or unsupported for {name}: {path}")
        paths.add(path)
    return paths


@contextmanager
def managed_runtime(
    ring_factory: Callable[[], Any],
    runtime_factory: Callable[[Any], Any],
    raw_path: Path,
    raw_stat: dict[str, Any],
    receipts: list[dict[str, Any]],
) -> Iterator[Any]:
    """Attempt every close and RAW stat check, preserving any primary failure."""
    ring = runtime = None
    failed = False
    try:
        ring = ring_factory()
        runtime = runtime_factory(ring)
        yield runtime
    except BaseException:
        failed = True
        raise
    finally:
        errors = []
        for name, resource in (("runtime", runtime), ("ring", ring)):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as error:
                    errors.append(f"{name}: {type(error).__name__}: {error}")
        check: dict[str, Any] = {"path": str(raw_path), "passed": False}
        try:
            after = raw_path.stat()
            check["passed"] = (after.st_size, after.st_mtime_ns) == (
                raw_stat["bytes"],
                raw_stat["mtime_ns"],
            )
        except OSError as error:
            check["error"] = f"{type(error).__name__}: {error}"
        receipts.append({"cleanup_errors": errors, "raw_stat_check": check})
        if not failed and (errors or not check["passed"]):
            raise RuntimeError("Case cleanup or RAW preservation failed; inspect cleanup receipts")


def arrival_partition(
    clocks: dict[int, int], query_us: int, step_us: int = 20_000
) -> tuple[list[int], list[int]]:
    """Partition real frame clocks at the preceding update, rejecting future RAW."""
    if step_us <= 0 or not clocks or any(value > query_us for value in clocks.values()):
        raise ValueError("Nonempty causal frame clocks and a positive step are required")
    previous = sorted(index for index, stamp in clocks.items() if stamp <= query_us - step_us)
    arriving = sorted(set(clocks) - set(previous))
    return previous, arriving


def register_observed_backend(backend: str) -> tuple[str, list[dict[str, Any]]]:
    """Record actual graph specializations; delegate every graph to the requested backend."""
    from torch._dynamo.backends.registry import lookup_backend, register_backend

    selected = lookup_backend(backend)
    records: list[dict[str, Any]] = []

    def observe(graph: Any, inputs: list[Any]) -> Any:
        started = time.perf_counter()
        record: dict[str, Any] = {
            "ordinal": len(records),
            "graph_sha256": hashlib.sha256(graph.code.encode()).hexdigest(),
            "inputs": [
                {
                    "shape": list(value.shape),
                    "stride": list(value.stride()),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "storage_offset": value.storage_offset(),
                }
                for value in inputs
                if hasattr(value, "shape")
            ],
        }
        records.append(record)
        try:
            result = selected(graph, inputs)
        except BaseException as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["backend_seconds"] = time.perf_counter() - started
        return result

    name = f"train40_pil_observer_{uuid.uuid4().hex}"
    register_backend(observe, name=name)
    return name, records


def seal_frames(raw: Any, selected: list[int], source: Any) -> dict[int, Any]:
    """Seal exactly the selected arriving RAW frames, with original IDs and clocks."""
    from operational.train40_runtime.rgb_raw_ring import freeze_rgb_frame

    return {
        index: freeze_rgb_frame(raw.frames[index], index, raw.timestamps_us[index], source)
        for index in selected
    }


def parity_receipt(prepared: Any, regions: list[Any], raw: Any) -> dict[str, Any]:
    """Compare all normalized pixels and deltas against independent PIL ROI-first."""
    import numpy as np

    from operational.train40_runtime.encoding import crop_uint8_roi_first

    expected, deltas = [], []
    for region in regions:
        unique = {
            index: crop_uint8_roi_first(frame, region.rgb_square)
            for index, frame in raw.frames.items()
        }
        for group in raw.triplets:
            expected.append(np.stack([unique[index] for index in group]))
            deltas.append(np.diff([raw.timestamps_us[index] for index in group]) / 1e6)
    expected_u8 = np.stack(expected)
    expected_f32 = expected_u8.astype(np.float32) / np.float32(255)
    actual = prepared.pixels.detach().cpu().numpy()
    actual_delta = prepared.delta.detach().cpu().numpy()
    finite = bool(np.isfinite(actual).all())
    normalized_exact = bool(np.array_equal(actual, expected_f32))
    # The LUT is injective over bytes; this additionally reports the byte image.
    byte_exact = finite and bool(
        np.array_equal(np.rint(actual * np.float32(255)).astype(np.uint8), expected_u8)
    )
    delta_exact = bool(np.array_equal(actual_delta, np.asarray(deltas, dtype=np.float32)))
    return {
        "passed": normalized_exact and byte_exact and delta_exact,
        "normalized_fp32_exact": normalized_exact,
        "uint8_exact": byte_exact,
        "delta_fp32_exact": delta_exact,
        "different_normalized_values": int(np.count_nonzero(actual != expected_f32)),
        "max_abs_normalized": float(np.max(np.abs(actual - expected_f32))) if finite else None,
        "shape": list(actual.shape),
        "actual_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
        "expected_sha256": hashlib.sha256(expected_f32.tobytes()).hexdigest(),
    }


def synchronize(device: Any) -> None:
    """Finish this experiment's device work before stopping the wall clock."""
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize(device)


def measure_update(
    runtime: Any,
    raw: Any,
    regions: list[Any],
    source: Any,
    *,
    prefix: bool,
    compilation_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure one update after an explicitly recorded optional causal prefix rebuild."""
    from operational.train40_runtime.rgb_pil_exact import FrozenRGBHistory

    runtime.ring.reset()
    prior, arriving = arrival_partition(dict(raw.timestamps_us), raw.query_us)
    setup_start = time.perf_counter()
    tokens: dict[int, Any] = {}
    prefix_receipt = None
    if prefix and prior:
        tokens = seal_frames(raw, prior, source)
        binding = runtime.ring.bind(list(tokens.values()), query_us=raw.query_us - 20_000)
        prefix_receipt = binding.receipt
        runtime.ring.mark_consumed(binding)
    synchronize(runtime.device)
    setup_seconds = time.perf_counter() - setup_start
    needed = arriving if prefix else sorted(raw.frames)
    before_compilations = len(compilation_records)
    started = time.perf_counter()
    tokens.update(seal_frames(raw, needed, source))
    sealing_seconds = time.perf_counter() - started
    history = FrozenRGBHistory(raw.query_us, tokens, raw.triplets, raw.unavailable_reason)
    prepared = runtime.prepare(regions, history)
    try:
        synchronize(runtime.device)
        wall_ms = (time.perf_counter() - started) * 1000
        receipt = {
            "scope": SCOPE,
            "mode": "causal_prefix_rebuilt" if prefix else "full_history_miss",
            "query_us": raw.query_us,
            "prefix_cutoff_us": raw.query_us - 20_000 if prefix else None,
            "prefix_setup_seconds_excluded": setup_seconds,
            "prefix_receipt": prefix_receipt,
            "newly_sealed_frame_ids": needed,
            "newly_sealed_bytes": sum(tokens[index].hashed_bytes for index in needed),
            "sealing_seconds_included": sealing_seconds,
            "wall_ms": wall_ms,
            "new_compilations_inside_clock": len(compilation_records) - before_compilations,
            "prepare_receipt": prepared.receipt,
            "parity": parity_receipt(prepared, regions, raw),
        }
    finally:
        prepared.release()
    return receipt


def prime_synthetic(runtime: Any, regions: list[Any], raw: Any) -> dict[str, Any]:
    """Prime a declared ROI shape profile using synthetic bytes, never real pixels.

    The profile is supplied by this microbenchmark's fixed cases. It does not
    claim arbitrary future ROI shapes are precompiled in a streaming service.
    """
    import numpy as np

    from operational.train40_runtime.multiobject import RawRGBHistory
    from operational.train40_runtime.rgb_pil_exact import FrozenRGBHistory
    from operational.train40_runtime.rgb_raw_ring import RGBSourceIdentity

    started = time.perf_counter()
    frame = next(iter(raw.frames.values()))
    synthetic = np.zeros(frame.shape, dtype=np.uint8)
    history = RawRGBHistory(
        raw.query_us,
        {index: synthetic for index in raw.frames},
        dict(raw.timestamps_us),
        raw.triplets,
    )
    source = RGBSourceIdentity("SYNTHETIC_DECLARED_SHAPE_PROFILE", "0" * 64)
    tokens = seal_frames(history, sorted(history.frames), source)
    runtime.ring.reset()
    prepared = runtime.prepare(
        regions, FrozenRGBHistory(history.query_us, tokens, history.triplets)
    )
    try:
        synchronize(runtime.device)
        return {
            "seconds": time.perf_counter() - started,
            "profile": "MANIFEST_ROI_SHAPES_DECLARED_AT_BENCHMARK_SETUP",
            "real_pixels_used": False,
            "covers_arbitrary_future_roi": False,
            "shape_buckets": prepared.receipt["shape_buckets"],
        }
    finally:
        prepared.release()
        runtime.ring.reset()


def run(args: argparse.Namespace) -> None:
    """Preserve cold failures and every warmed MISS; require an unused output directory."""
    if args.repeats < 1 or not args.indices or len(set(args.indices)) != len(args.indices):
        raise ValueError("Use positive repeats and nonempty unique query indices")
    if (
        not args.objects
        or len(set(args.objects)) != len(args.objects)
        or any(value not in (1, 2) for value in args.objects)
    ):
        raise ValueError("This milestone experiment supports one or two objects")
    output, manifest = args.output.resolve(), args.manifest.resolve()
    if output == manifest.parent or output.is_relative_to(manifest.parent):
        raise ValueError("Evidence must be separate from the frozen manifest directory")
    output.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "schema": "train40_rgb_pil_exact_probe_v1",
        "status": "FAILED",
        "scope": SCOPE,
        "complete_update_admitted": False,
        "sustained_50hz_demonstrated": False,
        "targets_read": False,
        "optimizer_updates": 0,
        "weights_or_normalizers_loaded": False,
        "records": [],
        "cold_records": [],
        "startup": [],
        "cleanup": [],
    }
    original: dict[str, Any] = {}
    caught: BaseException | None = None
    try:
        if args.compile:
            if args.compiler_root is None:
                raise ValueError("--compile requires --compiler-root")
            from scripts.benchmark_train40_encoder import configure_compiler

            summary["compiler_environment"] = configure_compiler(args.compiler_root, output)
        import numpy as np
        import PIL
        import torch

        from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
        from operational.train40_runtime.inputs import PersistentSensorInputs
        from operational.train40_runtime.measurement import (
            latency_distribution,
            snapshot_files,
            verify_snapshot,
        )
        from operational.train40_runtime.multiobject import read_raw_rgb_history
        from operational.train40_runtime.rgb_pil_exact import ExactPILRGBRuntime
        from operational.train40_runtime.rgb_raw_ring import ResidentRGBFrameRing, RGBSourceIdentity
        from scripts.benchmark_train40_multiobject import artificial_regions
        from scripts.final_train40_transfer import reject_targets

        torch.set_num_threads(2)
        torch.manual_seed(7)
        np.random.seed(7)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        data = json.loads(manifest.read_text(encoding="utf-8"))
        rows = data["rows"]
        reject_targets(rows)
        if any(index < 0 or index >= len(rows) for index in args.indices):
            raise ValueError("Query index is outside the supplied manifest")
        sources = {Path(__file__).resolve(), *project_source_paths(sys.modules, ROOT)}
        original = snapshot_files([manifest, *sources])
        (output / "source_snapshots").mkdir()
        source_snapshots = []
        for index, source in enumerate(sorted(sources)):
            payload = source.read_bytes()
            snapshot_path = Path("source_snapshots") / f"{index:03d}_{source.name}.txt"
            (output / snapshot_path).write_bytes(payload)
            source_snapshots.append(
                {
                    "source": str(source.relative_to(ROOT)),
                    "snapshot": snapshot_path.as_posix(),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        (output / "original_snapshot.json").write_text(json.dumps(original, indent=2))
        (output / "source_snapshots.json").write_text(
            json.dumps(source_snapshots, indent=2), encoding="utf-8"
        )
        backend, compilation_records = (
            register_observed_backend("inductor") if args.compile else (None, [])
        )
        device = torch.device(args.device)
        summary["binding"] = {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "command_argv": sys.argv,
            "cwd": str(Path.cwd()),
            "manifest": str(manifest),
            "output": str(output),
            "compiler_root": str(args.compiler_root) if args.compiler_root is not None else None,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "pillow": PIL.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "seed": 7,
            "torch_threads": 2,
            "compile": args.compile,
            "frame_chunk_size": 1,
            "ring_capacity": 16,
            "indices": args.indices,
            "objects": args.objects,
            "repeats": args.repeats,
            "compile_limit_unchanged": True,
            "cold_definition": (
                "FIRST_REAL_PREPARE_PER_RUNTIME; compiler code/cache may already exist "
                "from preceding cases"
            ),
            "warmed_definition": (
                "EXPLICIT_SYNTHETIC_PRIMING_OF_DECLARED_CASE_SHAPES; "
                "all measured RAW misses retained"
            ),
            "clock_excludes": ["disk", "prefix_rebuild", "post_clock_D2H_parity", "models", "TTC"],
        }
        summary["compilations"] = compilation_records
        with PersistentSensorInputs() as inputs:
            for count in args.objects:
                for index in args.indices:
                    row = rows[index]
                    raw = read_raw_rgb_history(inputs, row)
                    if not raw.triplets:
                        raise ValueError(f"Index {index} lacks RGB: {raw.unavailable_reason}")
                    boxes = [inputs._box(item) for item in list(row["bbox_sources"])[-2:]]
                    rgb_square = common_square_from_boxes(boxes, (0, 1), margin_fraction=0.25)
                    regions = artificial_regions(row["square_xyxy"], rgb_square, count)
                    path = Path(row["raw_path"]).resolve()
                    stat = path.stat()
                    raw_stat = {
                        "path": str(path),
                        "bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                    source = RGBSourceIdentity(
                        str(path),
                        hashlib.sha256(json.dumps(raw_stat, sort_keys=True).encode()).hexdigest(),
                    )
                    frame_shape = next(iter(raw.frames.values())).shape
                    with managed_runtime(
                        lambda frame_shape=frame_shape: ResidentRGBFrameRing(
                            device, frame_shape=frame_shape, capacity=16
                        ),
                        lambda ring, count=count: ExactPILRGBRuntime(
                            ring,
                            max_objects=count,
                            max_history=8,
                            frame_chunk_size=1,
                            compile_kernels=args.compile,
                            compile_backend=backend,
                        ),
                        path,
                        raw_stat,
                        summary["cleanup"],
                    ) as runtime:
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)
                        cold = measure_update(
                            runtime,
                            raw,
                            regions,
                            source,
                            prefix=False,
                            compilation_records=compilation_records,
                        )
                        cold.update(
                            index=index,
                            objects=count,
                            phase="cold_first_actual_prepare",
                            raw_stat=raw_stat,
                        )
                        summary["cold_records"].append(cold)
                        prime = prime_synthetic(runtime, regions, raw)
                        summary["startup"].append(
                            {
                                "index": index,
                                "objects": count,
                                "synthetic_priming": prime,
                                "allocation": runtime.allocation_receipt,
                            }
                        )
                        for repeat in range(args.repeats):
                            for prefix in (False, True):
                                record = measure_update(
                                    runtime,
                                    raw,
                                    regions,
                                    source,
                                    prefix=prefix,
                                    compilation_records=compilation_records,
                                )
                                record.update(
                                    index=index,
                                    objects=count,
                                    repeat=repeat,
                                    phase="synthetically_primed",
                                    raw_stat=raw_stat,
                                )
                                summary["records"].append(record)
                                print(
                                    json.dumps(
                                        {
                                            key: record[key]
                                            for key in (
                                                "index",
                                                "objects",
                                                "repeat",
                                                "mode",
                                                "wall_ms",
                                                "new_compilations_inside_clock",
                                            )
                                        }
                                    ),
                                    flush=True,
                                )
                        if device.type == "cuda":
                            summary["startup"][-1]["memory"] = {
                                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                            }
                    (output / "summary.json").write_text(
                        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
                    )
        all_records = [*summary["cold_records"], *summary["records"]]
        expected = len(args.objects) * len(args.indices) * (1 + 2 * args.repeats)
        summary["coverage"] = {
            "expected": expected,
            "measured": len(all_records),
            "passed": len(all_records) == expected,
        }
        summary["all_exact"] = all(record["parity"]["passed"] for record in all_records)
        summary["warmed_recompilations"] = sum(
            record["new_compilations_inside_clock"] for record in summary["records"]
        )
        summary["latency"] = [
            {
                "objects": count,
                "index": index,
                "mode": mode,
                **latency_distribution(
                    [
                        record["wall_ms"]
                        for record in summary["records"]
                        if (record["objects"], record["index"], record["mode"])
                        == (count, index, mode)
                    ]
                ),
            }
            for count in args.objects
            for index in args.indices
            for mode in ("full_history_miss", "causal_prefix_rebuilt")
        ]
        summary["status"] = (
            "COMPLETE"
            if summary["coverage"]["passed"] and summary["all_exact"]
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
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
    if caught is not None:
        raise caught
    if summary["status"] != "COMPLETE":
        raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    """Construct a bounded opt-in CLI without touching CUDA."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--indices", type=int, nargs="+", default=[0, 137, 553])
    result.add_argument("--objects", type=int, nargs="+", default=[1])
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--compile", action="store_true")
    result.add_argument("--compiler-root", type=Path)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
