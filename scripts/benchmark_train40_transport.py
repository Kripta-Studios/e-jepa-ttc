"""Measure complete bidirectional transport blocks with prepared GPU features.

This diagnostic includes correlation, both displacement fields, all four
global/foreground summaries and cycle sampling. It excludes CNN encoders,
sensor preparation, H2D and final TTC. It does not admit the frozen candidate.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def _short_windows_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer))
    if not length:
        raise OSError(f"Cannot resolve an existing short Windows path: {path}")
    result = Path(buffer.value)
    if not str(result).isascii():
        raise ValueError("The isolated compiler cache must have an ASCII short path")
    return result


def configure_isolated_compiler(dependencies: Path, cache: Path) -> dict[str, str | None]:
    """Select already installed dependencies and alter only this child process."""
    dependencies = dependencies.resolve(strict=True)
    if not (dependencies / "triton").is_dir():
        raise ValueError("--compiler-deps must contain the isolated triton package")
    cache.mkdir(parents=True, exist_ok=True)
    dependencies, cache = _short_windows_path(dependencies), _short_windows_path(cache.resolve())
    previous = {name: os.environ.get(name) for name in ("CUDA_PATH", "CC")}
    sys.path.insert(0, str(dependencies))
    os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    if os.name == "nt":
        compiler = dependencies / "triton/runtime/tcc/tcc.exe"
        cuda = dependencies / "triton/backends/nvidia"
        if not compiler.is_file() or not cuda.is_dir():
            raise ValueError("The isolated Windows compiler/CUDA bundle is incomplete")
        os.environ["CC"] = str(compiler)
        os.environ["CUDA_PATH"] = str(cuda)
    return previous


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parity(actual: tuple[Any, ...], expected: tuple[Any, ...]) -> dict[str, Any]:
    import torch

    results = []
    for name, a, b in zip(("forward", "reverse"), actual, expected, strict=True):
        finite = bool(torch.isfinite(a).all() & torch.isfinite(b).all())
        difference = (a - b).abs()
        tolerance = 1e-5 + 1e-5 * b.abs()
        results.append(
            {
                "name": name,
                "exact": bool(torch.equal(a, b)),
                "finite": finite,
                "max_abs": float(difference.max()),
                "outside_tolerance": int((difference > tolerance).sum()),
                "passed_1e5": finite and bool((difference <= tolerance).all()),
            }
        )
    return {
        "outputs": results,
        "exact": all(value["exact"] for value in results),
        "passed_1e5": all(value["passed_1e5"] for value in results),
    }


def _measure(call: Any, repeats: int) -> dict[str, Any]:
    import numpy as np
    import torch

    for _ in range(5):
        call()
    torch.cuda.synchronize()
    wall_ms, gpu_ms = [], []
    start_event, end_event = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    for _ in range(repeats):
        torch.cuda.synchronize()
        begin = time.perf_counter_ns()
        start_event.record()
        call()
        end_event.record()
        end_event.synchronize()
        wall_ms.append((time.perf_counter_ns() - begin) / 1e6)
        gpu_ms.append(start_event.elapsed_time(end_event))
    return {
        "wall_ms": wall_ms,
        "gpu_ms": gpu_ms,
        "median_ms": float(np.median(wall_ms)),
        "mean_ms": float(np.mean(wall_ms)),
        "p95_ms": float(np.quantile(wall_ms, 0.95)),
        "p99_ms": float(np.quantile(wall_ms, 0.99)),
        "max_ms": max(wall_ms),
        "median_gpu_ms": float(np.median(gpu_ms)),
    }


def run(args: argparse.Namespace) -> None:
    """Write both successful and failed probes, with immutable source bindings."""
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing experiment directory")
    args.output.mkdir(parents=True)
    sources = [
        Path(__file__),
        ROOT / "operational/train40_runtime/compiled_transport.py",
        ROOT / "operational/train40_runtime/transport.py",
        ROOT / "src/e_jepa_ttc/models/local_transport.py",
        ROOT / "src/e_jepa_ttc/models/causal_scale_ttc.py",
    ]
    source_hashes = {str(path.relative_to(ROOT)): _sha(path) for path in sources}
    config = {
        name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    report: dict[str, Any] = {
        "scope": "COMPLETE_BIDIRECTIONAL_TRANSPORT_PREPARED_SYNTHETIC_GPU_FEATURES_NO_TTC",
        "status": "RUNNING",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": source_hashes,
        "python": platform.python_version(),
        "host": platform.node(),
        "start_time": datetime.now(UTC).isoformat(),
        "experiment_id": args.output.name,
        "run_name": "train40_complete_transport_core",
        "config": config,
        "config_hash": config_hash,
        "dataset_manifest_hash": None,
        "split_version": None,
        "checkpoint_path": None,
        "metrics_path": str(args.output / "summary.json"),
        "seed": 7,
        "precision": "FP32_TF32_OFF",
        "objects": None,
        "batch_meaning": "independent endpoint pairs, not full objects",
        "synthetic_inputs": True,
        "final_ttc_parity_admitted": False,
        "records": [],
    }
    caught = None
    try:
        if any("compiled" in value for value in args.variants):
            if args.compiler_deps is None:
                raise ValueError("Compiled variants require an explicit --compiler-deps directory")
            report["previous_process_toolchain"] = configure_isolated_compiler(
                args.compiler_deps, args.output / "compiler_cache"
            )
        import torch

        from operational.train40_runtime.compiled_transport import TransportPairTensorCore

        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(7)
        report.update(
            device=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda
        )
        compile_options: dict[str, Any] = {
            "use_fast_math": False,
            "force_same_precision": True,
            "triton.cudagraphs": False,
        }
        report["compile_options"] = compile_options
        with torch.inference_mode():
            for batch in args.batches:
                for radius in args.radii:
                    shape = (batch, args.channels, args.size, args.size)
                    previous = torch.randn(shape, device="cuda")
                    current = torch.randn_like(previous)
                    foreground_previous = torch.rand(
                        (batch, 1, args.size, args.size), device="cuda"
                    )
                    foreground_current = torch.rand_like(foreground_previous)
                    inputs = (previous, current, foreground_previous, foreground_current)
                    oracle = TransportPairTensorCore(
                        radius,
                        args.temperature,
                        deterministic_sampling=args.deterministic_sampling,
                        reuse_reverse=False,
                        reuse_samples=False,
                    ).eval()
                    expected = tuple(value.clone() for value in oracle(*inputs))
                    for variant in args.variants:
                        record: dict[str, Any] = {
                            "batch": batch,
                            "radius": radius,
                            "shape": list(shape),
                            "variant": variant,
                        }
                        report["records"].append(record)
                        _write(args.output / "summary.json", report)
                        core = TransportPairTensorCore(
                            radius,
                            args.temperature,
                            deterministic_sampling=args.deterministic_sampling,
                            reuse_reverse="reverse" in variant,
                            reuse_samples=variant != "oracle",
                        ).eval()
                        if "compiled" in variant:
                            core = torch.compile(
                                core,
                                backend="inductor",
                                fullgraph=True,
                                dynamic=False,
                                options=compile_options,
                            )
                        torch.cuda.reset_peak_memory_stats()
                        begin = time.perf_counter()
                        actual = core(*inputs)
                        torch.cuda.synchronize()
                        record["setup_seconds"] = time.perf_counter() - begin
                        record["parity"] = _parity(actual, expected)
                        record["changed_cases"] = []
                        for a, b in (
                            (previous * 0.7, current * 0.4),
                            (torch.zeros_like(previous), torch.zeros_like(current)),
                        ):
                            changed = (a, b, foreground_previous, foreground_current)
                            record["changed_cases"].append(
                                _parity(core(*changed), oracle(*changed))
                            )
                        record["eager_or_compiled"] = _measure(partial(core, *inputs), args.repeats)
                        warmup_stream = torch.cuda.Stream()
                        warmup_stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(warmup_stream):
                            for _ in range(3):
                                core(*inputs)
                        torch.cuda.current_stream().wait_stream(warmup_stream)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            output = core(*inputs)
                        graph.replay()
                        record["graph_parity"] = _parity(output, expected)
                        record["graph"] = _measure(graph.replay, args.repeats)
                        record["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
                        record["passed_1e5"] = all(
                            item["passed_1e5"]
                            for item in (
                                record["parity"],
                                record["graph_parity"],
                                *record["changed_cases"],
                            )
                        )
                        print(json.dumps(record), flush=True)
                        _write(args.output / "summary.json", report)
                        del graph, core, output, actual
        report["status"] = "PROBE_COMPLETE_NOT_FINAL_TTC_ADMISSION"
    except BaseException as exc:
        caught = exc
        report.update(
            status="FAILED", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
        )
        print(report["traceback"], flush=True)
    finally:
        changed = [name for name, digest in source_hashes.items() if _sha(ROOT / name) != digest]
        report["changed_sources"] = changed
        if changed:
            report["status"] = "FAILED_SOURCE_CHANGED"
        report["end_time"] = datetime.now(UTC).isoformat()
        _write(args.output / "summary.json", report)
    if caught is not None:
        raise caught
    if report["changed_sources"]:
        raise RuntimeError("Sources changed during the experiment")


def main() -> None:
    """Parse a prepared-feature transport experiment with an explicit output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, default=[8])
    parser.add_argument("--radii", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--deterministic-sampling", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["oracle", "reuse_samples", "reuse_reverse", "compiled"],
        choices=["oracle", "reuse_samples", "reuse_reverse", "compiled", "compiled_reverse"],
    )
    parser.add_argument("--compiler-deps", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.batches, *args.radii, args.channels, args.size, args.repeats) < 1:
        parser.error("Batch, radius, channels, size and repetitions must be positive")
    if args.size <= 2 * max(args.radii):
        parser.error("Feature grid must exceed twice every radius")
    run(args)


if __name__ == "__main__":
    main()
