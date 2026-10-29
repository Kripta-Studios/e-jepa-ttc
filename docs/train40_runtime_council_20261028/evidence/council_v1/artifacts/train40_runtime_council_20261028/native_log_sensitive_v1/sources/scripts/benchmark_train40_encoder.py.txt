"""Compare isolated frozen encoders: CUDA Graph versus compiled CUDA Graph.

Inputs derive from one registered query, with declared artificial endpoint
batches for capacity. This excludes sensors, H2D, transport, context and TTC.
Compilation, graph setup and parity checks are outside the latency clock.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def configure_compiler(compiler_root: Path, output: Path) -> dict[str, str]:
    """Select the existing isolated Triton toolchain for this process only."""

    def ascii_path(path: Path) -> Path:
        resolved = path.resolve(strict=True)
        if os.name == "nt":
            buffer = ctypes.create_unicode_buffer(32768)
            if not ctypes.windll.kernel32.GetShortPathNameW(str(resolved), buffer, len(buffer)):
                raise OSError(f"Cannot resolve existing short path: {resolved}")
            resolved = Path(buffer.value)
        if not str(resolved).isascii():
            raise ValueError("Compiler paths must be ASCII in this isolated Windows experiment")
        return resolved

    isolated, cache = ascii_path(compiler_root), ascii_path(output)
    deps = isolated / "deps"
    if not (deps / "triton").is_dir():
        raise ValueError("compiler-root must contain the existing isolated deps/triton package")
    sys.path.insert(0, str(deps))
    selected = {
        "TRITON_CACHE_DIR": str(cache / "triton_cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor_cache"),
    }
    if os.name == "nt":
        selected.update(
            CUDA_PATH=str(deps / "triton/backends/nvidia"),
            CC=str(deps / "triton/runtime/tcc/tcc.exe"),
        )
        if not Path(selected["CC"]).is_file():
            raise ValueError("Isolated TinyCC executable is missing")
    os.environ.update(selected)
    return selected


def tensor_parity(actual: tuple[Tensor, ...], expected: tuple[Tensor, ...]) -> dict[str, Any]:
    """Compare every encoder output with the fixed 1e-5 combined tolerance."""
    import torch

    from operational.train40_runtime.compiled_encoder import ENCODER_OUTPUT_NAMES

    rows = []
    for name, value, reference in zip(ENCODER_OUTPUT_NAMES, actual, expected, strict=True):
        if value.shape != reference.shape:
            rows.append({"name": name, "passed": False, "reason": "shape"})
            continue
        lhs, rhs = value.detach().double().cpu(), reference.detach().double().cpu()
        finite = torch.isfinite(lhs) & torch.isfinite(rhs)
        difference = (lhs - rhs).abs()
        passed = finite & (difference <= 1e-5 + 1e-5 * rhs.abs())
        rows.append(
            {
                "name": name,
                "shape": list(value.shape),
                "passed": bool(passed.all()),
                "exact": bool(torch.equal(lhs, rhs)),
                "outside_tolerance": int((~passed).sum()),
                "max_abs": float(difference[finite].max()) if bool(finite.any()) else None,
            }
        )
    return {
        "passed": all(row["passed"] for row in rows),
        "outputs": rows,
        "atol": 1e-5,
        "rtol": 1e-5,
    }


def run(args: argparse.Namespace) -> None:
    """Preserve measured failures and never promote an encoder probe to TTC admission."""
    # Import torch only after selecting isolated Triton paths/environment.
    output = args.output.resolve()
    for protected in (args.candidate, args.manifest.parent, args.reference):
        path = protected.resolve()
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError("Probe output must be separate from preserved inputs")
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "scope": "PREPARED_ENDPOINT_ENCODERS_ONLY_NO_SENSOR_H2D_TRANSPORT_CONTEXT_TTC",
        "status": "FAILED",
        "final_ttc_admitted": False,
        "target_20_objects_50hz_demonstrated": False,
        "records": [],
    }
    original = {}
    caught: BaseException | None = None
    try:
        report["compiler_environment"] = configure_compiler(args.compiler_root, output)
        import numpy as np
        import torch
        import triton  # pyright: ignore[reportMissingImports]

        from operational.rgb_port import transfer as engine
        from operational.train40_runtime import compiled_encoder
        from operational.train40_runtime.compiled_encoder import (
            CompiledEndpointEncoder,
            EncoderTensorCore,
        )
        from operational.train40_runtime.measurement import (
            latency_distribution,
            snapshot_files,
            verify_snapshot,
        )
        from scripts.benchmark_train40_runtime import _candidate_inventory, _validate_reference
        from scripts.final_train40_transfer import (
            load_models,
            read,
            reject_targets,
            sha,
            verify_candidate,
            verify_frozen_runtime,
        )

        torch.set_num_threads(2)
        torch.manual_seed(7)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        paths, binding = verify_candidate(args.candidate)
        frozen = verify_frozen_runtime(args.candidate)
        rows = read(args.manifest)["rows"]
        reject_targets(rows)
        _, reference_files, reference_binding = _validate_reference(
            args.reference, args.manifest, rows, binding
        )
        if not 0 <= args.query_index < len(rows):
            raise ValueError("query-index must belong to the registered manifest")
        sources = {Path(__file__).resolve(), Path(compiled_encoder.__file__).resolve()}
        for name, expected_sha in {
            **frozen["source_sha256"],
            **reference_binding["sources"],
        }.items():
            path = Path(name)
            path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
            if not path.is_relative_to(ROOT) or sha(path) != expected_sha:
                raise ValueError(f"Frozen source identity differs: {name}")
            sources.add(path)
        sources.update(
            ROOT / name
            for name in (
                "scripts/final_train40_transfer.py",
                "scripts/benchmark_train40_runtime.py",
                "operational/train40_runtime/measurement.py",
            )
        )
        original = snapshot_files(
            [*_candidate_inventory(args.candidate), *reference_files, args.manifest, *sources]
        )
        (output / "original_snapshot.json").write_text(
            json.dumps(original, indent=2), encoding="utf-8"
        )
        row = rows[args.query_index]
        event = engine.prepare_event(row)["own_events"]
        rgb, reason = engine.raw_rgb_observations(row)
        if len(event) != 4 or len(rgb) != 5:
            raise ValueError(f"Probe requires H4 event/H5 RGB: RGB reason {reason}")
        sensors = {"E": event, "R": np.stack([value for value, _ in rgb])}
        endpoints = {}
        for key, sensor in sensors.items():
            if not np.array_equal(sensor[:-1, 1:], sensor[1:, :-1]):
                raise ValueError(f"{key}: registered endpoints do not overlap exactly")
            endpoint_values = (
                sensor.reshape(-1, *sensor.shape[2:])
                if args.endpoint_layout == "original"
                else np.concatenate((sensor[0], sensor[1:, -1]))
            )
            endpoints[key] = torch.from_numpy(endpoint_values).float()
        models, _ = load_models(paths, "cuda")
        report.update(
            model_binding=binding,
            query_id=row["query_id"],
            manifest_sha256=sha(args.manifest),
            source_sha256={str(path.relative_to(ROOT)): sha(path) for path in sorted(sources)},
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            python=platform.python_version(),
            torch=torch.__version__,
            triton=triton.__version__,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(),
            precision="FP32_TF32_OFF",
            cudnn_benchmark=args.cudnn_benchmark,
            endpoint_layout=args.endpoint_layout,
            input_scope=(
                "One real query; repeated endpoint groups receive deterministic "
                "per-group scaling for artificial capacity"
            ),
        )
        with torch.inference_mode():
            for name in args.models:
                source = models[name].encoder
                native = endpoints["E" if name.startswith("E_") else "R"]
                batches = args.batches or [len(native), 20 * len(native)]
                for batch in batches:
                    groups = (batch + len(native) - 1) // len(native)
                    values = native.repeat(groups, 1, 1, 1)[:batch]
                    factors = 1 - 0.001 * (torch.arange(batch) // len(native)).float()
                    values = (values * factors[:, None, None, None]).cuda()
                    baseline = EncoderTensorCore(source).eval()
                    proposed = CompiledEndpointEncoder(source, conv_backends=args.conv_backends)
                    record: dict[str, Any] = {
                        "model": name,
                        "batch": batch,
                        "shape": list(values.shape),
                        "options": proposed.options,
                    }
                    report["records"].append(record)
                    expected = tuple(x.clone() for x in baseline(values))
                    begin = time.perf_counter()
                    actual = proposed.flat(values)
                    torch.cuda.synchronize()
                    record["compile_and_first_execution_seconds"] = time.perf_counter() - begin
                    record["parity"] = tensor_parity(actual, expected)
                    graphs, outputs = {}, {}
                    graph_begin = time.perf_counter()
                    for label, call in (
                        ("baseline_graph", baseline),
                        ("compiled_graph", proposed.flat),
                    ):
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(3):
                                call(values)
                        torch.cuda.current_stream().wait_stream(stream)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            graph_output = call(values)
                        graphs[label], outputs[label] = graph, graph_output
                    torch.cuda.synchronize()
                    record["graph_setup_seconds"] = time.perf_counter() - graph_begin
                    record["changed_input_parity"] = []
                    for transformed in (values.clone() * 0.7, torch.zeros_like(values)):
                        values.copy_(transformed)
                        for graph in graphs.values():
                            graph.replay()
                        torch.cuda.synchronize()
                        record["changed_input_parity"].append(
                            tensor_parity(outputs["compiled_graph"], outputs["baseline_graph"])
                        )
                    values.copy_(
                        (
                            native.repeat(groups, 1, 1, 1)[:batch] * factors[:, None, None, None]
                        ).cuda()
                    )
                    timings: dict[str, list[float]] = {key: [] for key in graphs}
                    for repetition in range(args.repeats):
                        order = list(graphs) if repetition % 2 == 0 else list(reversed(graphs))
                        for label in order:
                            torch.cuda.synchronize()
                            begin = time.perf_counter_ns()
                            graphs[label].replay()
                            torch.cuda.synchronize()
                            timings[label].append((time.perf_counter_ns() - begin) / 1e6)
                    record["latency_ms"] = {
                        key: latency_distribution(value) for key, value in timings.items()
                    }
                    record["samples_ms"] = timings
                    record["memory_allocated_bytes"] = torch.cuda.memory_allocated()
                    record["all_encoder_parity_passed"] = record["parity"]["passed"] and all(
                        item["passed"] for item in record["changed_input_parity"]
                    )
                    (output / "progress.json").write_text(
                        json.dumps(report, indent=2), encoding="utf-8"
                    )
                    print(
                        json.dumps(
                            {
                                "model": name,
                                "batch": batch,
                                "parity": record["all_encoder_parity_passed"],
                                "latency_ms": record["latency_ms"],
                            }
                        ),
                        flush=True,
                    )
                    del graphs, outputs, proposed, baseline, values, actual, expected
                    del graph, graph_output, call, stream, transformed
                    gc.collect()
        report["encoder_parity_passed"] = bool(report["records"]) and all(
            item["all_encoder_parity_passed"] for item in report["records"]
        )
        report["status"] = (
            "ENCODER_PROBE_COMPLETE_NOT_TTC_ADMISSION"
            if report["encoder_parity_passed"]
            else "FAILED_ENCODER_PARITY"
        )
    except BaseException as exc:
        caught = exc
        report.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        if original:
            from operational.train40_runtime.measurement import verify_snapshot

            report["originals"] = verify_snapshot(original)
            if not report["originals"]["passed"]:
                report["status"] = "FAILED_PRESERVATION"
        (output / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
    if caught is not None:
        raise caught
    if not report.get("originals", {}).get("passed"):
        raise RuntimeError("Encoder probe preservation failed")
    if not report.get("encoder_parity_passed"):
        raise RuntimeError("Encoder probe failed the fixed output parity tolerance")


def main() -> None:
    """Parse an isolated encoder experiment; defaults cover native and twenty-object shapes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--compiler-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument(
        "--endpoint-layout",
        choices=["unique", "original"],
        default="unique",
        help="Unique uses6/7 endpoints; original retains every E12/R15 triplet slot",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["E_A5_MATCHED", "E_C2F_MATCHED", "R_A5", "R_C2F"],
        default=["E_A5_MATCHED", "E_C2F_MATCHED", "R_A5", "R_C2F"],
    )
    parser.add_argument("--batches", nargs="+", type=int)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--conv-backends", default="ATEN,TRITON")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.batches is not None and any(batch < 1 for batch in args.batches):
        parser.error("repeats and batches must be positive")
    run(args)


if __name__ == "__main__":
    main()
