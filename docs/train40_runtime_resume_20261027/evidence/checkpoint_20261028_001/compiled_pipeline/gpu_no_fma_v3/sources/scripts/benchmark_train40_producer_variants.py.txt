"""Compare frozen prepared multiobject producers, layouts and stream parity.

This capacity probe excludes sensors, H2D and final context from its latency.
The final TTC parity check is run outside that clock for every artificial object.
The oracle uses each object's four event observations together and each RGB
observation separately, matching the original producer batching. This does not
replace the complete 554-query admission or a sustained physical stream.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402

from operational.rgb_port import transfer as engine  # noqa: E402
from operational.train40_runtime.context import ContextRuntime  # noqa: E402
from operational.train40_runtime.measurement import (  # noqa: E402
    latency_distribution,
    prediction_parity,
    snapshot_files,
    verify_snapshot,
)
from operational.train40_runtime.producers import (  # noqa: E402
    ProducerRuntime,
    clone_producer_models_for_layout,
)
from scripts.benchmark_train40_runtime import (  # noqa: E402
    _candidate_inventory,
    _source_inventory,
    _validate_output,
    _validate_reference,
    _write,
)
from scripts.final_train40_transfer import (  # noqa: E402
    load_models,
    read,
    reject_targets,
    sha,
    verify_candidate,
    verify_frozen_runtime,
)

SCOPE = "PREPARED_GPU_MULTIOBJECT_PRODUCERS_PAIR_PHASE17_GUARDS_NO_SENSOR_H2D_CONTEXT"
FEATURE_KEYS = ("event_raw", "event_experts", "rgb_raw", "rgb_experts")
Features = tuple[Tensor, Tensor, Tensor, Tensor]


def tensor_comparison(actual: Tensor, reference: Tensor) -> dict[str, Any]:
    """Quantify exactness and fixed tolerance without hiding nonfinite values."""
    if actual.shape != reference.shape:
        return {"passed": False, "reason": "shape", "exact": False}
    left, right = actual.detach().double().cpu(), reference.detach().double().cpu()
    finite = torch.isfinite(left) & torch.isfinite(right)
    error = (left - right).abs()
    allowed = 1e-5 + 1e-5 * right.abs()
    mask = finite & (error <= allowed)
    return {
        "passed": bool(mask.all()),
        "exact": bool(torch.equal(left, right)),
        "shape": list(left.shape),
        "nonfinite_pairs": int((~finite).sum()),
        "outside_tolerance": int((~mask).sum()),
        "different_elements": int((left != right).sum()),
        "max_abs": float(error[finite].max()) if bool(finite.any()) else None,
        "max_relative": (
            float((error[finite] / right[finite].abs().clamp_min(1e-5)).max())
            if bool(finite.any())
            else None
        ),
    }


def _compare(actual: Features, reference: Features) -> dict[str, Any]:
    return {
        key: tensor_comparison(a, b)
        for key, a, b in zip(FEATURE_KEYS, actual, reference, strict=True)
    }


def _weight_values(models: Mapping[str, nn.Module]) -> dict[str, str]:
    result = {}
    for name, model in models.items():
        digest = hashlib.sha256()
        for key, value in model.state_dict().items():
            digest.update(key.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def _preserve_source_bytes(
    sources: Sequence[Path], original: Mapping[str, Any], output: Path
) -> None:
    """Keep executable source bytes alongside hashes before an experiment starts."""
    manifest = {}
    for source in sorted(set(sources)):
        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != original[str(source.resolve())]["sha256"]:
            raise ValueError(f"Source changed before evidence snapshot: {source}")
        relative = source.relative_to(ROOT)
        destination = output / "sources" / (str(relative) + ".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        manifest[str(destination.relative_to(output))] = digest
    _write(output / "source_snapshots.json", manifest)


def _route(
    runtime: ProducerRuntime,
    inputs: Features,
    *,
    parallel: bool,
    streams: tuple[torch.cuda.Stream, torch.cuda.Stream],
    object_chunk_size: int = 0,
) -> Features:
    event, event_delta, rgb, rgb_delta = inputs
    objects = len(event) // 4
    if object_chunk_size and objects > object_chunk_size:
        chunks = []
        for start in range(0, objects, object_chunk_size):
            end = min(start + object_chunk_size, objects)
            chunks.append(
                _route(
                    runtime,
                    (
                        event[start * 4 : end * 4],
                        event_delta[start * 4 : end * 4],
                        rgb[start * 5 : end * 5],
                        rgb_delta[start * 5 : end * 5],
                    ),
                    parallel=parallel,
                    streams=streams,
                )
            )
        return tuple(torch.cat([item[index] for item in chunks]) for index in range(4))  # type: ignore[return-value]
    if not parallel:
        eraw, eexperts = runtime.expert_block("E", event, event_delta, history_group_size=4)
        rraw, rexperts = runtime.expert_block("R", rgb, rgb_delta, history_group_size=5)
        return eraw, eexperts, rraw, rexperts
    current = torch.cuda.current_stream()
    event_stream, rgb_stream = streams
    event_stream.wait_stream(current)
    rgb_stream.wait_stream(current)
    with torch.cuda.stream(event_stream):
        et = runtime.expert_block_async("E", event, event_delta, history_group_size=4)
    with torch.cuda.stream(rgb_stream):
        rt = runtime.expert_block_async("R", rgb, rgb_delta, history_group_size=5)
    current.wait_stream(event_stream)
    current.wait_stream(rgb_stream)
    runtime.validate_async_blocks(et, rt)
    return et.raw, et.experts, rt.raw, rt.experts


def _gpu_snapshot() -> dict[str, Any]:
    """Record untimed hardware context without controlling global GPU settings."""
    fields = "timestamp,power.draw,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu,memory.used"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv"],
        capture_output=True,
        text=True,
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    return {"returncode": result.returncode, "csv": result.stdout, "stderr": result.stderr}


def _oracle(models: Mapping[str, nn.Module], inputs: Features, objects: int) -> Features:
    runtime = ProducerRuntime(models, "cuda", graphs=False)
    event, event_delta, rgb, rgb_delta = inputs
    outputs: list[list[Tensor]] = [[], [], [], []]
    for index in range(objects):
        start = index * 4
        eraw, eexperts = runtime.expert_block(
            "E", event[start : start + 4], event_delta[start : start + 4]
        )
        outputs[0].append(eraw)
        outputs[1].append(eexperts)
        for observation in range(index * 5, (index + 1) * 5):
            rraw, rexperts = runtime.expert_block(
                "R", rgb[observation : observation + 1], rgb_delta[observation : observation + 1]
            )
            outputs[2].append(rraw)
            outputs[3].append(rexperts)
    return tuple(torch.cat(values) for values in outputs)  # type: ignore[return-value]


def _ttc(
    context: ContextRuntime,
    features: Features,
    row: dict[str, Any],
    metadata: Sequence[dict[str, Any]],
    objects: int,
) -> list[dict[str, Any]]:
    eraw, eexperts, rraw, rexperts = features
    query_us = int(row["anchor_us"])
    results = []
    for index in range(objects):
        result = context.predict(
            eraw[index * 4 : (index + 1) * 4],
            eexperts[index * 4 : (index + 1) * 4],
            [query_us - int(lag) for lag in engine.EVENT_LAGS_US],
            rraw[index * 5 : (index + 1) * 5],
            rexperts[index * 5 : (index + 1) * 5],
            metadata,
            query_us,
        )
        result.update(
            query_id=f"{row['query_id']}/capacity-object-{index}", rgb_unavailable_reason=""
        )
        results.append(result)
    return results


def _profile_family(name: str) -> str:
    lower = name.lower()
    if "conv" in lower:
        return "convolution"
    if any(part in lower for part in ("linear", "gemm", "matmul", "::mm", "::bmm", "addmm")):
        return "linear_matmul"
    if any(part in lower for part in ("copy", "memcpy", "::to", "_to_copy")):
        return "copies"
    if any(part in lower for part in ("gather", "grid_sampler", "index_select", "index_put")):
        return "sampling_indexing"
    if any(part in lower for part in ("sum", "mean", "norm", "amax", "amin", "softmax")):
        return "reductions_normalization"
    return "other_pointwise_and_runtime"


def _profile(
    runtime: ProducerRuntime,
    inputs: Features,
    streams: tuple[torch.cuda.Stream, torch.cuda.Stream],
    output: Path,
) -> dict[str, Any]:
    """Attribute a diagnostic eager-core execution; never substitute for timing.

    CUDA graph replay has no per-operator CPU dispatch to attribute. A warm eager
    execution of the same two graph cores exposes their kernel families; this
    diagnostic therefore does not claim the graph route's latency or overhead.
    """
    from operational.train40_runtime.producers import ExpertBlockTensorCore

    event, event_delta, rgb, rgb_delta = inputs
    event_core = ExpertBlockTensorCore(
        runtime.models,
        "E",
        reuse_shape=(len(event), int(event.shape[1])) if runtime.reuse_by_modality["E"] else None,
        history_group_size=4,
    ).eval()
    rgb_core = ExpertBlockTensorCore(
        runtime.models,
        "R",
        reuse_shape=(len(rgb), int(rgb.shape[1])) if runtime.reuse_by_modality["R"] else None,
        history_group_size=5,
    ).eval()
    event_core(event, event_delta)
    rgb_core(rgb, rgb_delta)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profiler:
        with torch.profiler.record_function("TRAIN40_EVENT_CORE"):
            event_core(event, event_delta)
        with torch.profiler.record_function("TRAIN40_RGB_CORE"):
            rgb_core(rgb, rgb_delta)
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(output / "producer_profile.json"))
    entries, families = [], {}
    for item in profiler.key_averages(group_by_input_shape=True):
        # CPU operator self-device totals attribute child kernels once. CUDA
        # kernel rows are retained in the Chrome trace, not counted twice here.
        if item.device_type.name != "CPU":
            continue
        milliseconds = float(item.self_device_time_total) / 1000
        if not milliseconds:
            continue
        family = _profile_family(item.key)
        families[family] = families.get(family, 0.0) + milliseconds
        entries.append(
            {
                "name": item.key,
                "calls": item.count,
                "input_shapes": item.input_shapes,
                "cuda_self_ms": milliseconds,
                "cuda_total_ms": float(item.device_time_total) / 1000,
                "cpu_self_ms": float(item.self_cpu_time_total) / 1000,
                "family": family,
            }
        )
    # Keep the graph-replay trace separate: kernel counts and times are useful
    # even though host operator ancestry is absent during replay.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as replay_profiler:
        _route(runtime, inputs, parallel=True, streams=streams)
        torch.cuda.synchronize()
    replay_profiler.export_chrome_trace(str(output / "producer_graph_replay_profile.json"))
    return {
        "scope": "DIAGNOSTIC_EAGER_SAME_CORE_CUDA_OPERATOR_ATTRIBUTION_NOT_LATENCY_ADMISSION",
        "cuda_self_ms_by_family": families,
        "operators": sorted(entries, key=lambda item: item["cuda_self_ms"], reverse=True),
        "graph_replay_trace": "producer_graph_replay_profile.json",
        "warning": (
            "Kernel work totals and profiled runs are not service response-time measurements"
        ),
    }


def run(args: argparse.Namespace) -> None:
    """Retain candidate identity, negative parity results and full workload scope."""
    if not args.objects or len(set(args.objects)) != len(args.objects):
        raise ValueError("Object counts must be nonempty and unique")
    if any(isinstance(count, bool) or not 1 <= count <= 20 for count in args.objects):
        raise ValueError("Object counts must lie in [1,20]")
    if args.repeats < 1 or args.warmup < 1:
        raise ValueError("At least one repetition and warmup are required")
    if not 0 <= args.object_chunk_size <= 20:
        raise ValueError("Object chunk size must be zero (disabled) or in [1,20]")
    if args.profile and args.object_chunk_size:
        raise ValueError("Profile the unchunked core separately from chunked capacity timing")
    _validate_output(args.output, [args.candidate, args.reference, args.manifest.parent])
    args.output.mkdir(parents=True, exist_ok=False)
    original: dict[str, Any] = {}
    records = []
    summary: dict[str, Any] = {"scope": SCOPE, "status": "RUNNING", "records": records}
    try:
        paths, binding = verify_candidate(args.candidate)
        frozen = verify_frozen_runtime(args.candidate)
        rows = read(args.manifest)["rows"]
        reject_targets(rows)
        if not 0 <= args.index < len(rows):
            raise ValueError("Query index outside frozen manifest")
        reference, reference_files, reference_binding = _validate_reference(
            args.reference, args.manifest, rows, binding
        )
        used_runtime = {"context.py", "measurement.py", "producers.py"}
        if args.transport != "original":
            used_runtime.update(("transport.py", "compiled_transport.py"))
        sources = [
            path
            for path in _source_inventory(frozen, reference_binding)
            if path.parent.name != "train40_runtime" or path.name in used_runtime
        ] + [Path(__file__).resolve()]
        original = snapshot_files(
            [*_candidate_inventory(args.candidate), args.manifest, *reference_files, *sources]
        )
        _write(args.output / "original_snapshot.json", original)
        _preserve_source_bytes(sources, original, args.output)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(7)
        binding.update(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            sources={str(path.relative_to(ROOT)): sha(path) for path in sources},
            manifest_sha256=sha(args.manifest),
            device=torch.cuda.get_device_name(),
            python=platform.python_version(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            precision="FP32_TF32_OFF",
            seed=7,
            objects=args.objects,
            object_chunk_size=args.object_chunk_size,
            query_index=args.index,
            encoder_layout=args.layout,
            transport_variant=args.transport,
            endpoint_reuse=not args.no_reuse,
            event_endpoint_reuse=not (args.no_reuse or args.no_event_reuse),
            cudnn_benchmark=args.cudnn_benchmark,
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            scope=SCOPE,
            input_population=(
                "Artificial capacity objects: real histories scaled differently per ID"
            ),
            scaling={"event": "1+0.001*object_index", "rgb": "1-0.001*object_index"},
            target_20_objects_50hz_demonstrated=False,
            acceptance_tolerance={"atol": 1e-5, "rtol": 1e-5},
        )
        _write(args.output / "binding.json", binding)
        models, normalizers = load_models(paths, "cuda")
        weights_before = _weight_values(models)
        proposed = clone_producer_models_for_layout(
            models, channels_last=args.layout == "channels-last"
        )
        if args.transport != "original":
            from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
            from operational.train40_runtime.compiled_transport import attach_transport_pair_core

            proposed = {
                name: attach_transport_pair_core(model, reuse_reverse=args.transport == "reverse")
                if isinstance(model, CausalScaleTTC)
                else model
                for name, model in proposed.items()
            }
        proposed_before = _weight_values(proposed)
        context = ContextRuntime(models, normalizers, "cuda", graphs=True)
        row = rows[args.index]
        prepared = engine.prepare_event(row)
        rgb, reason = engine.raw_rgb_observations(row)
        if reason or len(rgb) != 5 or len(prepared["own_events"]) != 4:
            raise ValueError(
                "Capacity probe requires four event and five available RGB observations"
            )
        event_base = torch.from_numpy(prepared["own_events"]).cuda()
        rgb_base = torch.from_numpy(np.stack([pixels for pixels, _ in rgb])).cuda()
        rgb_clock = torch.tensor(
            [item["delta_t_s"] for _, item in rgb], dtype=torch.float32, device="cuda"
        )
        metadata = [item for _, item in rgb]
        streams = (torch.cuda.Stream(), torch.cuda.Stream())
        with torch.inference_mode():
            for objects in args.objects:
                runtime = None
                record: dict[str, Any] = {"objects": objects}
                try:
                    event = torch.cat([event_base * (1 + 0.001 * i) for i in range(objects)])
                    sensor_rgb = torch.cat([rgb_base * (1 - 0.001 * i) for i in range(objects)])
                    inputs = (
                        event,
                        torch.full((len(event), 2), 0.1, device="cuda"),
                        sensor_rgb,
                        rgb_clock.repeat(objects, 1),
                    )
                    torch.backends.cudnn.benchmark = False
                    reference_features = _oracle(models, inputs, objects)
                    reference_ttc = _ttc(context, reference_features, row, metadata, objects)
                    # Object zero remains the unmodified physical query. Verify
                    # its prepared oracle against the preserved original TTC.
                    first = dict(reference_ttc[0], query_id=row["query_id"])
                    record["unmodified_object_original_parity"] = prediction_parity(
                        [reference[args.index]], [first], [row["query_id"]]
                    )
                    torch.backends.cudnn.benchmark = args.cudnn_benchmark
                    runtime = ProducerRuntime(
                        proposed,
                        "cuda",
                        graphs=True,
                        block_graphs=True,
                        reuse_encodings=not args.no_reuse,
                        reuse_event_encodings=False if args.no_event_reuse else None,
                    )
                    start = time.perf_counter()
                    cold = _route(
                        runtime,
                        inputs,
                        parallel=False,
                        streams=streams,
                        object_chunk_size=args.object_chunk_size,
                    )
                    torch.cuda.synchronize()
                    record["capture_seconds"] = time.perf_counter() - start
                    for _ in range(args.warmup):
                        _route(
                            runtime,
                            inputs,
                            parallel=False,
                            streams=streams,
                            object_chunk_size=args.object_chunk_size,
                        )
                        _route(
                            runtime,
                            inputs,
                            parallel=True,
                            streams=streams,
                            object_chunk_size=args.object_chunk_size,
                        )
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    times: dict[str, list[float]] = {"serial": [], "parallel": []}
                    outputs: dict[str, Features] = {}
                    comparisons: list[dict[str, Any]] = []
                    record["gpu_before_repetitions"] = _gpu_snapshot()
                    for repeat in range(args.repeats):
                        order = (
                            ("serial", "parallel") if repeat % 2 == 0 else ("parallel", "serial")
                        )
                        for route in order:
                            torch.cuda.synchronize()
                            start_ns = time.perf_counter_ns()
                            output = _route(
                                runtime,
                                inputs,
                                parallel=route == "parallel",
                                streams=streams,
                                object_chunk_size=args.object_chunk_size,
                            )
                            torch.cuda.synchronize()
                            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
                            times[route].append(elapsed_ms)
                            outputs[route] = output
                        comparisons.append(_compare(outputs["parallel"], outputs["serial"]))
                    record["gpu_after_repetitions"] = _gpu_snapshot()
                    actual = _ttc(context, outputs["parallel"], row, metadata, objects)
                    serial_ttc = _ttc(context, outputs["serial"], row, metadata, objects)
                    ids = [item["query_id"] for item in reference_ttc]
                    record.update(
                        status="MEASURED",
                        event_shape=list(event.shape),
                        rgb_shape=list(sensor_rgb.shape),
                        times_ms=times,
                        latency_ms={
                            name: latency_distribution(values) for name, values in times.items()
                        },
                        cold_vs_serial=_compare(outputs["serial"], cold),
                        serial_parallel=comparisons,
                        original_producer_features=_compare(
                            outputs["parallel"], reference_features
                        ),
                        final_ttc_parity=prediction_parity(reference_ttc, actual, ids),
                        serial_parallel_ttc=prediction_parity(serial_ttc, actual, ids),
                        reference_ttc=reference_ttc,
                        actual_ttc=actual,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        graphs=runtime.diagnostics(),
                    )
                    if args.profile and objects == max(args.objects):
                        record["profile"] = _profile(runtime, inputs, streams, args.output)
                except Exception as error:
                    record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
                records.append(record)
                _write(args.output / "summary.json", summary)
                print(
                    json.dumps(
                        {
                            "objects": objects,
                            "status": record["status"],
                            "latency_ms": record.get("latency_ms"),
                            "error": record.get("error"),
                        }
                    ),
                    flush=True,
                )
                del runtime
                gc.collect()
                torch.cuda.empty_cache()
        summary["weight_values_unchanged"] = weights_before == _weight_values(models)
        summary["variant_weight_values_unchanged"] = proposed_before == _weight_values(proposed)
        summary["status"] = (
            "COMPLETE" if all(item["status"] == "MEASURED" for item in records) else "FAILED"
        )
        summary["all_final_ttc_parity_passed"] = all(
            item.get("final_ttc_parity", {}).get("passed", False)
            and item.get("unmodified_object_original_parity", {}).get("passed", False)
            for item in records
        )
        if summary["status"] == "COMPLETE" and not summary["all_final_ttc_parity_passed"]:
            summary["status"] = "COMPLETE_WITH_PARITY_FAILURES"
    except BaseException as error:
        summary.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        torch.backends.cudnn.benchmark = False
        summary["preservation"] = verify_snapshot(original) if original else {"passed": False}
        if not summary["preservation"]["passed"]:
            summary["status"] = "FAILED"
        _write(args.output / "summary.json", summary)
    if summary["status"] != "COMPLETE":
        raise SystemExit(2)


def main() -> None:
    """Require explicit frozen inputs and a fresh evidence directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--objects", nargs="+", type=int, default=[1, 2, 5, 10, 20])
    parser.add_argument("--layout", choices=("contiguous", "channels-last"), default="contiguous")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--no-event-reuse", action="store_true")
    parser.add_argument("--transport", choices=("original", "pair", "reverse"), default="original")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--object-chunk-size", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
