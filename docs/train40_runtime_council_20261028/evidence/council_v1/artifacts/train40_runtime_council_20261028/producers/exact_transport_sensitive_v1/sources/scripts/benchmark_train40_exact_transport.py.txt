"""Audit opt-in transport reuse on prepared, real one-object TRAIN40 histories.

Every explicit initial, warmup and measured return is checked before the next
producer call. Exact producer-feature comparison uses original eager cores with
the same E OFF / R ON endpoint reuse and batch shapes. Final TTC uses the sealed
historical reference at unchanged 1e-5 tolerances. This is a prepared-producer
diagnostic: its clock excludes sensors, H2D, context and interleaved validation;
it cannot admit complete updates, sustained 50 Hz, or two objects.
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
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC  # noqa: E402
from operational.rgb_port import transfer as engine  # noqa: E402
from operational.train40_runtime.compiled_transport import (  # noqa: E402
    TransportCoreBinding,
    attach_transport_pair_core,
)
from operational.train40_runtime.context import ContextRuntime  # noqa: E402
from operational.train40_runtime.measurement import (  # noqa: E402
    latency_distribution,
    prediction_parity,
    snapshot_files,
    verify_snapshot,
)
from operational.train40_runtime.producers import (  # noqa: E402
    ExpertBlockTensorCore,
    ProducerRuntime,
)
from operational.train40_runtime.transport import (  # noqa: E402
    TransportReuseOptions,
    attach_transport_reuse,
)
from scripts.benchmark_train40_compiled_pipeline import (  # noqa: E402
    capture_replay_parity,
    replay_parity_coverage,
)
from scripts.benchmark_train40_producer_variants import (  # noqa: E402
    FEATURE_KEYS,
    Features,
    _gpu_snapshot,
    _oracle,
    _preserve_source_bytes,
    _route,
    _ttc,
    _weight_values,
    tensor_comparison,
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

SCOPE = "PREPARED_ONE_OBJECT_EXACT_TRANSPORT_ALL_EXPLICIT_CALLS_NOT_COMPLETE_UPDATE"
SENSITIVE_INDICES = (0, 137, 488, 505, 514, 533, 553)


def replay_schedule(
    variants: Sequence[str], warmup: int, repeats: int
) -> list[tuple[str, str, int]]:
    """List every required return, alternating measured order between repeats."""
    if not variants or len(set(variants)) != len(variants) or warmup < 1 or repeats < 1:
        raise ValueError("Replay schedule requires unique variants and positive call counts")
    return [
        (name, stage, iteration)
        for name in variants
        for stage, count in (("initial_call", 1), ("warmup", warmup))
        for iteration in range(count)
    ] + [
        (name, "measured", iteration)
        for iteration in range(repeats)
        for name in (list(variants) if iteration % 2 == 0 else list(reversed(variants)))
    ]


def check_exact_replay_coverage(
    receipts: Sequence[dict[str, Any]],
    variants: Sequence[str],
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Require full ordered TTC coverage and all four matched features bitexact."""
    result = replay_parity_coverage(receipts, variants, warmup=warmup, repeats=repeats)
    exact = bool(receipts) and all(
        item.get("feature_parity", {}).get("passed") is True
        and item.get("feature_parity", {}).get("exact") is True
        and set(item.get("feature_parity", {}).get("features", {})) == set(FEATURE_KEYS)
        and all(
            item["feature_parity"]["features"][key].get("passed") is True
            and item["feature_parity"]["features"][key].get("exact") is True
            for key in FEATURE_KEYS
        )
        for item in receipts
    )
    result.update(all_matched_features_exact=exact, passed=result["passed"] and exact)
    return result


def compare_features(actual: Features, expected: Features) -> dict[str, Any]:
    """Compare matched cores, separate from historical-batching TTC parity."""
    comparisons = {
        key: tensor_comparison(a, b)
        for key, a, b in zip(FEATURE_KEYS, actual, expected, strict=True)
    }
    return {
        "passed": all(item["passed"] for item in comparisons.values()),
        "exact": all(item["exact"] for item in comparisons.values()),
        "features": comparisons,
    }


def matched_eager_features(models: Mapping[str, nn.Module], inputs: Features) -> Features:
    """Run original unfused cores with the exact candidate encoder batching."""
    event, event_delta, rgb, rgb_delta = inputs
    eraw, eexperts, eguard = ExpertBlockTensorCore(models, "E").eval()(event, event_delta)
    rraw, rexperts, rguard = ExpertBlockTensorCore(
        models, "R", reuse_shape=(len(rgb), int(rgb.shape[1])), history_group_size=5
    ).eval()(rgb, rgb_delta)
    if not bool(torch.stack((eguard, rguard)).all()):
        raise ValueError("Original matched eager core rejected sensor/clock/overlap/output")
    return eraw, eexperts, rraw, rexperts


def transport_view(
    models: Mapping[str, nn.Module], variant: str
) -> tuple[dict[str, nn.Module], dict[str, Any]]:
    """Attach only private, frozen adapters; never deduplicate observation pairs."""
    if variant not in ("original", "reverse", "reverse-summaries"):
        raise ValueError(f"Unknown transport variant: {variant}")
    proposed, bindings = {}, {}
    for name, model in models.items():
        view = model
        if isinstance(model, CausalScaleTTC) and variant != "original":
            if variant == "reverse":
                view = attach_transport_pair_core(model, reuse_reverse=True)
                adapter = view._train40_transport_binding
                if not isinstance(adapter, TransportCoreBinding):
                    raise TypeError("Transport-pair view did not bind its adapter identity")
                bindings[name] = asdict(adapter)
            else:
                view = attach_transport_reuse(model, history_group_size=None, summaries=True)
                options = view._train40_transport_options
                if not isinstance(options, TransportReuseOptions):
                    raise TypeError("Transport-reuse view did not bind its adapter options")
                bindings[name] = asdict(options)
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError(f"Model must be frozen and in eval mode: {name}")
        original_parameters, view_parameters = (
            dict(model.named_parameters()),
            dict(view.named_parameters()),
        )
        if original_parameters.keys() != view_parameters.keys() or any(
            value is not view_parameters[key] for key, value in original_parameters.items()
        ):
            raise ValueError(f"Transport view did not share exact frozen parameters: {name}")
        proposed[name] = view
    return proposed, bindings


def _normalizer_hashes(normalizers: Mapping[str, Any]) -> dict[str, list[str]]:
    return {
        key: [hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest() for value in pair]
        for key, pair in normalizers.items()
    }


def run(args: argparse.Namespace) -> None:
    """Save frozen identities, complete receipts and negative results fail-closed."""
    variants = ["original", *args.variants]
    routes = args.routes
    names = [f"{variant}/{route}" for variant in variants for route in routes]
    schedule = replay_schedule(names, args.warmup, args.repeats)
    if len(set(routes)) != len(routes) or not routes:
        raise ValueError("Routes must be nonempty and unique")
    if not args.indices or len(set(args.indices)) != len(args.indices):
        raise ValueError("Query indices must be nonempty and unique")
    _validate_output(args.output, [args.candidate, args.reference, args.manifest.parent])
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "scope": SCOPE,
        "status": "RUNNING",
        "run_name": args.output.name,
        "start_time": datetime.now(UTC).isoformat(),
        "records": records,
        "complete_update_admitted": False,
        "sustained_50hz_demonstrated": False,
        "two_object_admitted": False,
        "measurement_cadence": "Per-call context/parity and owned copies outside clock",
        "initial_call_definition": (
            "First explicit call for each variant/route/query; runtime may warm, trace and "
            "capture internally or reuse prior shapes. Not process-cold latency."
        ),
    }
    try:
        paths, binding = verify_candidate(args.candidate)
        frozen = verify_frozen_runtime(args.candidate)
        rows = read(args.manifest)["rows"]
        reject_targets(rows)
        if any(index < 0 or index >= len(rows) for index in args.indices):
            raise ValueError("Query index outside frozen manifest")
        reference, reference_files, reference_binding = _validate_reference(
            args.reference, args.manifest, rows, binding
        )
        used_runtime = {
            "context.py",
            "measurement.py",
            "producers.py",
            "transport.py",
            "compiled_transport.py",
        }
        sources = [
            path
            for path in _source_inventory(frozen, reference_binding)
            if path.parent.name != "train40_runtime" or path.name in used_runtime
        ] + [
            Path(__file__),
            ROOT / "scripts/benchmark_train40_producer_variants.py",
            ROOT / "scripts/benchmark_train40_compiled_pipeline.py",
        ]
        sources = sorted(set(path.resolve() for path in sources))
        snapshot = snapshot_files(
            [*_candidate_inventory(args.candidate), args.manifest, *reference_files, *sources]
        )
        _write(args.output / "original_snapshot.json", snapshot)
        _preserve_source_bytes(sources, snapshot, args.output)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(7)
        binding.update(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            sources={str(path.relative_to(ROOT)): sha(path) for path in sources},
            manifest_sha256=sha(args.manifest),
            python=platform.python_version(),
            host=platform.node(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(),
            precision="FP32_TF32_OFF",
            seed=7,
            objects=1,
            indices=args.indices,
            variants=variants,
            routes=routes,
            warmup=args.warmup,
            repeats=args.repeats,
            expected_explicit_calls_per_query=len(schedule),
            event_endpoint_reuse=False,
            rgb_endpoint_reuse=True,
            event_history_batch=4,
            rgb_history_batch=5,
            adjacent_transport_pair_deduplication=False,
            compiler_backend=None,
            jit_optimized_execution=False,
            exactness_reference="ORIGINAL_EAGER_EXPERT_BLOCK_SAME_BATCH_AND_ENDPOINT_REUSE",
            ttc_reference="SEALED_ORIGINAL_HISTORICAL_BATCHING",
            acceptance_tolerance={"atol": 1e-5, "rtol": 1e-5},
            optimizer_updates=0,
            targets_read=False,
            scope=SCOPE,
        )
        models, normalizers = load_models(paths, "cuda")
        before, norms_before = _weight_values(models), _normalizer_hashes(normalizers)
        runtimes, variant_models, adapters = {}, {}, {}
        for variant in variants:
            view, adapters[variant] = transport_view(models, variant)
            variant_models[variant] = view
            runtimes[variant] = ProducerRuntime(
                view,
                "cuda",
                graphs=True,
                block_graphs=True,
                reuse_event_encodings=False,
                reuse_rgb_encodings=True,
            )
        binding["transport_adapters"] = adapters
        binding["config_hash"] = hashlib.sha256(
            json.dumps(binding, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest()
        _write(args.output / "binding.json", binding)
        context = ContextRuntime(models, normalizers, "cuda", graphs=True)
        streams = (torch.cuda.Stream(), torch.cuda.Stream())
        with torch.inference_mode():
            for index in args.indices:
                record: dict[str, Any] = {"index": index, "objects": 1, "status": "FAILED"}
                records.append(record)
                try:
                    row = rows[index]
                    prepared = engine.prepare_event(row)
                    rgb, reason = engine.raw_rgb_observations(row)
                    if reason or len(rgb) != 5 or len(prepared["own_events"]) != 4:
                        raise ValueError("Requires four event and five available RGB observations")
                    event = torch.from_numpy(prepared["own_events"]).cuda()
                    pixels = torch.from_numpy(np.stack([value for value, _ in rgb])).cuda()
                    inputs: Features = (
                        event,
                        torch.full((4, 2), 0.1, device="cuda"),
                        pixels,
                        torch.tensor(
                            [item["delta_t_s"] for _, item in rgb],
                            dtype=torch.float32,
                            device="cuda",
                        ),
                    )
                    metadata = [item for _, item in rgb]
                    predict = partial(_ttc, context, row=row, metadata=metadata, objects=1)
                    query_id = f"{row['query_id']}/capacity-object-0"
                    expected_ttc = [dict(reference[index], query_id=query_id)]
                    compare_ttc = partial(prediction_parity, expected_ttc, expected_ids=[query_id])
                    historical = _oracle(models, inputs, 1)
                    record["historical_original_parity"] = compare_ttc(predict(historical))
                    del historical
                    matched = matched_eager_features(models, inputs)
                    record["matched_eager_original_parity"] = compare_ttc(predict(matched))
                    record["input_shapes"] = [list(value.shape) for value in inputs]
                    record["input_sha256"] = [
                        hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()
                        for value in inputs
                    ]
                    receipts: list[dict[str, Any]] = []
                    record["replay_parity"] = receipts
                    times: dict[str, list[float]] = {name: [] for name in names}
                    setup: dict[str, float] = {}
                    check = partial(
                        capture_replay_parity,
                        predict=predict,
                        compare_ttc=compare_ttc,
                        compare_features=partial(compare_features, expected=matched),
                    )
                    record["gpu_before"] = _gpu_snapshot()
                    for name, stage, iteration in schedule:
                        variant, route = name.split("/")
                        torch.cuda.synchronize()
                        started = time.perf_counter_ns()
                        output = _route(
                            runtimes[variant], inputs, parallel=route == "parallel", streams=streams
                        )
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter_ns() - started) / 1e6
                        if stage == "measured":
                            times[name].append(elapsed)
                        elif stage == "initial_call":
                            setup[name] = elapsed
                        receipts.append(
                            check(output, variant=name, stage=stage, iteration=iteration)
                        )
                        del output
                    coverage = check_exact_replay_coverage(
                        receipts, names, args.warmup, args.repeats
                    )
                    record.update(
                        status="MEASURED",
                        replay_parity_coverage=coverage,
                        setup_milliseconds=setup,
                        times_ms=times,
                        latency_ms={
                            name: latency_distribution(value) for name, value in times.items()
                        },
                        graphs={name: runtime.diagnostics() for name, runtime in runtimes.items()},
                        gpu_after=_gpu_snapshot(),
                    )
                    del matched, inputs
                except Exception as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                _write(args.output / "summary.json", summary)
                print(
                    json.dumps(
                        {
                            "index": index,
                            "status": record["status"],
                            "coverage": record.get("replay_parity_coverage"),
                            "latency_ms": record.get("latency_ms"),
                            "error": record.get("error"),
                        }
                    ),
                    flush=True,
                )
        after = _weight_values(models)
        summary["weight_values_sha256_before"] = before
        summary["weight_values_sha256_after"] = after
        summary["weights_unchanged"] = before == after
        summary["variant_weights_equal_original"] = all(
            before == _weight_values(view) for view in variant_models.values()
        )
        summary["normalizers_unchanged"] = norms_before == _normalizer_hashes(normalizers)
        summary["query_coverage_exact"] = [record["index"] for record in records] == args.indices
        summary["all_replay_gates_passed"] = all(
            record["status"] == "MEASURED"
            and record.get("replay_parity_coverage", {}).get("passed") is True
            and record.get("historical_original_parity", {}).get("passed") is True
            and record.get("matched_eager_original_parity", {}).get("passed") is True
            for record in records
        )
        summary["status"] = (
            "COMPLETE"
            if all(
                summary[key] is True
                for key in (
                    "weights_unchanged",
                    "variant_weights_equal_original",
                    "normalizers_unchanged",
                    "query_coverage_exact",
                    "all_replay_gates_passed",
                )
            )
            else "COMPLETE_WITH_GATE_FAILURES"
        )
    except BaseException as error:
        summary.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["end_time"] = datetime.now(UTC).isoformat()
        summary["preservation"] = verify_snapshot(snapshot) if snapshot else {"passed": False}
        if not summary["preservation"]["passed"]:
            summary["status"] = "FAILED"
        _write(args.output / "summary.json", summary)
    if summary["status"] != "COMPLETE":
        raise SystemExit(2)


def main() -> None:
    """Run opt-in one-object probes without changing service defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "manifest", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--indices", nargs="+", type=int, default=list(SENSITIVE_INDICES))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("reverse", "reverse-summaries"),
        default=["reverse", "reverse-summaries"],
    )
    parser.add_argument(
        "--routes", nargs="+", choices=("serial", "parallel"), default=["serial", "parallel"]
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
