"""Measure actual native predict latency locally, separately from instrumented stage timing."""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import psutil
import torch

from e_jepa_ttc.rgb_port.data import TarFrameReader
from operational.final_refit.test12 import TRUSTED_INPUTS_SHA256, FinalV16Train40Predictor
from operational.phase_robust.bundle import verified_upstream_paths
from operational.phase_timewarp.canonical_pair import CanonicalPair
from operational.release_research.protocol import ResearchRun, sha256, write_json
from operational.rgb_port import transfer as engine
from operational.rgb_port_streaming.contracts import NativeObservation
from scripts.audit_phase_native_source_closure import DEPENDENCIES
from scripts.final_train40_transfer import load_models
from scripts.release_research import read, sources, verify_complete

PLAN = Path("docs/release_research_implementation_20261031/NATIVE_LOCAL_LATENCY_PROTOCOL.md")


def select_queries(queries: list[dict]) -> dict[str, list[dict]]:
    """Select deterministic physical prefixes per sequence, never by prediction or measured cost."""
    if len({q["sample_token"] for q in queries}) != len(queries):
        raise ValueError("Collapse aliases before measuring service")
    groups = defaultdict(list)
    for query in queries:
        groups[query["sequence_id"]].append(query)
    return {
        k: sorted(groups[k], key=lambda q: (q["query_time_us"], q["sample_token"]))[:10]
        for k in sorted(groups)
    }


def queue_delay(arrivals_ms: list[float], service_ms: list[float]) -> list[float]:
    """Simulate one sequential server on measured service durations and fixed arrival times."""
    if (
        len(arrivals_ms) != len(service_ms)
        or not arrivals_ms
        or any(b < a for a, b in zip(arrivals_ms, arrivals_ms[1:], strict=False))
        or not np.isfinite(arrivals_ms).all()
        or not np.isfinite(service_ms).all()
        or any(t < 0 for t in service_ms)
    ):
        raise ValueError("Expected finite ordered arrivals and nonnegative aligned service times")
    end = arrivals_ms[0]
    delays = []
    for arrival, service in zip(arrivals_ms, service_ms, strict=True):
        end = max(arrival, end) + service
        delays.append(end - arrival)
    return delays


def quantiles(values: list[float]) -> list[float] | None:
    return np.quantile(values, [0.5, 0.95, 0.99]).tolist() if values else None


def native_model(models: dict, normalizers: dict, data: dict) -> FinalV16Train40Predictor:
    native = FinalV16Train40Predictor.__new__(FinalV16Train40Predictor)
    native.models, native.normalizers, native.inputs = models, normalizers, data
    native.device = torch.device("cuda")
    native.event_values, native.rgb_values = {}, {}
    native.reader = TarFrameReader(
        Path(data["media_root"]), max_open=2, max_frame_cache_bytes=128 * 1024**2
    )
    return native


def timed_predict(
    native: FinalV16Train40Predictor, query: dict, data: dict, instrumented: bool
) -> dict:
    """Time the unmodified native prediction; optional stage wrappers form a separate pass."""
    accounting = {"observations_ms": 0.0, "producers_ms": 0.0}

    def observation(function: Callable[[str], tuple]) -> Callable[[str], tuple]:
        def call(identity: str) -> tuple:
            start = time.perf_counter()
            value = function(identity)
            accounting["observations_ms"] += (time.perf_counter() - start) * 1000
            return value

        return call

    original = engine._expert_block

    def expert(models: dict, prefix: str, pixels: torch.Tensor, delta: torch.Tensor) -> tuple:
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = original(models, prefix, pixels, delta)
        torch.cuda.synchronize()
        accounting["producers_ms"] += (time.perf_counter() - start) * 1000
        return value

    old_counts = len(native.event_values), len(native.rgb_values)
    with ExitStack() as stack:
        stack.enter_context(patch.object(engine, "ProducerObservation", NativeObservation))
        if instrumented:
            for name in ("_event_observation", "_rgb_observation"):
                stack.enter_context(patch.object(native, name, observation(getattr(native, name))))
            stack.enter_context(patch.object(engine, "_expert_block", expert))
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = native.predict(query, data)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
    result = {
        "prediction": value,
        "elapsed_ms": elapsed,
        "new_event_observations": len(native.event_values) - old_counts[0],
        "new_rgb_observations": len(native.rgb_values) - old_counts[1],
    }
    if instrumented:
        result.update(
            producers_ms=accounting["producers_ms"],
            preparation_and_cache_ms=accounting["observations_ms"] - accounting["producers_ms"],
            context_fusion_host_ms=elapsed - accounting["observations_ms"],
        )
    return result


def benchmark(parent: Path, upstream: Path, output: Path) -> None:
    """Measure fixed prefixes only after the full exact-local paired diagnostic completes."""
    verify_complete(parent)
    paired, capture = parent / "paired_replay", parent / "capture"
    if (
        read(parent / "DECISION.json")["outcome"] != "local_sensitivity_complete"
        or not read(paired / "RESULTS.json")["local_original_exact"]
    ):
        raise ValueError("Require complete exact-local paired sensitivity")
    capture_freeze = read(capture / "FREEZE.json")
    public = Path(capture_freeze["input_paths"]["public_inputs"])
    if sha256(public) != TRUSTED_INPUTS_SHA256:
        raise ValueError("Public inputs changed")
    data = read(public)
    selected = select_queries(read(capture / "QUERIES.json")["queries"])
    predictions = {r["token"]: r for r in read(paired / "PREDICTIONS.json")["records"]}
    paths = verified_upstream_paths(upstream)
    files = {
        **sources(),
        "preregistration": PLAN,
        "parent": parent / "COMPLETE.json",
        "public_inputs": public,
        "paired_predictions": paired / "PREDICTIONS.json",
        "upstream_bundle": upstream / "BUNDLE.json",
        **{"model:" + k: v for k, v in paths.items()},
    }
    for name in (
        *DEPENDENCIES,
        "scripts/release_native_local_latency.py",
        "scripts/final_train40_transfer.py",
        "operational/phase_timewarp/canonical_pair.py",
        "operational/phase_robust/bundle.py",
    ):
        files["source:" + name] = Path(name)
    inventory_path = Path(capture_freeze["input_paths"]["media_files"])
    if sha256(inventory_path) != capture_freeze["inputs"]["media_files"]:
        raise ValueError("Public media inventory changed")
    files["media_inventory"] = inventory_path
    inventory = {str(Path(r["path"]).resolve()): r for r in read(inventory_path)["verified"]}
    for rows in selected.values():
        for query in rows:
            for identity in query["event_history"]:
                item = data["event_observations"][identity]
                path = Path(item["current_tensor_path"])
                if sha256(path) != item["current_tensor_sha256"]:
                    raise ValueError("Event bytes changed")
                files["event:" + identity] = path
            for identity in query["rgb_history"]:
                for shard in data["rgb_observations"][identity]["producer_shards"]:
                    if "rgb:" + shard not in files:
                        path = (Path(data["media_root"]) / shard).resolve()
                        if sha256(path) != inventory[str(path)]["sha256"]:
                            raise ValueError("RGB shard changed")
                        files["rgb:" + shard] = path
    run = ResearchRun(
        output,
        files,
        {
            "experiment_id": "native-local-latency-v1",
            "seed": 7,
            "targets_read": False,
            "optimizer_updates": 0,
            "queries_per_sequence": 10,
            "scope": "local prefix latency, no historical accuracy claim",
            "device": "cuda",
            "torch_version": torch.__version__,
            "tf32": False,
            "deterministic_algorithms": True,
            "benchmark": False,
            "threads": 1,
            "batch_size": 1,
        },
    )
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(1)
        torch.manual_seed(7)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        start = time.perf_counter()
        models, normalizers = load_models(paths, "cuda")
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - start
        variants = {
            "original": models,
            "canonical": {
                **models,
                **{k: CanonicalPair(models[k]) for k in ("PAIR_E_MATCHED", "PAIR_R")},
            },
        }
        warmup = read(capture / "QUERIES.json")["queries"][:2]
        for current in variants.values():
            native = native_model(current, normalizers, data)
            try:
                for query in warmup:
                    timed_predict(native, query, data, False)
            finally:
                native.close()
        torch.cuda.reset_peak_memory_stats()
        records = []
        for instrumented in (False, True):
            for sequence_index, (sequence, rows) in enumerate(selected.items()):
                natives = {
                    arm: native_model(current, normalizers, data)
                    for arm, current in variants.items()
                }
                seen_tracks = set()
                try:
                    for index, query in enumerate(rows):
                        first_track_query = query["track_id"] not in seen_tracks
                        seen_tracks.add(query["track_id"])
                        order = ["original", "canonical"]
                        if (index + sequence_index) % 2:
                            order.reverse()
                        for position, arm in enumerate(order):
                            result = timed_predict(natives[arm], query, data, instrumented)
                            if result["prediction"] != predictions[query["sample_token"]][arm]:
                                raise ValueError("Timed output differs from exact local replay")
                            records.append(
                                {
                                    **result,
                                    "token": query["sample_token"],
                                    "sequence": sequence,
                                    "track_id": query["track_id"],
                                    "query_us": query["query_time_us"],
                                    "arm": arm,
                                    "instrumented": instrumented,
                                    "first_track_query": first_track_query,
                                    "execution_position": position,
                                }
                            )
                finally:
                    for native in natives.values():
                        native.close()
                print(f"instrumented={instrumented} {sequence} {len(rows)} queries", flush=True)
        summaries, queues = {}, {}
        for arm in variants:
            rows = [r for r in records if r["arm"] == arm and not r["instrumented"]]
            summaries[arm] = {
                "queries": len(rows),
                "p50_p95_p99_ms": quantiles([r["elapsed_ms"] for r in rows]),
                "first_track_query_ms": quantiles(
                    [r["elapsed_ms"] for r in rows if r["first_track_query"]]
                ),
                "subsequent_query_ms": quantiles(
                    [r["elapsed_ms"] for r in rows if not r["first_track_query"]]
                ),
            }
            queues[arm] = {}
            for sequence in selected:
                part = [r for r in rows if r["sequence"] == sequence]
                delays = queue_delay(
                    [(r["query_us"] - part[0]["query_us"]) / 1000 for r in part],
                    [r["elapsed_ms"] for r in part],
                )
                queues[arm][sequence] = {
                    "p50_p95_p99_delay_ms": quantiles(delays),
                    "outputs_after_20ms": sum(d > 20 for d in delays),
                }
        ratio = (
            summaries["canonical"]["p50_p95_p99_ms"][1] / summaries["original"]["p50_p95_p99_ms"][1]
        )
        memory = psutil.Process().memory_info()
        write_json(output / "MEASUREMENTS.json", records)
        write_json(
            output / "RESULTS.json",
            {
                "service": summaries,
                "simulated_queue": queues,
                "candidate_over_original_p95": ratio,
                "within_descriptive_10pct_budget": ratio <= 1.1,
                "load_weights_seconds": load_seconds,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "process_peak_rss_bytes": getattr(memory, "peak_wset", memory.rss),
                "gpu": torch.cuda.get_device_name(),
                "native_accuracy_evaluated": False,
                "promotion_admitted": False,
                "all_predictions_exact_to_local_replay": True,
            },
        )
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument(
        "--upstream", type=Path, default=Path("output/phase_submission/FINAL_MODEL/upstream")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark(args.parent, args.upstream, args.output)


if __name__ == "__main__":
    main()
