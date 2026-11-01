"""Verify RAW event tensor identity and measure local RAW-to-output latency."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from types import MethodType

import numpy as np
import psutil
import torch

from e_jepa_ttc.efficient_context.mapped_union import encode_mapped, map_roi
from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool
from e_jepa_ttc.simplex_t.context_raw_union import _read_crop
from operational.final_refit.test12 import TRUSTED_INPUTS_SHA256, FinalV16Train40Predictor
from operational.phase_robust.bundle import verified_upstream_paths
from operational.phase_timewarp.canonical_pair import CanonicalPair
from operational.release_research.protocol import ResearchRun, sha256, write_json
from operational.rgb_port import transfer as engine
from operational.rgb_port.infer_experts import delta_matrix
from scripts.audit_phase_native_source_closure import DEPENDENCIES
from scripts.final_train40_transfer import load_models
from scripts.release_native_local_latency import (
    native_model,
    quantiles,
    queue_delay,
    select_queries,
    timed_predict,
)
from scripts.release_research import read, sources, verify_complete

PLAN = Path("docs/release_research_implementation_20261031/NATIVE_RAW_LATENCY_PROTOCOL.md")
PREPARATION_SOURCES = (
    "src/e_jepa_ttc/efficient_context/mapped_union.py",
    "src/e_jepa_ttc/simplex_t/cached_event_reader.py",
    "src/e_jepa_ttc/simplex_t/context_raw_union.py",
    "src/e_jepa_ttc/simplex_t/query_context_voxel.py",
    "src/e_jepa_ttc/data/eap.py",
    "src/e_jepa_ttc/data/eap_representation.py",
)


def prepare(pool: ReaderPool, item: dict) -> torch.Tensor:
    """Rebuild only the current three-window observation, without changing reductions."""
    windows = np.asarray(item["windows_us"], dtype=np.int64)
    if windows.shape != (3, 2) or np.any(windows[:, 1] <= windows[:, 0]):
        raise ValueError("Three positive sensor windows required")
    if np.any(np.diff(windows[:, 1]) <= 0) or windows.max() != item["sensor_anchor_us"]:
        raise ValueError("Invalid causal sensor endpoints")
    reader = pool.get(item["raw_path"])
    square = tuple(item["square_xyxy"])
    raw = _read_crop(reader, int(windows.min()), int(windows.max()), square, 5.0, 256 * 1024**2)
    if raw is None:
        raise MemoryError("Current event crop exceeds registered 256 MiB bound")
    mapped = map_roi(raw, square, 128, 5.0)
    return torch.stack(
        [encode_mapped(mapped, int(a), int(b), 128, item["sequence_id"]) for a, b in windows]
    )[None]


def public_inputs(public: Path) -> tuple[dict, dict]:
    """Resolve the same physical prefix cohort without opening any target table."""
    if sha256(public) != TRUSTED_INPUTS_SHA256:
        raise ValueError("Untrusted public inventory")
    data = read(public)
    physical = set(data["physical_aliases"].values())
    queries = [q for q in data["queries"] if q["sample_token"] in physical]
    if len(queries) != 5805:
        raise ValueError("Physical population changed")
    return data, select_queries(queries)


def bindings(public: Path, data: dict, selected: dict) -> dict[str, Path]:
    """Hash all selected RAW bytes and reference NPZs outside the service clock."""
    files = {**sources(), "public": public, "protocol": PLAN}
    for name in (
        *PREPARATION_SOURCES,
        *DEPENDENCIES,
        "scripts/release_native_raw_latency.py",
        "scripts/release_native_local_latency.py",
        "scripts/final_train40_transfer.py",
        "operational/phase_timewarp/canonical_pair.py",
        "operational/phase_robust/bundle.py",
    ):
        files["source:" + name] = Path(name)
    for rows in selected.values():
        for query in rows:
            for identity in query["event_history"]:
                item = data["event_observations"][identity]
                path = Path(item["current_tensor_path"])
                if sha256(path) != item["current_tensor_sha256"]:
                    raise ValueError("Reference NPZ changed")
                files["npz:" + identity] = path
                files["raw:" + item["sequence_id"]] = Path(item["raw_path"])
    return files


def parity(public: Path, output: Path) -> None:
    """Admit RAW preparation only after exact comparison on every requested observation."""
    torch.set_num_threads(1)
    data, selected = public_inputs(public)
    run = ResearchRun(
        output,
        bindings(public, data, selected),
        {
            "experiment_id": "native-raw-preparation-parity-v1",
            "targets_read": False,
            "optimizer_updates": 0,
            "seed": 7,
        },
    )
    pool, records = ReaderPool(), []
    try:
        seen = set()
        for sequence, queries in selected.items():
            for query in queries:
                for identity in query["event_history"]:
                    if identity in seen:
                        continue
                    seen.add(identity)
                    item = data["event_observations"][identity]
                    start = time.perf_counter()
                    actual = prepare(pool, item).numpy()
                    elapsed = time.perf_counter() - start
                    with np.load(item["current_tensor_path"], allow_pickle=False) as archive:
                        expected = archive["events"][-1:]
                        if not archive["valid"][-1] or not np.array_equal(actual, expected):
                            raise ValueError("RAW tensor differs: " + identity)
                    records.append({"identity": identity, "exact": True, "seconds": elapsed})
            print(f"RAW parity {sequence}: {len(records)} observations", flush=True)
        write_json(output / "RECORDS.json", records)
        write_json(
            output / "RESULTS.json",
            {
                "all_exact": True,
                "observations": len(records),
                "queries": sum(map(len, selected.values())),
                "targets_read": False,
            },
        )
        run.finish()
    except Exception as error:
        run.fail(error)
        raise
    finally:
        pool.close()


def raw_native(models: dict, normalizers: dict, data: dict) -> tuple:
    """Use the native head and RGB path with a bounded RAW event input adapter."""
    native, pool = native_model(models, normalizers, data), ReaderPool()

    def event(self: FinalV16Train40Predictor, identity: str) -> tuple[np.ndarray, np.ndarray]:
        if identity not in self.event_values:
            item = self.inputs["event_observations"][identity]
            sensor = prepare(pool, item).to(self.device)
            delta = delta_matrix(torch.tensor([item["delta_t_s"]]), 3).to(self.device)
            with torch.inference_mode():
                raw, experts = engine._expert_block(self.models, "E", sensor, delta)
            self.event_values[identity] = raw[0].cpu().numpy(), experts[0].cpu().numpy()
        return self.event_values[identity]

    native._event_observation = MethodType(event, native)
    return native, pool


def benchmark(
    public: Path, parity_run: Path, completed: Path, output: Path, upstream: Path
) -> None:
    """Measure both arms after all prior GPU work has completed, retaining exact outputs."""
    verify_complete(parity_run)
    verify_complete(completed)
    parent = Path(read(completed / "FREEZE.json")["input_paths"]["native_freeze"]).parent
    verify_complete(parent)
    paired = parent / "paired_replay"
    if (
        not read(parity_run / "RESULTS.json")["all_exact"]
        or not read(paired / "RESULTS.json")["local_original_exact"]
    ):
        raise ValueError("Exact RAW and local replay admission required")
    data, selected = public_inputs(public)
    files = bindings(public, data, selected)
    previous = read(parity_run / "FREEZE.json")["inputs"]
    if any(sha256(path) != previous[key] for key, path in files.items()):
        raise ValueError("RAW parity input/source binding changed")
    paths = verified_upstream_paths(upstream)
    files.update(
        {
            "parity": parity_run / "COMPLETE.json",
            "previous": completed / "COMPLETE.json",
            "predictions": paired / "PREDICTIONS.json",
            **{"model:" + k: v for k, v in paths.items()},
        }
    )
    for queries in selected.values():
        for query in queries:
            for identity in query["rgb_history"]:
                for shard in data["rgb_observations"][identity]["producer_shards"]:
                    files["rgb:" + shard] = Path(data["media_root"]) / shard
    run = ResearchRun(
        output,
        files,
        {
            "experiment_id": "native-raw-latency-v1",
            "seed": 7,
            "targets_read": False,
            "optimizer_updates": 0,
            "threads": 1,
            "tf32": False,
            "torch_version": torch.__version__,
            "scope": "local RAW+RGB replay; supplied boxes; no sensor acquisition",
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
        models, normalizers = load_models(paths, "cuda")
        variants = {
            "original": models,
            "canonical": {
                **models,
                **{k: CanonicalPair(models[k]) for k in ("PAIR_E_MATCHED", "PAIR_R")},
            },
        }
        expected = {r["token"]: r for r in read(paired / "PREDICTIONS.json")["records"]}
        warmup = next(iter(selected.values()))[:2]
        for current in variants.values():
            native, pool = raw_native(current, normalizers, data)
            try:
                for query in warmup:
                    timed_predict(native, query, data, False)
            finally:
                native.close()
                pool.close()
        torch.cuda.reset_peak_memory_stats()
        records = []
        for sequence_index, (sequence, queries) in enumerate(selected.items()):
            arms = {
                arm: raw_native(current, normalizers, data) for arm, current in variants.items()
            }
            try:
                for index, query in enumerate(queries):
                    order = ["original", "canonical"]
                    if (index + sequence_index) % 2:
                        order.reverse()
                    for position, arm in enumerate(order):
                        result = timed_predict(arms[arm][0], query, data, False)
                        if result["prediction"] != expected[query["sample_token"]][arm]:
                            raise ValueError("RAW-to-output prediction differs from local replay")
                        records.append(
                            {
                                **result,
                                "arm": arm,
                                "sequence": sequence,
                                "token": query["sample_token"],
                                "query_us": query["query_time_us"],
                                "execution_position": position,
                            }
                        )
            finally:
                for native, pool in arms.values():
                    native.close()
                    pool.close()
            print(f"RAW latency {sequence} {len(queries)} queries", flush=True)
        service, queues = {}, {}
        for arm in variants:
            rows = [r for r in records if r["arm"] == arm]
            service[arm] = {
                "queries": len(rows),
                "p50_p95_p99_ms": quantiles([r["elapsed_ms"] for r in rows]),
            }
            queues[arm] = {}
            for sequence in selected:
                part = [r for r in rows if r["sequence"] == sequence]
                delays = queue_delay(
                    [(r["query_us"] - part[0]["query_us"]) / 1000 for r in part],
                    [r["elapsed_ms"] for r in part],
                )
                queues[arm][sequence] = {
                    "p50_p95_p99_ms": quantiles(delays),
                    "after_20ms": sum(d > 20 for d in delays),
                }
        memory = psutil.Process().memory_info()
        write_json(output / "MEASUREMENTS.json", records)
        write_json(
            output / "RESULTS.json",
            {
                "service": service,
                "queues": queues,
                "candidate_over_original_p95": service["canonical"]["p50_p95_p99_ms"][1]
                / service["original"]["p50_p95_p99_ms"][1],
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "process_peak_rss_bytes": getattr(memory, "peak_wset", memory.rss),
                "gpu": torch.cuda.get_device_name(),
                "all_predictions_exact": True,
                "native_accuracy_evaluated": False,
                "promotion_admitted": False,
            },
        )
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("parity", "benchmark"))
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parity", type=Path)
    parser.add_argument("--completed", type=Path)
    parser.add_argument(
        "--upstream", type=Path, default=Path("output/phase_submission/FINAL_MODEL/upstream")
    )
    args = parser.parse_args()
    if args.mode == "parity":
        parity(args.public, args.output)
    else:
        if args.parity is None or args.completed is None:
            parser.error("benchmark requires --parity and --completed")
        benchmark(args.public, args.parity, args.completed, args.output, args.upstream)


if __name__ == "__main__":
    main()
