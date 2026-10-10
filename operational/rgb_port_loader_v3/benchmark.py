"""CPU-only real-data paired loader benchmark; never executes optimizer updates."""

# ruff: noqa: ANN401
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import psutil


def fingerprint(batch: Any) -> str:
    """Hash every returned field including tensor shape, dtype and exact bytes."""
    import torch

    digest = hashlib.sha256()
    for field in dataclasses.fields(batch):
        value = getattr(batch, field.name)
        digest.update(field.name.encode())
        if isinstance(value, torch.Tensor):
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.contiguous().numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    """Measure one role with the same canonical order and original cache cap."""
    import torch

    from operational.rgb_port.train_producers import (
        EventProducerSource,
        ProducerSource,
        _epoch_order,
        _windows_commit_headroom,
    )
    from operational.rgb_port_loader_v3.loader import DepthTwoEventPrefetch, ParallelGroupRowCache
    from operational.rgb_port_revision.cache import GroupRowCache

    torch.set_num_threads(args.torch_threads)
    source = EventProducerSource(args.manifest, args.event_source)
    source._inputs.limit = (512 if args.fit == "A5" else 1024) * 1024**2
    cache = (
        GroupRowCache(source)
        if args.mode == "baseline"
        else ParallelGroupRowCache(source, workers=2)
    )
    source._audit_row_cache = cache
    generator = torch.Generator().manual_seed(
        491 + args.group_seed + (1000 if args.distinct_orders and args.fit == "C2F" else 0)
    )
    order = _epoch_order(cast(ProducerSource, source), generator)[: args.samples].clone()
    view = SimpleNamespace(
        population_size=len(order), frame_counts=[3] * len(order), batch=source.batch
    )
    loader = DepthTwoEventPrefetch(view, batch_size=32) if args.mode == "prefetch" else None
    cursor = {"order": order, "position": 0}
    peak = {
        "rss": 0,
        "private": 0,
        "available_min": psutil.virtual_memory().available,
        "commit_headroom_min": _windows_commit_headroom() or 0,
    }
    done = threading.Event()

    def monitor() -> None:
        process = psutil.Process()
        while not done.wait(0.1):
            peak["rss"] = max(peak["rss"], process.memory_info().rss)
            peak["private"] = max(peak["private"], getattr(process.memory_info(), "private", 0))
            peak["commit_headroom_min"] = min(
                peak["commit_headroom_min"], _windows_commit_headroom() or 0
            )
            peak["available_min"] = min(peak["available_min"], psutil.virtual_memory().available)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    hashes, waits = [], []
    hash_seconds = 0.0
    start = time.perf_counter()
    try:
        if loader is not None:
            loader.bind(cursor)
        for position in range(0, len(order), 32):
            ids = order[position : position + 32].tolist()
            tick = time.perf_counter()
            batch = (loader or source).batch(ids, "event")
            waits.append(time.perf_counter() - tick)
            tick = time.perf_counter()
            hashes.append(fingerprint(batch))
            hash_seconds += time.perf_counter() - tick
            del batch
            cursor["position"] = position + len(ids)
            print(
                json.dumps(
                    {
                        "fit": args.fit,
                        "mode": args.mode,
                        "rows": cursor["position"],
                        "wait_s": waits[-1],
                    }
                ),
                flush=True,
            )
        elapsed = time.perf_counter() - start
    finally:
        if loader is not None:
            loader.close()
        if isinstance(cache, ParallelGroupRowCache):
            cache.close()
        source.close()
        done.set()
        monitor_thread.join()
    return {
        "schema": "rgb_port_loader_v3_cpu_trial_v1",
        "utc": datetime.now(UTC).isoformat(),
        "fit": args.fit,
        "mode": args.mode,
        "samples": len(order),
        "group_seed": args.group_seed,
        "torch_threads": args.torch_threads,
        "distinct_orders": args.distinct_orders,
        "order_sha256": hashlib.sha256(order.numpy().tobytes()).hexdigest(),
        "batch_sha256": hashes,
        "wall_s_including_hash": elapsed,
        "hash_s": hash_seconds,
        "consumer_wait_s": sum(waits),
        "batch_wait_s": waits,
        "prepare_s": None if loader is None else loader.prepare_seconds,
        "shard_reads": cache.reads,
        "row_hits": cache.hits,
        "row_cache_peak_bytes": cache.peak_bytes,
        "memory": peak,
        "decoder_workers": 1 if args.mode == "baseline" else 2,
        "prefetch_depth": 2 if loader else 0,
        "max_queued_batches": 0 if loader is None else loader.outstanding_max,
        "optimizer_updates": 0,
        "cuda_used": False,
        "note": "Live training continued. Hashing is consumer work and overlaps prefetch; "
        "these timings are not training throughput or GPU latency.",
    }


def main() -> int:
    """Run one trial or two isolated CPU consumers under the same live load."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--event-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["baseline", "parallel", "prefetch"], required=True)
    parser.add_argument("--fit", choices=["A5", "C2F"])
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--group-seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--distinct-orders", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.fit:
        result = run_case(args)
        (args.output / f"{args.fit}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return 0
    children, streams = [], []
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for fit in ("A5", "C2F"):
        log = (args.output / f"{fit}.log").open("w", encoding="utf-8")
        streams.append(log)
        children.append(
            subprocess.Popen(
                [sys.executable, "-B", "-m", __spec__.name, *sys.argv[1:], "--fit", fit],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
        )
    codes = [child.wait() for child in children]
    for stream in streams:
        stream.close()
    return int(any(codes))


if __name__ == "__main__":
    raise SystemExit(main())
