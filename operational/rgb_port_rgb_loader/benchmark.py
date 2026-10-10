"""Reproducible CPU ABBA comparison from the exact durable RGB sampler cursor."""

# ruff: noqa: ANN401 -- recursively inspect the frozen dataclass batch contract
from __future__ import annotations

import argparse
import cProfile
import dataclasses
import gc
import hashlib
import json
import os
import pickle
import platform
import random
import subprocess
import time
from collections import Counter
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch


def fingerprint(value: Any) -> str:
    """Include every nested field, tensor dtype, shape, metadata and exact bytes."""
    import torch

    digest = hashlib.sha256()

    def add(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            if item.device.type != "cpu":
                raise ValueError("CPU-only benchmark received a non-CPU tensor")
            digest.update(str((str(item.dtype), tuple(item.shape))).encode())
            digest.update(item.contiguous().numpy().tobytes())
        elif dataclasses.is_dataclass(item) and not isinstance(item, type):
            digest.update(type(item).__name__.encode())
            for field in dataclasses.fields(item):
                digest.update(field.name.encode())
                add(getattr(item, field.name))
        elif isinstance(item, (list, tuple)):
            digest.update(str((type(item).__name__, len(item))).encode())
            for child in item:
                add(child)
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())

    add(value)
    return digest.hexdigest()


def rng_fingerprints() -> dict[str, str]:
    """Snapshot process RNGs without using any CUDA APIs."""
    import numpy as np
    import torch

    return {
        "python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
        "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_trial(
    *,
    run: Path,
    ids: list[int],
    source_identity: dict[str, Any],
    optimized: bool,
    name: str,
    output: Path,
    profile: bool,
) -> dict[str, Any]:
    """Start each trial with fresh logical caches; OS caches are not flushed."""
    import numpy as np

    from operational.rgb_port.run import make_bound_rgb_producer_source
    from operational.rgb_port_revision.loss import EffectiveSource
    from operational.rgb_port_rgb_loader.collate import installed_collate
    from operational.rgb_port_rgb_loader.loader import installed_dino_cache

    source = make_bound_rgb_producer_source(run / "P_MANIFEST.json")
    revision = json.loads((run / "fits/R_A5/AUDIT_REVISION_RUNTIME.json").read_text())
    effective = EffectiveSource(
        source,
        SimpleNamespace(microbatch_size=32, effective_batch_size=32),
        revision["migration_sha256"],
    )
    if dict(effective.identity) != source_identity:
        raise ValueError("Benchmark source identity differs from the durable training checkpoint")
    if source.dataset.prepared_cache is None:
        raise ValueError("Benchmark requires the admitted prepared RGB cache")
    rgb_paths = {
        os.path.normcase(str(item[0]["path"]))
        for item in source.dataset.prepared_cache._lookup.values()
    }
    dino_paths = {os.path.normcase(str(item["path"])) for item in source._dino_shards}
    reads: Counter[str] = Counter()
    by_path: Counter[str] = Counter()
    original_load = np.load

    def counted_load(file: Any, *args: Any, **kwargs: Any) -> Any:
        # Paths already come from the admitted absolute-path manifests. Do not
        # resolve them again here: filesystem probes would bias measured time.
        path = os.path.normcase(os.fspath(file))
        reads["rgb" if path in rgb_paths else "dino" if path in dino_paths else "other"] += 1
        by_path[path] += 1
        return original_load(file, *args, **kwargs)

    timings: list[float] = []
    fingerprints: list[str] = []
    batch_details: list[dict[str, Any]] = []
    hash_seconds = 0.0
    before = rng_fingerprints()
    profiler = cProfile.Profile() if profile else None
    started = time.perf_counter()
    with ExitStack() as stack:
        cache = stack.enter_context(installed_dino_cache(source)) if optimized else None
        if optimized:
            stack.enter_context(installed_collate(source))
        stack.enter_context(patch.object(np, "load", counted_load))
        for position in range(0, len(ids), 32):
            selected = ids[position : position + 32]
            if profiler is not None:
                profiler.enable()
            tick = time.perf_counter()
            batch = effective.batch(selected, "rgb")
            timings.append(time.perf_counter() - tick)
            if profiler is not None:
                profiler.disable()
            tick = time.perf_counter()
            fingerprints.append(fingerprint(batch))
            hash_seconds += time.perf_counter() - tick
            batch_details.append(
                {
                    "indices": selected,
                    "parts": [
                        {
                            "frames": int(part.events.shape[1]),
                            "tokens": list(part.sample_tokens),
                            "rows": int(part.events.shape[0]),
                        }
                        for part in batch.parts
                    ],
                }
            )
            del batch
            print(
                json.dumps({"trial": name, "rows": position + len(selected), "s": timings[-1]}),
                flush=True,
            )
        stats = None if cache is None else cache.snapshot()
    elapsed = time.perf_counter() - started
    after = rng_fingerprints()
    if before != after:
        raise AssertionError("Input loading changed a process random-number state")
    if profiler is not None:
        profiler.dump_stats(str(output / f"{name}.pstats"))
    result = {
        "name": name,
        "optimized": optimized,
        "rows": len(ids),
        "source_identity": dict(effective.identity),
        "source_seconds": sum(timings),
        "batch_source_seconds": timings,
        "batch_sha256": fingerprints,
        "batch_details": batch_details,
        "hash_seconds": hash_seconds,
        "wall_seconds_including_hash": elapsed,
        "np_load_calls": dict(reads),
        "np_load_calls_by_path": dict(by_path),
        "dino_cache": stats,
        "rng_before": before,
        "rng_after": after,
        "rng_unchanged": before == after,
        "profile_enabled": profile,
    }
    _json(output / f"{name}.json", result)
    del effective, source
    gc.collect()
    return result


def main() -> int:
    """Compare A1/B1/B2/A2 on exactly the next saved-cursor examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.samples <= 0 or args.samples % 32 or args.torch_threads < 1:
        parser.error("Positive samples divisible by 32 and positive thread count required")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(args.torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.torch_threads)
    import numpy as np
    import torch

    torch.set_num_threads(args.torch_threads)
    run = args.run.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    fit = run / "fits/R_A5"
    pointer_path = fit / "CHECKPOINT_POINTER.json"
    pointer_bytes = pointer_path.read_bytes()
    pointer = json.loads(pointer_bytes)
    checkpoint = fit / "checkpoint_versions" / pointer["version"]
    if _sha256(checkpoint) != pointer["checkpoint_sha256"]:
        raise ValueError("Immutable checkpoint hash differs from pointer")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cursor = payload["cursor"]
    position = int(cursor["position"])
    ids = cursor["order"][position : position + args.samples].tolist()
    if len(ids) != args.samples:
        raise ValueError("Requested rows cross epoch; use a shorter slice without changing order")
    source_identity = dict(payload["identity"]["source_identity"])
    epoch = int(cursor["epoch"])
    del payload, cursor
    gc.collect()
    trials = []
    for name, optimized in (("A1", False), ("B1", True), ("B2", True), ("A2", False)):
        trials.append(
            run_trial(
                run=run,
                ids=ids,
                source_identity=source_identity,
                optimized=optimized,
                name=name,
                output=args.output,
                profile=args.profile,
            )
        )
    parity = all(trial["batch_sha256"] == trials[0]["batch_sha256"] for trial in trials)
    baseline = sum(item["source_seconds"] for item in trials if not item["optimized"]) / 2
    optimized_s = sum(item["source_seconds"] for item in trials if item["optimized"]) / 2
    result = {
        "schema": "rgb_port_rgb_loader_cpu_abba_v1",
        "status": "COMPLETE" if parity else "FAILED_PARITY",
        "utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_sha256": {
            name: _sha256(Path(__file__).with_name(name))
            for name in ("benchmark.py", "loader.py", "collate.py")
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": pointer["checkpoint_sha256"],
        "completed_updates": pointer["completed_updates"],
        "epoch": epoch,
        "cursor_position": position,
        "indices": ids,
        "order_slice_sha256": hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest(),
        "all_fields_exact": parity,
        "rng_unchanged": all(item["rng_unchanged"] for item in trials),
        "checkpoint_pointer_unchanged": pointer_path.read_bytes() == pointer_bytes,
        "optimizer_updates": 0,
        "cuda_initialized": torch.cuda.is_initialized(),
        "baseline_mean_source_seconds": baseline,
        "optimized_mean_source_seconds": optimized_s,
        "source_time_reduction_percent": 100 * (1 - optimized_s / baseline),
        "trials": trials,
        "limitations": "Fresh logical caches each trial; OS file cache not flushed. "
        "Source-only CPU times exclude hashing, GPU computation and checkpoint saving. "
        "Two trials per condition; no claim of whole-training speedup. "
        "cProfile overhead is included if requested.",
    }
    _json(args.output / "RESULT.json", result)
    return int(
        not parity or result["cuda_initialized"] or not result["checkpoint_pointer_unchanged"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
