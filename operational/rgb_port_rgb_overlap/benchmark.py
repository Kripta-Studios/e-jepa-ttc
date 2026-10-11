"""CPU-only RGB decoder and effective-prefetch parity from an immutable origin."""

# ruff: noqa: ANN401
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch


def run_case(
    run: Path,
    output: Path,
    name: str,
    *,
    workers: int,
    prefetch: bool,
    order: Any,
    position: int,
    samples: int,
    identity: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    from operational.rgb_port.run import make_bound_rgb_producer_source
    from operational.rgb_port_revision.loss import EffectiveSource
    from operational.rgb_port_rgb_loader.benchmark import fingerprint, rng_fingerprints
    from operational.rgb_port_rgb_loader.collate import installed_collate
    from operational.rgb_port_rgb_loader.loader import installed_dino_cache
    from operational.rgb_port_rgb_overlap.decoder import installed_decoder
    from operational.rgb_port_rgb_overlap.prefetch import capture_effective_prefetch

    source = make_bound_rgb_producer_source(run / "P_MANIFEST.json")
    revision = json.loads((run / "fits/R_A5/AUDIT_REVISION_RUNTIME.json").read_text())
    original_load = np.load
    calls: Counter[str] = Counter()
    lock = Lock()
    prepared_cache = source.dataset.prepared_cache
    if prepared_cache is None:
        raise ValueError("Benchmark requires the admitted prepared RGB cache")
    rgb_paths = {os.path.normcase(str(item[0]["path"])) for item in prepared_cache._lookup.values()}

    def counted_load(file: Any, *args: Any, **kwargs: Any) -> Any:
        kind = "rgb" if os.path.normcase(os.fspath(file)) in rgb_paths else "dino"
        with lock:
            calls[kind] += 1
        return original_load(file, *args, **kwargs)

    cursor = {"order": order.clone(), "position": position, "epoch": 1}
    original_order = cursor["order"].clone()
    ids = order[position : position + samples].tolist()
    times, hashes, parts = [], [], []
    before = rng_fingerprints()
    with ExitStack() as stack:
        dino = stack.enter_context(installed_dino_cache(source))
        stack.enter_context(installed_collate(source))
        decoder = (
            stack.enter_context(installed_decoder(source, workers=workers)) if workers else None
        )
        capture = (
            stack.enter_context(
                capture_effective_prefetch(
                    EffectiveSource, expected_source=source, batch_size=32, depth=2
                )
            )
            if prefetch
            else None
        )
        effective = EffectiveSource(
            source, SimpleNamespace(microbatch_size=32), revision["migration_sha256"]
        )
        if dict(effective.identity) != identity:
            raise ValueError("Source identity differs from the immutable origin")
        stack.enter_context(patch.object(np, "load", counted_load))
        if capture is not None:
            capture.bind(cursor)
        for start in range(0, samples, 32):
            requested = ids[start : start + 32]
            cursor_before = int(cursor["position"])
            tick = time.perf_counter()
            batch = effective.batch(requested, "rgb")
            times.append(time.perf_counter() - tick)
            if int(cursor["position"]) != cursor_before or not cursor["order"].equal(
                original_order
            ):
                raise AssertionError("Input pipeline mutated the canonical sampler cursor")
            hashes.append(fingerprint(batch))
            parts.append(
                [
                    {
                        "shape": list(part.events.shape),
                        "token_sha256": hashlib.sha256(
                            json.dumps(part.sample_tokens).encode()
                        ).hexdigest(),
                    }
                    for part in batch.parts
                ]
            )
            del batch
            cursor["position"] += len(requested)
            print(
                json.dumps({"case": name, "rows": start + len(requested), "wait_s": times[-1]}),
                flush=True,
            )
        if capture is not None:
            capture.close()
        stats = {
            "dino": dino.snapshot(),
            "decoder": None if decoder is None else decoder.snapshot(),
            "prefetch": None if capture is None else capture.snapshot(),
        }
    after = rng_fingerprints()
    if before != after:
        raise AssertionError("Input pipeline changed process RNG states")
    result = {
        "name": name,
        "workers": workers,
        "prefetch": prefetch,
        "source_seconds": sum(times),
        "batch_seconds": times,
        "batch_sha256": hashes,
        "parts": parts,
        "np_load_calls": dict(calls),
        "stats": stats,
        "rng_before": before,
        "rng_after": after,
        "cursor_only_advanced_by_consumer": True,
        "consumer_end_position": cursor["position"],
    }
    (output / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    del source, effective
    gc.collect()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    args = parser.parse_args()
    if args.samples < 32 or args.samples % 32:
        parser.error("Samples must be a positive multiple of 32")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"
    import torch

    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    origin_sha = hashlib.sha256(args.origin.read_bytes()).hexdigest()
    payload = torch.load(args.origin, weights_only=False, map_location="cpu")
    order = payload["cursor"]["order"].clone()
    position = int(payload["cursor"]["position"])
    if position + args.samples > len(order):
        raise ValueError("Samples must remain inside the saved epoch")
    identity = payload["identity"]["source_identity"]
    completed = int(payload["completed_updates"])
    del payload
    gc.collect()
    trials = [
        run_case(
            args.run,
            args.output,
            name,
            workers=workers,
            prefetch=prefetch,
            order=order,
            position=position,
            samples=args.samples,
            identity=identity,
        )
        for name, workers, prefetch in (
            ("baseline_A", 0, False),
            ("decoder_2", 2, False),
            ("decoder_4", 4, False),
            ("baseline_B", 0, False),
            ("decoder_2_prefetch_2", 2, True),
        )
    ]
    exact = all(item["batch_sha256"] == trials[0]["batch_sha256"] for item in trials)
    origin_unchanged = hashlib.sha256(args.origin.read_bytes()).hexdigest() == origin_sha
    result = {
        "schema": "rgb_overlap_cpu_decoder_v1",
        "status": "COMPLETE" if exact and origin_unchanged else "FAILED",
        "optimizer_updates": 0,
        "cuda_initialized": torch.cuda.is_initialized(),
        "torch_threads": torch.get_num_threads(),
        "origin_sha256": origin_sha,
        "origin_unchanged": origin_unchanged,
        "origin_completed_updates": completed,
        "cursor_position": position,
        "sample_count": args.samples,
        "exact_batch_parity": exact,
        "cursor_rng_unchanged": all(
            item["rng_before"] == item["rng_after"] and item["cursor_only_advanced_by_consumer"]
            for item in trials
        ),
        "code_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("benchmark.py", "decoder.py", "prefetch.py")
        },
        "trials": trials,
        "limitations": "Active RGB training can contend for CPU and disk; fresh logical caches, "
        "OS cache not flushed. Decoder variants have one trial each; source-only measurements. "
        "Prefetch case validates parity without simulated model work, so it does not measure "
        "whole-training acceleration. Its digest work can overlap pending input preparation.",
    }
    (args.output / "RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return int(not exact or not origin_unchanged or result["cuda_initialized"])


if __name__ == "__main__":
    raise SystemExit(main())
