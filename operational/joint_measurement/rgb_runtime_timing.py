"""Observe durable RGB update timings before/after residency without training."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import torch

from operational.range_transfer.common import freeze, read, sha, utc, write
from operational.rgb_port.accounting import read_bytes_shared

from .common import RUN
from .qa import exact
from .train import committed_sources

RUNTIME = RUN / "runtime/rgb_resident_v1"
OUT = RUNTIME / "timing"
FIT = RUN / "rgb/fits/RGB-PAIR-CORR-FROZEN-S7"


def describe(values: list[float]) -> dict:
    """Summarize finite positive per-update wall times with their explicit scope."""
    times = np.asarray(values, dtype=np.float64)
    if not len(times) or not np.isfinite(times).all() or (times <= 0).any():
        raise ValueError("Nonempty finite positive update timings required")
    return {
        "updates": len(times),
        "mean_seconds": float(times.mean()),
        "median_seconds": float(np.median(times)),
        "p90_seconds": float(np.quantile(times, 0.9)),
        "min_seconds": float(times.min()),
        "max_seconds": float(times.max()),
        "updates_per_second": float(len(times) / times.sum()),
    }


def register() -> dict:
    """Fix windows and scope before reading any post-transition timing series."""
    sources = {"operational/joint_measurement/rgb_runtime_timing.py": sha(Path(__file__))}
    committed_sources(sources)
    record = {
        "sources": sources,
        "origin_sha256": read(RUNTIME / "DURABLE_ORIGIN.json")["checkpoint_sha256"],
        "runtime_registration_sha256": sha(RUNTIME / "REGISTRATION.json"),
        "warmup_updates": 256,
        "comparison_window_updates": 256,
        "minimum_post_transition_updates": 512,
        "scope": "Trainer per-update wall time: input load through optimizer/journal commit",
        "excluded": "Checkpoint saves, parent restoration, scope setup and inter-scope idle",
        "interpretation": "Observational before/after, not a controlled causal speedup experiment",
        "optimizer_updates": 0,
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def snapshot() -> tuple[dict, str]:
    """Read an immutable pointed version, retrying only a concurrent version cleanup."""
    for _ in range(3):
        pointer = read(FIT / "CHECKPOINT_POINTER.json")
        name = pointer["version"]
        if Path(name).name != name:
            raise ValueError("Invalid checkpoint version name")
        try:
            data = read_bytes_shared(FIT / "checkpoint_versions" / name)
        except FileNotFoundError:
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest != pointer["checkpoint_sha256"]:
            raise ValueError("Checkpoint bytes differ from the durable pointer")
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        if payload["completed_updates"] != pointer["completed_updates"]:
            raise ValueError("Checkpoint update cursor differs from the durable pointer")
        return payload, digest
    raise RuntimeError("Checkpoint version advanced during all snapshot attempts; retry later")


def run() -> dict:
    """Verify the historical prefix, then summarize a sufficiently warmed durable suffix."""
    record = register()
    origin_path = RUNTIME / "origin_checkpoint.pt"
    if sha(origin_path) != record["origin_sha256"]:
        raise ValueError("Preserved runtime origin changed")
    origin = torch.load(origin_path, map_location="cpu", weights_only=False)
    current, current_sha = snapshot()
    old_count, new_count = origin["completed_updates"], current["completed_updates"]
    if current["identity"] != origin["identity"] or new_count < old_count:
        raise ValueError("Timing snapshot belongs to another trajectory")
    for key in ("curve", "timings"):
        if len(current["cursor"][key]) != new_count or not exact(
            origin["cursor"][key], current["cursor"][key][:old_count]
        ):
            raise ValueError("Historical trajectory prefix changed across migration")
    result = {
        "utc": utc(),
        "origin_updates": old_count,
        "durable_updates": new_count,
        "post_transition_updates": new_count - old_count,
        "checkpoint_sha256": current_sha,
        "history_prefix_exact": True,
        "registration_sha256": sha(OUT / "REGISTRATION.json"),
        "optimizer_updates": 0,
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    if new_count - old_count < record["minimum_post_transition_updates"]:
        result["status"] = "WAITING_FOR_WARMED_DURABLE_WINDOW"
    else:
        width = record["comparison_window_updates"]
        before = describe(origin["cursor"]["timings"][-width:])
        after = describe(current["cursor"]["timings"][-width:])
        result.update(
            status="OBSERVED",
            before=before,
            after=after,
            mean_time_ratio=after["mean_seconds"] / before["mean_seconds"],
            compared_before_updates=[old_count - width + 1, old_count],
            compared_after_updates=[new_count - width + 1, new_count],
            limitation=record["interpretation"],
        )
    write(OUT / f"OBSERVATION_{new_count:06d}.json", result)
    print(result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    print(register()) if args.register_only else run()
