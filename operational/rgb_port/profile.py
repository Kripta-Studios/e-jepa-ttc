"""Measured, scope-explicit latency profiling for frozen RGB-PORT endpoints."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .accounting import atomic_write_json, read_json_shared


def profile_callable(
    function: Callable[[], Any],
    *,
    scope: str,
    device: str,
    warm_iterations: int = 50,
    synchronize: Callable[[], None] | None = None,
    memory_reader: Callable[[], int] | None = None,
) -> dict[str, Any]:
    """Measure one cold and repeated warm batch-1 calls without inferred costs."""
    if not scope or warm_iterations < 1:
        raise ValueError("A non-empty scope and at least one warm iteration are required")
    import torch

    cuda = str(device).startswith("cuda")
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    vram_before = torch.cuda.memory_allocated(device) if cuda else None
    sync = synchronize or (lambda: torch.cuda.synchronize(device) if cuda else None)
    sync()
    memory_before = memory_reader() if memory_reader else None
    start = time.perf_counter_ns()
    function()
    sync()
    cold_ms = (time.perf_counter_ns() - start) / 1e6
    samples: list[float] = []
    for _ in range(warm_iterations):
        sync()
        start = time.perf_counter_ns()
        function()
        sync()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    memory_after = memory_reader() if memory_reader else None
    return {
        "schema": "rgb_port_profile_v1",
        "status": "MEASURED",
        "scope": scope,
        "device": device,
        "batch_size": 1,
        "cold_ms": cold_ms,
        "warm_iterations": warm_iterations,
        "warm_ms": {
            "mean": statistics.fmean(samples),
            "median": statistics.median(samples),
            "p90": _quantile(samples, 0.90),
            "p95": _quantile(samples, 0.95),
            "min": min(samples),
            "max": max(samples),
        },
        "vram_allocated_before_bytes": vram_before,
        "vram_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
        "vram_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None,
        "cold_definition": (
            "first_timed_call_not_OS_cache_flush; other_routes_or_parity_may_have_warmed_sources"
        ),
        "warm_definition": "subsequent_calls_with_application_and_OS_caches",
        "memory_bytes_before": memory_before,
        "memory_bytes_after": memory_after,
    }


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def validate_profile(path: Path) -> dict[str, Any]:
    profile = read_json_shared(path)
    if profile.get("schema") != "rgb_port_profile_v1" or profile.get("status") != "MEASURED":
        raise ValueError(f"Not a measured RGB-PORT profile: {path}")
    if profile.get("batch_size") != 1 or not profile.get("scope"):
        raise ValueError("Profile lacks its batch-1 scope")
    values = [profile.get("cold_ms"), *profile.get("warm_ms", {}).values()]
    if any(not isinstance(value, (int, float)) or value < 0 for value in values):
        raise ValueError("Profile contains an invented/missing/negative timing")
    return profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and consolidate measured RGB-PORT batch-1 profiles."
    )
    parser.add_argument("--profile", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    profiles = [validate_profile(path) for path in args.profile]
    scopes = [str(item["scope"]) for item in profiles]
    if len(scopes) != len(set(scopes)):
        raise ValueError("Profile scopes must be unique")
    atomic_write_json(
        args.output,
        {
            "schema": "rgb_port_profile_bundle_v1",
            "status": "COMPLETE",
            "profiles": profiles,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
