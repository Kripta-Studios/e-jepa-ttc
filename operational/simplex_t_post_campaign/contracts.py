"""Durable fragments, matched timing order and fail-closed route admission."""

# JSON receipts and producer tensors intentionally meet at dynamic I/O boundaries.
# ruff: noqa: ANN401

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

MODELS = (
    "H1_SEED7",
    "H8_SEED7",
    "H16_SEED7",
    "FULL_C0",
    "A5_ONLY_C0",
    "C2F_ONLY_C0",
    "A5_PAIR_C0",
    "SET_AGE_C0",
    "SET_NOTIME_C0",
)


def digest(path: Path) -> str:
    """Hash a complete file with bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> Any:
    """Read an explicitly named JSON receipt."""
    return json.loads(path.read_bytes())


def save(path: Path, value: Any) -> None:
    """Flush before replacing a complete receipt; never expose a partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def checked(path: Path, expected: str) -> Path:
    """Reject changed evidence before loading its payload."""
    if digest(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return path


def order(block: int, fragment: int) -> tuple[str, ...]:
    """Rotate predetermined model order across 60 matched fragments."""
    shift = (block * 20 + fragment) % len(MODELS)
    return MODELS[shift:] + MODELS[:shift]


def timed(
    call: Callable[[], Any], clock: Callable[[], int] = time.perf_counter_ns
) -> tuple[Any, float]:
    """Time only head forward plus canonical emission, excluding admission/checks."""
    start = clock()
    result = call()
    return result, (clock() - start) / 1_000_000


def fragment(
    path: Path, binding: dict[str, Any], produce: Callable[[], Sequence[float]]
) -> tuple[dict[str, Any], bool]:
    """Reuse only complete bound fragments; reject changed or invalid measurements."""
    if path.exists():
        result = read(path)
        reused = True
    else:
        result = dict(binding=binding, raw_ms=list(produce()))
        validate_fragment(result, binding)
        save(path, result)
        reused = False
    validate_fragment(result, binding)
    return result, reused


def validate_fragment(result: dict, binding: dict) -> None:
    """Require exact identity and 25 positive finite durations."""
    import math

    values = result.get("raw_ms", [])
    if result.get("binding") != binding or len(values) != 25:
        raise ValueError("fragment binding/count changed")
    if any(not isinstance(x, (float, int)) or not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("nonfinite/nonpositive timing")


def route_admission(ack: dict) -> dict:
    """Require a prospective exclusive slot; idle hardware is not permission."""
    slot = ack.get("resources", {}).get("exclusive_gpu_and_heavy_io_slot")
    return dict(
        status="BLOCKED_NO_EXCLUSIVE_GPU_HEAVY_IO_LEASE"
        if not slot
        else "LEASE_REQUIRES_LIVE_VALIDATION",
        acknowledged_slot=slot,
        raw_payloads_read=False,
        system_latency_measured=False,
        missing_dependency=(
            "Explicit exclusive GPU/heavy-I/O lease from Stage70–76 resource owner; "
            "then validate TRAIN raw sources and frozen producer bindings."
        ),
    )


def segmented_route(
    *,
    admitted: bool,
    supplied_roi: Any,
    events: Callable[[], Any],
    context: Callable[[Any, Any], Any],
    experts: Callable[[Any], Any],
    features: Callable[[Any], Any],
    head: Callable[[Any], Any],
    synchronize: Callable[[], None],
    clock: Callable[[], int] = time.perf_counter_ns,
) -> tuple[Any, dict[str, float]]:
    """Measure an admitted frozen route conditioned on external ROI.

    No detector, tracker or AEB is included. Callers must supply verified canonical
    producers and synchronize transfers. This interface grants no resource lease.
    """
    if not admitted:
        raise PermissionError("exclusive lease and verified sources required before raw calls")
    durations = {}
    synchronize()
    total_start = clock()
    value = None
    for name, call in (
        ("raw_slice", lambda _: events()),
        ("roi_context", lambda current: context(current, supplied_roi)),
        ("experts_and_transfers", experts),
        ("features", features),
        ("head_and_emission", head),
    ):
        synchronize()
        start = clock()
        value = call(value)
        synchronize()
        durations[name] = (clock() - start) / 1_000_000
    durations["total_conditioned_on_supplied_roi"] = (clock() - total_start) / 1_000_000
    return value, durations
