"""Run the frozen overlap trainer with an admitted bounded adaptive input cache."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from operational.efficient_context.common import Lease, digest
from operational.train40_system import engine_overlap
from operational.train40_system.checkpoint import DurableState
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json


def verify_execution(output: Path) -> dict:
    """Verify the additive cache admission and preserve the original OpenMP policy."""
    paths = {
        name: output / name
        for name in (
            "COORDINATION_FREEZE.json",
            "CONTROLLER_PAUSE_SAFE_FREEZE.json",
            "ADAPTIVE_CACHE_FREEZE.json",
        )
    }
    freezes = {name: read(path) for name, path in paths.items()}
    for freeze in freezes.values():
        verify_sources(freeze)
    cache = freezes["ADAPTIVE_CACHE_FREEZE.json"]
    for field, name in (
        ("coordination_freeze_sha256", "COORDINATION_FREEZE.json"),
        ("controller_pause_safe_freeze_sha256", "CONTROLLER_PAUSE_SAFE_FREEZE.json"),
    ):
        if cache[field] != digest(paths[name]):
            raise ValueError(f"Adaptive cache admission differs: {field}")
    if cache["engine_sha256"] != digest(Path(__file__)):
        raise ValueError("Adaptive cache engine differs from its admission")
    if os.environ.get("OMP_WAIT_POLICY", "ACTIVE").upper() != "ACTIVE":
        raise ValueError("Adaptive cache comparison requires the original active OpenMP policy")
    if os.environ.get("KMP_BLOCKTIME", "200") != "200":
        raise ValueError("Adaptive cache comparison requires the original OpenMP block time")
    return cache


@contextmanager
def installed_inputs(directory: Path) -> Iterator[list[Any]]:
    """Substitute only input caching and append telemetry after an original durable save."""
    from operational.train40_system.adaptive_cache import AdaptiveSealedInputs

    instances: list[Any] = []
    original_inputs = engine_overlap.SealedInputs
    original_save = cast(Any, DurableState.save)

    class RecordingInputs(AdaptiveSealedInputs):
        def __init__(self, output: Path) -> None:
            super().__init__(output)
            instances.append(self)

    def save(self: DurableState, *args: object, **kwargs: object) -> None:
        original_save(self, *args, **kwargs)
        atomic_json(
            directory / "ADAPTIVE_CACHE_COUNTERS.json",
            {
                "committed_updates": self.committed,
                "durable_updates": self.durable,
                "inputs": [value.snapshot() for value in instances],
                "checkpoint_contract_unchanged": True,
            },
        )

    engine_overlap.SealedInputs = RecordingInputs
    DurableState.save = cast(Any, save)
    try:
        yield instances
    finally:
        engine_overlap.SealedInputs = original_inputs
        DurableState.save = original_save


def run(output: Path, arm: str) -> None:
    """Resume the exact scientific cursor using the byte-identical admitted inputs."""
    from datetime import UTC, datetime

    verify_execution(output)
    directory = output / "fits" / f"{arm}_seed7"
    runtime = {
        "schema": "train40_adaptive_cache_runtime_v1",
        "started_utc": datetime.now(UTC).isoformat(),
        "adaptive_cache_freeze_sha256": digest(output / "ADAPTIVE_CACHE_FREEZE.json"),
        "environment": {key: os.environ.get(key) for key in ("OMP_WAIT_POLICY", "KMP_BLOCKTIME")},
        "checkpoint_contract_unchanged": True,
        "model_loss_optimizer_sampler_and_precision_unchanged": True,
        "extra_optimizer_updates": 0,
    }
    atomic_json(directory / "ADAPTIVE_CACHE_RUNTIME.json", {**runtime, "status": "RUNNING"})
    try:
        with installed_inputs(directory):
            engine_overlap.run(output, arm)
    finally:
        atomic_json(
            directory / "ADAPTIVE_CACHE_RUNTIME.json",
            {
                **runtime,
                "status": "EXITED",
                "ended_utc": datetime.now(UTC).isoformat(),
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--arm", choices=("a5", "c2f"), required=True)
    args = parser.parse_args()
    with Lease(args.output.resolve()):
        run(args.output.resolve(), args.arm)
