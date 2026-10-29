"""Run the frozen overlap engine with passive OpenMP worker waiting."""

from __future__ import annotations

import os
from pathlib import Path

IDLE_WAIT_ENVIRONMENT = {
    "OMP_WAIT_POLICY": "PASSIVE",
    "KMP_BLOCKTIME": "0",
}


def configure_environment() -> None:
    """Set and verify only the two admitted process-local idle-wait controls."""
    for name, value in IDLE_WAIT_ENVIRONMENT.items():
        os.environ[name] = value
    if {name: os.environ.get(name) for name in IDLE_WAIT_ENVIRONMENT} != IDLE_WAIT_ENVIRONMENT:
        raise RuntimeError("Passive OpenMP idle-wait environment was not installed")


def run(output: Path, arm: str) -> None:
    """Verify the additive admission and invoke the otherwise frozen trainer."""
    configure_environment()
    from operational.efficient_context.common import ROOT, digest
    from operational.train40_system import engine_overlap
    from operational.train40_system.contracts import read, verify_sources
    from operational.train40_system.durable_io import atomic_json

    freeze_paths = {
        name: output / name
        for name in (
            "COORDINATION_FREEZE.json",
            "CONTROLLER_PAUSE_SAFE_FREEZE.json",
            "IDLE_WAIT_FREEZE.json",
        )
    }
    for path in freeze_paths.values():
        verify_sources(read(path))
    source = ROOT / "operational/train40_system/engine_idle_wait.py"
    atomic_json(
        output / "fits" / f"{arm}_seed7" / "IDLE_WAIT_RUNTIME.json",
        {
            "schema": "train40_idle_wait_runtime_v1",
            "source_sha256": digest(source),
            "environment": dict(IDLE_WAIT_ENVIRONMENT),
            "coordination_freeze_sha256": digest(freeze_paths["COORDINATION_FREEZE.json"]),
            "controller_pause_safe_freeze_sha256": digest(
                freeze_paths["CONTROLLER_PAUSE_SAFE_FREEZE.json"]
            ),
            "idle_wait_freeze_sha256": digest(freeze_paths["IDLE_WAIT_FREEZE.json"]),
            "threads_batch_math_rng_precision_and_checkpoint_contract_unchanged": True,
        },
    )
    engine_overlap.run(output, arm)


if __name__ == "__main__":
    configure_environment()
    import argparse

    from operational.efficient_context.common import Lease
    from operational.train40_system.data_audit import OUTPUT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--arm", choices=("a5", "c2f"), required=True)
    args = parser.parse_args()
    with Lease(args.output.resolve()):
        run(args.output.resolve(), args.arm)
