"""Compose admitted passive worker waiting with same-loss relation-map reuse."""

from __future__ import annotations

from pathlib import Path

from operational.train40_system import engine_idle_wait


def _verify_execution(output: Path) -> dict:
    """Bind this runtime to the admitted coordination, pause and reuse sources."""
    from operational.efficient_context.common import digest
    from operational.train40_system.contracts import read, verify_sources

    coordination_path = output / "COORDINATION_FREEZE.json"
    pause_path = output / "CONTROLLER_PAUSE_SAFE_FREEZE.json"
    idle_path = output / "IDLE_WAIT_FREEZE.json"
    reuse_path = output / "RELATION_REUSE_FREEZE.json"
    coordination, pause, idle, reuse = map(
        read, (coordination_path, pause_path, idle_path, reuse_path)
    )
    for freeze in (coordination, pause, idle, reuse):
        verify_sources(freeze)
    if reuse["coordination_freeze_sha256"] != digest(coordination_path):
        raise ValueError("Relation reuse does not bind the coordination freeze")
    if reuse["controller_pause_safe_freeze_sha256"] != digest(pause_path):
        raise ValueError("Relation reuse does not bind the pause-safe freeze")
    if reuse["idle_wait_freeze_sha256"] != digest(idle_path):
        raise ValueError("Relation reuse does not bind the idle-wait freeze")
    if reuse["engine_relation_reuse_sha256"] != digest(Path(__file__)):
        raise ValueError("Relation-reuse engine differs from its freeze")
    return reuse


def run(output: Path, arm: str) -> None:
    """Configure the admitted environment before importing Torch-backed modules."""
    engine_idle_wait.configure_environment()
    from datetime import UTC, datetime
    from typing import Any, cast

    import e_jepa_ttc.training.causal_scale_eap as training
    from operational.efficient_context.common import digest
    from operational.train40_system.checkpoint import DurableState
    from operational.train40_system.durable_io import atomic_json
    from operational.train40_system.relation_reuse import (
        ReuseCounters,
        TrainingModule,
        scoped_relation_reuse,
    )

    _verify_execution(output)
    directory = output / "fits" / f"{arm}_seed7"
    runtime = directory / "RELATION_REUSE_RUNTIME.json"
    freeze_path = output / "RELATION_REUSE_FREEZE.json"
    idle_freeze_path = output / "IDLE_WAIT_FREEZE.json"
    counters = ReuseCounters()
    atomic_json(
        runtime,
        {
            "schema": "train40_relation_reuse_runtime_v1",
            "status": "RUNNING",
            "started_utc": datetime.now(UTC).isoformat(),
            "relation_reuse_freeze_sha256": digest(freeze_path),
            "idle_wait_freeze_sha256": digest(idle_freeze_path),
            "idle_wait_engine_sha256": digest(Path(engine_idle_wait.__file__)),
            "environment": dict(engine_idle_wait.IDLE_WAIT_ENVIRONMENT),
            "checkpoint_contract_unchanged": True,
            "objective_operations_unchanged": True,
            "additional_optimizer_updates": 0,
        },
    )
    original_save = cast(Any, DurableState.save)

    def save(self: DurableState, *args: object, **kwargs: object) -> None:
        original_save(self, *args, **kwargs)
        atomic_json(
            self.directory / "RELATION_REUSE_COUNTERS.json",
            {
                "schema": "train40_relation_reuse_counters_v1",
                "committed_updates": self.committed,
                "durable_updates": self.durable,
                "checkpoint": str(self.path),
                "checkpoint_receipt": str(self.directory / "CHECKPOINT_RECEIPT.json"),
                "counters": counters.snapshot(),
            },
        )

    DurableState.save = cast(Any, save)
    try:
        with scoped_relation_reuse(cast(TrainingModule, training), counters):
            engine_idle_wait.run(output, arm)
    finally:
        DurableState.save = original_save
        atomic_json(
            runtime,
            {
                "schema": "train40_relation_reuse_runtime_v1",
                "status": "EXITED",
                "ended_utc": datetime.now(UTC).isoformat(),
                "relation_reuse_freeze_sha256": digest(freeze_path),
                "idle_wait_freeze_sha256": digest(idle_freeze_path),
                "idle_wait_engine_sha256": digest(Path(engine_idle_wait.__file__)),
                "environment": dict(engine_idle_wait.IDLE_WAIT_ENVIRONMENT),
                "checkpoint_contract_unchanged": True,
                "objective_operations_unchanged": True,
                "additional_optimizer_updates": 0,
                "counters": counters.snapshot(),
            },
        )


if __name__ == "__main__":
    engine_idle_wait.configure_environment()
    import argparse

    from operational.efficient_context.common import Lease
    from operational.train40_system.data_audit import OUTPUT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--arm", choices=("a5", "c2f"), required=True)
    args = parser.parse_args()
    with Lease(args.output.resolve()):
        run(args.output.resolve(), args.arm)
