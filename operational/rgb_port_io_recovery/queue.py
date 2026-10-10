"""Preserve graph/concurrent routing and retry supervisor disk interruptions."""

# ruff: noqa: ANN401

from __future__ import annotations

import argparse
import copy
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from operational.rgb_port import run as frozen_queue
from operational.rgb_port.accounting import (
    OwnerLease,
    atomic_write_json,
    identity_is_live,
    read_json_shared,
)
from operational.rgb_port_c2f_graph import queue as graph_queue
from operational.rgb_port_concurrent import queue as concurrent_queue

from .common import FREEZE_NAME, retryable_read_error, validate_freeze
from .producer import DELEGATES, FIT_IDS

PRODUCER_MODULE = "operational.rgb_port_io_recovery.producer"


def route(command: list[str], freeze: Path) -> list[str]:
    """Wrap exactly the admitted four producers, preserving all original args."""
    if "-m" not in command or "--fit-id" not in command:
        return command
    index = command.index("-m") + 1
    if command[index] not in DELEGATES or command[command.index("--fit-id") + 1] not in FIT_IDS:
        return command
    return [
        *command[:index],
        PRODUCER_MODULE,
        "--io-freeze",
        str(freeze),
        "--delegate-module",
        command[index],
        "--",
        *command[index + 1 :],
    ]


def detach_factory(original: Callable[..., Any]) -> Callable[..., Any]:
    """Detach only A5/C2F; subsequent RGB trainers remain mutually exclusive."""
    previous = graph_queue._detach_factory(original)

    def launch(command: Sequence[Any], *args: Any, **kwargs: Any) -> Any:
        values = [str(v) for v in command]
        if "-m" not in values or values[values.index("-m") + 1] != PRODUCER_MODULE:
            return previous(command, *args, **kwargs)
        process = original(command, *args, **kwargs)
        if values[values.index("--fit-id") + 1] not in concurrent_queue.FIT_IDS:
            return process

        class Detached:
            pid = process.pid

            def __getattr__(self, name: str) -> Any:
                return getattr(process, name)

            def wait(self, *_args: Any, **_kwargs: Any) -> int:
                raise concurrent_queue._DetachedChildError

        return Detached()

    return launch


@contextmanager
def installed(run: Path) -> Iterator[None]:
    """Keep every original scientific and numerical lineage validator enabled."""
    validate_freeze(run)
    with graph_queue._patched_runtime(run):
        original_resolve = frozen_queue._resolve_command
        original_load = frozen_queue._load_config
        original_freeze = frozen_queue._freeze_inputs
        original_cycle = frozen_queue._one_cycle

        def resolve(command: list[Any], repository: Path) -> list[str]:
            return route(original_resolve(command, repository), run / FREEZE_NAME)

        def load(path: Path) -> dict[str, Any]:
            config = copy.deepcopy(original_load(path))
            for field in ("heavy_command_markers", "project_command_markers"):
                if PRODUCER_MODULE not in config["resources"][field]:
                    config["resources"][field].append(PRODUCER_MODULE)
            for member in [
                FREEZE_NAME,
                "repo-tree:operational/rgb_port_io_recovery",
                "repo:tests/test_rgb_port_io_recovery.py",
                "io_recovery/QA_JUNIT.xml",
                "io_recovery/QA_RECEIPT.json",
            ]:
                if member not in config["package"]["members"]:
                    config["package"]["members"].append(member)
            for fit_id in FIT_IDS:
                member = f"optional:fits/{fit_id}/IO_RECOVERY_RUNTIME.json"
                if member not in config["package"]["members"]:
                    config["package"]["members"].append(member)
            return config

        def freeze(*args: Any, **kwargs: Any) -> Any:
            validate_freeze(run)
            return original_freeze(*args, **kwargs)

        def cycle(*args: Any, **kwargs: Any) -> bool:
            # An explicit resume of dead children should not insert another
            # 15-minute wait after native reconciliation. Repeated resource
            # pauses retain the original bounded exponential backoff.
            state = args[3]
            dead = {
                name
                for name, task in state["tasks"].items()
                if task["status"] == "RUNNING"
                and task.get("child")
                and not identity_is_live(task["child"])
            }
            progressed = bool(original_cycle(*args, **kwargs))
            expedited = expedite_reconciled_orphans(state, dead)
            if expedited:
                atomic_write_json(run / "RGB_PORT_STATE.json", state)
            return progressed or expedited

        with ExitStack() as stack:
            stack.enter_context(patch.object(frozen_queue, "_resolve_command", resolve))
            stack.enter_context(patch.object(frozen_queue, "_load_config", load))
            stack.enter_context(patch.object(frozen_queue, "_freeze_inputs", freeze))
            stack.enter_context(patch.object(frozen_queue, "_one_cycle", cycle))
            stack.enter_context(patch.object(concurrent_queue, "_detach_factory", detach_factory))
            yield


def expedite_reconciled_orphans(state: dict[str, Any], dead: set[str]) -> bool:
    """Clear only the delay of children already reconciled by the native ledger."""
    changed = False
    for name in dead:
        record = state["tasks"][name]
        if (
            record.get("status") == "PAUSED_RESOURCE"
            and record.get("reason") == "orphaned child reconciled for durable resume"
        ):
            record["retry_after_unix"] = 0
            changed = True
    return changed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["resume"])
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run.resolve(strict=True)
    validate_freeze(run)
    lease = OwnerLease(run / "IO_RECOVERY_OWNER.json")
    lease.acquire()
    try:
        while True:
            owner_path = run / "OWNER.json"
            if owner_path.exists() and identity_is_live(read_json_shared(owner_path)):
                raise RuntimeError("Existing live supervisor: refusing duplicate")
            try:
                with installed(run):
                    return frozen_queue.main(["resume", "--run", str(run)])
            except OSError as error:
                if not retryable_read_error(error):
                    raise
                atomic_write_json(
                    run / "io_recovery/SUPERVISOR_DISK_PAUSE.json",
                    {
                        "status": "WAITING_DISK_IO",
                        "errno": error.errno,
                        "message": str(error),
                        "checked_utc": datetime.now(UTC).isoformat(),
                        "retry_seconds": 30,
                        "scientific_changes": False,
                    },
                )
                time.sleep(30)
    finally:
        lease.release()


if __name__ == "__main__":
    raise SystemExit(main())
