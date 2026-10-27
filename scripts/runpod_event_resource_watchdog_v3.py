"""Admit an exact zero-update EVENT preflight pause with inherited old progress."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

V2_SHA256 = "98d7add8a234a5fe72a0a18d1835580877ee3e18a36b0e0215171ac355a66070"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install_v3() -> Any:
    source = Path(__file__).resolve()
    v2_path = source.with_name("runpod_event_resource_watchdog_v2.py")
    if digest(v2_path) != V2_SHA256:
        raise ValueError("Frozen EVENT watchdog V2 bytes differ")
    spec = importlib.util.spec_from_file_location("ejepa_event_resource_watchdog_v2_v3", v2_path)
    if spec is None or spec.loader is None:
        raise ImportError(str(v2_path))
    wrapper: Any = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)
    module: Any = wrapper.install_v2()
    original_terminal = module.producer_terminal

    def producer_terminal(task: dict[str, Any], managed: dict[str, Any]) -> tuple[str, str]:
        terminal, reason = original_terminal(task, managed)
        if (terminal, reason) != ("BLOCKED", "PROGRESS_PID_OR_FRESHNESS_MISMATCH"):
            return terminal, reason
        directory = Path(task["run"]) / f"rotations/{task['rotation']}/fits/{task['fit']}"
        started, pid = float(managed["started_epoch"]), int(managed["pid"])
        receipt_path = directory / "CHECKPOINT_RECEIPT.json"
        pointer_path = directory / "CHECKPOINT_POINTER.json"
        journal_path = directory / "UPDATE_JOURNAL.json"
        pause_path = directory / "RESOURCE_PAUSE.json"
        progress_path = directory / "PROGRESS.json"
        transaction = (receipt_path, pointer_path, journal_path, pause_path)
        if any(not module.fresh(path, started) for path in transaction):
            return terminal, reason
        if not progress_path.is_file() or progress_path.stat().st_mtime > started:
            return terminal, reason
        durable = wrapper.durable_producer_updates(module, task, None)
        if durable is None:
            return terminal, reason
        receipt = module.read(receipt_path)
        pause = module.read(pause_path)
        progress = module.read(progress_path)
        if not (
            receipt.get("status") == "PAUSED_RESOURCE"
            and pause.get("manual") is False
            and pause.get("completed_updates") == durable
            and progress.get("completed_updates") == durable
            and isinstance(progress.get("pid"), int)
            and progress.get("pid") != pid
        ):
            return terminal, reason
        return "RESTARTABLE", "FRESH_DURABLE_ZERO_UPDATE_PREFLIGHT_PAUSE"

    module.producer_terminal = producer_terminal
    module.__file__ = str(source)
    return module


def main() -> int:
    return int(install_v3().main())


if __name__ == "__main__":
    raise SystemExit(main())
