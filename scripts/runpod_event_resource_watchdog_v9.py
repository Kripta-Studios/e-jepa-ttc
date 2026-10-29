"""Defer one durable EVENT producer until admitted capacity becomes available."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

V8_SHA256 = "e252f6956589dfd1f291b4db53979dcd120c8d0cd501ccd8e4db64d0c31845b9"
DEFERRED = "R3_E_A5"
CANDIDATES = ["R4_E_A5", "R2_E_C2F", "R3_E_C2F", "R4_E_C2F"]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install_v9() -> Any:
    source = Path(__file__).resolve()
    v8_path = source.with_name("runpod_event_resource_watchdog_v8.py")
    if digest(v8_path) != V8_SHA256:
        raise ValueError("Frozen EVENT watchdog V8 bytes differ")
    spec = importlib.util.spec_from_file_location("ejepa_event_resource_watchdog_v8_v9", v8_path)
    if spec is None or spec.loader is None:
        raise ImportError(v8_path)
    wrapper: Any = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)
    module: Any = wrapper.install_v8()
    original_validate = module.validate_config
    original_step = module.Watchdog.step

    def validate_config(path: Path) -> dict[str, Any]:
        value = original_validate(path)
        if (
            value.get("watchdog_capacity_overlay_source_sha256") != digest(source)
            or value.get("deferred_task") != DEFERRED
            or value.get("release_candidates") != CANDIDATES
            or value.get("capacity_max_active") != 4
        ):
            raise ValueError("EVENT watchdog V9 capacity policy differs")
        task_ids = {task["id"] for task in value["tasks"]}
        if DEFERRED not in task_ids or not set(CANDIDATES).issubset(task_ids):
            raise ValueError("EVENT watchdog V9 capacity tasks are absent")
        return value

    def step(self: Any) -> dict[str, Any]:
        tasks = self.state.setdefault("tasks", {})
        task_by_id = {task["id"]: task for task in self.config["tasks"]}
        deferred_task = task_by_id[DEFERRED]
        current = tasks.setdefault(DEFERRED, self._initial(deferred_task))
        already_released = bool(current.get("capacity_release_completed"))
        original_tasks = self.config["tasks"]
        if not already_released:
            current.update(state="WAITING_CAPACITY", reason="CAPACITY_WAIT")
            self.config["tasks"] = [task for task in original_tasks if task["id"] != DEFERRED]
        try:
            value = original_step(self)
        finally:
            self.config["tasks"] = original_tasks
        if already_released:
            return value

        deadline = module.parse_deadline(
            self.config["authority"]["global_deadline_utc"]
        ).timestamp()
        if self.now() >= deadline:
            current.update(state="DEADLINE_EXPIRED", reason="AUTHORIZED_DEADLINE_REACHED")
            return self._publish()
        if any(path.exists() for path in module.stop_paths(deferred_task)):
            current.update(state="BLOCKED", reason="STOP_REQUEST_PRESENT")
            return self._publish()

        released_by: str | None = None
        for candidate_id in CANDIDATES:
            candidate = tasks.get(candidate_id, {})
            if candidate.get("state") != "COMPLETE":
                continue
            task = task_by_id[candidate_id]
            terminal, _ = module.producer_terminal(task, candidate)
            if terminal == "COMPLETE":
                released_by = candidate_id
                break
        if released_by is None:
            current.update(state="WAITING_CAPACITY", reason="CAPACITY_WAIT")
            return self._publish()

        terminal, reason = module.producer_terminal(deferred_task, current)
        if terminal != "RESTARTABLE":
            current.update(state="BLOCKED", reason=f"CAPACITY_RELEASE_{reason}")
            return self._publish()
        previous_pid = current.get("pid")
        previous_start_ticks = current.get("start_ticks")
        self._restart(deferred_task, current)
        if current.get("state") != "RESTARTED":
            return self._publish()
        current.update(
            capacity_release_completed=True,
            capacity_released_by=released_by,
            capacity_previous_pid=previous_pid,
            capacity_previous_start_ticks=previous_start_ticks,
        )
        receipt = {
            "schema": "ejepa_event_capacity_release_v1",
            "status": "RESTARTED",
            "deferred_task": DEFERRED,
            "released_by": released_by,
            "capacity_max_active": 4,
            "previous_pid": previous_pid,
            "previous_start_ticks": previous_start_ticks,
            "new_pid": current["pid"],
            "new_start_ticks": current["start_ticks"],
            "checkpoint_terminal_reason": reason,
            "watchdog_source_sha256": digest(source),
            "timestamp_utc": datetime.fromtimestamp(self.now(), UTC).isoformat(),
            "optimizer_updates_performed_by_watchdog": 0,
            "stop_files_removed": 0,
        }
        receipt_path = self.status_path.with_name("EVENT_CAPACITY_RELEASE_RECEIPT.json")
        module.atomic_json(receipt_path, receipt)
        current["capacity_release_receipt_sha256"] = digest(receipt_path)
        return self._publish()

    module.validate_config = validate_config
    module.Watchdog.step = step
    module.__file__ = str(source)
    return module


def main() -> int:
    return int(install_v9().main())


if __name__ == "__main__":
    raise SystemExit(main())
