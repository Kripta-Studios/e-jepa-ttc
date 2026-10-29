"""Hand off one durably paused fit to the already authorized quota wrapper."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import runpod_resource_resume_watchdog_v1 as watchdog


def handoff(config: Path, task_id: str, expected_stop: str) -> dict:
    controller = watchdog.Watchdog(config)
    with watchdog.Lock(Path(controller.config["lock_path"])):
        task = next(t for t in controller.config["tasks"] if t["id"] == task_id)
        if task["fit"] not in watchdog.PRODUCERS:
            raise ValueError("Operator quota handoff is limited to producers")
        old = dict(controller.state["tasks"][task_id])
        process = watchdog.proc_identity(old["pid"])
        if process is not None and process["state"] != "Z":
            raise ValueError("Producer has not stopped durably")
        run = Path(task["run"])
        directory = run / f"rotations/{task['rotation']}/fits/{task['fit']}"
        stop = run / "STOP_REQUEST"
        if not expected_stop.startswith("OPERATIONS_QUOTA_POLICY_") or (
            stop.read_text().strip() != expected_stop
        ):
            raise ValueError("Refusing to remove another user or operator stop")
        clean, reason = watchdog.clean_checkpoint(directory, old["started_epoch"], old["pid"])
        receipt = watchdog.read(directory / "CHECKPOINT_RECEIPT.json")
        pause = watchdog.read(directory / "RESOURCE_PAUSE.json")
        if not (
            clean
            and receipt["status"] == "PAUSED_RESOURCE"
            and pause.get("manual") is True
            and pause["completed_updates"] == receipt["completed_updates"]
            and receipt["checkpoint_sha256"] == watchdog.digest(directory / "checkpoint_last.pt")
        ):
            raise ValueError(f"Requested quota handoff is not a durable manual boundary: {reason}")
        if controller.now() >= watchdog.parse_deadline(task["deadline_utc"]).timestamp():
            raise ValueError("Authorized deadline has expired")
        if Path(task["argv"][1]).name != "runpod_quota_race_overlay_v1.py":
            raise ValueError("Future entrypoint is not the admitted quota wrapper")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        archive = stop.with_name(f"STOP_REQUEST_APPLIED_QUOTA_{stamp}")
        stop.rename(archive)
        current = {**old, "next_retry_epoch": 0}
        controller._restart(task, current)
        if current.get("state") != "RESTARTED":
            raise RuntimeError(f"Quota wrapper could not start: {current}")
        current.update(
            reason="AUTHORIZED_OPERATOR_QUOTA_POLICY_RESTART",
            retries=old.get("retries", 0),
            operator_policy_restarts=old.get("operator_policy_restarts", 0) + 1,
        )
        controller.state["tasks"][task_id] = current
        controller._publish()
        value = {
            "schema": "ejepa_operator_quota_handoff_v1",
            "status": "STARTED_FROM_DURABLE_BOUNDARY",
            "task": task_id,
            "completed_updates": receipt["completed_updates"],
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "old_pid": old["pid"],
            "new_pid": current["pid"],
            "archived_stop": str(archive),
            "archived_stop_sha256": watchdog.digest(archive),
            "operator_stop_token": expected_stop,
            "checkpoint_bytes_rewritten": False,
            "config_sha256": controller.config["config_sha256"],
            "source_sha256": watchdog.digest(Path(__file__)),
        }
        watchdog.atomic_json(
            Path(controller.config["status_path"]).parent / f"HANDOFF_{task_id}_{stamp}.json",
            value,
        )
        return value


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--expected-stop", required=True)
    args = parser.parse_args()
    print(json.dumps(handoff(args.config, args.task, args.expected_stop), indent=2))


if __name__ == "__main__":
    main()
