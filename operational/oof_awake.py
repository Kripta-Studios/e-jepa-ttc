"""Scoped Windows idle-sleep protection for an active, AC-powered OOF queue.

This sidecar never changes power plans, scientific state, or manual sleep policy.
The Windows request belongs to its thread and is released on exit or loss of work.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import psutil

CONTINUOUS = 0x80000000
SYSTEM_REQUIRED = 0x00000001
DISPLAY_REQUIRED = 0x00000002


class AwakeRequest:
    """Own one thread's execution-state request with explicit cleanup."""

    def __init__(self, setter: Callable[[int], int], *, display: bool = False) -> None:
        self.setter = setter
        self.display = display
        self.active = False

    def update(self, active: bool) -> None:
        """Acquire or release the request; fail explicitly if Windows refuses."""
        if active == self.active:
            return
        flags = CONTINUOUS
        if active:
            flags |= SYSTEM_REQUIRED
            if self.display:
                flags |= DISPLAY_REQUIRED
        if not self.setter(flags):
            raise OSError("SetThreadExecutionState failed")
        self.active = active


def should_hold(*, queue_status: str, on_ac: bool, stopped: bool, alive: bool) -> bool:
    """Only active work on external power can inhibit automatic idle sleep."""
    return queue_status == "RUNNING" and on_ac and not stopped and alive


def main() -> None:
    """Watch one queue process, releasing protection on pause, battery, or exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--queue-pid", type=int, required=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--seconds", type=float, default=0, help="Optional bounded diagnostic")
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("This sidecar requires Windows")
    from operational.final_refit.continuation_queue import queue_lock

    run = args.run.resolve()
    queue = psutil.Process(args.queue_pid)
    queue_birth = queue.create_time()
    command = queue.cmdline()
    if "operational.final_refit.scientific_oof_queue" not in command:
        raise ValueError("PID does not identify the scientific queue")
    plan = json.loads((run / "SCIENTIFIC_OOF_QUEUE_PLAN.json").read_text(encoding="utf-8"))
    if (
        str(Path(command[command.index("--plan") + 1]).resolve())
        != str(run / "SCIENTIFIC_OOF_QUEUE_PLAN.json")
        or Path(plan["run"]).resolve() != run
    ):
        raise ValueError("Queue PID belongs to a different run")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    setter = kernel.SetThreadExecutionState
    setter.argtypes = [ctypes.c_uint32]
    setter.restype = ctypes.c_uint32
    request = AwakeRequest(setter, display=args.display)
    started = time.monotonic()
    status = run / "AWAKE_STATUS.json"

    def write_status(reason: str) -> None:
        value = dict(
            pid=os.getpid(),
            queue_pid=queue.pid,
            queue_birth=queue_birth,
            updated_unix=time.time(),
            active=request.active,
            display=args.display,
            reason=reason,
            scientific_state_changed=False,
        )
        temporary = status.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(status)

    with queue_lock(run / "AWAKE.lock"):
        try:
            while queue.is_running():
                stopped = (run / "STOP_REQUEST").exists() or (run / "AWAKE_STOP").exists()
                if stopped or (args.seconds and time.monotonic() - started >= args.seconds):
                    break
                try:
                    state = json.loads((run / "SCIENTIFIC_OOF_QUEUE_STATE.json").read_text())
                    battery = psutil.sensors_battery()
                    hold = should_hold(
                        queue_status=state["status"],
                        on_ac=battery is None or battery.power_plugged,
                        stopped=stopped,
                        alive=queue.is_running(),
                    )
                except (OSError, ValueError, KeyError):
                    hold = False
                request.update(hold)
                write_status("active_work" if hold else "released_no_active_ac_work")
                time.sleep(5)
        finally:
            request.update(False)
            write_status("released_on_exit")


if __name__ == "__main__":
    main()
