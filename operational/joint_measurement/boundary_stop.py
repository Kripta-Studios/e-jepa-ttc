"""Stop only this worktree's primary runner after a settled, fully durable scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/joint_measurement_20261012"


def read(path: Path) -> dict:
    """Read an atomic campaign receipt without importing Torch or opening CUDA."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value: dict) -> None:
    """Persist the controller's own evidence atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha(path: Path) -> str:
    """Hash only the small durable checkpoint at this transition boundary."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_owner(process: psutil.Process, created: float) -> None:
    """Never terminate a recycled PID, another worktree or another command."""
    command = process.cmdline()
    if (
        process.create_time() != created
        or Path(process.cwd()).resolve() != ROOT
        or "operational.joint_measurement.run" not in command
        or "-m" not in command
        or "python" not in process.name().lower()
    ):
        raise ValueError("Refuse to stop a process outside this exact primary runner")


def durable_boundary(
    journal: dict, receipt: dict, state: dict, minimum: int, scope_end: str
) -> bool:
    """Require post-scope evidence, never a stale state with the same update count."""
    count = journal["completed_updates"]
    return (
        count >= minimum
        and journal["durable_updates"] == count
        and journal["pending_update_upper"] == 0
        and receipt["completed_updates"] == count
        and receipt["complete_state"]
        and state.get("fit_id") == "NATIVE-P2-FT-S7"
        and state.get("updates") == count
        and "utc" in state
        and datetime.fromisoformat(state["utc"]) >= datetime.fromisoformat(scope_end)
    )


def stop(pid: int, minimum_updates: int) -> dict:
    """Wait for exclusive ownership and complete state; never interrupt an optimizer step."""
    import msvcrt

    directory = RUN / "scheduling/PAIRS_FIRST"
    if (directory / "STOP_RESULT.json").exists():
        raise ValueError("Transition already has a result; inspect it instead of stopping again")
    order_path = RUN / "PRIMARY_EXECUTION_ORDER.json"
    if not order_path.is_file():
        raise ValueError("Register the complete unchanged matrix order before transition")
    process = psutil.Process(pid)
    created = process.create_time()
    check_owner(process, created)
    write(
        directory / "STOP_REQUEST.json",
        {
            "pid": pid,
            "create_time": created,
            "minimum_durable_updates": minimum_updates,
            "utc": datetime.now(UTC).isoformat(),
            "order_sha256": sha(order_path),
            "reason": "Resource-aware ordering only; resume all four unchanged 12000-update fits",
        },
    )
    expires = time.monotonic() + 1200
    fit = RUN / "fits/NATIVE-P2-FT-S7"
    with (RUN / "V18_GPU.lock").open("r+b") as stream:
        while time.monotonic() < expires:
            check_owner(process, created)
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                time.sleep(0.2)
                continue
            try:
                book = read(RUN / "GPU_BUDGET.json")
                if any(r["pid"] == pid and r["status"] == "RESERVED" for r in book["entries"]):
                    raise ValueError("Owner still has an unsettled GPU reservation")
                scope_end = max(r["end_utc"] for r in book["entries"] if r["pid"] == pid)
                # fit() writes its curve and runner state after releasing the GPU lock.
                # Wait for that post-scope evidence while preventing another GPU scope.
                for _ in range(200):
                    journal = read(fit / "UPDATE_JOURNAL.json")
                    receipt = read(fit / "CHECKPOINT_RECEIPT.json")
                    state = read(RUN / "RUNNER_STATE.json")
                    count = journal["completed_updates"]
                    if durable_boundary(journal, receipt, state, minimum_updates, scope_end):
                        break
                    time.sleep(0.1)
                else:
                    raise ValueError("Primary scope did not produce complete durable evidence")
                checkpoint_sha = sha(fit / "checkpoint_last.pt")
                if checkpoint_sha != receipt["checkpoint_sha256"]:
                    raise ValueError("Durable checkpoint bytes differ from receipt")
                try:
                    check_owner(process, created)
                    process.terminate()
                    process.wait(timeout=15)
                    status = "STOPPED_AT_DURABLE_BOUNDARY"
                except psutil.NoSuchProcess:
                    status = "EXITED_AT_DURABLE_BOUNDARY"
                result = {
                    "status": status,
                    "pid": pid,
                    "create_time": created,
                    "completed_updates": count,
                    "durable_updates": count,
                    "pending_update_upper": 0,
                    "recovery_upper": journal["recovery_upper"],
                    "checkpoint_sha256": checkpoint_sha,
                    "identity_sha256": journal["identity_sha256"],
                    "order_sha256": sha(order_path),
                    "utc": datetime.now(UTC).isoformat(),
                    "scope": "Execution-order transition; native scientific endpoint still 12000",
                }
                write(directory / "STOP_RESULT.json", result)
                return result
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    raise TimeoutError("No safe primary boundary observed; no process was terminated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--minimum-updates", type=int, required=True)
    args = parser.parse_args()
    print(stop(args.pid, args.minimum_updates), flush=True)
