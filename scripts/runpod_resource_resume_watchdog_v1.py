"""Linux-only fail-closed watchdog for admitted Runpod resource pauses."""

# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - imported on Windows only for unit tests
    fcntl = None  # type: ignore[assignment]

SCHEMA = "ejepa_resource_resume_watchdog_config_v1"
ADMISSION_SCHEMA = "ejepa_resource_resume_watchdog_admission_v1"
BUDGET_SCHEMA = "ejepa_cloud_budget_authorization_v1"
DEADLINE = "2026-10-27T12:02:19.738064+00:00"
PRODUCERS = ("R_A5", "R_C2F")
HEAD = "RGB_DOWNSTREAM"
REQUIRED_ENVIRONMENT = {
    "PYTHONPATH": (
        "/workspace/ejepa_bundle/workspace/e-jepa-ttc-v18-joint-measurement/src:"
        "/workspace/ejepa_bundle/workspace/e-jepa-ttc-v18-joint-measurement"
    ),
    "PYTHONDONTWRITEBYTECODE": "1",
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_deadline(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Deadline must be an ISO-8601 string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Deadline must be timezone-aware")
    return parsed


def option(argv: list[str], name: str) -> str:
    if argv.count(name) != 1:
        raise ValueError(f"Command requires exactly one {name}")
    index = argv.index(name)
    if index + 1 >= len(argv):
        raise ValueError(f"Command option has no value: {name}")
    return argv[index + 1]


def bound_entrypoint(argv: list[str], allowed_sha256: list[str]) -> Path | None:
    """Locate a hash-allowed script after optional interpreter flags such as ``-u``."""
    matches: list[Path] = []
    for value in argv[1:]:
        path = Path(value)
        if path.is_absolute() and path.is_file() and digest(path) in allowed_sha256:
            matches.append(path.resolve(strict=True))
    return matches[0] if len(matches) == 1 else None


def safe_linux_path(value: object) -> Path:
    if not isinstance(value, str) or not re.fullmatch(r"/[A-Za-z0-9_./-]+", value):
        raise ValueError(f"Unsafe Linux path: {value!r}")
    pure = PurePosixPath(value)
    if ".." in pure.parts:
        raise ValueError(f"Linux path traversal is forbidden: {value}")
    return Path(value)


def expected_run(rotation: int, fit: str) -> str:
    suffix = "rgb_v3" if fit == "R_A5" else "c2f_v3"
    if fit == HEAD:
        suffix = "rgb_downstream_v1"
    return f"/workspace/ejepa_technical/r{rotation}_{suffix}"


def validate_config(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    value = read(path)
    if value.get("schema") != SCHEMA:
        raise ValueError("Unknown watchdog config schema")
    authority = value.get("authority")
    tasks = value.get("tasks")
    if not isinstance(authority, dict) or not isinstance(tasks, list):
        raise ValueError("Watchdog config requires authority and tasks")
    if value.get("backoff_seconds") != 60:
        raise ValueError("Resource restart backoff must be exactly 60 seconds")
    poll = value.get("poll_seconds")
    retries = value.get("max_retries")
    if not isinstance(poll, int) or not 1 <= poll <= 60:
        raise ValueError("Watchdog poll_seconds must be within 1..60")
    if not isinstance(retries, int) or not 1 <= retries <= 20:
        raise ValueError("Watchdog max_retries must be within 1..20")
    if value.get("required_environment") != REQUIRED_ENVIRONMENT:
        raise ValueError("Watchdog inherited environment contract is invalid")
    global_deadline = authority.get("global_deadline_utc")
    if global_deadline != DEADLINE:
        raise ValueError("Watchdog cannot extend or replace the authorized global deadline")
    parse_deadline(global_deadline)
    budget_path = safe_linux_path(authority.get("budget_path"))
    admission_path = safe_linux_path(authority.get("admission_path"))
    if digest(budget_path) != authority.get("budget_sha256"):
        raise ValueError("Global budget bytes differ from watchdog config")
    if digest(admission_path) != authority.get("admission_sha256"):
        raise ValueError("Parent watchdog admission bytes differ")
    budget, admission = read(budget_path), read(admission_path)
    if (
        budget.get("schema") != BUDGET_SCHEMA
        or budget.get("maximum_total_usd") != 24
        or budget.get("execution_deadline_utc") != global_deadline
        or admission.get("schema") != ADMISSION_SCHEMA
        or admission.get("status") != "PASS"
        or admission.get("budget_sha256") != authority.get("budget_sha256")
        or admission.get("global_deadline_utc") != global_deadline
    ):
        raise ValueError("Watchdog admission/global budget binding is invalid")
    bundle = safe_linux_path(value.get("bundle_root"))
    expected_tasks = {
        f"R{rotation}_{fit}"
        for rotation in (2, 3, 4)
        for fit in (*PRODUCERS, HEAD)
    }
    if len(tasks) != 9:
        raise ValueError("Watchdog requires exactly six producers and three RGB heads")
    normalized: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Watchdog task must be an object")
        identifier, rotation, fit = task.get("id"), task.get("rotation"), task.get("fit")
        if (
            not isinstance(identifier, str)
            or rotation not in {2, 3, 4}
            or fit not in {*PRODUCERS, HEAD}
            or identifier != f"R{rotation}_{fit}"
        ):
            raise ValueError("Watchdog task identity is invalid")
        identifiers.add(identifier)
        run = safe_linux_path(task.get("run"))
        if str(run) != expected_run(rotation, fit):
            raise ValueError(f"Task run root differs from fixed assignment: {identifier}")
        argv = task.get("argv")
        if (
            not isinstance(argv, list)
            or len(argv) < 4
            or any(not isinstance(item, str) or not item or "\x00" in item for item in argv)
        ):
            raise ValueError(f"Task command is not an explicit argv list: {identifier}")
        if not Path(argv[0]).is_absolute():
            raise ValueError("Python path must be absolute")
        future_sha = task.get("entrypoint_sha256")
        adopted_hashes = task.get("adopt_entrypoint_sha256", [task.get("entrypoint_sha256")])
        if (
            not isinstance(adopted_hashes, list)
            or not adopted_hashes
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(item)) for item in adopted_hashes)
            or future_sha not in adopted_hashes
        ):
            raise ValueError(f"Adopted entrypoint allowlist is invalid: {identifier}")
        entrypoint = bound_entrypoint(argv, [str(future_sha)])
        if entrypoint is None:
            raise ValueError(f"Future entrypoint changed or is ambiguous: {identifier}")
        if option(argv, "--run") != str(run) or option(argv, "--rotation") != str(rotation):
            raise ValueError(f"Task command run/rotation differs: {identifier}")
        if option(argv, "--bundle") != str(bundle):
            raise ValueError(f"Task bundle differs: {identifier}")
        if fit in PRODUCERS and option(argv, "--fit") != fit:
            raise ValueError(f"Producer fit differs: {identifier}")
        if fit == HEAD and "--fit" in argv:
            raise ValueError(f"RGB downstream command must not inject a fit: {identifier}")
        deadline = task.get("deadline_utc")
        if parse_deadline(deadline) > parse_deadline(global_deadline):
            raise ValueError(f"Task extends the global deadline: {identifier}")
        pid = task.get("pid")
        if not isinstance(pid, int) or pid <= 1:
            raise ValueError(f"Task PID is invalid: {identifier}")
        normalized.append(
            {
                **task,
                "run": str(run),
                "argv": argv,
                "log": str(safe_linux_path(task.get("log"))),
                "entrypoint": str(entrypoint),
                "adopt_entrypoint_sha256": adopted_hashes,
            }
        )
    if identifiers != expected_tasks or set(admission.get("task_ids", [])) != expected_tasks:
        raise ValueError("Watchdog admission task closure differs")
    for field in ("status_path", "lock_path"):
        safe_linux_path(value.get(field))
    return {**value, "tasks": normalized, "config_sha256": digest(path)}


def proc_identity(
    pid: int, proc_root: Path = Path("/proc"), clock_ticks: int | None = None
) -> dict[str, Any] | None:
    root = proc_root / str(pid)
    try:
        raw_stat = (root / "stat").read_text(encoding="utf-8")
        tail = raw_stat[raw_stat.rindex(")") + 2 :].split()
        state = tail[0]
        start_ticks = int(tail[19])
        argv = [
            item.decode("utf-8")
            for item in (root / "cmdline").read_bytes().split(b"\0")
            if item
        ]
        btime_line = next(
            line
            for line in (proc_root / "stat").read_text().splitlines()
            if line.startswith("btime ")
        )
        boot_epoch = int(btime_line.split()[1])
        ticks = clock_ticks if clock_ticks is not None else int(os.sysconf("SC_CLK_TCK"))
    except (FileNotFoundError, PermissionError, ProcessLookupError, StopIteration, ValueError):
        return None
    return {
        "pid": pid,
        "state": state,
        "start_ticks": start_ticks,
        "started_epoch": boot_epoch + start_ticks / ticks,
        "argv": argv,
        "cmdline_sha256": hashlib.sha256(b"\0".join(x.encode() for x in argv)).hexdigest(),
    }


def command_matches(task: dict[str, Any], process: dict[str, Any]) -> bool:
    argv = process.get("argv")
    if not isinstance(argv, list) or process.get("state") == "Z":
        return False
    if len(argv) < 2 or bound_entrypoint(argv, task["adopt_entrypoint_sha256"]) is None:
        return False
    try:
        return (
            option(argv, "--run") == task["run"]
            and option(argv, "--rotation") == str(task["rotation"])
            and option(argv, "--bundle") == task["argv"][task["argv"].index("--bundle") + 1]
            and (task["fit"] == HEAD or option(argv, "--fit") == task["fit"])
        )
    except ValueError:
        return False


def fresh(path: Path, started_epoch: float) -> bool:
    return path.is_file() and path.stat().st_mtime >= started_epoch


def clean_checkpoint(directory: Path, started_epoch: float, pid: int) -> tuple[bool, str]:
    receipt_path = directory / "CHECKPOINT_RECEIPT.json"
    journal_path = directory / "UPDATE_JOURNAL.json"
    if not fresh(receipt_path, started_epoch) or not fresh(journal_path, started_epoch):
        return False, "STALE_OR_MISSING_CHECKPOINT_TRANSACTION"
    receipt, journal = read(receipt_path), read(journal_path)
    completed = receipt.get("completed_updates")
    if not (
        isinstance(completed, int)
        and completed == journal.get("completed_updates") == journal.get("durable_updates")
        and journal.get("pending_update_upper") == 0
        and receipt.get("complete_state") is True
    ):
        return False, "CHECKPOINT_TRANSACTION_NOT_DURABLE"
    progress_path = directory / "PROGRESS.json"
    if progress_path.is_file():
        if not fresh(progress_path, started_epoch) or read(progress_path).get("pid") != pid:
            return False, "PROGRESS_PID_OR_FRESHNESS_MISMATCH"
    return True, "DURABLE"


def immutable_complete_head_endpoint(
    directory: Path, rotation: int, fit: str
) -> tuple[bool, str]:
    """Verify an older sealed head endpoint without requiring current-process mtimes."""
    receipt_path = directory / "CHECKPOINT_RECEIPT.json"
    journal_path = directory / "UPDATE_JOURNAL.json"
    admission_path = directory / "ENDPOINT_ADMISSION.json"
    if any(not path.is_file() for path in (receipt_path, journal_path, admission_path)):
        return False, "SEALED_ENDPOINT_EVIDENCE_MISSING"
    receipt, journal, admission = read(receipt_path), read(journal_path), read(admission_path)
    completed = receipt.get("completed_updates")
    checkpoint_raw = receipt.get("checkpoint_path")
    if not isinstance(checkpoint_raw, str):
        return False, "SEALED_ENDPOINT_CHECKPOINT_PATH_INVALID"
    try:
        checkpoint = Path(checkpoint_raw).resolve(strict=True)
        contained = checkpoint.is_relative_to(directory.resolve(strict=True))
    except (FileNotFoundError, OSError):
        return False, "SEALED_ENDPOINT_CHECKPOINT_MISSING"
    checkpoint_sha = digest(checkpoint) if contained and checkpoint.is_file() else None
    if not (
        isinstance(completed, int)
        and receipt.get("status") == "COMPLETE"
        and receipt.get("scientific_endpoint") is True
        and receipt.get("complete_state") is True
        and completed == journal.get("completed_updates") == journal.get("durable_updates")
        and journal.get("pending_update_upper") == 0
        and checkpoint_sha == receipt.get("checkpoint_sha256")
        and admission.get("schema") == "final_refit_fold_endpoint_admission_v1"
        and admission.get("status") == "PASS"
        and admission.get("scientific_endpoint") is True
        and admission.get("rotation") == rotation
        and admission.get("fit_id") == fit
        and admission.get("updates") == completed
        and admission.get("checkpoint_path") == checkpoint_raw
        and admission.get("checkpoint_sha256") == checkpoint_sha
        and admission.get("checkpoint_receipt_path") == str(receipt_path.resolve())
        and admission.get("checkpoint_receipt_sha256") == digest(receipt_path)
    ):
        return False, "SEALED_ENDPOINT_IDENTITY_OR_DURABILITY_MISMATCH"
    return True, "SEALED_COMPLETE_ENDPOINT"


def stop_paths(task: dict[str, Any]) -> list[Path]:
    run = Path(task["run"])
    if task["fit"] in PRODUCERS:
        fit = run / f"rotations/{task['rotation']}/fits/{task['fit']}"
        return [run / "STOP_REQUEST", fit / "STOP_REQUEST"]
    root = run / f"rotations/{task['rotation']}"
    return [
        run / "STOP_REQUEST",
        root / "STOP_REQUEST",
        root / "fits/PAIR_R/STOP_REQUEST",
        root / "fits/R_CTX/STOP_REQUEST",
    ]


def producer_terminal(task: dict[str, Any], managed: dict[str, Any]) -> tuple[str, str]:
    run = Path(task["run"])
    directory = run / f"rotations/{task['rotation']}/fits/{task['fit']}"
    pid, started = int(managed["pid"]), float(managed["started_epoch"])
    clean, reason = clean_checkpoint(directory, started, pid)
    if not clean:
        return "BLOCKED", reason
    receipt = read(directory / "CHECKPOINT_RECEIPT.json")
    if receipt.get("status") == "COMPLETE" and receipt.get("scientific_endpoint") is True:
        return "COMPLETE", "SEALED_ENDPOINT"
    pause_path = directory / "RESOURCE_PAUSE.json"
    if not fresh(pause_path, started):
        return "BLOCKED", "RESOURCE_PAUSE_NOT_FRESH"
    pause = read(pause_path)
    if receipt.get("status") != "PAUSED_RESOURCE" or pause.get("manual") is not False:
        return "BLOCKED", "NOT_AN_AUTOMATIC_RESOURCE_PAUSE"
    if pause.get("completed_updates") != receipt.get("completed_updates"):
        return "BLOCKED", "PAUSE_CHECKPOINT_UPDATE_MISMATCH"
    return "RESTARTABLE", "FRESH_DURABLE_RESOURCE_PAUSE"


def head_terminal(task: dict[str, Any], managed: dict[str, Any]) -> tuple[str, str]:
    run = Path(task["run"])
    started, pid = float(managed["started_epoch"]), int(managed["pid"])
    result_path = run / "RGB_DOWNSTREAM_RESULT.json"
    if fresh(result_path, started):
        result = read(result_path)
        if (
            result.get("schema") == "ejepa_cloud_rgb_downstream_queue_v1"
            and result.get("stage") == "RGB_OOF"
            and result.get("status") == "COMPLETE"
            and result.get("rotation") == task["rotation"]
        ):
            return "COMPLETE", "RGB_OOF_COMPLETE"
        return "BLOCKED", "MALFORMED_FRESH_RGB_RESULT"
    pause_path = run / "RGB_DOWNSTREAM_PAUSE.json"
    if not fresh(pause_path, started):
        return "BLOCKED", "RGB_PAUSE_NOT_FRESH"
    pause = read(pause_path)
    if (
        pause.get("status") != "PAUSED_RECOVERABLE"
        or pause.get("manual") is not False
        or pause.get("rotation") != task["rotation"]
        or pause.get("optimizer_state_durable_at_last_scientific_checkpoint") is not True
    ):
        return "BLOCKED", "RGB_PAUSE_NOT_AUTOMATIC_OR_DURABLE"
    pipeline = run / f"rotations/{task['rotation']}/RGB_PIPELINE_RECEIPT.json"
    if pipeline.is_file() and (not fresh(pipeline, started) or read(pipeline).get("pid") != pid):
        return "BLOCKED", "RGB_PIPELINE_PID_OR_FRESHNESS_MISMATCH"
    observed = 0
    for fit in ("PAIR_R", "R_CTX"):
        directory = run / f"rotations/{task['rotation']}/fits/{fit}"
        if (directory / "CHECKPOINT_RECEIPT.json").is_file():
            observed += 1
            receipt = read(directory / "CHECKPOINT_RECEIPT.json")
            clean, reason = (
                immutable_complete_head_endpoint(directory, task["rotation"], fit)
                if receipt.get("status") == "COMPLETE"
                else clean_checkpoint(directory, started, pid)
            )
            if not clean:
                return "BLOCKED", f"{fit}_{reason}"
    if (
        observed == 0
        and pause.get("optimizer_state_durable_at_last_scientific_checkpoint") is not True
    ):
        return "BLOCKED", "NO_DURABLE_RGB_HEAD_BOUNDARY"
    return "RESTARTABLE", "FRESH_DURABLE_RGB_RESOURCE_PAUSE"


def campaign_terminal_code(states: list[str]) -> int | None:
    """Keep supervising active tasks even when a sibling is permanently blocked."""
    terminal = {"COMPLETE", "DEADLINE_EXPIRED", "BLOCKED"}
    if not states or not all(state in terminal for state in states):
        return None
    return 5 if "BLOCKED" in states else 0


class Lock:
    def __init__(self, path: Path) -> None:
        self.path, self.stream = path, None

    def __enter__(self) -> Lock:
        if fcntl is None:
            raise RuntimeError("Resource resume watchdog lock requires Linux fcntl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise RuntimeError("Another resource resume watchdog owns the lock") from error
        return self

    def __exit__(self, *_args: object) -> None:
        assert fcntl is not None
        assert self.stream is not None
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


class Watchdog:
    def __init__(
        self,
        config_path: Path,
        *,
        proc_root: Path = Path("/proc"),
        now: Callable[[], float] = time.time,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        if os.name != "posix":
            raise RuntimeError("Resource resume watchdog requires Linux")
        self.config = validate_config(config_path)
        if any(os.environ.get(key) != expected for key, expected in REQUIRED_ENVIRONMENT.items()):
            raise ValueError("Watchdog process did not inherit the frozen runtime environment")
        self.proc_root, self.now, self.popen = proc_root, now, popen
        self.source_sha256 = digest(Path(__file__).resolve())
        self.status_path = Path(self.config["status_path"])
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if not self.status_path.is_file():
            return {"tasks": {}}
        value = read(self.status_path)
        if (
            value.get("schema") != "ejepa_resource_resume_watchdog_status_v1"
            or value.get("config_sha256") != self.config["config_sha256"]
            or value.get("source_sha256") != self.source_sha256
        ):
            raise ValueError("Existing watchdog status belongs to another config/source")
        return value

    def _publish(self) -> dict[str, Any]:
        states = self.state.setdefault("tasks", {})
        summary = {
            name: sum(item.get("state") == name for item in states.values())
            for name in ("RUNNING", "RESTARTED", "COMPLETE", "BLOCKED", "DEADLINE_EXPIRED")
        }
        value = {
            "schema": "ejepa_resource_resume_watchdog_status_v1",
            "updated_utc": datetime.fromtimestamp(self.now(), UTC).isoformat(),
            "config_sha256": self.config["config_sha256"],
            "source_sha256": self.source_sha256,
            "global_deadline_utc": self.config["authority"]["global_deadline_utc"],
            "budget_sha256": self.config["authority"]["budget_sha256"],
            "tasks": states,
            "summary": summary,
            "optimizer_updates_performed_by_watchdog": 0,
            "processes_killed": 0,
            "stop_files_removed": 0,
        }
        atomic_json(self.status_path, value)
        self.state = value
        return value

    def _adopt(self, task: dict[str, Any], current: dict[str, Any]) -> dict[str, Any] | None:
        pid = int(current.get("pid", task["pid"]))
        process = proc_identity(pid, self.proc_root)
        if process is None:
            return None
        if "start_ticks" in current and process["start_ticks"] != current["start_ticks"]:
            current.update(state="BLOCKED", reason="PID_REUSED")
            return process
        if process.get("state") == "Z":
            return None
        if not command_matches(task, process):
            current.update(state="BLOCKED", reason="PROCESS_IDENTITY_MISMATCH")
            return process
        current.update(
            pid=pid,
            start_ticks=process["start_ticks"],
            started_epoch=process["started_epoch"],
            cmdline_sha256=process["cmdline_sha256"],
            state="RUNNING",
            reason="ADOPTED_OR_STILL_RUNNING",
        )
        return process

    def _restart(self, task: dict[str, Any], current: dict[str, Any]) -> None:
        retries = int(current.get("retries", 0))
        if retries >= self.config["max_retries"]:
            current.update(state="BLOCKED", reason="MAX_RETRIES_EXHAUSTED")
            return
        if self.now() < float(current.get("next_retry_epoch", 0)):
            current.update(state="RESTART_BACKOFF", reason="WAITING_60_SECOND_BACKOFF")
            return
        log = Path(task["log"])
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab", buffering=0) as output:
            process = self.popen(
                task["argv"],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=True,
            )
        identity = proc_identity(process.pid, self.proc_root)
        if identity is None or not command_matches(task, identity):
            current.update(state="BLOCKED", reason="RESTARTED_PROCESS_IDENTITY_UNAVAILABLE")
            return
        current.update(
            pid=process.pid,
            start_ticks=identity["start_ticks"],
            started_epoch=identity["started_epoch"],
            cmdline_sha256=identity["cmdline_sha256"],
            retries=retries + 1,
            next_retry_epoch=self.now() + 60,
            state="RESTARTED",
            reason="AUTOMATIC_RESOURCE_PAUSE_RESUME",
        )

    def step(self) -> dict[str, Any]:
        states = self.state.setdefault("tasks", {})
        global_deadline = parse_deadline(
            self.config["authority"]["global_deadline_utc"]
        ).timestamp()
        for task in self.config["tasks"]:
            current = states.setdefault(task["id"], {"pid": task["pid"], "retries": 0})
            if current.get("state") in {"COMPLETE", "BLOCKED", "DEADLINE_EXPIRED"}:
                continue
            if any(path.exists() for path in stop_paths(task)):
                current.update(state="BLOCKED", reason="STOP_REQUEST_PRESENT")
                continue
            process = self._adopt(task, current)
            if current.get("state") == "BLOCKED":
                continue
            if process is not None:
                continue
            if "started_epoch" not in current:
                current.update(state="BLOCKED", reason="INITIAL_PID_NOT_ADOPTABLE")
                continue
            deadline = min(global_deadline, parse_deadline(task["deadline_utc"]).timestamp())
            terminal, reason = (
                producer_terminal(task, current)
                if task["fit"] in PRODUCERS
                else head_terminal(task, current)
            )
            if terminal == "COMPLETE":
                current.update(state="COMPLETE", reason=reason)
            elif terminal != "RESTARTABLE":
                current.update(state="BLOCKED", reason=reason)
            elif self.now() >= deadline:
                current.update(state="DEADLINE_EXPIRED", reason="AUTHORIZED_DEADLINE_REACHED")
            else:
                self._restart(task, current)
        return self._publish()

    def run(self, *, once: bool = False) -> int:
        with Lock(Path(self.config["lock_path"])):
            while True:
                value = self.step()
                states = [item["state"] for item in value["tasks"].values()]
                terminal_code = campaign_terminal_code(states)
                if terminal_code is not None:
                    return terminal_code
                if once:
                    return 3
                time.sleep(self.config["poll_seconds"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--validate-config", action="store_true")
    args = parser.parse_args()
    if args.validate_config:
        config = validate_config(args.config)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "config_sha256": config["config_sha256"],
                    "tasks": len(config["tasks"]),
                    "global_deadline_utc": config["authority"]["global_deadline_utc"],
                    "optimizer_updates": 0,
                },
                indent=2,
            )
        )
        return 0
    return Watchdog(args.config).run(once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
