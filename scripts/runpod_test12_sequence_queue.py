"""Autonomous, resumable two-worker queue for sealed Test12 inference."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil

OPS = Path("/workspace/ejepa_final_operations")
RUN = Path("/workspace/ejepa_final_runs")
EVENT = Path("/workspace/ejepa_test12_events")
RGB = Path("/workspace/ejepa_test12_rgb")
STAGED_CURRENT = OPS / "runpod_test12_staged.py"
STAGED_PREVIOUS = OPS / "runpod_test12_staged_59037295.py"
STAGED = OPS / "runpod_test12_staged_64a020ff.py"
INVENTORY = Path("/workspace/ejepa_test12_probe/TEST12_EXACT_INVENTORY.json")
PYTHON = "/opt/ejepa_env/bin/python"
DEADLINE = 1_793_145_600.0
MAX_WORKERS = 2
RAM_RESERVE = 2 * 1024**3
STAGED_SHA256 = "64a020ff65ad9528056dda0fc3ac2e5b14b195e2cc9d398ee71b8a3921ed3eb4"
INVENTORY_SHA256 = "ab749a052855ea6e68e6eb4e6e1a19a27b6e0524d93c8ccce0952a90f77f4a95"
PACKAGE = "__PACKAGE__"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def frozen_sequences() -> list[str]:
    if digest(INVENTORY) != INVENTORY_SHA256:
        raise ValueError("Frozen Test12 inventory bytes differ")
    inventory = read(INVENTORY)
    sequences: set[str] = set()
    events: list[str] = []
    for item in inventory.get("files", []):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("Malformed frozen Test12 inventory")
        parts = Path(item["path"]).parts
        if len(parts) >= 4 and parts[:2] == ("data", "test"):
            sequences.add(parts[2])
            if len(parts) == 4 and parts[3] == "events.h5":
                events.append(parts[2])
    if len(sequences) != 12 or len(events) != 12 or set(events) != sequences:
        raise ValueError("Frozen Test12 inventory must contain exactly 12 sequences")
    return sorted(sequences)


def expected_command(name: str) -> list[str]:
    suffix = ["package"] if name == PACKAGE else ["predict", "--sequence", name]
    return [PYTHON, "-u", str(STAGED), *suffix]


def command_name(command: list[str], sequences: set[str]) -> str | None:
    if command == expected_command(PACKAGE):
        return PACKAGE
    prefix = [PYTHON, "-u", str(STAGED), "predict", "--sequence"]
    if len(command) == len(prefix) + 1 and command[:-1] == prefix and command[-1] in sequences:
        return command[-1]
    current_prefix = [PYTHON, "-u", str(STAGED_CURRENT), "predict", "--sequence"]
    previous_prefix = [PYTHON, "-u", str(STAGED_PREVIOUS), "predict", "--sequence"]
    for admitted in (current_prefix, previous_prefix):
        if (
            len(command) == len(admitted) + 1
            and command[:-1] == admitted
            and command[-1] in sequences
        ):
            return command[-1]
    if any(str(path) in command for path in (STAGED, STAGED_PREVIOUS, STAGED_CURRENT)):
        raise ValueError(f"Unrecognized concurrent staged Test12 command: {command}")
    return None


def process_record(process: psutil.Process, sequences: set[str]) -> dict[str, Any] | None:
    command = process.cmdline()
    name = command_name(command, sequences)
    if name is None or process.status() == psutil.STATUS_ZOMBIE:
        return None
    return {
        "name": name,
        "pid": process.pid,
        "created": process.create_time(),
        "cmdline": command,
    }


def wait_registration(
    child: subprocess.Popen[bytes],
    name: str,
    sequences: set[str],
    *,
    timeout: float = 2.0,
) -> dict[str, Any] | None:
    """Wait across fork/exec without admitting a partial or different command."""
    deadline = time.monotonic() + timeout
    last_error: ValueError | None = None
    while time.monotonic() < deadline:
        if child.poll() is not None:
            return None
        try:
            record = process_record(psutil.Process(child.pid), sequences)
        except psutil.NoSuchProcess:
            record = None
        except ValueError as error:
            record, last_error = None, error
        if record is not None:
            if record["name"] != name:
                raise ValueError("Launched Test12 worker identity differs")
            return record
        time.sleep(0.02)
    if last_error is not None:
        raise last_error
    return None


def sequence_complete(sequence: str) -> bool:
    path = RUN / "final/test12/staged_sequences" / f"{sequence}.complete.json"
    if not path.is_file():
        return False
    value = read(path)
    admission = path.with_name(f"{sequence}.admission.json")
    feature = path.with_name(f"{sequence}.features.npz")
    consumed_path = RGB / "consumed" / f"{sequence}.json"
    tokens = value.get("canonical_tokens")
    if not (
        value.get("schema") == "test12_staged_sequence_complete_v1"
        and value.get("status") == "COMPLETE"
        and value.get("sequence_id") == sequence
        and value.get("private_targets_read") is False
        and value.get("external_submission") is False
        and isinstance(tokens, list)
        and tokens
        and len(tokens) == len(set(tokens))
        and admission.is_file()
        and value.get("admission_sha256") == digest(admission)
        and feature.is_file()
        and value.get("feature_cache_sha256") == digest(feature)
    ):
        raise ValueError(f"Malformed Test12 completion: {sequence}")
    if not consumed_path.is_file():
        return False
    consumed = read(consumed_path)
    outputs = consumed.get("outputs")
    completion_output = (
        [item for item in outputs if item.get("path") == str(path)]
        if isinstance(outputs, list) and all(isinstance(item, dict) for item in outputs)
        else []
    )
    if not (
        consumed.get("schema") == "test12_staged_rgb_consumed_v1"
        and consumed.get("status") == "PASS"
        and consumed.get("sequence_id") == sequence
        and consumed.get("private_targets_read") is False
        and len(completion_output) == 1
        and completion_output[0].get("sha256") == digest(path)
        and completion_output[0].get("bytes") == path.stat().st_size
    ):
        raise ValueError(f"Malformed Test12 consumed receipt: {sequence}")
    return True


def source_ready(sequence: str) -> bool:
    event_path, rgb_path = (
        EVENT / "sequences" / f"{sequence}.json",
        RGB / "sequences" / f"{sequence}.json",
    )
    if not event_path.is_file() or not rgb_path.is_file():
        return False
    event, rgb = read(event_path), read(rgb_path)
    if event.get("status") != "COMPLETE" or rgb.get("status") != "READY":
        raise ValueError(f"Malformed staged input readiness: {sequence}")
    for value in (event, rgb):
        declared = value.get("sequence_id", value.get("sequence"))
        if declared is not None and declared != sequence:
            raise ValueError(f"Readiness marker belongs to another sequence: {sequence}")
    return True


def package_complete(sequences: list[str]) -> bool:
    receipt_path = RUN / "final/test12/PREDICTION_RECEIPT.json"
    package_path = RUN / "final/test12/PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF.json"
    zip_path = package_path.with_suffix(".zip")
    if not package_path.is_file():
        return False
    if not receipt_path.is_file():
        raise ValueError("Partial Test12 package evidence")
    receipt, package = read(receipt_path), read(package_path)
    completion = receipt.get("staged_sequence_completion_sha256")
    if not (
        receipt.get("status") == "COMPLETE"
        and receipt.get("private_targets_read") is False
        and receipt.get("external_submission") is False
        and isinstance(completion, dict)
        and set(completion) == set(sequences)
        and package.get("schema") == "scientific_oof_test12_package_v1"
        and package.get("status") == "PACKAGED_NOT_SUBMITTED"
        and package.get("prediction_receipt_sha256") == digest(receipt_path)
        and package.get("private_targets_read") is False
        and package.get("external_submission") is False
        and zip_path.is_file()
        and package.get("zip_sha256") == digest(zip_path)
        and package.get("receipt_sha256")
        == canonical({key: value for key, value in package.items() if key != "receipt_sha256"})
    ):
        raise ValueError("Malformed Test12 package evidence")
    for sequence in sequences:
        path = RUN / "final/test12/staged_sequences" / f"{sequence}.complete.json"
        if completion[sequence] != digest(path):
            raise ValueError("Packaged sequence completion identity differs")
    return True


class Adopted:
    def __init__(self, record: dict[str, Any], sequences: set[str]) -> None:
        self.record, self.sequences = record, sequences

    def poll(self) -> int | None:
        try:
            process = psutil.Process(int(self.record["pid"]))
            if (
                process.create_time() == self.record["created"]
                and process.status() != psutil.STATUS_ZOMBIE
            ):
                if process.cmdline() != self.record["cmdline"]:
                    raise ValueError("Adopted Test12 worker changed command")
                return None
        except psutil.NoSuchProcess:
            pass
        name = str(self.record["name"])
        done = (
            package_complete(sorted(self.sequences)) if name == PACKAGE else sequence_complete(name)
        )
        return 0 if done else 1


def discover(saved: dict[str, Any], sequences: set[str]) -> dict[str, Adopted]:
    workers: dict[str, Adopted] = {}
    for process in psutil.process_iter():
        try:
            record = process_record(process, sequences)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            raise RuntimeError("Cannot establish exclusive Test12 worker inventory") from None
        if record is None:
            continue
        name = str(record["name"])
        if name in workers:
            raise RuntimeError(f"Duplicate live Test12 worker: {name}")
        previous = saved.get(name)
        if (
            previous
            and previous.get("pid") == record["pid"]
            and previous.get("created") == record["created"]
            and previous.get("cmdline", record["cmdline"]) != record["cmdline"]
        ):
            raise ValueError(f"Persisted Test12 worker command differs: {name}")
        workers[name] = Adopted(record, sequences)
    return workers


def resource_available() -> bool:
    headrooms: list[int] = []
    for current_path, limit_path in (
        (Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory.max")),
        (
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        ),
    ):
        if not current_path.is_file() or not limit_path.is_file():
            continue
        current = int(current_path.read_text().strip())
        raw_limit = limit_path.read_text().strip()
        if raw_limit == "max":
            continue
        limit = int(raw_limit)
        if 0 < limit < 2**62 and 0 <= current <= limit:
            headrooms.append(limit - current)
    if not headrooms:
        return False
    return min(min(headrooms), int(psutil.virtual_memory().available)) >= RAM_RESERVE


def resource_pause(log: Path) -> bool:
    if not log.is_file():
        return False
    tail = log.read_bytes()[-32768:].decode("utf-8", errors="replace")
    latest = tail.rsplit("Traceback", 1)[-1]
    return "ResourcePause" in latest or "resource reserve unavailable" in latest


def main() -> int:
    import fcntl

    if digest(STAGED) != STAGED_SHA256:
        raise ValueError("Staged Test12 adapter bytes differ")
    sequences = frozen_sequences()
    sequence_set = set(sequences)
    OPS.mkdir(parents=True, exist_ok=True)
    lock_path = OPS / "TEST12_SEQUENCE_QUEUE.lock"
    registry_path = OPS / "TEST12_SEQUENCE_QUEUE_WORKERS.json"
    stop_paths = (RUN / "STOP_REQUEST", OPS / "TEST12_QUEUE_STOP_REQUEST")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = read(registry_path) if registry_path.is_file() else {}
        children = discover(dict(prior.get("workers", {})), sequence_set)
        failures = dict(prior.get("failures", {}))

        def persist(status: str) -> None:
            completed = [sequence for sequence in sequences if sequence_complete(sequence)]
            atomic(
                registry_path,
                {
                    "schema": "ejepa_test12_sequence_queue_v1",
                    "status": status,
                    "timestamp": time.time(),
                    "deadline": DEADLINE,
                    "max_workers": MAX_WORKERS,
                    "workers": {name: child.record for name, child in children.items()},
                    "completed_sequences": completed,
                    "failures": failures,
                    "optimizer_updates": 0,
                },
            )

        def start(name: str) -> bool:
            if (
                name in children
                or time.time() >= DEADLINE
                or any(path.exists() for path in stop_paths)
            ):
                return False
            inventory = discover(
                {key: child.record for key, child in children.items()}, sequence_set
            )
            for key, child in inventory.items():
                children.setdefault(key, child)
            if name in children:
                persist("RUNNING")
                return True
            log = OPS / ("test12_package.log" if name == PACKAGE else f"test12_{name}.log")
            with log.open("ab", buffering=0) as output:
                child = subprocess.Popen(
                    expected_command(name),
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=dict(
                        os.environ,
                        OMP_NUM_THREADS="2",
                        MKL_NUM_THREADS="2",
                        OPENBLAS_NUM_THREADS="2",
                    ),
                )
            record = wait_registration(child, name, sequence_set)
            if record is None:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                if name != PACKAGE and sequence_complete(name):
                    failures.pop(name, None)
                elif name == PACKAGE and package_complete(sequences):
                    failures.pop(name, None)
                elif resource_pause(log):
                    failures[name] = {
                        "classification": "RETRYABLE_RESOURCE_PAUSE",
                        "next_retry_epoch": time.time() + 60,
                    }
                else:
                    failures[name] = {
                        "classification": "FATAL_SCIENTIFIC_OR_RUNTIME_FAILURE",
                        "reason": "WORKER_EXITED_OR_TIMED_OUT_BEFORE_EXACT_REGISTRATION",
                        "pid": child.pid,
                        "log": str(log),
                    }
                persist("RUNNING_OR_WAITING_INPUT_RESOURCE")
                return False
            children[name] = Adopted(record, sequence_set)
            failures.pop(name, None)
            persist("RUNNING")
            return True

        persist("RUNNING")
        if any(
            failure.get("classification") == "FATAL_SCIENTIFIC_OR_RUNTIME_FAILURE"
            for failure in failures.values()
        ):
            persist("FAILED_REQUIRES_REVIEW")
            return 5
        while time.time() < DEADLINE and not any(path.exists() for path in stop_paths):
            for name, child in list(children.items()):
                result = child.poll()
                if result is None:
                    continue
                del children[name]
                if result == 0:
                    failures.pop(name, None)
                    continue
                log = OPS / ("test12_package.log" if name == PACKAGE else f"test12_{name}.log")
                if resource_pause(log):
                    failures[name] = {
                        "classification": "RETRYABLE_RESOURCE_PAUSE",
                        "next_retry_epoch": time.time() + 60,
                    }
                else:
                    failures[name] = {
                        "classification": "FATAL_SCIENTIFIC_OR_RUNTIME_FAILURE",
                        "log": str(log),
                    }
                    persist("FAILED_REQUIRES_REVIEW")
                    return 5
            completed = [sequence for sequence in sequences if sequence_complete(sequence)]
            if len(completed) == len(sequences):
                prediction_workers = [name for name in children if name != PACKAGE]
                if prediction_workers:
                    raise RuntimeError("Package cannot start while prediction workers remain")
                if PACKAGE in children:
                    pass
                elif package_complete(sequences):
                    admission = {
                        "schema": "ejepa_test12_package_admission_v1",
                        "status": "COMPLETE",
                        "prediction_receipt_sha256": digest(
                            RUN / "final/test12/PREDICTION_RECEIPT.json"
                        ),
                        "package_receipt_sha256": digest(
                            RUN / "final/test12/PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF.json"
                        ),
                        "sequence_completion_sha256": {
                            sequence: digest(
                                RUN / "final/test12/staged_sequences" / f"{sequence}.complete.json"
                            )
                            for sequence in sequences
                        },
                        "optimizer_updates": 0,
                    }
                    atomic(OPS / "TEST12_PACKAGE_ADMISSION.json", admission)
                    persist("COMPLETE")
                    return 0
                elif (
                    PACKAGE not in children
                    and resource_available()
                    and time.time() >= float(failures.get(PACKAGE, {}).get("next_retry_epoch", 0))
                ):
                    start(PACKAGE)
            elif PACKAGE in children:
                raise RuntimeError("Package worker exists before all sequence completions")
            else:
                active = sum(name != PACKAGE for name in children)
                if active < MAX_WORKERS and resource_available():
                    for sequence in sequences:
                        if active >= MAX_WORKERS:
                            break
                        if (
                            sequence not in children
                            and not sequence_complete(sequence)
                            and source_ready(sequence)
                            and time.time()
                            >= float(failures.get(sequence, {}).get("next_retry_epoch", 0))
                        ):
                            if start(sequence):
                                active += 1
            persist("RUNNING_OR_WAITING_INPUT_RESOURCE")
            time.sleep(5)
        persist("STOPPED_DEADLINE_OR_OPERATOR_REQUEST")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
