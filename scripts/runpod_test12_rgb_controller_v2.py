"""Prefetch at most 8GB RGB and evict staged public TARs only after exact predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
from runpod_test12_rgb_worker_v1 import QUOTA, ROOT, SHARED, atomic, digest, quota_available

RESIDENT_LIMIT = 8_000_000_000
FINAL = Path("/workspace/ejepa_final_runs/final")


def worker_alive(
    pid: int, worker: Path, sequence: str, children: dict[int, subprocess.Popen]
) -> bool:
    """Reap owned children first and tolerate exit between process inspection calls."""
    child = children.get(pid)
    if child is not None and child.poll() is not None:
        children.pop(pid, None)
        return False
    try:
        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            if child is not None:
                child.poll()
            return False
        command = process.cmdline()
        return str(worker) in command and sequence in command
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False


def recoverable_stop(state: dict) -> bool:
    """Only resource pauses and unsuccessful download transport may resume automatically."""
    if state.get("status") != "STOPPED_REQUIRES_REVIEW":
        return True
    message = str(state.get("error", ""))
    return message.startswith(
        (
            "RuntimeError: Quota reserve unavailable:",
            "RuntimeError: RAM reserve below 2 GiB",
            "RuntimeError: hf download failed:",
        )
    )


def canonical(value: object) -> str:
    """Use the frozen final recipe JSON representation."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def verify_consumed(
    sequence: str, catalog: dict, ready: dict, receipt: dict, seal_sha: str
) -> dict:
    """Verify each canonical token and all durable output bytes before raw eviction."""
    expected_binding = canonical(
        {"model_seal_sha256": seal_sha, "inputs_sha256": catalog["inputs_sha256"]}
    )
    if (
        receipt.get("schema") != "test12_staged_rgb_consumed_v1"
        or receipt.get("status") != "PASS"
        or receipt.get("sequence_id") != sequence
        or receipt.get("private_targets_read") is not False
        or receipt.get("source_receipts") != ready["source_receipts"]
        or receipt.get("run_binding_sha256") != expected_binding
    ):
        raise ValueError("Consumed RGB receipt is not bound to exact inputs and model")
    records = receipt["outputs"]
    paths = [Path(x["path"]) for x in records]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicated durable output")
    outputs = {}
    for item, path in zip(records, paths, strict=True):
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(FINAL):
            raise ValueError("Output escaped final inference run")
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError("Consumed output byte identity differs")
        outputs[str(path)] = json.loads(path.read_text()) if path.suffix == ".json" else None
    complete_path = FINAL / "test12/staged_sequences" / f"{sequence}.complete.json"
    admission_path = complete_path.with_name(f"{sequence}.admission.json")
    complete = outputs[str(complete_path)]
    if (
        complete.get("schema") != "test12_staged_sequence_complete_v1"
        or complete.get("status") != "COMPLETE"
        or complete.get("sequence_id") != sequence
        or complete.get("run_binding_sha256") != expected_binding
        or complete.get("private_targets_read") is not False
        or complete.get("external_submission") is not False
        or complete.get("admission_sha256") != digest(admission_path)
        or complete.get("canonical_tokens") != catalog["sequences"][sequence]["canonical_tokens"]
    ):
        raise ValueError("Sequence completion does not cover frozen canonical tokens")
    predictions = {}
    for token in complete["canonical_tokens"]:
        path = FINAL / "test12/fragments" / (hashlib.sha256(token.encode()).hexdigest() + ".json")
        value = outputs[str(path)]
        if (
            value.get("schema") != "final_refit_test12_prediction_fragment_v1"
            or value.get("sample_token") != token
            or value.get("binding_sha256") != expected_binding
            or value.get("private_targets_read") is not False
            or not math.isfinite(float(value["prediction"]))
        ):
            raise ValueError("Canonical prediction fragment differs")
        predictions[token] = float(value["prediction"])
    if complete["predictions_sha256"] != canonical(predictions):
        raise ValueError("Sequence prediction mapping hash differs")
    return {
        "binding_sha256": expected_binding,
        "predictions": len(predictions),
        "outputs": records,
        "all_tokens": len(catalog["sequences"][sequence]["tokens"]),
    }


def main() -> int:
    """Adopt live downloads and advance sequences without touching other workloads."""
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--worker-sha256", required=True)
    args = parser.parse_args()
    worker = ROOT / "runpod_test12_rgb_worker_v1.py"
    for path, sha in [
        (args.plan, args.plan_sha256),
        (args.catalog, args.catalog_sha256),
        (worker, args.worker_sha256),
    ]:
        if digest(path) != sha:
            raise ValueError("Controller source/config SHA differs")
    plan, catalog = json.loads(args.plan.read_text()), json.loads(args.catalog.read_text())
    if catalog["inputs_sha256"] != plan["inputs_sha256"] or set(catalog["sequences"]) != set(
        plan["sequences"]
    ):
        raise ValueError("Sequence catalog differs from frozen plan")
    with (ROOT / "CONTROLLER.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempts = {}
        launched = {}
        children: dict[int, subprocess.Popen] = {}
        sys.path.insert(0, str(SHARED))
        from resource_reader import memory_snapshot

        while time.time() < plan["deadline_epoch"] and not (ROOT / "STOP_REQUEST").exists():
            for child_pid, child in list(children.items()):
                if child.poll() is not None:
                    children.pop(child_pid)
            resident = sum(
                (ROOT / rec["path"]).stat().st_size
                for records in plan["sequences"].values()
                for rec in records
                if (ROOT / rec["path"]).is_file()
            )
            running, waiting, completed = [], [], []
            for sequence, records in plan["sequences"].items():
                evicted = ROOT / "evicted" / f"{sequence}.json"
                consumed = ROOT / "consumed" / f"{sequence}.json"
                ready_path = ROOT / "sequences" / f"{sequence}.json"
                if evicted.exists():
                    previous = json.loads(evicted.read_text())
                    if (
                        previous.get("status") != "EVICTED_PUBLIC_RAW_ONLY"
                        or previous.get("sequence") != sequence
                        or previous.get("consumed_sha256") != digest(consumed)
                    ):
                        raise ValueError("Historical raw eviction receipt differs")
                    completed.append(sequence)
                    continue
                if consumed.exists() and ready_path.exists():
                    receipt = json.loads(consumed.read_text())
                    ready = json.loads(ready_path.read_text())
                    verified = verify_consumed(
                        sequence, catalog, ready, receipt, digest(FINAL / "MODEL_SEAL.json")
                    )
                    intent = {
                        "schema": "ejepa_test12_rgb_eviction_v1",
                        "status": "VERIFIED_INTENT",
                        "sequence": sequence,
                        "consumed_sha256": digest(consumed),
                        "verification": verified,
                        "paths": [str(ROOT / x["path"]) for x in records],
                    }
                    atomic(ROOT / "eviction_intents" / f"{sequence}.json", intent)
                    for record in records:
                        path = ROOT / record["path"]
                        if path.exists():
                            if path.is_symlink() or not path.resolve().is_relative_to(
                                ROOT / "data/test" / sequence
                            ):
                                raise ValueError("Staged raw path escaped sequence root")
                            if (
                                path.stat().st_size != record["size_bytes"]
                                or digest(path) != record["sha256"]
                            ):
                                raise ValueError("Eviction source differs from public frozen TAR")
                            resident -= path.stat().st_size
                            path.unlink()
                    atomic(
                        evicted,
                        {**intent, "status": "EVICTED_PUBLIC_RAW_ONLY", "timestamp": time.time()},
                    )
                    completed.append(sequence)
                    continue
                if ready_path.exists():
                    waiting.append(sequence)
                    continue
                state_path = ROOT / "status" / f"{sequence}.json"
                state = json.loads(state_path.read_text()) if state_path.exists() else {}
                pid = launched.get(sequence, state.get("pid"))
                if pid and worker_alive(pid, worker, sequence, children):
                    running.append(sequence)
                    continue
                if not recoverable_stop(state):
                    raise RuntimeError(
                        f"Worker integrity/manual failure requires review: {sequence}: {state}"
                    )
                if running:
                    continue
                size = sum(x["size_bytes"] for x in records if not (ROOT / x["path"]).is_file())
                if resident + size > RESIDENT_LIMIT:
                    continue
                ledger_path = SHARED / "DOWNLOAD_RESERVATIONS.json"
                ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
                reserved = sum(int(v["bytes"]) for v in ledger.values())
                if (
                    quota_available(
                        json.loads(QUOTA.read_text()), time.time(), reserved, plan["reserve_bytes"]
                    )
                    < size
                ):
                    continue
                available, _ = memory_snapshot(
                    Path("/sys/fs/cgroup"), Path("/proc/self/cgroup").read_text(), 2**63 - 1
                )
                if min(available, psutil.virtual_memory().available) < plan["ram_reserve_bytes"]:
                    continue
                attempts[sequence] = attempts.get(sequence, 0) + 1
                if attempts[sequence] > 3:
                    raise RuntimeError(f"Staging failed repeatedly: {sequence}")
                argv = [
                    "/opt/ejepa_env/bin/python",
                    "-u",
                    str(worker),
                    "--plan",
                    str(args.plan),
                    "--plan-sha256",
                    args.plan_sha256,
                    "--sequence",
                    sequence,
                ]
                with (ROOT / f"{sequence}.log").open("a") as log:
                    process = subprocess.Popen(
                        argv,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                    )
                launched[sequence] = process.pid
                children[process.pid] = process
                running.append(sequence)
            stage = (
                "COMPLETE"
                if len(completed) == len(plan["sequences"])
                else "RUNNING_OR_WAITING_CONSUMER"
            )
            atomic(
                ROOT / "CONTROLLER_STATUS.json",
                {
                    "schema": "ejepa_test12_rgb_controller_v2",
                    "status": stage,
                    "pid": os.getpid(),
                    "timestamp": time.time(),
                    "resident_bytes": resident,
                    "limit_bytes": RESIDENT_LIMIT,
                    "running": running,
                    "waiting_consumer": waiting,
                    "completed": completed,
                    "optimizer_updates": 0,
                },
            )
            if stage == "COMPLETE":
                return 0
            time.sleep(10)
    return 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        atomic(
            ROOT / "CONTROLLER_STATUS.json",
            {
                "schema": "ejepa_test12_rgb_controller_v2",
                "status": "STOPPED_REQUIRES_REVIEW",
                "timestamp": time.time(),
                "pid": os.getpid(),
                "optimizer_updates": 0,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
