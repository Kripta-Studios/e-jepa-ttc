"""Recover frozen Test12 event tensors one public sequence at a time."""

from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing
import os
import platform
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PureWindowsPath

import psutil

PROBE = Path("/workspace/ejepa_test12_probe")
ROOT = Path("/workspace/ejepa_test12_events")
DEADLINE = 1793145600.0
QUOTA = 600_000_000_000
RESERVE = 10_000_000_000 + 12 * 1024**3
sys.path.insert(0, str(PROBE))
from resource_reader import memory_snapshot  # noqa: E402
from runpod_test12_parity_probe import (  # noqa: E402
    FREEZE_SHA,
    INPUTS_SHA,
    MEDIA_REVISION,
    digest,
    normalize_npz_platform,
    read_bound,
    validate_sources,
)

_pool = None


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(f".{os.getpid()}.pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(pending, path)


def status(**values: object) -> None:
    values.update(timestamp=time.time(), pid=os.getpid(), optimizer_updates=0)
    atomic(ROOT / "STATUS.json", values)
    print(json.dumps(values), flush=True)


def resource_guard() -> dict:
    if time.time() >= DEADLINE or (ROOT / "STOP_REQUEST").exists():
        raise RuntimeError("Execution deadline or manual stop")
    headroom, _ = memory_snapshot(
        Path("/sys/fs/cgroup"), Path("/proc/self/cgroup").read_text(), 2**63 - 1
    )
    available = min(headroom, psutil.virtual_memory().available)
    if available < 2 * 1024**3:
        raise RuntimeError("RESOURCE_WAIT: less than 2 GiB RAM")
    snapshot = json.loads(
        Path("/workspace/ejepa_event_runs/EVENT_VOLUME_SNAPSHOT.json").read_text()
    )
    if (
        int(snapshot["quota_bytes"]) != QUOTA
        or not 0 <= time.time() - float(snapshot["observed_timestamp"]) <= 120
    ):
        raise RuntimeError("RESOURCE_WAIT: invalid or stale volume quota snapshot")
    if int(snapshot["used_bytes"]) + RESERVE >= QUOTA:
        raise RuntimeError("RESOURCE_WAIT: volume reserve")
    return snapshot


def wait_resources(sequence: str) -> dict:
    while True:
        try:
            return resource_guard()
        except RuntimeError as error:
            if not str(error).startswith("RESOURCE_WAIT:"):
                raise
            status(status="RESOURCE_WAIT", sequence=sequence, reason=str(error))
            time.sleep(10)


def reserve_download(key: str, required: int) -> None:
    with (PROBE / "DOWNLOAD_RESERVATIONS.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = PROBE / "DOWNLOAD_RESERVATIONS.json"
        ledger = json.loads(path.read_text()) if path.exists() else {}
        if required:
            snapshot = resource_guard()
            others = sum(int(v["bytes"]) for k, v in ledger.items() if k != key)
            if int(snapshot["used_bytes"]) + others + required + RESERVE > QUOTA:
                raise RuntimeError("RESOURCE_WAIT: download reservation exceeds quota")
            ledger[key] = {"bytes": required, "pid": os.getpid(), "updated_timestamp": time.time()}
        else:
            ledger.pop(key, None)
        atomic(path, ledger)


def download(sequence: str, pin: dict) -> Path:
    raw = PROBE / "raw" / pin["path"]
    if raw.exists():
        return raw
    key = "event_h5_" + sequence
    while True:
        wait_resources(sequence)
        try:
            reserve_download(key, int(pin["bytes"]))
            break
        except RuntimeError as error:
            status(status="RESOURCE_WAIT", sequence=sequence, reason=str(error))
            time.sleep(10)
    command = [
        "/opt/ejepa_env/bin/hf",
        "download",
        "NAIL-HNU/eAP-dataset",
        pin["path"],
        "--repo-type",
        "dataset",
        "--revision",
        MEDIA_REVISION,
        "--local-dir",
        str(PROBE / "raw"),
    ]
    env = dict(
        os.environ,
        HF_HUB_DISABLE_XET="1",
        HF_HUB_ENABLE_HF_TRANSFER="0",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    child = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        while child.poll() is None:
            resource_guard()
            status(
                status="DOWNLOADING_ONE_H5",
                sequence=sequence,
                child_pid=child.pid,
                expected_raw_bytes=pin["bytes"],
            )
            time.sleep(10)
        if child.returncode:
            raise RuntimeError(f"HF download exited {child.returncode}")
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        reserve_download(key, 0)
    return raw


def initialize_worker() -> None:
    global _pool
    sys.path[:0] = [str(PROBE / "code"), str(PROBE / "code/src")]
    import torch

    from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    _pool = ReaderPool()


def materialize(task: dict) -> dict:
    import numpy as np
    import torch

    from e_jepa_ttc.efficient_context.mapped_union import encode_union
    from operational.sota_evidence.test_inputs import supported_job

    begin = time.monotonic()
    destination = ROOT / "tensors" / task["name"]
    receipt_path = destination.with_suffix(".json")
    if destination.exists():
        if digest(destination) != task["expected"]:
            raise ValueError("Existing tensor does not match frozen SHA: " + str(destination))
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt["sha256"] != task["expected"]
                or receipt["sample_token"] != task["job"]["sample_token"]
                or receipt["input_freeze_sha256"] != FREEZE_SHA
            ):
                raise ValueError("Tensor provenance mismatch")
            return receipt
    if _pool is None:
        raise RuntimeError("No initialized H5 reader")
    row = dict(task["job"], path=task["raw"])
    reader = _pool.get(row["path"])
    timestamps = reader.datasets["events/t"]
    job = supported_job(row, int(timestamps[0]), int(timestamps[-1]))
    events = encode_union(
        reader,
        job["windows"],
        np.arange(750000, -1, -50000, dtype=np.int64),
        np.concatenate((np.zeros(8, bool), job["valid"])),
        tuple(job["square"]),
        sequence_id=job["sequence"],
        roi_size=128,
        event_pixel_diff=5,
    ).numpy()[-8:]
    if events.shape != (8, 3, 12, 128, 128) or not np.isfinite(events).all():
        raise ValueError("Invalid frozen H8 tensor")
    temporary = destination.with_suffix(f".{os.getpid()}.pending.npz")
    np.savez_compressed(temporary, events=events, valid=job["valid"])
    native = digest(temporary)
    changed = normalize_npz_platform(temporary)
    actual = digest(temporary)
    if actual != task["expected"]:
        atomic(
            ROOT / "mismatches" / f"{task['name']}.json",
            {
                "status": "PARITY_MISMATCH",
                "sample_token": job["sample_token"],
                "expected": task["expected"],
                "actual": actual,
                "path": str(temporary),
            },
        )
        raise ValueError("Exact frozen tensor parity failed: " + task["name"])
    os.replace(temporary, destination)
    receipt = {
        "schema": "test12_exact_event_tensor_v1",
        "status": "EXACT_NPZ_SHA_MATCH",
        "input_freeze_sha256": FREEZE_SHA,
        "inputs_sha256": INPUTS_SHA,
        "sample_token": job["sample_token"],
        "sha256": actual,
        "path": str(destination),
        "raw_sha256": task["raw_sha256"],
        "raw_relative_path": task["raw_relative_path"],
        "native_platform_sha256": native,
        "zip_creator_os_bytes_normalized": changed,
        "source_bindings": task["sources"],
        "seconds": time.monotonic() - begin,
        "bytes": destination.stat().st_size,
        "valid": job["valid"].tolist(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "optimizer_updates": 0,
        "test_labels_read": False,
    }
    atomic(receipt_path, receipt)
    return receipt


def run(args: argparse.Namespace) -> None:
    ROOT.mkdir(exist_ok=True)
    (ROOT / "tensors").mkdir(exist_ok=True)
    argv = json.loads((PROBE / "PROBE_ARGV.json").read_text())
    inputs = read_bound(Path(argv[argv.index("--frozen-inputs") + 1]), INPUTS_SHA)
    freeze = read_bound(Path(argv[argv.index("--event-freeze") + 1]), FREEZE_SHA)
    sources = validate_sources(PROBE / "code", freeze)
    inventory = json.loads((PROBE / "TEST12_EXACT_INVENTORY.json").read_text())
    if inventory["revision"] != MEDIA_REVISION:
        raise ValueError("Media revision changed")
    pins = {x["path"]: x for x in inventory["files"]}
    jobs = {}
    for number, row in enumerate(freeze["rows"]):
        observation = inputs["event_observations"].get(row["sample_token"])
        if observation is None:
            continue
        name = f"query_{number:05d}.npz"
        if PureWindowsPath(observation["current_tensor_path"]).name != name:
            raise ValueError("Frozen tensor row order changed")
        jobs.setdefault(row["sequence"], []).append(
            {
                "job": row,
                "name": name,
                "expected": observation["current_tensor_sha256"],
                "sources": sources,
            }
        )
    if sum(map(len, jobs.values())) != len(inputs["event_observations"]):
        raise ValueError("Physical observation coverage mismatch")
    manifest = {
        "schema": "test12_event_staging_v1",
        "inputs_sha256": INPUTS_SHA,
        "input_freeze_sha256": FREEZE_SHA,
        "revision": MEDIA_REVISION,
        "status": "IN_PROGRESS",
        "raw_sources": {},
        "sequences": {},
        "tensors": {},
    }
    manifest_path = ROOT / "EVENT_INPUT_MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("inputs_sha256") != INPUTS_SHA
            or manifest.get("input_freeze_sha256") != FREEZE_SHA
            or manifest.get("revision") != MEDIA_REVISION
        ):
            raise ValueError("Existing event manifest belongs to a different frozen input")
    for sequence, tasks in sorted(jobs.items()):
        if args.sequence and sequence != args.sequence:
            continue
        begin = time.monotonic()
        wait_resources(sequence)
        if sequence in manifest["sequences"]:
            receipt = json.loads(Path(manifest["sequences"][sequence]).read_text())
            expected = {task["job"]["sample_token"]: task["expected"] for task in tasks}
            if (
                receipt.get("inputs_sha256") != INPUTS_SHA
                or receipt.get("input_freeze_sha256") != FREEZE_SHA
                or set(receipt["tensors"]) != set(expected)
            ):
                raise ValueError("Completed sequence provenance or coverage changed")
            if receipt["status"] == "COMPLETE" and all(
                x["sha256"] == expected[token] and digest(Path(x["path"])) == expected[token]
                for token, x in receipt["tensors"].items()
            ):
                relative = f"data/test/{sequence}/events.h5"
                raw_record_path = Path(receipt["raw_receipt_path"])
                raw_record = json.loads(raw_record_path.read_text())
                if (
                    raw_record["sha256"] != pins[relative]["sha256"]
                    or raw_record["size_bytes"] != pins[relative]["bytes"]
                ):
                    raise ValueError("Completed sequence raw provenance changed")
                if not (PROBE / "raw" / relative).exists():
                    raw_record["storage_state"] = "EVICTED_AFTER_TENSOR_VERIFICATION"
                    atomic(raw_record_path, raw_record)
                    manifest["raw_sources"][relative].update(raw_record)
                    atomic(manifest_path, manifest)
                continue
        relative = f"data/test/{sequence}/events.h5"
        pin = pins[relative]
        raw = download(sequence, pin)
        status(status="VERIFYING_RAW_SHA256", sequence=sequence)
        if raw.stat().st_size != pin["bytes"] or digest(raw) != pin["sha256"]:
            raise ValueError("Public H5 identity mismatch: " + sequence)
        raw_receipt = {
            "schema": "test12_raw_source_verified_v1",
            "relative_path": relative,
            "sha256": pin["sha256"],
            "size_bytes": pin["bytes"],
            "revision": MEDIA_REVISION,
            "status": "SHA256_VERIFIED",
            "storage_state": "PRESENT",
            "inputs_sha256": INPUTS_SHA,
            "input_freeze_sha256": FREEZE_SHA,
            "verified_timestamp": time.time(),
            "path": str(raw),
        }
        raw_receipt_path = ROOT / "raw_receipts" / f"{sequence}.json"
        atomic(raw_receipt_path, raw_receipt)
        records = {}
        sequence_bytes = 0
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker,
        ) as executor:
            for offset in range(0, len(tasks), args.workers):
                wait_resources(sequence)
                batch = [
                    dict(t, raw=str(raw), raw_sha256=pin["sha256"], raw_relative_path=relative)
                    for t in tasks[offset : offset + args.workers]
                ]
                futures = [executor.submit(materialize, t) for t in batch]
                for future in futures:
                    result = future.result()
                    records[result["sample_token"]] = {
                        "path": result["path"],
                        "sha256": result["sha256"],
                        "receipt_path": str(Path(result["path"]).with_suffix(".json")),
                    }
                    sequence_bytes += result["bytes"]
                status(
                    status="MATERIALIZING_EXACT_TENSORS",
                    sequence=sequence,
                    completed=len(records),
                    total=len(tasks),
                    bytes=sequence_bytes,
                    elapsed_seconds=time.monotonic() - begin,
                    workers=args.workers,
                    threads_per_worker=2,
                )
        # Durable proof precedes eviction. Public COMPLETE is published only
        # after raw storage metadata reaches its stable final state, so consumers
        # can bind the raw receipt hash without racing the eviction transaction.
        atomic(
            ROOT / "tensor_checkpoints" / f"{sequence}.json",
            {
                "status": "TENSORS_VERIFIED_PENDING_RAW_FINALIZATION",
                "sequence": sequence,
                "inputs_sha256": INPUTS_SHA,
                "input_freeze_sha256": FREEZE_SHA,
                "raw_receipt_path": str(raw_receipt_path),
                "tensors": records,
            },
        )
        # Only this worker's publicly downloaded H5 is disposable; frozen tensors,
        # all receipts, original datasets, and RGB inputs are never removed.
        if not args.keep_raw:
            if not raw.resolve().is_relative_to((PROBE / "raw/data/test").resolve()):
                raise ValueError("Raw eviction escaped owned Test12 download root")
            raw.unlink()
            raw_receipt["storage_state"] = "EVICTED_AFTER_TENSOR_VERIFICATION"
            atomic(raw_receipt_path, raw_receipt)
        sequence_receipt = ROOT / "sequences" / f"{sequence}.json"
        atomic(
            sequence_receipt,
            {
                "status": "COMPLETE",
                "sequence": sequence,
                "inputs_sha256": INPUTS_SHA,
                "input_freeze_sha256": FREEZE_SHA,
                "raw_receipt_path": str(raw_receipt_path),
                "tensors": records,
                "count": len(records),
                "bytes": sequence_bytes,
                "seconds": time.monotonic() - begin,
            },
        )
        manifest["sequences"][sequence] = str(sequence_receipt)
        manifest["tensors"].update(records)
        manifest["raw_sources"][relative] = {**raw_receipt, "receipt_path": str(raw_receipt_path)}
        atomic(manifest_path, manifest)
    manifest["status"] = (
        "COMPLETE"
        if len(manifest["tensors"]) == len(inputs["event_observations"])
        else "IN_PROGRESS"
    )
    atomic(manifest_path, manifest)
    status(
        status=manifest["status"],
        completed=len(manifest["tensors"]),
        total=len(inputs["event_observations"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--workers", type=int, choices=(2, 4), default=2)
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    with (ROOT / "WORKER.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                try:
                    run(args)
                    break
                except RuntimeError as error:
                    if not str(error).startswith("RESOURCE_WAIT:"):
                        raise
                    # download() already stopped its own child and released its
                    # reservation in finally. Exact cached tensors and HF partials
                    # remain durable; resume revalidates their original bindings.
                    status(status="RETRYING_RESOURCE_WAIT", reason=str(error))
                    wait_resources("resume_after_resource_wait")
        except Exception as error:
            status(status="STOPPED_REQUIRES_REVIEW", error=str(error))
            raise


if __name__ == "__main__":
    main()
