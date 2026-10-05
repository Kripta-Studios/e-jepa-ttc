"""Resume guards for native outputs and the owned head accounting transaction lock."""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
from pathlib import Path

from .common import Campaign, digest, publish_json, read


def recover_head_transaction(c: Campaign) -> None:
    """Archive a proven dead local transaction owner only under our live writer lease."""
    import psutil

    path = c.out / "garl_heads/PHYSICAL_WORK.lock"
    if not path.exists():
        return
    writer = read(c.out / "WRITER.lock")
    if writer["pid"] != os.getpid() or writer["create_time"] != psutil.Process().create_time():
        raise RuntimeError("head journal recovery requires this process's campaign writer lease")
    owner = read(path)
    if owner.get("host") != socket.gethostname() or type(owner.get("pid")) is not int:
        raise RuntimeError("head journal transaction owner cannot be proven local")
    if psutil.pid_exists(owner["pid"]):
        raise RuntimeError("head journal transaction owner is still live; preserve its lock")
    pin = digest(path)
    ledger = path.with_suffix(".json")
    if ledger.exists():
        value = read(ledger)
        if value.get("schema") != "simplex_t_physical_work_v1":
            raise ValueError("unrecognized head physical-work ledger; preserve its lock")
    archive = c.out / f"garl_heads/recovery/STALE_TRANSACTION_{pin}.json"
    proof = {
        "lock_sha256": pin,
        "owner": owner,
        "ledger_sha256": digest(ledger) if ledger.exists() else None,
        "reason": "proven_dead_local_owner_under_live_campaign_writer_lease",
        "ledger_checkpoint_and_pending_work_changed": False,
    }
    if archive.exists():
        # A crash after publishing this receipt but before unlink must be resumable.
        # Keep the first observer's PID/time; a later legitimate writer has a new identity.
        previous = read(archive)
        if any(previous.get(k) != v for k, v in proof.items()):
            raise ValueError("archived head transaction recovery proof changed")
    else:
        publish_json(archive, {**proof, "writer_owner": writer})
    if digest(path) != pin:
        raise RuntimeError("head journal transaction lock changed during recovery")
    path.unlink()


def verified_fragment(path: Path, expected: dict) -> bool:
    """Reuse a committed payload only when its query/source/checkpoint binding and bytes match."""
    receipt = path.with_suffix(".json")
    if not receipt.exists():
        return False
    value = read(receipt)
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("native fragment binding changed: " + str(receipt))
    if not path.is_file() or digest(path) != value.get("sha256"):
        raise ValueError("native committed fragment bytes missing or changed: " + str(path))
    return True


def completed_source(folder: Path, binding: str, queries: int | None = None) -> dict | None:
    """Reuse an atomic compact source without rewriting bytes already bound to a head fit."""
    path = folder / "COMPLETE.json"
    if not path.exists():
        return None
    value = read(path)
    if value.get("status") != "COMPLETE" or value.get("binding_sha256") != binding:
        raise ValueError("native compact source completion binding changed")
    if queries is not None and value.get("queries") != queries:
        raise ValueError("native compact source population changed")
    for filename, key in (("SOURCE.npz", "source_sha256"), ("METADATA.csv", "metadata_sha256")):
        if not (folder / filename).is_file() or digest(folder / filename) != value.get(key):
            raise ValueError("native compact source bytes missing or changed: " + filename)
    return value


def runtime_measurements(receipt: Path, expected: dict) -> list[dict]:
    """Validate saved timing identity and completeness before aggregating resumed measurements."""
    value = read(receipt)
    if value.get("status") != "PASSED" or any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("native runtime receipt failed or binding changed")
    regimes = ["R0", "R2_PREPARED_PRODUCER_AND_HEAD"]
    if expected["label"] != "GARL_NATIVE":
        regimes.append("R2_HEAD_ONLY_CPU_FP32")
    rows = value.get("measurements", [])
    checksum = hashlib.sha256(
        json.dumps(rows, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if checksum != value.get("measurements_sha256"):
        raise ValueError("native runtime measurement bytes changed")
    if len(rows) != len(regimes) or [r.get("regime") for r in rows] != regimes:
        raise ValueError("native runtime receipt measurements are incomplete")
    for row in rows:
        if any(row.get(k) != expected[k] for k in ("label", "block", "query")):
            raise ValueError("native runtime measurement query identity changed")
        if (
            not isinstance(row.get("milliseconds"), (float, int))
            or not math.isfinite(row["milliseconds"])
            or row["milliseconds"] < 0
        ):
            raise ValueError("native runtime measurement cost is invalid")
    if (
        not 0 <= value.get("feature_max_abs", float("inf")) <= 1e-4
        or not 0 <= value.get("phase_max_abs", float("inf")) <= 1e-5
        or value.get("repeated_output_max_abs") != 0
    ):
        raise ValueError("native runtime saved parity limits failed")
    return rows
