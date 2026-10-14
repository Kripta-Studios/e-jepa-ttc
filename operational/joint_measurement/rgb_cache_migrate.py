"""One-time durable boundary migration; never terminate a training process."""

from __future__ import annotations

import argparse
import shutil

import psutil

from operational.range_transfer.common import freeze, read, sha, utc, write
from operational.rgb_port.accounting import atomic_write_bytes

from .common import ROOT, RUN

OUT = RUN / "runtime/rgb_resident_v2"
AUTH = RUN / "BUDGET_AUTHORIZATION_20261014.json"
FIT = RUN / "rgb/fits/RGB-PAIR-CORR-FROZEN-S7"


def hold(pid: int) -> dict:
    """Let the old driver finish its current scope and stop at its next admission."""
    if not (OUT / "REGISTRATION.json").is_file():
        raise ValueError("Admit the cache runtime before requesting a boundary")
    if (OUT / "MIGRATION.json").exists():
        raise ValueError("Migration already recorded; inspect existing state instead")
    process = psutil.Process(pid)
    if not any(
        module in process.cmdline()
        for module in (
            "operational.joint_measurement.rgb_run",
            "operational.joint_measurement.rgb_cached_run",
        )
    ) or process.cwd() != str(ROOT):
        raise ValueError("Expected the original live RGB driver")
    original = read(AUTH)
    if original["status"] != "APPLIED":
        raise ValueError("Full program authorization must already be applied")
    backup = OUT / "authorization_original.bytes"
    if backup.exists():
        raise ValueError("Existing authorization backup requires manual reconciliation")
    atomic_write_bytes(backup, AUTH.read_bytes())
    migration = {
        "old_pid": pid,
        "old_created": process.create_time(),
        "utc": utc(),
        "original_authorization_sha256": sha(backup),
        "runtime_registration_sha256": sha(OUT / "REGISTRATION.json"),
        "reason": "User-authorized lossless input cache optimization at a fully durable boundary",
        "user_authorization_unchanged": True,
        "migration_script_sha256": sha(ROOT / "operational/joint_measurement/rgb_cache_migrate.py"),
    }
    freeze(OUT / "MIGRATION.json", migration)
    held = {**original, "status": "RUNTIME_MIGRATION_HOLD"}
    freeze(OUT / "authorization_hold.json", held)
    write(AUTH, held)
    return {"status": "WAITING_FOR_ORIGINAL_DRIVER_SCOPE_END", **migration}


def seal() -> dict:
    """Preserve the stopped writer's complete checkpoint and restore original authority."""
    migration = read(OUT / "MIGRATION.json")
    try:
        process = psutil.Process(migration["old_pid"])
        if process.is_running() and abs(process.create_time() - migration["old_created"]) < 0.001:
            raise ValueError("Original RGB driver remains live; do not seal or restart")
    except psutil.NoSuchProcess:
        pass
    if read(AUTH) != read(OUT / "authorization_hold.json"):
        raise ValueError("Unexpected authorization change during transition")
    journal = read(FIT / "UPDATE_JOURNAL.json")
    pointer = read(FIT / "CHECKPOINT_POINTER.json")
    receipt = read(FIT / "CHECKPOINT_RECEIPT.json")
    if (
        journal["completed_updates"] != journal["durable_updates"]
        or journal["pending_update_upper"] != 0
        or journal["completed_updates"] != pointer["completed_updates"]
        or journal["completed_updates"] != receipt["completed_updates"]
    ):
        raise ValueError("Cannot migrate an undurable trajectory")
    origin = FIT / "checkpoint_versions" / pointer["version"]
    if sha(origin) != pointer["checkpoint_sha256"]:
        raise ValueError("Original checkpoint does not match its durable pointer")
    preserved = OUT / "origin_checkpoint.pt"
    if preserved.exists():
        if sha(preserved) != sha(origin):
            raise ValueError("Preserved origin differs")
    else:
        shutil.copyfile(origin, preserved)
    if sha(preserved) != pointer["checkpoint_sha256"]:
        raise ValueError("Preserved checkpoint copy failed verification")
    backup = OUT / "authorization_original.bytes"
    if sha(backup) != migration["original_authorization_sha256"]:
        raise ValueError("Original authorization backup changed")
    result = {
        "status": "DURABLE_ORIGIN_PRESERVED",
        "utc": utc(),
        "origin_updates": journal["completed_updates"],
        "checkpoint_sha256": pointer["checkpoint_sha256"],
        "recovery_upper": journal["recovery_upper"],
        "pending_updates": 0,
        "lost_updates": 0,
        "migration_sha256": sha(OUT / "MIGRATION.json"),
    }
    durable_record = OUT / "DURABLE_ORIGIN.json"
    if durable_record.exists():
        result["utc"] = read(durable_record)["utc"]
    freeze(durable_record, result)
    atomic_write_bytes(AUTH, backup.read_bytes())
    if sha(AUTH) != migration["original_authorization_sha256"]:
        raise ValueError("Authorization was not restored byte for byte")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["hold", "seal"])
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()
    print(hold(args.pid) if args.action == "hold" else seal())
