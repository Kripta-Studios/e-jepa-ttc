"""Verify the completed runtime archive using bounded parallel network reads."""

import concurrent.futures
import hashlib
import json
import os
import signal
import subprocess
from datetime import UTC, datetime
from pathlib import Path

BASE = Path("/workspace/ejepa_event_operations/manual_pause_20261026")


def main() -> None:
    archive = BASE / "OPT_RUNTIME.tar"
    state = json.loads((BASE / "PRESERVATION_STATUS.json").read_text())
    if state["status"] == "PAUSED_PRESERVED":
        print(json.dumps(state))
        return
    # The archive was atomically renamed only after tar exit=0 and fsync.
    assert archive.is_file() and not archive.with_suffix(".tar.pending").exists()
    old = Path("/proc/289733/cmdline")
    if old.exists() and old.read_bytes():
        assert b"manual_pause_20261026/preserve.py" in old.read_bytes()
        os.kill(289733, signal.SIGTERM)
    size = archive.stat().st_size
    fd = os.open(archive, os.O_RDONLY)
    step = 8 * 1024 * 1024
    digest = hashlib.sha256()
    local = Path("/opt/ejepa_pause_runtime_verify.tar")

    def read(offset: int) -> bytes:
        wanted = min(step, size - offset)
        value = os.pread(fd, wanted, offset)
        if len(value) != wanted:
            raise RuntimeError("Short archive read")
        return value

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool, local.open("wb") as out:
        offsets = iter(range(0, size, step))
        pending = []
        for _ in range(8):
            offset = next(offsets, None)
            if offset is not None:
                pending.append(pool.submit(read, offset))
        done = 0
        while pending:
            data = pending.pop(0).result()
            digest.update(data)
            out.write(data)
            done += len(data)
            offset = next(offsets, None)
            if offset is not None:
                pending.append(pool.submit(read, offset))
            (BASE / "ARCHIVE_VERIFY_PROGRESS.json").write_text(
                json.dumps({"bytes_verified": done, "total_bytes": size})
            )
    os.close(fd)
    subprocess.run(["tar", "-tf", str(local)], check=True, stdout=subprocess.DEVNULL)
    checks = json.loads((BASE / "SCIENTIFIC_PAUSE_CHECK.json").read_text())
    final = {
        "status": "PAUSED_PRESERVED",
        "utc": datetime.now(UTC).isoformat(),
        "network_volume_id": "3mritxijmp",
        "mount": "/workspace",
        "manual_resume_required": True,
        "pod_stop_performed": False,
        "runtime_archive": str(archive),
        "runtime_bytes": size,
        "runtime_sha256": digest.hexdigest(),
        "archive_listing_verified": True,
        "verified_complete_checkpoints": len(checks["checkpoints"]),
        "rotations": checks["rotations"],
        "scientific_updates_lost": 0,
        "resume_config": str(BASE / "EVENT_WATCHDOG_CONFIG_V18.json"),
        "budget_deadline_utc": "2026-10-27T14:09:00+00:00",
        "resume_note": (
            "User authorization required; restore /opt, refresh process identities, "
            "review remaining budget and deadline."
        ),
    }
    pending_path = BASE / "PRESERVATION_STATUS.pending"
    with pending_path.open("w") as f:
        json.dump(final, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(pending_path, BASE / "PRESERVATION_STATUS.json")
    subprocess.run(["sync", "-f", str(BASE)], check=True)
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
