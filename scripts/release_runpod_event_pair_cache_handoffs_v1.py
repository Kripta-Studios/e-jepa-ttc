"""Release the two temporary C2F handoffs after verified R2 PAIR_E sealing."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

OPS = Path("/workspace/ejepa_event_operations")
RUN = Path("/workspace/ejepa_event_runs")
CONFIG = OPS / "EVENT_WATCHDOG_CONFIG_V9.json"
STATUS = OPS / "EVENT_WATCHDOG_STATUS_V9.json"
WATCHDOG = OPS / "runpod_event_resource_watchdog_v7.py"
STOP_TEXT = "temporary operator resource handoff for R2 PAIR_E cache\n"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic(path: Path, value: object) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with tmp.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> None:
    cache = RUN / "rotations/2/caches/PAIR_E.npz"
    side = cache.with_suffix(".json")
    receipt = json.loads(side.read_text())
    payload = dict(receipt)
    claimed = payload.pop("receipt_payload_sha256")
    if not (
        receipt.get("schema") == "final_refit_pair_cache_v1"
        and receipt.get("status") == "COMPLETE"
        and receipt.get("role") == "producer_pair"
        and receipt.get("output") == str(cache.resolve())
        and receipt.get("output_sha256") == sha(cache)
        and receipt.get("targets_used_for_feature_construction") is False
        and claimed == canonical(payload)
    ):
        raise ValueError("PAIR_E cache seal differs")
    stops = [
        RUN / "rotations/3/fits/E_C2F/STOP_REQUEST",
        RUN / "rotations/4/fits/E_C2F/STOP_REQUEST",
    ]
    if any(path.read_text() != STOP_TEXT for path in stops):
        raise ValueError("handoff STOP identity differs")
    status = json.loads(STATUS.read_text())
    config = json.loads(CONFIG.read_text())
    w = int((OPS / "EVENT_WATCHDOG_V9.pid").read_text())
    expected = ["/opt/ejepa_env/bin/python", "-u", str(WATCHDOG), "--config", str(CONFIG)]
    if Path(f"/proc/{w}/cmdline").read_bytes().split(b"\0")[:-1] != [x.encode() for x in expected]:
        raise ValueError("watchdog identity differs")
    os.kill(w, signal.SIGTERM)
    for _ in range(100):
        if not Path(f"/proc/{w}").exists():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("watchdog did not exit")
    for path in stops:
        path.unlink()
    for task in ("R3_E_C2F", "R4_E_C2F"):
        status["tasks"][task].update(
            state="RESTART_BACKOFF",
            reason="PAIR_E_CACHE_SEALED_HANDOFF_RELEASE",
            next_retry_epoch=0,
        )
    atomic(STATUS, status)
    environment = os.environ.copy()
    environment.update(config["required_environment"])
    environment.update(
        {
            "EJEPA_EVENT_SUPPLEMENT_ROOT": "/opt/ejepa_event_supplement",
            "EJEPA_EVENT_SUPPLEMENT_ADMISSION": str(
                OPS / "supplement/EVENT_DOWNSTREAM_SUPPLEMENTAL_ADMISSION.json"
            ),
            "EJEPA_EVENT_AUTHORITY_SUPPLEMENT_ROOT": "/opt/ejepa_event_authority_supplement_v2",
            "EJEPA_EVENT_AUTHORITY_SUPPLEMENT_ADMISSION": str(
                OPS / "authority_supplement_v2/EVENT_AUTHORITY_SUPPLEMENT_ADMISSION.json"
            ),
        }
    )
    proc = subprocess.Popen(
        expected,
        stdin=subprocess.DEVNULL,
        stdout=(OPS / "EVENT_WATCHDOG_V9.log").open("ab", buffering=0),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=environment,
    )
    (OPS / "EVENT_WATCHDOG_V9.pid").write_text(f"{proc.pid}\n")
    time.sleep(4)
    if proc.poll() is not None:
        raise RuntimeError("watchdog relaunch failed")
    atomic(
        OPS / "PAIR_E_CACHE_HANDOFF_RELEASE.json",
        {
            "schema": "ejepa_event_pair_cache_handoff_release_v1",
            "status": "RELEASED",
            "cache_sha256": sha(cache),
            "cache_receipt_sha256": sha(side),
            "tasks": ["R3_E_C2F", "R4_E_C2F"],
            "watchdog_pid": proc.pid,
            "optimizer_updates_reset": False,
        },
    )


if __name__ == "__main__":
    main()
