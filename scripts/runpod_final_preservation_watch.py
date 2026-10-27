"""Preserve each completed Test12 event sequence as it becomes immutable."""

from __future__ import annotations

import json
import subprocess
import sys
import time

from runpod_final_preservation import DEADLINE, RECEIPTS, ROOT, call, publish


def main() -> None:
    while time.time() < DEADLINE and not (RECEIPTS / "STOP_BACKUP_REQUEST").exists():
        state = call("""import pathlib,json
root=pathlib.Path('/workspace/ejepa_test12_events');ready=[]
for p in (root/'sequences').glob('*.json'):
 value=json.loads(p.read_text())
 if value.get('status')=='COMPLETE':ready.append(p.stem)
print('PODSTATUS:'+json.dumps({'ready':sorted(ready),'timestamp':__import__('time').time()}))""")
        verified = []
        for sequence in state["ready"]:
            group = "test12_events_" + sequence
            path = RECEIPTS / (group + ".json")
            if path.exists() and json.loads(path.read_text()).get("status") == "LOCAL_VERIFIED":
                verified.append(sequence)
                continue
            publish(
                RECEIPTS / "WATCH_STATUS.json",
                {
                    "status": "PRESERVING_SEQUENCE",
                    "sequence": sequence,
                    "verified": verified,
                    "timestamp": time.time(),
                },
            )
            command = [
                sys.executable,
                str(ROOT / "scripts/runpod_final_preservation.py"),
                "--groups",
                group,
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode and "banner exchange" in result.stderr:
                publish(
                    RECEIPTS / "WATCH_STATUS.json",
                    {
                        "status": "RETRYING_SSH_TRANSPORT",
                        "sequence": sequence,
                        "timestamp": time.time(),
                    },
                )
                time.sleep(15)
                break
            if result.returncode:
                raise RuntimeError(result.stderr[-8000:])
            print(result.stdout[-1500:], flush=True)
            verified.append(sequence)
        status = (
            "ALL_12_EVENT_SEQUENCES_LOCAL_VERIFIED"
            if len(verified) == 12
            else "WAITING_FOR_COMPLETED_SEQUENCES"
        )
        publish(
            RECEIPTS / "WATCH_STATUS.json",
            {
                "status": status,
                "ready": state["ready"],
                "verified": verified,
                "timestamp": time.time(),
            },
        )
        if len(verified) == 12:
            return
        time.sleep(30)
    publish(
        RECEIPTS / "WATCH_STATUS.json",
        {"status": "DEADLINE_OR_MANUAL_STOP", "timestamp": time.time()},
    )


if __name__ == "__main__":
    main()
