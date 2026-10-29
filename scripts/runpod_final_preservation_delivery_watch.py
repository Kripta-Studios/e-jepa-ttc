"""Preserve admitted per-sequence predictions, then close the local final delivery."""

from __future__ import annotations

import json
import subprocess
import sys
import time

from runpod_final_preservation import DEADLINE, RECEIPTS, ROOT, call, publish
from runpod_final_preservation_close import REQUIRED


def preserve(groups: list[str], *, refresh: bool = False) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/runpod_final_preservation.py"),
        "--groups",
        *groups,
    ]
    if refresh:
        command.append("--refresh")
    for attempt in range(3):
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if result.returncode == 0:
            print(result.stdout[-2000:], flush=True)
            return
        if "banner exchange" not in result.stderr or attempt == 2:
            raise RuntimeError(result.stderr[-8000:])
        time.sleep(15)


def admitted(group: str) -> bool:
    path = RECEIPTS / f"{group}.json"
    return path.exists() and json.loads(path.read_text()).get("status") == "LOCAL_VERIFIED"


def main() -> None:
    while time.time() < DEADLINE and not (RECEIPTS / "STOP_BACKUP_REQUEST").exists():
        state = call("""import pathlib,json
def read(path):
 return json.loads(path.read_text()) if path.exists() else {}
root=pathlib.Path('/workspace/ejepa_test12_rgb/consumed')
ready=[p.stem for p in root.glob('*.json') if read(p).get('status')=='PASS']
package=read(pathlib.Path('/workspace/ejepa_final_operations/TEST12_PACKAGE_ADMISSION.json'))
stopped=read(pathlib.Path('/workspace/ejepa_final_diagnostics/COLLECTOR_STOPPED.json'))
print('PODSTATUS:'+json.dumps({'ready':sorted(ready),'package':package,'stopped':stopped}))""")
        for sequence in state["ready"]:
            group = "test12_predictions_" + sequence
            if not admitted(group):
                publish(
                    RECEIPTS / "DELIVERY_WATCH_STATUS.json",
                    {
                        "status": "PRESERVING_PREDICTION_SEQUENCE",
                        "sequence": sequence,
                        "timestamp": time.time(),
                    },
                )
                preserve([group])
        events = sorted(
            path.stem
            for path in RECEIPTS.glob("test12_events_*.json")
            if ".manifest" not in path.stem and ".cleanup" not in path.stem and admitted(path.stem)
        )
        if (
            len(events) == 12
            and len(state["ready"]) == 12
            and state["package"].get("status") == "COMPLETE"
            and state["stopped"].get("status") == "STOPPED"
        ):
            preserve(
                [
                    "final_delivery",
                    "test12_event_metadata",
                    "test12_rgb_metadata",
                    "final_runtime",
                    "final_diagnostics",
                ],
                refresh=True,
            )
            missing = [group for group in REQUIRED if not admitted(group)]
            if missing:
                publish(
                    RECEIPTS / "DELIVERY_WATCH_STATUS.json",
                    {
                        "status": "WAITING_OTHER_BACKUPS",
                        "missing": missing,
                        "timestamp": time.time(),
                    },
                )
                time.sleep(15)
                continue
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/runpod_final_preservation_close.py")],
                check=True,
            )
            publish(
                RECEIPTS / "DELIVERY_WATCH_STATUS.json",
                {"status": "LOCAL_FINAL_PRESERVATION_COMPLETE", "timestamp": time.time()},
            )
            return
        publish(
            RECEIPTS / "DELIVERY_WATCH_STATUS.json",
            {
                "status": "WAITING_PACKAGE_AND_FINAL_DIAGNOSTICS",
                "prediction_sequences": len(state["ready"]),
                "event_sequences_local": len(events),
                "package_status": state["package"].get("status"),
                "collector_status": state["stopped"].get("status"),
                "timestamp": time.time(),
            },
        )
        time.sleep(30)
    publish(
        RECEIPTS / "DELIVERY_WATCH_STATUS.json",
        {"status": "DEADLINE_OR_MANUAL_STOP", "timestamp": time.time()},
    )


if __name__ == "__main__":
    main()
