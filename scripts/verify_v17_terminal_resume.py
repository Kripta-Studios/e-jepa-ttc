"""Execute the real resume driver and verify that a closed campaign spends no updates."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from operational.range_transfer.common import read, sha, write
from operational.representation_audit.run import ROOT, RUN


def snapshot() -> dict:
    receipts = sorted((RUN / "fits").glob("*/CHECKPOINT_RECEIPT.json"))
    states = {}
    for path in receipts:
        receipt = read(path)
        if receipt["status"] != "COMPLETE":
            raise ValueError(f"Incomplete fit: {path.parent.name}")
        checkpoint = Path(receipt["checkpoint_path"])
        if sha(checkpoint) != receipt["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash changed: {checkpoint}")
        states[path.parent.name] = {
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "updates": receipt["completed_updates"],
        }
    return {"fits": states, "accounting": read(RUN / "ACCOUNTING.json")}


def main() -> None:
    if (RUN / "PROCESS.lock").exists():
        raise ValueError("A live worker owns the campaign")
    if read(RUN / "CAMPAIGN_NEXT_DECISION.json")["status"] != "NO_CANDIDATE":
        raise ValueError("This proof requires a closed negative campaign")
    before = snapshot()
    command = ["pwsh", "-NoProfile", "-File", "scripts/Resume-ResearchV17.ps1", "-Queue", "all"]
    started = datetime.now(UTC).isoformat()
    with (RUN / "TERMINAL_RESUME.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    after = snapshot()
    if before != after:
        raise AssertionError("Terminal resume changed fit weights or update accounting")
    write(
        RUN / "TERMINAL_RESUME_PROOF.json",
        {
            "status": "PASS",
            "started_utc": started,
            "finished_utc": datetime.now(UTC).isoformat(),
            "command": command,
            "source_sha256": sha(Path(__file__)),
            "before": before,
            "after": after,
            "new_optimizer_updates": 0,
            "scope": "idempotent completed-campaign resume; not continuous scientific retraining",
            "GPU_scope_cost": "cache QA forward is charged separately in GPU_BUDGET.json",
        },
    )
    print(f"PASS: {len(after['fits'])} final checkpoint hashes preserved; zero new updates")


if __name__ == "__main__":
    main()
