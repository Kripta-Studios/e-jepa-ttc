import json
import time
from pathlib import Path


def main() -> None:
    runs_dir = Path("artifacts/runs")
    audit_dir = Path("artifacts/audit")
    audit_dir.mkdir(parents=True, exist_ok=True)

    interrupted_runs = []

    # We look for anything that might have been spawned recently (last 10 minutes)
    # or matches the post_fix_v3_cache_verified suffix
    now = time.time()

    if runs_dir.exists():
        for d in runs_dir.iterdir():
            if not d.is_dir():
                continue

            mtime = d.stat().st_mtime
            if (now - mtime) < 600 or "v3_cache_verified" in d.name:
                run_info = {
                    "path": d.as_posix(),
                    "command": "unknown",
                    "start_time": mtime,
                    "checkpoint_files_present": [f.name for f in d.glob("*.pt")],
                    "metrics_files_present": [f.name for f in d.glob("*.json")],
                    "completion_status": "interrupted",
                    "reason_for_invalidation": "invalid_workflow_smoke_not_completed",
                    "consumed_final_test_split": False,
                }
                interrupted_runs.append(run_info)

                # Mark as invalid
                (d / "INVALIDATED_invalid_workflow_smoke_not_completed").touch()

    with open(audit_dir / "task_827_interrupted_runs.json", "w") as f:
        json.dump(interrupted_runs, f, indent=2)


if __name__ == "__main__":
    main()
