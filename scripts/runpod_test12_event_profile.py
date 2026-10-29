"""Measure sixty active preprocessing seconds without replaying any inputs."""

from __future__ import annotations

import json
import time
from pathlib import Path

import psutil

ROOT = Path("/workspace/ejepa_test12_events")
DEADLINE = 1793145600.0


def snapshot() -> dict:
    status = json.loads((ROOT / "STATUS.json").read_text())
    processes = {}
    for process in psutil.process_iter(["pid", "ppid", "cpu_times", "memory_info", "cmdline"]):
        command = process.info["cmdline"] or []
        if command and "python" in command[0]:
            processes[str(process.pid)] = {
                "cpu": sum(process.info["cpu_times"][:2]),
                "rss": process.info["memory_info"].rss,
                "ppid": process.info["ppid"],
                "cmd": command,
            }
    memory = Path("/sys/fs/cgroup/memory")
    return {
        "timestamp": time.time(),
        "status": status,
        "processes": processes,
        "headroom": int((memory / "memory.limit_in_bytes").read_text())
        - int((memory / "memory.usage_in_bytes").read_text()),
    }


def main() -> None:
    started = time.time()
    previous = snapshot()
    active = 0.0
    queries = 0
    records = []
    while time.time() < DEADLINE and active < 60:
        time.sleep(1)
        current = snapshot()
        before = previous["status"]
        after = current["status"]
        same = (
            before.get("status") == "MATERIALIZING_EXACT_TENSORS"
            and after.get("status") == "MATERIALIZING_EXACT_TENSORS"
            and before.get("sequence") == after.get("sequence")
            and before.get("pid") == after.get("pid")
            and after.get("workers") == 4
        )
        if same:
            elapsed = current["timestamp"] - previous["timestamp"]
            count = after["completed"] - before["completed"]
            if count >= 0:
                cores = []
                for pid, item in current["processes"].items():
                    if pid in previous["processes"]:
                        value = (item["cpu"] - previous["processes"][pid]["cpu"]) / elapsed
                        if value > 0.01:
                            cores.append(
                                {
                                    "pid": pid,
                                    "cores": value,
                                    "rss": item["rss"],
                                    "event_child": item["ppid"] == after["pid"],
                                    "cmd": item["cmd"],
                                }
                            )
                records.append(
                    {
                        "seconds": elapsed,
                        "queries": count,
                        "sequence": after["sequence"],
                        "headroom": current["headroom"],
                        "processes": cores,
                    }
                )
                active += elapsed
                queries += count
        previous = current
    result = {
        "schema": "test12_event_four_worker_profile_v1",
        "started": started,
        "finished": time.time(),
        "wall_seconds": time.time() - started,
        "active_preprocessing_seconds": active,
        "new_queries": queries,
        "queries_per_active_second": queries / active if active else None,
        "minimum_headroom": min((r["headroom"] for r in records), default=None),
        "event_core_seconds": sum(
            sum(p["cores"] for p in r["processes"] if p["event_child"]) * r["seconds"]
            for r in records
        ),
        "records": records,
        "replayed_queries": 0,
        "interpretation": (
            "First sixty active seconds; download gaps excluded, wall time separately recorded; "
            "no matched same-query speedup claim"
        ),
    }
    temporary = ROOT / "FOUR_WORKER_PROFILE.pending"
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(ROOT / "FOUR_WORKER_PROFILE.json")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
