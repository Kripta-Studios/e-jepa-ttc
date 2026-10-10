"""Verify the public memory tables directly from the captured samples."""

import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
result = json.loads((root / "MEMORY_OBSERVATION.json").read_text(encoding="utf-8"))
series = json.loads((root / "MEMORY_TIMESERIES.json").read_text(encoding="utf-8"))
for summary, data in zip(result["intervals"], series, strict=True):
    rows = data["samples"]
    assert data["interval"] == summary["interval"]
    assert all(
        row["commit_limit_bytes"] - row["commit_bytes"] == row["commit_headroom_bytes"]
        for row in rows
    )
    assert summary["commit_headroom_min_gib"] == min(
        row["commit_headroom_bytes"] for row in rows
    ) / 1024**3
    assert summary["commit_headroom_final_gib"] == rows[-1]["commit_headroom_bytes"] / 1024**3
    assert summary["ram_available_min_gib"] == min(
        row["available_ram_bytes"] for row in rows
    ) / 1024**3
    steady = [row for row in rows if row["steady_elapsed_s"] is not None]
    assert summary["steady_s"] == steady[-1]["steady_elapsed_s"]
    head = [row for row in steady if row["steady_elapsed_s"] <= 30]
    tail = [
        row for row in steady
        if row["steady_elapsed_s"] >= steady[-1]["steady_elapsed_s"] - 30
    ]
    for fit, memory in summary["private_memory_gib"].items():
        assert len({row["trainers"][fit]["pid"] for row in steady}) == 1
        assert rows[-1]["updates"][fit] > rows[0]["updates"][fit]
        for key, samples in (("first_30s_median", head), ("last_30s_median", tail)):
            assert memory[key] == statistics.median(
                row["trainers"][fit]["private_bytes"] / 1024**3 for row in samples
            )
print("Verified: two intervals, four producer records, memory bounds and 30-second medians.")
