"""Bounded A/B/A process-QoS trial while the admitted H8 process keeps running."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

from operational.efficient_context.common import ROOT, read
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.process_qos_probe import run

OUTPUT = ROOT / "artifacts/train40_system_20261005"
DIRECTORY = OUTPUT / "h8_bottleneck_20261007"


def phase(name: str, seconds: int) -> dict:
    """Measure matching runtime counters, avoiding stale-progress timing bias."""
    started = time.monotonic()
    samples = []
    while True:
        runtime = read(OUTPUT / "H8_FAST_RUNTIME.json")
        samples.append(runtime)
        atomic_json(
            DIRECTORY / "QOS_TRIAL_PROGRESS.json",
            {
                "phase": name,
                "seconds_elapsed": time.monotonic() - started,
                "checked_utc": datetime.now(UTC).isoformat(),
                "runtime": runtime,
            },
        )
        if time.monotonic() - started >= seconds:
            break
        time.sleep(10)
    first, last = samples[0], samples[-1]
    rows = last["extract_calls"] - first["extract_calls"]
    elapsed = (
        datetime.fromisoformat(last["checked_utc"]) - datetime.fromisoformat(first["checked_utc"])
    ).total_seconds()
    if (first["started_utc"], first["freeze_sha256"]) != (
        last["started_utc"], last["freeze_sha256"]
    ):
        raise ValueError("H8 process or execution contract changed during the phase")
    if rows <= 0 or elapsed <= 0:
        raise ValueError("No fresh advancing H8 runtime samples during the phase")
    result = {
        "phase": name,
        "first": first,
        "last": last,
        "queries": rows,
        "elapsed_seconds": elapsed,
        "queries_per_minute": rows * 60 / elapsed,
    }
    for field in ("guard_seconds", "extract_seconds", "progress_publication_seconds"):
        result[field + "_per_query"] = (last[field] - first[field]) / rows
    for field in ("worker_prepare_seconds", "arena_copy_seconds"):
        result[field + "_per_query"] = (
            last["shared_pool"][field] - first["shared_pool"][field]
        ) / rows
    atomic_json(DIRECTORY / f"QOS_PHASE_{name}.json", result)
    return result


def main() -> None:
    """Restore the initial process policy even when sampling or activation fails."""
    results = []
    results.append(phase("A_before", 90))
    try:
        run(OUTPUT, "enable")
        # Runtime publishes every thirty seconds. Skip one interval after a switch.
        time.sleep(35)
        results.append(phase("B_high_qos", 120))
    finally:
        if (DIRECTORY / "QOS_BASELINE.json").exists():
            run(OUTPUT, "restore")
    time.sleep(35)
    results.append(phase("A_restored", 90))
    atomic_json(
        DIRECTORY / "QOS_TRIAL_REPORT.json",
        {
            "status": "COMPLETE_ORIGINAL_POLICY_RESTORED",
            "phases": results,
            "optimizer_updates_added": 0,
            "original_H8_process_never_restarted": True,
            "comparison_limit": "Sequential production queries, not identical-input replay.",
        },
    )
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "phase",
                        "queries_per_minute",
                        "extract_seconds_per_query",
                        "worker_prepare_seconds_per_query",
                    )
                }
                for r in results
            ]
        )
    )


if __name__ == "__main__":
    main()
