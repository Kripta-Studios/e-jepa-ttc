"""Summarize recorded production costs without rerunning models or equating unlike workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def summary(values: list[float]) -> dict[str, float | int]:
    """Describe nonnegative finite measured durations, including zero-duration entries."""
    a = np.asarray(values, dtype=np.float64)
    if not len(a) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("timing observations must be nonempty, finite and nonnegative")
    return {
        "count": len(values),
        "sum_seconds": float(a.sum()),
        "mean_seconds": float(a.mean()),
        "p50_seconds": float(np.median(a)),
        "p95_seconds": float(np.percentile(a, 95)),
        "max_seconds": float(a.max()),
    }


def collect(root: Path, kind: str) -> dict[str, Any]:
    """Read small timing receipts and bind their exact bytes with an ordered SHA-256 digest."""
    folder = root / ("h8_feature_fragments" if kind == "H8" else "garl_train_predictions")
    files = sorted(folder.glob("query_*.json" if kind == "H8" else "batch_*.json"))
    groups: dict[str, list[float]] = {}
    all_times, compute, stamps = [], [], []
    population, next_ordinal = 0, 0
    receipt_hash = hashlib.sha256()
    for path in files:
        payload = path.read_bytes()
        receipt_hash.update(path.name.encode("utf-8") + b"\0" + payload + b"\0")
        value = json.loads(payload)
        ordinal = int(path.stem.split("_")[-1])
        count = 1 if kind == "H8" else value["rows"]
        if ordinal != next_ordinal or value["optimizer_updates"] != 0 or count <= 0:
            raise ValueError("production receipts are not contiguous zero-update inference")
        next_ordinal += count
        population += count
        elapsed = float(value["seconds"])
        all_times.append(elapsed)
        key = value.get(
            "four_worker_execution_freeze_sha256",
            value.get(
                "memory_execution_freeze_sha256",
                value.get("execution_freeze_sha256", "initial_route"),
            ),
        )
        groups.setdefault(key, []).append(elapsed)
        if "forward_and_transfer_seconds" in value:
            compute.append(float(value["forward_and_transfer_seconds"]))
        stamps.append(path.stat().st_mtime)
    if population != 88744:
        raise ValueError("complete TRAIN40 population required")
    result: dict[str, Any] = {
        "population": population,
        "receipt_count": len(files),
        "ordered_receipt_sha256": receipt_hash.hexdigest(),
        "recorded_wait_compute_write_per_fragment": summary(all_times),
        "engineering_regimes": {key: summary(v) for key, v in groups.items()},
        "first_receipt_local": datetime.fromtimestamp(min(stamps)).astimezone().isoformat(),
        "last_receipt_local": datetime.fromtimestamp(max(stamps)).astimezone().isoformat(),
        "receipt_span_hours_including_pauses": (max(stamps) - min(stamps)) / 3600,
        "fragment_rows": "1 query with eight history slots" if kind == "H8" else 8,
        "is_isolated_latency_benchmark": False,
        "includes_all_CPU_worker_time": False,
        "includes_all_startup_guard_and_queue_time": False,
    }
    if compute:
        result["GPU_forward_and_transfer_per_batch"] = summary(compute)
    return result


def main() -> None:
    """Emit an additive analysis; never modify completed campaign files or predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    costs = {key: collect(args.campaign, key) for key in ("H8", "Garl_event_only")}
    report = {
        "status": "COMPLETE",
        "checked_utc": datetime.now(UTC).isoformat(),
        "optimizer_updates": 0,
        "new_model_inferences": 0,
        "costs": costs,
        "interpretation": [
            "This is a production accounting audit, not an equal-context latency comparison.",
            "H8 prepares history and features; Garl emits predictions with native inputs.",
            "Fragment time includes unhidden wait, compute and writes; preparation overlaps.",
            "Receipt span includes pauses/resumption, excluding work before the first receipt.",
            "No training-time or inference-speed ratio is claimed between these workloads.",
        ],
    }
    (args.output / "COST_AUDIT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Recorded TRAIN40 production cost",
        "",
        *report["interpretation"],
        "",
        "| Route | Queries | Recorded fragment seconds | Receipt span (hours) |",
        "|---|---:|---:|---:|",
    ]
    for name, item in costs.items():
        seconds = item["recorded_wait_compute_write_per_fragment"]["sum_seconds"]
        lines.append(
            f"| {name} | {item['population']} | {seconds:.2f} | "
            f"{item['receipt_span_hours_including_pauses']:.3f} |"
        )
    (args.output / "COST_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                name: {
                    "receipt_span_hours": item["receipt_span_hours_including_pauses"],
                    "recorded_seconds": item["recorded_wait_compute_write_per_fragment"][
                        "sum_seconds"
                    ],
                }
                for name, item in costs.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
