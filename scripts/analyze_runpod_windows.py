"""Validate and compare one or more concurrent Runpod measurement windows.

The Pod-level throughput of concurrent workers is the sum of their update
deltas divided by their shared wall-clock interval.  Worker-seconds are not
used as the denominator for aggregate throughput.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _finite_number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def read_samples(path: Path) -> list[dict[str, Any]]:
    """Read a nonempty JSONL telemetry stream."""
    samples: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number} is not a JSON object")
        samples.append(value)
    if len(samples) < 2:
        raise ValueError(f"{path} needs at least two samples")
    return samples


def _compute_pids(sample: dict[str, Any]) -> set[int] | None:
    for key in ("cuda_compute_pids", "gpu_compute_pids", "compute_pids"):
        if key in sample:
            values = sample[key]
            if not isinstance(values, list):
                raise ValueError(f"{key} must be a list")
            return {_integer(value, key) for value in values}
    values = sample.get("gpu_compute_processes")
    if values is not None:
        if not isinstance(values, list):
            raise ValueError("gpu_compute_processes must be a list")
        return {_integer(value.get("pid"), "gpu_compute_processes.pid") for value in values}
    return None


def analyze_window(
    name: str, path: Path, *, expected_seconds: float, duration_tolerance: float
) -> dict[str, Any]:
    """Validate stable workers and calculate concurrent Pod throughput."""
    samples = read_samples(path)
    elapsed = [_finite_number(sample.get("elapsed_s"), "elapsed_s") for sample in samples]
    if any(right <= left for left, right in zip(elapsed, elapsed[1:], strict=False)):
        raise ValueError(f"{name}: elapsed_s is not strictly increasing")
    duration = elapsed[-1] - elapsed[0]

    first_fits = samples[0].get("fits")
    if not isinstance(first_fits, list) or not first_fits:
        raise ValueError(f"{name}: fits must be a nonempty list")
    directories = [str(fit.get("directory")) for fit in first_fits]
    if len(set(directories)) != len(directories) or "None" in directories:
        raise ValueError(f"{name}: fit directories must be unique strings")

    histories: dict[str, dict[str, list[int]]] = {
        directory: {"pids": [], "updates": [], "pending": []} for directory in directories
    }
    observed_compute_pids: list[set[int]] = []
    missing_compute_pid_samples = 0
    for sample_number, sample in enumerate(samples):
        fits = sample.get("fits")
        if not isinstance(fits, list):
            raise ValueError(f"{name}: sample {sample_number} has no fits list")
        by_directory = {str(fit.get("directory")): fit for fit in fits}
        if set(by_directory) != set(directories):
            raise ValueError(f"{name}: worker set changed inside the window")
        for directory in directories:
            fit = by_directory[directory]
            progress, journal = fit.get("progress"), fit.get("journal")
            if not isinstance(progress, dict) or not isinstance(journal, dict):
                raise ValueError(f"{name}: missing progress/journal for {directory}")
            histories[directory]["pids"].append(_integer(progress.get("pid"), "progress.pid"))
            histories[directory]["updates"].append(
                _integer(journal.get("completed_updates"), "journal.completed_updates")
            )
            histories[directory]["pending"].append(
                _integer(journal.get("pending_update_upper"), "journal.pending_update_upper")
            )
        compute = _compute_pids(sample)
        if compute is None:
            missing_compute_pid_samples += 1
        else:
            observed_compute_pids.append(compute)

    fits_result = []
    expected_pids: set[int] = set()
    violations: list[str] = []
    total_updates = 0
    for directory in directories:
        history = histories[directory]
        stable_pids = set(history["pids"])
        if len(stable_pids) != 1:
            violations.append(f"PID changed for {directory}: {sorted(stable_pids)}")
        expected_pids.update(stable_pids)
        updates = history["updates"]
        if any(right < left for left, right in zip(updates, updates[1:], strict=False)):
            violations.append(f"completed_updates decreased for {directory}")
        if history["pending"][-1] != 0:
            violations.append(f"terminal pending_update_upper is nonzero for {directory}")
        delta = updates[-1] - updates[0]
        total_updates += delta
        fits_result.append(
            {
                "directory": directory,
                "pid": history["pids"][0] if len(stable_pids) == 1 else None,
                "before": updates[0],
                "after": updates[-1],
                "delta_updates": delta,
                "updates_per_min": delta * 60.0 / duration,
                "terminal_pending_update_upper": history["pending"][-1],
            }
        )

    cuda_validation = "NOT_OBSERVED"
    if observed_compute_pids:
        if missing_compute_pid_samples:
            violations.append("CUDA PID inventory is missing from some samples")
        mismatches = [
            sorted(value) for value in observed_compute_pids if value != expected_pids
        ]
        if mismatches:
            violations.append(
                f"CUDA PID set differs from expected {sorted(expected_pids)}: {mismatches[:3]}"
            )
        cuda_validation = "PASS" if not missing_compute_pid_samples and not mismatches else "FAIL"
    else:
        violations.append("CUDA PID inventory was not observed in the window")

    duration_ok = abs(duration - expected_seconds) <= duration_tolerance
    if not duration_ok:
        violations.append(
            f"observed duration {duration:.6f}s is outside "
            f"{expected_seconds:.6f}+/-{duration_tolerance:.6f}s"
        )
    worker_count = len(directories)
    aggregate = total_updates * 60.0 / duration
    return {
        "name": name,
        "samples_path": str(path.resolve()),
        "samples": len(samples),
        "actual_seconds": duration,
        "expected_seconds": expected_seconds,
        "duration_tolerance_seconds": duration_tolerance,
        "worker_count": worker_count,
        "expected_cuda_compute_pids": sorted(expected_pids),
        "cuda_pid_validation": cuda_validation,
        "fits": fits_result,
        "total_updates": total_updates,
        "pod_aggregate_updates_per_min": aggregate,
        "average_worker_updates_per_min": aggregate / worker_count,
        "aggregate_denominator": "one shared wall-clock interval",
        "latency_p95_supported": False,
        "violations": violations,
        "status": "PASS" if not violations else "FAIL",
    }


def read_baseline(path: Path) -> dict[str, Any]:
    """Normalize the v1 measurement result used as the single-worker baseline."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Baseline result must be a JSON object")
    duration_value = value.get(
        "actual_seconds",
        value.get("window_seconds", value.get("duration_seconds", value.get("duration_s"))),
    )
    duration = _finite_number(duration_value, "baseline actual_seconds")
    fits = value.get("fits")
    if isinstance(fits, list) and fits:
        updates = sum(
            _integer(fit.get("updates", fit.get("delta_updates")), "baseline fits.updates")
            for fit in fits
        )
        workers = len(fits)
    else:
        updates = _integer(
            value.get("total_updates", value.get("updates", value.get("delta_updates"))),
            "baseline total_updates",
        )
        workers = _integer(value.get("worker_count", 1), "baseline worker_count")
    rate = updates * 60.0 / duration
    return {
        "source": str(path.resolve()),
        "actual_seconds": duration,
        "worker_count": workers,
        "total_updates": updates,
        "pod_aggregate_updates_per_min": rate,
        "average_worker_updates_per_min": rate / workers,
    }


def parse_named_path(value: str) -> tuple[str, Path]:
    """Parse NAME=PATH without guessing a name from mutable filenames."""
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("window must have the form NAME=PATH")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", action="append", required=True, type=parse_named_path)
    parser.add_argument("--baseline-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-seconds", type=float, default=360.0)
    parser.add_argument("--duration-tolerance", type=float, default=2.0)
    args = parser.parse_args()
    if args.expected_seconds <= 0 or args.duration_tolerance < 0:
        raise ValueError("Expected duration must be positive and tolerance nonnegative")

    baseline = read_baseline(args.baseline_result)
    windows = [
        analyze_window(
            name,
            path,
            expected_seconds=args.expected_seconds,
            duration_tolerance=args.duration_tolerance,
        )
        for name, path in args.window
    ]
    for window in windows:
        baseline_rate = baseline["pod_aggregate_updates_per_min"]
        window["comparison_to_baseline"] = {
            "pod_aggregate_throughput_ratio": window["pod_aggregate_updates_per_min"]
            / baseline_rate,
            "average_worker_throughput_ratio": window["average_worker_updates_per_min"]
            / baseline["average_worker_updates_per_min"],
        }
    report = {
        "schema": "ejepa_runpod_concurrent_window_analysis_v1",
        "status": "PASS" if all(item["status"] == "PASS" for item in windows) else "FAIL",
        "baseline": baseline,
        "windows": windows,
        "math_contract": {
            "pod_aggregate_updates_per_min": "sum(worker_delta_updates) * 60 / shared_wall_seconds",
            "average_worker_updates_per_min": "pod_aggregate_updates_per_min / worker_count",
            "forbidden_denominator": "sum of concurrent worker durations",
        },
        "scope": "Observed Pod/system throughput under the recorded operational overlay",
        "universal_hardware_or_model_claim": False,
        "update_latency_p95_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
