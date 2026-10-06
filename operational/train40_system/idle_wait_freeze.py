"""Admit passive OpenMP waiting only after CPU evidence and source QA."""

from __future__ import annotations

import argparse
from pathlib import Path

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.freeze import dependency_files

QA_CHECKS = {
    "IDLE_WAIT_PYTEST.txt": "[100%]",
    "IDLE_WAIT_RUFF.txt": "All checks passed",
    "IDLE_WAIT_PYRIGHT.txt": "0 errors",
}
BENCHMARKS = {
    "baseline": "IDLE_WAIT_BASELINE.json",
    "passive": "IDLE_WAIT_PASSIVE.json",
}
PILOT_20MS_ARTIFACTS = (
    "IDLE_WAIT_20MS_BASELINE.json",
    "IDLE_WAIT_20MS_PASSIVE.json",
    "IDLE_WAIT_20MS_PYTEST.txt",
    "IDLE_WAIT_20MS_RUFF.txt",
    "IDLE_WAIT_20MS_PYRIGHT.txt",
)


def run(output: Path) -> None:
    """Bind a measured CPU improvement without asserting unmeasured training acceleration."""
    for name, marker in QA_CHECKS.items():
        if marker not in (output / name).read_text(encoding="utf-8"):
            raise ValueError(f"Required idle-wait QA failed: {name}")
    coordination_path = output / "COORDINATION_FREEZE.json"
    pause_path = output / "CONTROLLER_PAUSE_SAFE_FREEZE.json"
    for path in (coordination_path, pause_path):
        verify_sources(read(path))
    benchmark_paths = {name: output / filename for name, filename in BENCHMARKS.items()}
    benchmarks = {name: read(path) for name, path in benchmark_paths.items()}
    baseline, passive = benchmarks["baseline"], benchmarks["passive"]
    if baseline.get("status") != "COMPLETE" or baseline.get("mode") != "baseline":
        raise ValueError("Complete baseline idle-wait benchmark required")
    if passive.get("status") != "COMPLETE" or passive.get("mode") != "passive":
        raise ValueError("Complete passive idle-wait benchmark required")
    required_contract = {
        "schema": "train40_idle_wait_cpu_benchmark_v1",
        "iterations": 30,
        "sleep_seconds_per_iteration": 0.100,
        "calibration": {
            "basis": "measured full-step GPU active-union arithmetic",
            "measured_seconds": 0.1093,
            "benchmark_sleep_seconds": 0.100,
            "earlier_20ms_pilot_was_not_representative": True,
        },
        "torch_num_threads": 4,
        "cuda_apis_called": False,
        "optimizer_updates": 0,
    }
    for mode, benchmark in benchmarks.items():
        if any(benchmark.get(key) != value for key, value in required_contract.items()):
            raise ValueError(f"{mode} idle-wait benchmark contract differs")
    if baseline.get("batch_shapes") != passive.get("batch_shapes"):
        raise ValueError("Idle-wait benchmarks used different batch shapes")
    if baseline.get("tensor_sha256") != passive.get("tensor_sha256"):
        raise ValueError("Idle-wait benchmarks produced different CPU tensors")
    if passive.get("environment") != {
        "OMP_WAIT_POLICY": "PASSIVE",
        "KMP_BLOCKTIME": "0",
        "OMP_NUM_THREADS": passive.get("environment", {}).get("OMP_NUM_THREADS"),
    }:
        raise ValueError("Passive benchmark did not use the admitted environment")
    runtime = passive.get("kmp_blocktime_after", {})
    if runtime.get("available") is True and runtime.get("value") != 0:
        raise ValueError("Intel OpenMP runtime did not apply KMP_BLOCKTIME=0")
    baseline_cpu = float(baseline["process_cpu_seconds"])
    passive_cpu = float(passive["process_cpu_seconds"])
    baseline_wall = float(baseline["wall_seconds"])
    passive_wall = float(passive["wall_seconds"])
    if not (0 < passive_cpu < baseline_cpu):
        raise ValueError("Passive idle waiting did not lower measured process CPU cost")
    if not (0 < passive_wall <= baseline_wall * 1.10 + 0.01):
        raise ValueError("Passive idle waiting regressed wall time beyond tolerance")
    pilot_paths = {name: output / name for name in PILOT_20MS_ARTIFACTS}
    pilot_baseline = read(pilot_paths["IDLE_WAIT_20MS_BASELINE.json"])
    pilot_passive = read(pilot_paths["IDLE_WAIT_20MS_PASSIVE.json"])
    if (
        pilot_baseline.get("sleep_seconds_per_iteration") != 0.020
        or pilot_passive.get("sleep_seconds_per_iteration") != 0.020
        or pilot_baseline.get("tensor_sha256") != pilot_passive.get("tensor_sha256")
    ):
        raise ValueError("Preserved 20ms pilot contract differs")
    if not (
        float(pilot_passive["process_cpu_seconds"])
        < float(pilot_baseline["process_cpu_seconds"])
        and float(pilot_passive["wall_seconds"])
        > float(pilot_baseline["wall_seconds"]) * 1.10
    ):
        raise ValueError("Preserved 20ms pilot must retain its unsuccessful wall-time result")
    source_names = (
        "idle_wait.py",
        "engine_idle_wait.py",
        "controller_idle_wait.py",
        "idle_wait_freeze.py",
    )
    sources = [ROOT / "operational/train40_system" / name for name in source_names]
    cpu_ratio = passive_cpu / baseline_cpu
    wall_ratio = passive_wall / baseline_wall
    contract = {
        "schema": "train40_idle_wait_freeze_v1",
        "files": dependency_files(sources),
        "source_sha256": {path.name: digest(path) for path in sources},
        "coordination_freeze_sha256": digest(coordination_path),
        "controller_pause_safe_freeze_sha256": digest(pause_path),
        "QA": {name: digest(output / name) for name in QA_CHECKS},
        "benchmarks": {
            name: {"sha256": digest(path), "tensor_sha256": benchmarks[name]["tensor_sha256"]}
            for name, path in benchmark_paths.items()
        },
        "unsuccessful_20ms_pilot_artifacts": {
            name: digest(path) for name, path in pilot_paths.items()
        },
        "process_cpu_ratio_passive_over_baseline": cpu_ratio,
        "wall_ratio_passive_over_baseline": wall_ratio,
        "measured_process_cpu_reduction": True,
        "measured_wall_acceleration": wall_ratio < 1.0,
        "training_throughput_acceleration_claimed": False,
        "unsuccessful_20ms_pilot_retained": True,
        "benchmark_calibrated_to_profiled_gpu_arithmetic_seconds": 0.1093,
        "environment": {"OMP_WAIT_POLICY": "PASSIVE", "KMP_BLOCKTIME": "0"},
        "torch_threads_unchanged": 4,
        "scientific_math_batch_rng_precision_and_checkpoint_unchanged": True,
        "optimizer_updates_during_admission": 0,
    }
    path = output / "IDLE_WAIT_FREEZE.json"
    if path.exists() and read(path) != contract:
        raise ValueError("Preserve existing idle-wait freeze")
    atomic_json(path, contract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output.resolve())
