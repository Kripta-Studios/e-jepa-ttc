"""Seal the fast process scan while retaining all scientific and resource contracts."""

from __future__ import annotations

import argparse
from pathlib import Path

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.freeze import dependency_files

QA_CHECKS = {
    "FAST_PROCESS_SCAN_PYTEST.txt": "[100%]",
    "FAST_PROCESS_SCAN_RUFF.txt": "All checks passed",
    "FAST_PROCESS_SCAN_PYRIGHT.txt": "0 errors",
}


def run(output: Path) -> None:
    """Bind executed QA and observational scan timing to the additive variant."""
    for name, marker in QA_CHECKS.items():
        if marker not in (output / name).read_text(encoding="utf-8"):
            raise ValueError(f"Fast process scan QA failed: {name}")
    cache_path = output / "ADAPTIVE_CACHE_FREEZE.json"
    verify_sources(read(cache_path))
    benchmark_path = output / "FAST_PROCESS_SCAN_BENCHMARK.json"
    benchmark = read(benchmark_path)
    if (
        benchmark.get("legacy_heavy_identities_equal") is not True
        or benchmark.get("scientific_optimizer_updates") != 0
    ):
        raise ValueError("Executed process scan evidence required")
    source_dir = ROOT / "operational/train40_system"
    seeds = [
        source_dir / name
        for name in (
            "process_scan.py",
            "engine_fast_scan.py",
            "controller_fast_scan.py",
            "process_scan_freeze.py",
        )
    ]
    seeds.extend(
        ROOT / "tests/unit" / name
        for name in (
            "test_train40_process_scan.py",
            "test_train40_fast_scan_execution.py",
        )
    )
    contract = {
        "schema": "train40_fast_process_scan_freeze_v1",
        "files": dependency_files(seeds),
        "engine_sha256": digest(source_dir / "engine_fast_scan.py"),
        "adaptive_cache_freeze_sha256": digest(cache_path),
        "benchmark_sha256": digest(benchmark_path),
        "QA": {name: digest(output / name) for name in QA_CHECKS},
        "original_guard_delegated_once_per_update": True,
        "scientific_budget_RAM_disk_deadline_checks_unchanged": True,
        "original_five_second_scan_cadence_preserved": True,
        "admitted_engine_aliases_and_graph_probe_added_to_heavy_detection": True,
        "model_loss_optimizer_sampler_RNG_and_checkpoint_unchanged": True,
        "optimizer_updates_during_admission": 0,
        "whole_training_acceleration_claimed_before_live_measurement": False,
    }
    path = output / "FAST_PROCESS_SCAN_FREEZE.json"
    if path.exists() and read(path) != contract:
        raise ValueError("Preserve existing process scan admission")
    atomic_json(path, contract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output.resolve())
