"""Publish the additive authenticated TRAIN40 parallel-pilot admission."""

from __future__ import annotations

import argparse
from pathlib import Path

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.freeze import dependency_files

QA = {
    "PARALLEL_PILOT_V3_PYTEST.txt": "[100%]",
    "PARALLEL_PILOT_V3_RUFF.txt": "All checks passed",
    "PARALLEL_PILOT_V3_PYRIGHT.txt": "0 errors",
    "PARALLEL_PILOT_V3_SPAWN.txt": '"status": "PASSED"',
}


def run(output: Path) -> None:
    """Require focused CPU QA, explicit authorization and the frozen host-fast trainer."""
    for name, marker in QA.items():
        if marker not in (output / name).read_text(encoding="utf-8"):
            raise ValueError(f"Parallel pilot QA failed: {name}")
    predecessor = output / "HOST_FAST_4_FREEZE.json"
    verify_sources(read(predecessor))
    pilot_v1 = output / "PARALLEL_PILOT_FREEZE.json"
    verify_sources(read(pilot_v1))
    authorization = read(output / "PARALLEL_PILOT_AUTHORIZATION.json")
    if (
        authorization.get("per_arm_scientific_quota") != {"a5": 1000, "c2f": 1000}
        or authorization.get("scientific_updates_in_measurement_total") != 2000
        or authorization.get("aggregate_trainer_tree_rss_max_bytes") != 24 * 1024**3
    ):
        raise ValueError("Parallel authorization contract differs")
    source = ROOT / "operational/train40_system"
    test = ROOT / "tests/unit/test_train40_parallel_pilot_v3.py"
    seeds = [
        source / "parallel_runtime_v3.py",
        source / "parallel_engine_v3.py",
        source / "parallel_controller_v3.py",
        source / "parallel_freeze_v3.py",
        test,
        output / "parallel_spawn_probe_v3.py",
    ]
    contract = {
        "schema": "train40_parallel_pilot_v3_freeze_v1",
        "files": dependency_files(seeds),
        "engine_sha256": digest(source / "parallel_engine_v3.py"),
        "controller_sha256": digest(source / "parallel_controller_v3.py"),
        "runtime_sha256": digest(source / "parallel_runtime_v3.py"),
        "publisher_sha256": digest(source / "parallel_freeze_v3.py"),
        "authorization_sha256": digest(output / "PARALLEL_PILOT_AUTHORIZATION.json"),
        "host_fast_4_freeze_sha256": digest(predecessor),
        "parallel_pilot_v1_freeze_sha256": digest(pilot_v1),
        "predecessor_freezes": {
            "HOST_FAST_4_FREEZE.json": digest(predecessor),
            "PARALLEL_PILOT_FREEZE.json": digest(pilot_v1),
        },
        "QA": {name: digest(output / name) for name in QA},
        "per_arm_scientific_quota": {"a5": 1000, "c2f": 1000},
        "combined_scientific_updates": 2000,
        "aggregate_tree_rss_limit_bytes": 24 * 1024**3,
        "host_available_min_bytes": 2 * 1024**3,
        "disk_free_min_bytes": 20_000_000_000,
        "physical_updates_global_limit": 240_000,
        "a5_warmup_before_c2f_and_post_restore_barrier": True,
        "fixed_science_model_loss_sampler_rng_precision_unchanged": True,
        "additional_optimizer_updates": 0,
    }
    path = output / "PARALLEL_PILOT_V3_FREEZE.json"
    if path.exists() and read(path) != contract:
        raise ValueError("Preserve existing parallel-pilot freeze")
    atomic_json(path, contract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output.resolve())
