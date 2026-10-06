"""Publish the additive authenticated TRAIN40 parallel-production admission."""

from __future__ import annotations

import argparse
from pathlib import Path

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.freeze import dependency_files

QA = {
    "PARALLEL_PRODUCTION_PYTEST.txt": "[100%]",
    "PARALLEL_PRODUCTION_RUFF.txt": "All checks passed",
    "PARALLEL_PRODUCTION_PYRIGHT.txt": "0 errors",
    "PARALLEL_PRODUCTION_SPAWN.txt": '"status": "PASSED"',
}


def run(output: Path) -> None:
    """Require focused CPU QA, explicit authorization and the frozen host-fast trainer."""
    for name, marker in QA.items():
        if marker not in (output / name).read_text(encoding="utf-8"):
            raise ValueError(f"Parallel production QA failed: {name}")
    predecessor = output / "HOST_FAST_4_FREEZE.json"
    verify_sources(read(predecessor))
    pilot_v1 = output / "PARALLEL_PILOT_FREEZE.json"
    verify_sources(read(pilot_v1))
    authorization = read(output / "PARALLEL_PRODUCTION_AUTHORIZATION.json")
    campaign_authorization = read(output / "AUTHORIZATION.json")
    if (
        authorization.get("allowed_arms") != ["a5", "c2f"]
        or authorization.get("per_arm_scientific_endpoints") != {"a5": 49_932, "c2f": 49_932}
        or authorization.get("max_concurrent_heavy_trainers") != 2
        or authorization.get("aggregate_rss_limit_bytes") != 24 * 1024**3
        or authorization.get("min_host_available_bytes") != 2 * 1024**3
        or authorization.get("min_output_free_bytes") != 20 * 1024**3
        or authorization.get("physical_update_cap") != 240_000
        or authorization.get("recovery_upper_limit") != 2_000
        or authorization.get("deadline_utc") != campaign_authorization.get("deadline_utc")
        or authorization.get("scientific_protocol_unchanged") is not True
        or authorization.get("new_scientific_arms") is not False
        or authorization.get("pilot_not_repeated") is not True
        or authorization.get("holdout_access_authorized") is not False
        or authorization.get("push_or_submission_authorized") is not False
        or authorization.get("historical_roots_read_only") is not True
        or authorization.get("checkpoint_and_partial_analysis_preservation_required")
        is not True
    ):
        raise ValueError("Parallel authorization contract differs")
    source = ROOT / "operational/train40_system"
    test = ROOT / "tests/unit/test_train40_parallel_production.py"
    seeds = [
        source / "parallel_runtime_production.py",
        source / "parallel_engine_production.py",
        source / "parallel_controller_production.py",
        source / "parallel_freeze_production.py",
        test,
        output / "parallel_spawn_probe_production.py",
    ]
    contract = {
        "schema": "train40_parallel_production_freeze_v1",
        "files": dependency_files(seeds),
        "engine_sha256": digest(source / "parallel_engine_production.py"),
        "controller_sha256": digest(source / "parallel_controller_production.py"),
        "runtime_sha256": digest(source / "parallel_runtime_production.py"),
        "publisher_sha256": digest(source / "parallel_freeze_production.py"),
        "authorization_sha256": digest(output / "PARALLEL_PRODUCTION_AUTHORIZATION.json"),
        "host_fast_4_freeze_sha256": digest(predecessor),
        "parallel_pilot_v1_freeze_sha256": digest(pilot_v1),
        "predecessor_freezes": {
            "HOST_FAST_4_FREEZE.json": digest(predecessor),
            "PARALLEL_PILOT_FREEZE.json": digest(pilot_v1),
        },
        "QA": {name: digest(output / name) for name in QA},
        "per_arm_scientific_endpoints": {"a5": 49_932, "c2f": 49_932},
        "reserve_exact_remaining_updates_at_activation": True,
        "aggregate_tree_rss_limit_bytes": 24 * 1024**3,
        "host_available_min_bytes": 2 * 1024**3,
        "disk_free_min_bytes": 20 * 1024**3,
        "physical_updates_global_limit": 240_000,
        "a5_warmup_before_c2f_and_post_restore_barrier": True,
        "fixed_science_model_loss_sampler_rng_precision_unchanged": True,
        "additional_optimizer_updates": 0,
    }
    path = output / "PARALLEL_PRODUCTION_FREEZE.json"
    if path.exists() and read(path) != contract:
        raise ValueError("Preserve existing parallel-production freeze")
    atomic_json(path, contract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output.resolve())
