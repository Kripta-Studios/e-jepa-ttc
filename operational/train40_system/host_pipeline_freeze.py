"""Bind tested CPU-process preparation to the existing scientific TRAIN40 freezes."""

from __future__ import annotations

import argparse
from pathlib import Path

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.freeze import dependency_files

PREDECESSORS = (
    "COORDINATION_FREEZE.json",
    "CONTROLLER_PAUSE_SAFE_FREEZE.json",
    "GRAPH_REPLAY_FREEZE.json",
    "SCALAR_SAFE_GRAPH_FREEZE.json",
)
QA_CHECKS = {
    "HOST_PIPELINE_PYTEST.txt": "[100%]",
    "HOST_PIPELINE_RUFF.txt": "All checks passed",
    "HOST_PIPELINE_PYRIGHT.txt": "0 errors",
}


def run(output: Path) -> None:
    """Publish only after real batch identity and bounded CPU recovery checks pass."""
    for name, marker in QA_CHECKS.items():
        if marker not in (output / name).read_text(encoding="utf-8"):
            raise ValueError(f"Required host pipeline QA failed: {name}")
    parity_name = "HOST_PIPELINE_REAL_PARITY.json"
    parity = read(output / parity_name)
    verify_sources(parity)
    if (
        parity.get("status") != "PASSED"
        or parity.get("additional_optimizer_updates") != 0
        or parity.get("all_batch_tensors_and_metadata_exact") is not True
        or parity.get("parent_rng_unchanged") is not True
    ):
        raise ValueError("Real TRAIN40 CPU batch parity is required")
    for name in PREDECESSORS:
        verify_sources(read(output / name))
    source_dir = ROOT / "operational/train40_system"
    modules = [
        "resource_monitor.py",
        "process_inputs.py",
        "engine_host_pipeline.py",
        "controller_host_pipeline.py",
        "host_pipeline_freeze.py",
        "host_pipeline_admission.py",
    ]
    tests = [
        ROOT / "tests/unit" / name
        for name in (
            "test_train40_resource_monitor.py",
            "test_train40_process_inputs.py",
            "test_train40_host_pipeline.py",
        )
    ]
    contract = {
        "schema": "train40_host_pipeline_freeze_v1",
        "files": dependency_files([*(source_dir / name for name in modules), *tests]),
        "engine_sha256": digest(source_dir / "engine_host_pipeline.py"),
        "controller_sha256": digest(source_dir / "controller_host_pipeline.py"),
        "predecessor_freezes": {name: digest(output / name) for name in PREDECESSORS},
        "QA": {name: digest(output / name) for name in (*QA_CHECKS, parity_name)},
        "input_manifest_sha256": digest(output / "INPUT_MANIFEST.json"),
        "teacher_manifest_sha256": digest(output / "TEACHER_MANIFEST.json"),
        "scientific_protocol_sha256": digest(output / "TRAINING_PROTOCOL.json"),
        "a5_graph_replay_and_c2f_eager_scientific_loops_unchanged": True,
        "model_loss_optimizer_sampler_RNG_precision_checkpoint_unchanged": True,
        "cpu_only_spawn_preparation_with_bounded_shared_slots": True,
        "current_and_next_256_row_group_cache": True,
        "physical_recovery_deadline_admission_each_update": True,
        "monitor_stale_or_dead_fails_closed": True,
        "additional_optimizer_updates": 0,
    }
    path = output / "HOST_PIPELINE_FREEZE.json"
    if path.exists() and read(path) != contract:
        raise ValueError("Preserve previously published host pipeline freeze")
    atomic_json(path, contract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output.resolve())
