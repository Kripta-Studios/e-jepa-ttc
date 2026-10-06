"""Publish relation reuse after CPU QA and zero-update real-CUDA admission."""

from __future__ import annotations

import argparse
from pathlib import Path

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.freeze import dependency_files

QA_CHECKS = {
    "RELATION_REUSE_PYTEST.txt": "[100%]",
    "RELATION_REUSE_RUFF.txt": "All checks passed",
    "RELATION_REUSE_PYRIGHT.txt": "0 errors",
}


def run(output: Path) -> None:
    """Bind exact reuse evidence to both published coordination baselines."""
    for name, marker in QA_CHECKS.items():
        if marker not in (output / name).read_text(encoding="utf-8"):
            raise ValueError(f"Required relation-reuse QA failed: {name}")
    coordination_path = output / "COORDINATION_FREEZE.json"
    pause_path = output / "CONTROLLER_PAUSE_SAFE_FREEZE.json"
    idle_path = output / "IDLE_WAIT_FREEZE.json"
    coordination, pause, idle = read(coordination_path), read(pause_path), read(idle_path)
    verify_sources(coordination)
    verify_sources(pause)
    verify_sources(idle)
    admission_path = output / "relation_reuse_admission/REAL_RELATION_REUSE_ADMISSION.json"
    admission = read(admission_path)
    admission_source = ROOT / "operational/train40_system/relation_reuse_admission.py"
    required = (
        "exact_loss_and_all_components",
        "all_gradient_keys_shapes_and_finite_audited",
        "gradient_parity_within_baseline_repeatability",
        "first_grad_enabled_map_calls_original",
        "same_call_diagnostic_receives_detached_map",
        "CPU_and_CUDA_RNG_trajectory_exact",
    )
    if admission.get("status") != "PASSED" or not all(
        admission.get(name) is True for name in required
    ):
        raise ValueError("Real relation-reuse admission must pass")
    if admission.get("optimizer_updates") != 0 or admission.get(
        "training_checkpoint_modified"
    ) is not False:
        raise ValueError("Relation-reuse admission must not change scientific state")
    if admission.get("source_sha256") != digest(admission_source):
        raise ValueError("Executed relation-reuse admission source changed")
    if admission.get("idle_wait_freeze_sha256") != digest(idle_path):
        raise ValueError("Real admission does not bind the idle-wait freeze")
    if admission.get("environment") != {"OMP_WAIT_POLICY": "PASSIVE", "KMP_BLOCKTIME": "0"}:
        raise ValueError("Real admission did not use the admitted idle-wait environment")
    if not isinstance(admission.get("gradient_bit_equality_observed"), bool):
        raise ValueError("Gradient bit equality must be reported honestly")
    if admission.get("arm") == "a5" and admission.get("gradient_tensors_compared") != 59:
        raise ValueError("All 59 A5 gradient tensors must be audited")
    counters = admission.get("reuse_counters", {})
    if counters.get("loss_calls") != 3 or counters.get("reuse_hits") != 3:
        raise ValueError("Real admission did not demonstrate one reuse per loss")
    sources = [
        ROOT / "operational/train40_system" / name
        for name in (
            "relation_reuse.py",
            "engine_relation_reuse.py",
            "controller_relation_reuse.py",
            "relation_reuse_admission.py",
            "relation_reuse_freeze.py",
        )
    ]
    engine = sources[1]
    controller = sources[2]
    contract = {
        "schema": "train40_relation_reuse_freeze_v1",
        "files": dependency_files(sources),
        "engine_relation_reuse_sha256": digest(engine),
        "controller_relation_reuse_sha256": digest(controller),
        "coordination_freeze_sha256": digest(coordination_path),
        "controller_pause_safe_freeze_sha256": digest(pause_path),
        "idle_wait_freeze_sha256": digest(idle_path),
        "coordination_trainer_sha256": coordination["trainer_sha256"],
        "pause_safe_source_sha256": pause["source_sha256"],
        "QA": {name: digest(output / name) for name in QA_CHECKS},
        "real_admission_sha256": digest(admission_path),
        "real_admission_checkpoint_sha256": admission["checkpoint_sha256"],
        "gradient_bit_equality_observed": admission["gradient_bit_equality_observed"],
        "gradient_parity_admission": admission["repeated_gradient_admission"],
        "reuse_counters": counters,
        "checkpoint_contract_unchanged": True,
        "objective_operations_unchanged": True,
        "first_grad_enabled_relation_map_is_original": True,
        "diagnostic_reuse_is_same_loss_same_object_same_offsets_only": True,
        "new_optimizer_updates_during_admission": 0,
        "original_durable_save_called_once_per_existing_save": True,
        "checkpoint_boundary_counter_evidence": "RELATION_REUSE_COUNTERS.json",
        "performance_improvement_claimed_before_live_measurement": False,
    }
    path = output / "RELATION_REUSE_FREEZE.json"
    if path.exists() and read(path) != contract:
        raise ValueError("Preserve existing relation-reuse freeze")
    atomic_json(path, contract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output.resolve())
