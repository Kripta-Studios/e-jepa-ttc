"""Publish the overlap trainer only after exact real and CPU coordination admission."""

from __future__ import annotations

import argparse
from pathlib import Path

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.freeze import dependency_files

QA_CHECKS = {
    "COORDINATION_PYTEST.txt": "[100%]",
    "COORDINATION_RUFF.txt": "All checks passed",
    "COORDINATION_PYRIGHT.txt": "0 errors",
}


def _real_admission(output: Path) -> tuple[Path, dict]:
    candidates = (
        output / "REAL_COORDINATION_ADMISSION.json",
        output / "coordination_admission/REAL_COORDINATION_ADMISSION.json",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    admission = read(path)
    required_true = (
        "exact_loss_and_all_components",
        "all_gradient_keys_shapes_and_finite_audited",
        "gradient_parity_within_baseline_repeatability",
        "prefetch_exact_all_tensor_fields_and_metadata",
        "prefetch_preserves_CPU_and_CUDA_RNG",
    )
    if admission.get("status") != "PASSED" or not all(
        admission.get(name) is True for name in required_true
    ):
        raise ValueError("Exact real coordination admission must pass")
    if admission.get("optimizer_updates") != 0:
        raise ValueError("Coordination admission must execute zero optimizer updates")
    if not isinstance(
        admission.get("exact_loss_all_components_and_all_parameter_gradients"), bool
    ):
        raise ValueError("Gradient bit equality must be reported honestly")
    repeated = admission.get("repeated_gradient_admission")
    if not isinstance(repeated, dict) or not repeated:
        raise ValueError("Repeated original/optimized gradient controls are required")
    profile = ROOT / "operational/train40_system/profile_coordination.py"
    if admission.get("source_sha256") != digest(profile):
        raise ValueError("Executed real coordination admission source changed")
    return path, admission


def run(output: Path) -> None:
    """Bind the reviewed variant while retaining the original scientific checkpoint contract."""
    for name, marker in QA_CHECKS.items():
        if marker not in (output / name).read_text(encoding="utf-8"):
            raise ValueError(f"Required coordination QA failed: {name}")
    admission_path, admission = _real_admission(output)
    models_path = output / "MODELS_FREEZE.json"
    models = read(models_path)
    verify_sources(models)
    original_engine = ROOT / "operational/train40_system/engine.py"
    overlap_engine = ROOT / "operational/train40_system/engine_overlap.py"
    if models["trainer_sha256"] != digest(original_engine):
        raise ValueError("Original tested TRAIN40 engine changed")
    if models["protocol_sha256"] != digest(output / "TRAINING_PROTOCOL.json"):
        raise ValueError("Original scientific protocol changed")
    engineering_path = output / "ENGINEERING_FREEZE.json"
    if models["engineering_freeze_sha256"] != digest(engineering_path):
        raise ValueError("Original engineering baseline changed")
    seeds = [
        ROOT / "operational/train40_system" / name
        for name in (
            "coordination.py",
            "engine_overlap.py",
            "controller_overlap.py",
            "coordination_freeze.py",
            "profile_coordination.py",
        )
    ]
    contract = {
        "schema": "train40_coordination_overlap_freeze_v1",
        "files": dependency_files(seeds),
        "trainer_sha256": digest(overlap_engine),
        "controller_sha256": digest(
            ROOT / "operational/train40_system/controller_overlap.py"
        ),
        "coordination_sha256": digest(
            ROOT / "operational/train40_system/coordination.py"
        ),
        "original_engine_sha256": digest(original_engine),
        "models_freeze_sha256": digest(models_path),
        "engineering_freeze_sha256": digest(engineering_path),
        "protocol_sha256": digest(output / "TRAINING_PROTOCOL.json"),
        "QA": {name: digest(output / name) for name in QA_CHECKS},
        "real_coordination_admission_sha256": digest(admission_path),
        "real_coordination_checkpoint_sha256": admission["checkpoint_sha256"],
        "real_coordination_checkpoint_full_state_sha256": admission[
            "checkpoint_full_state_sha256"
        ],
        "gradient_bit_equality_observed": admission[
            "exact_loss_all_components_and_all_parameter_gradients"
        ],
        "gradient_parity_admission": admission["repeated_gradient_admission"],
        "optimizer_updates_during_admission": 0,
        "checkpoint_contract_unchanged": True,
        "old_models_freeze_remains_checkpoint_engineering_freeze": True,
        "differences": {
            "relational_diagnostic": "vectorized floor/ceil half-open bbox mask",
            "prefetch": "one collator, pinned host tensors, one copy stream, depth 3",
            "scalar_metrics": "one ordered FP32 CPU transfer",
            "synchronization": "current compute stream only",
            "pause_guard": ["COORDINATION_PAUSE_REQUEST.json", "STOP_REQUEST"],
            "runtime_provenance_sidecar": "COORDINATION_RUNTIME.json",
        },
        "pinned_prefetch_depth": 3,
        "new_scientific_optimizer_updates_before_publication": 0,
        "scientific_protocol_and_model_freeze_preserved": True,
    }
    path = output / "COORDINATION_FREEZE.json"
    if path.exists() and read(path) != contract:
        raise ValueError("Preserve existing coordination freeze")
    atomic_json(path, contract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output.resolve())
