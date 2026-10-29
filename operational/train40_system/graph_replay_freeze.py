"""Publish A5 CUDA-graph replay after exact zero-update real-CUDA admission."""

from __future__ import annotations

import argparse
from pathlib import Path

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.freeze import dependency_files

QA_CHECKS = {
    "GRAPH_REPLAY_PYTEST.txt": "[100%]",
    "GRAPH_REPLAY_RUFF.txt": "All checks passed",
    "GRAPH_REPLAY_PYRIGHT.txt": "0 errors",
}


def run(output: Path) -> None:
    """Bind production replay sources to exact parity and launch evidence."""
    for name, marker in QA_CHECKS.items():
        if marker not in (output / name).read_text(encoding="utf-8"):
            raise ValueError(f"Graph replay QA failed: {name}")
    dependencies = {
        "coordination_freeze_sha256": output / "COORDINATION_FREEZE.json",
        "controller_pause_safe_freeze_sha256": output / "CONTROLLER_PAUSE_SAFE_FREEZE.json",
        "adaptive_cache_freeze_sha256": output / "ADAPTIVE_CACHE_FREEZE.json",
        "fast_process_scan_freeze_sha256": output / "FAST_PROCESS_SCAN_FREEZE.json",
    }
    for path in dependencies.values():
        verify_sources(read(path))

    admission_path = output / "graph_launch_admission/REAL_GRAPH_LAUNCH_ADMISSION.json"
    admission = read(admission_path)
    required = (
        "exact_loss_and_all_components",
        "exact_all_model_output_tensors",
        "all_gradient_keys_shapes_and_finite_audited",
        "gradient_parity_within_baseline_repeatability",
        "repeated_dropout_CPU_and_CUDA_RNG_trajectory_exact",
        "B32_B8_B32_output_loss_components_and_RNG_trajectory_exact",
        "batch8_uses_original_eager_forward",
        "model_parameter_identity_and_state_dict_keys_unchanged",
        "original_shape_and_delta_t_guards_retained",
    )
    if (
        admission.get("status") != "PASSED"
        or admission.get("compatible") is not True
        or not all(admission.get(name) is True for name in required)
        or admission.get("arm") != "a5"
        or admission.get("gradient_tensors_compared") != 59
        or admission.get("checkpoint_updates", 0) < 8322
        or admission.get("optimizer_updates") != 0
        or admission.get("training_checkpoint_modified") is not False
        or admission.get("steady_recompiles") != 0
    ):
        raise ValueError("Exact zero-update A5 graph admission must pass")
    launch_profiles = admission.get("launch_profiles", {})
    compiled_launches = launch_profiles.get("compiled", {})
    if not any("GraphLaunch" in name for name in compiled_launches):
        raise ValueError("Real admission did not observe CUDA graph replay")
    admission_source = ROOT / "operational/train40_system/graph_launch_admission.py"
    if admission.get("source_sha256") != digest(admission_source):
        raise ValueError("Executed graph admission source changed")
    if admission.get("backend") != "cudagraphs":
        raise ValueError("Only the installed cudagraphs backend was admitted")

    source_dir = ROOT / "operational/train40_system"
    sources = [
        source_dir / name
        for name in (
            "engine_graph_replay.py",
            "controller_graph_replay.py",
            "graph_replay_freeze.py",
            "graph_launch_admission.py",
            "engine_fast_scan_safe.py",
        )
    ]
    sources.extend(
        ROOT / "tests/unit" / name
        for name in ("test_train40_graph_replay.py", "test_train40_fast_scan_safe.py")
    )
    contract = {
        "schema": "train40_graph_replay_freeze_v1",
        "files": dependency_files(sources),
        "engine_sha256": digest(source_dir / "engine_graph_replay.py"),
        "controller_sha256": digest(source_dir / "controller_graph_replay.py"),
        "fast_scan_safe_engine_sha256": digest(source_dir / "engine_fast_scan_safe.py"),
        **{field: digest(path) for field, path in dependencies.items()},
        "real_admission_sha256": digest(admission_path),
        "real_admission_checkpoint_sha256": admission["checkpoint_sha256"],
        "backend_source_sha256": admission["backend_source_sha256"],
        "gradient_parity_admission": admission["repeated_gradient_admission"],
        "steady_eager_seconds": admission["steady_eager_seconds"],
        "steady_compiled_seconds": admission["steady_compiled_seconds"],
        "first_compiled_forward_backward_seconds": admission[
            "first_compiled_forward_backward_seconds"
        ],
        "launch_profiles": launch_profiles,
        "peak_reserved_vram_bytes": admission["peak_reserved_vram_bytes"],
        "QA": {name: digest(output / name) for name in QA_CHECKS},
        "a5_only_graph_replay": True,
        "requires_existing_post_foreground_warmup_checkpoint": True,
        "c2f_uses_fast_process_scan_without_graph_replay": True,
        "warmup_forward_backward_clip_calls": 3,
        "warmup_optimizer_updates": 0,
        "full_checkpoint_state_restored_before_prefetch": True,
        "checkpoint_contract_unchanged": True,
        "b8_original_eager_fallback": True,
        "performance_claim_limited_to_zero_update_admission": True,
    }
    path = output / "GRAPH_REPLAY_FREEZE.json"
    if path.exists() and read(path) != contract:
        raise ValueError("Preserve existing graph replay freeze")
    atomic_json(path, contract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    run(arguments.output.resolve())
