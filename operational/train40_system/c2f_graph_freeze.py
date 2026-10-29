"""Freeze C2F CUDA-graph replay after real zero-update parity admission."""

from __future__ import annotations

import argparse
from pathlib import Path

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.freeze import dependency_files

QA_CHECKS = {
    "C2F_GRAPH_PYTEST.txt": "[100%]",
    "C2F_GRAPH_RUFF.txt": "All checks passed",
    "C2F_GRAPH_PYRIGHT.txt": "0 errors",
}

REQUIRED_PARITY = (
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


def run(output: Path) -> None:
    """Bind replay code, predecessors, QA, and exact real-CUDA evidence."""
    for name, marker in QA_CHECKS.items():
        if marker not in (output / name).read_text(encoding="utf-8"):
            raise ValueError(f"C2F graph QA failed: {name}")

    dependencies = {
        "coordination_freeze_sha256": output / "COORDINATION_FREEZE.json",
        "controller_pause_safe_freeze_sha256": output / "CONTROLLER_PAUSE_SAFE_FREEZE.json",
        "adaptive_cache_freeze_sha256": output / "ADAPTIVE_CACHE_FREEZE.json",
        "fast_process_scan_freeze_sha256": output / "FAST_PROCESS_SCAN_FREEZE.json",
        "host_fast_4_freeze_sha256": output / "HOST_FAST_4_FREEZE.json",
    }
    for path in dependencies.values():
        verify_sources(read(path))

    admission_path = output / "c2f_graph_launch_admission/REAL_GRAPH_LAUNCH_ADMISSION.json"
    admission = read(admission_path)
    backend_source = Path(str(admission.get("backend_source", "")))
    peak_vram = admission.get("peak_reserved_vram_bytes")
    vram_admissible = (
        isinstance(peak_vram, dict)
        and set(peak_vram) == {"eager_after_graph_capture", "compiled_replay"}
        and all(isinstance(value, int) for value in peak_vram.values())
        and max(peak_vram.values(), default=13 * 1024**3) < 12 * 1024**3
    )
    if (
        admission.get("status") != "PASSED"
        or admission.get("compatible") is not True
        or admission.get("arm") != "c2f"
        or not all(admission.get(name) is True for name in REQUIRED_PARITY)
        or int(admission.get("gradient_tensors_compared", 0)) != 63
        or int(admission.get("checkpoint_updates", 0)) < 13_817
        or admission.get("optimizer_updates") != 0
        or admission.get("training_checkpoint_modified") is not False
        or admission.get("steady_recompiles") != 0
        or admission.get("backend") != "cudagraphs"
        or not backend_source.is_file()
        or digest(backend_source) != admission.get("backend_source_sha256")
        or not vram_admissible
    ):
        raise ValueError("Exact zero-update C2F graph admission must pass")
    launches = admission.get("launch_profiles", {}).get("compiled", {})
    if not any("GraphLaunch" in name for name in launches):
        raise ValueError("C2F admission did not observe CUDA graph replay")

    source_dir = ROOT / "operational/train40_system"
    admission_source = source_dir / "c2f_graph_admission.py"
    if admission.get("source_sha256") != digest(admission_source):
        raise ValueError("Executed C2F admission source changed")
    sources = [
        source_dir / "c2f_graph_core.py",
        source_dir / "engine_c2f_graph_replay.py",
        source_dir / "controller_c2f_graph.py",
        source_dir / "c2f_graph_freeze.py",
        admission_source,
        source_dir / "process_inputs_fast_4.py",
        source_dir / "resource_monitor.py",
        ROOT / "tests/unit/test_train40_c2f_graph_replay.py",
        ROOT / "tests/unit/test_train40_c2f_graph_controller.py",
        ROOT / "tests/unit/test_train40_c2f_graph_admission.py",
    ]
    contract = {
        "schema": "train40_c2f_graph_replay_freeze_v1",
        "files": dependency_files(sources),
        "engine_sha256": digest(source_dir / "c2f_graph_core.py"),
        "wrapper_engine_sha256": digest(source_dir / "engine_c2f_graph_replay.py"),
        "controller_sha256": digest(source_dir / "controller_c2f_graph.py"),
        **{field: digest(path) for field, path in dependencies.items()},
        "real_graph_launch_admission_sha256": digest(admission_path),
        "real_admission_checkpoint_sha256": admission["checkpoint_sha256"],
        "backend_source_sha256": admission["backend_source_sha256"],
        "backend_source": str(backend_source.resolve()),
        "gradient_parity_admission": admission["repeated_gradient_admission"],
        "steady_eager_seconds": admission["steady_eager_seconds"],
        "steady_compiled_seconds": admission["steady_compiled_seconds"],
        "launch_profiles": admission["launch_profiles"],
        "peak_reserved_vram_bytes": admission["peak_reserved_vram_bytes"],
        "QA": {name: digest(output / name) for name in QA_CHECKS},
        "c2f_only_graph_replay": True,
        "requires_existing_post_foreground_warmup_checkpoint": True,
        "warmup_forward_backward_clip_calls": 3,
        "warmup_optimizer_updates": 0,
        "full_checkpoint_state_restored_before_prefetch": True,
        "checkpoint_contract_unchanged": True,
        "b32_cudagraph_replay": True,
        "b8_original_eager_fallback": True,
        "process_inputs_fast_4_and_planned_prefetch": True,
        "bounded_single_arm_inventory_repair": True,
        "performance_claim_limited_to_zero_update_admission": True,
        "additional_optimizer_updates": 0,
    }
    path = output / "C2F_GRAPH_REPLAY_FREEZE.json"
    if path.exists() and read(path) != contract:
        raise ValueError("Preserve existing C2F graph replay freeze")
    atomic_json(path, contract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    run(arguments.output.resolve())
