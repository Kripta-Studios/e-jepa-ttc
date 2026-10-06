"""Publish cache-only admission without altering scientific or historical freezes."""

from __future__ import annotations

import argparse
from pathlib import Path

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.freeze import dependency_files

QA_CHECKS = {
    "ADAPTIVE_CACHE_PYTEST.txt": "[100%]",
    "ADAPTIVE_CACHE_RUFF.txt": "All checks passed",
    "ADAPTIVE_CACHE_PYRIGHT.txt": "0 errors",
}


def run(output: Path) -> None:
    """Require executed input parity and QA before routing a scientific fit."""
    for name, marker in QA_CHECKS.items():
        if marker not in (output / name).read_text(encoding="utf-8"):
            raise ValueError(f"Required adaptive cache QA failed: {name}")
    for name in ("COORDINATION_FREEZE.json", "CONTROLLER_PAUSE_SAFE_FREEZE.json"):
        verify_sources(read(output / name))
    admission_path = output / "REAL_ADAPTIVE_CACHE_ADMISSION.json"
    admission = read(admission_path)
    source_dir = ROOT / "operational/train40_system"
    if (
        admission.get("status") != "PASSED"
        or admission.get("exact_all_tensor_bytes_and_metadata") is not True
        or admission.get("CPU_RNG_unchanged") is not True
        or admission.get("optimizer_updates") != 0
        or admission.get("training_checkpoint_modified") is not False
        or admission.get("cuda_initialized") is not False
        or admission.get("batch_sizes") != [32, 8]
    ):
        raise ValueError("Exact CPU-only real input admission must pass")
    for key, path in (
        ("source_sha256", source_dir / "adaptive_cache_admission.py"),
        ("cache_source_sha256", source_dir / "adaptive_cache.py"),
        ("coordination_freeze_sha256", output / "COORDINATION_FREEZE.json"),
        ("input_manifest_sha256", output / "INPUT_MANIFEST.json"),
        ("teacher_manifest_sha256", output / "TEACHER_MANIFEST.json"),
    ):
        if admission.get(key) != digest(path):
            raise ValueError(f"Executed cache admission differs: {key}")
    sources = [
        source_dir / name
        for name in (
            "adaptive_cache.py",
            "engine_adaptive_cache.py",
            "controller_adaptive_cache.py",
            "adaptive_cache_admission.py",
            "adaptive_cache_freeze.py",
        )
    ]
    sources.extend(
        ROOT / "tests/unit" / name
        for name in (
            "test_train40_adaptive_cache.py",
            "test_train40_adaptive_cache_execution.py",
        )
    )
    contract = {
        "schema": "train40_adaptive_cache_freeze_v1",
        "files": dependency_files(sources),
        "engine_sha256": digest(source_dir / "engine_adaptive_cache.py"),
        "coordination_freeze_sha256": digest(output / "COORDINATION_FREEZE.json"),
        "controller_pause_safe_freeze_sha256": digest(output / "CONTROLLER_PAUSE_SAFE_FREEZE.json"),
        "real_input_admission_sha256": digest(admission_path),
        "QA": {name: digest(output / name) for name in QA_CHECKS},
        "checkpoint_contract_unchanged": True,
        "model_loss_optimizer_sampler_precision_and_prefetch_unchanged": True,
        "original_resource_guard_16GB_tree_RSS_and_2GiB_host_reserve_unchanged": True,
        "optimizer_updates_during_admission": 0,
        "performance_improvement_claimed_before_live_measurement": False,
    }
    path = output / "ADAPTIVE_CACHE_FREEZE.json"
    if path.exists() and read(path) != contract:
        raise ValueError("Preserve existing adaptive cache freeze")
    atomic_json(path, contract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output.resolve())
