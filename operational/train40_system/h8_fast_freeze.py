"""Seal executed H8 parity and system QA while retaining every historic contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.freeze import dependency_files


def run(output: Path, mode: str, *, shared_memory: bool) -> None:
    """Freeze the selected execution only after all required real-data checks pass."""
    baseline = read(output / "FEATURES_EIGHT_WORKER_FREEZE.json")
    verify_sources(baseline)
    admission_directory = output / "h8_fast_admission"
    candidate = admission_directory / f"H8_FAST_{mode.upper()}_ADMISSION.json"
    report = read(candidate)
    directory = ROOT / "operational/train40_system"
    for field, source in (
        ("source_sha256", "h8_fast_admission.py"),
        ("extractor_source_sha256", "h8_fast_extract.py"),
    ):
        if report[field] != digest(directory / source):
            raise ValueError("Admission source changed: " + source)
    if report["status"] != "PASSED" or not report["all_16_features_exact"]:
        raise ValueError("Real FP32 H8 feature parity is not admitted")
    if report["stored_fragments_checked"] < 2 or not report["model_state_unchanged"]:
        raise ValueError("Historic fragment and frozen-weight parity required")
    reports = {candidate.name: digest(candidate)}
    if shared_memory:
        raw = admission_directory / "SHARED_RAW_ADMISSION.json"
        raw_report = read(raw)
        if raw_report["status"] != "PASSED":
            raise ValueError("Real shared-buffer raw parity is not admitted")
        for field, source in (
            ("source_sha256", "h8_shared_admission.py"),
            ("pool_source_sha256", "h8_shared_pool.py"),
        ):
            if raw_report[field] != digest(directory / source):
                raise ValueError("Shared-buffer admission source changed: " + source)
        reports[raw.name] = digest(raw)
    for name, marker in (
        ("H8_FAST_PYTEST.txt", "[100%]"),
        ("H8_FAST_RUFF.txt", "All checks passed"),
        ("H8_FAST_PYRIGHT.txt", "0 errors"),
        ("H8_FAST_SPAWN.txt", "PASSED"),
    ):
        path = output / name
        if marker not in path.read_text(encoding="utf-8"):
            raise ValueError("H8 execution QA failed: " + name)
        reports[name] = digest(path)
    atomic_json(
        admission_directory / "ADMISSION.json",
        {
            "status": "PASSED",
            "mode": mode,
            "shared_memory": shared_memory,
            "reports": reports,
            "baseline_binding_sha256": digest(output / "h8_feature_fragments/BINDING.json"),
            "optimizer_updates": 0,
            "original_receipts_and_parents_preserved": True,
        },
    )
    seeds = [
        directory / name
        for name in (
            "history_resources8_fast.py",
            "controller_h8_fast.py",
            "hourly_supervisor_h8.py",
            "h8_fast_extract.py",
            "h8_fast_admission.py",
            "h8_shared_pool.py",
            "h8_shared_admission.py",
            "h8_fast_freeze.py",
        )
    ]
    backend = read(output / "C2F_GRAPH_REPLAY_FREEZE.json")
    freeze = {
        "schema": "train40_h8_exact_features_accelerated_execution_v1",
        "files": dependency_files(seeds),
        "mode": mode,
        "shared_memory": shared_memory,
        "entrypoint_sha256": digest(directory / "history_resources8_fast.py"),
        "controller_sha256": digest(directory / "controller_h8_fast.py"),
        "protocol_sha256": digest(output / "TRAINING_PROTOCOL.json"),
        "baseline_feature_freeze_sha256": digest(output / "FEATURES_EIGHT_WORKER_FREEZE.json"),
        "baseline_binding_sha256": digest(output / "h8_feature_fragments/BINDING.json"),
        "admission_sha256": digest(admission_directory / "ADMISSION.json"),
        "backend_source": backend["backend_source"],
        "backend_source_sha256": backend["backend_source_sha256"],
        "precision": "float32",
        "batch_shape": [16, 3, 12, 128, 128],
        "history_slots": 8,
        "producer_execution_order": ["A5", "PAIR", "C2F"],
        "raw_workers_max": 8,
        "pending_raw_queries_max": 8,
        "progress_publication_max_interval_seconds": 5,
        "fragment_receipts_durable_each_row": True,
        "new_receipts_bind_execution_freeze": True,
        "original_feature_binding_preserved": True,
        "original_fragment_data_preserved": True,
        "optimizer_updates": 0,
        "QA": reports,
    }
    if digest(Path(freeze["backend_source"])) != freeze["backend_source_sha256"]:
        raise ValueError("Installed CUDA graph backend changed")
    path = output / "H8_FAST_FREEZE.json"
    if path.exists() and read(path) != freeze:
        raise ValueError("Preserve existing H8 execution freeze")
    atomic_json(path, freeze)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/train40_system_20261005")
    parser.add_argument("--mode", choices=("packed", "graph"), required=True)
    parser.add_argument("--shared-memory", action="store_true")
    args = parser.parse_args()
    run(args.output.resolve(), args.mode, shared_memory=args.shared_memory)
