"""Seal exact H8 transport replay evidence without changing canonical science."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.freeze import dependency_files

_TRUE_GATES = (
    "compatible",
    "public_TRAIN40_only",
    "canonical_all_16_exact",
    "historic_masked_last_8_exact",
    "exact_bytes_A_B_A",
    "retained_features_unchanged",
    "model_state_unchanged",
    "CPU_CUDA_RNG_unchanged",
    "aliases_restored",
    "alternating_same_inputs",
    "steady_graph_cache_no_new_captures",
)


def _require_qa(output: Path) -> dict[str, str]:
    receipts: dict[str, str] = {}
    for name in ("H8_TRANSPORT_PYTEST.txt", "H8_TRANSPORT_RUFF.txt", "H8_TRANSPORT_PYRIGHT.txt"):
        path = output / name
        text = path.read_text(encoding="utf-8")
        passed = (
            "[100%]" in text
            and re.search(r"\b\d+ passed\b", text) is not None
            and re.search(
                r"\b\d+ (?:failed|errors?)\b|\b(?:ERRORS|FAILURES)\b",
                text,
                re.IGNORECASE,
            )
            is None
            if name.endswith("PYTEST.txt")
            else (
                "All checks passed" in text
                if name.endswith("RUFF.txt")
                else "0 errors, 0 warnings" in text
            )
        )
        if not passed:
            raise ValueError("H8 transport QA failed: " + name)
        receipts[name] = digest(path)
    return receipts


def run(output: Path) -> None:
    """Publish a new freeze only after real exact-byte inference admission."""

    base_path = output / "H8_FAST_FREEZE.json"
    base = read(base_path)
    verify_sources(base)
    admission_path = output / "h8_transport_admission/H8_TRANSPORT_ADMISSION.json"
    admission = read(admission_path)
    if admission["status"] != "PASSED" or any(not admission[field] for field in _TRUE_GATES):
        raise ValueError("Exact H8 transport parity and lifecycle admission required")
    if admission["optimizer_updates"] != 0 or admission["targets_read"]:
        raise ValueError("Transport admission must be inference-only and label-blind")
    if int(admission["peak_reserved_bytes"]) >= 3 * 1024**3:
        raise ValueError("Transport replay exceeded its admitted 3 GiB VRAM bound")

    bounded_path = output / "h8_transport_admission/H8_BOUNDED_RAW_ADMISSION.json"
    bounded = read(bounded_path)
    if (
        bounded["status"] != "PASSED"
        or not bounded["all_tensors_exact_bytes"]
        or not bounded["rng_unchanged"]
        or bounded["optimizer_updates"] != 0
        or bounded["targets_read"]
        or int(bounded["oversized_union_cases"]) < 2
        or int(bounded["retained_bytes_max"]) > 256 * 1024**2
    ):
        raise ValueError("Exact bounded H8 raw preparation admission required")

    directory = ROOT / "operational/train40_system"
    for field, source in (
        ("source_sha256", "h8_transport_admission.py"),
        ("graph_source_sha256", "h8_transport_graph.py"),
        ("extractor_source_sha256", "h8_fast_extract.py"),
        (
            "canonical_adapter_source_sha256",
            "../simplex_t_shared_route/adapter.py",
        ),
        ("history_source_sha256", "history_features.py"),
    ):
        path = (directory / source).resolve()
        if admission[field] != digest(path):
            raise ValueError("H8 transport admission source changed: " + source)
    for field, source in (
        ("source_sha256", "h8_bounded_admission.py"),
        ("candidate_source_sha256", "h8_bounded_union.py"),
        ("canonical_source_sha256", "history_resources8.py"),
    ):
        if bounded[field] != digest(directory / source):
            raise ValueError("Bounded H8 raw admission source changed: " + source)

    qa = _require_qa(output)
    seeds = [
        directory / name
        for name in (
            "h8_transport_graph.py",
            "history_resources8_transport_graph.py",
            "controller_transport_graph.py",
            "hourly_supervisor_transport.py",
            "h8_transport_freeze.py",
            "h8_transport_admission.py",
            "h8_bounded_union.py",
            "h8_bounded_admission.py",
            "history_resources8_fast.py",
            "controller_h8_fast.py",
            "h8_fast_extract.py",
        )
    ]
    test = ROOT / "tests/unit/test_train40_h8_transport_graph.py"
    seeds.append(test)
    seeds.append(ROOT / "tests/unit/test_train40_h8_bounded_union.py")
    freeze: dict[str, Any] = {
        "schema": "train40_h8_transport_cuda_graph_execution_v1",
        "files": dependency_files(seeds),
        "entrypoint_sha256": digest(directory / "history_resources8_transport_graph.py"),
        "controller_sha256": digest(directory / "controller_transport_graph.py"),
        "supervisor_sha256": digest(directory / "hourly_supervisor_transport.py"),
        "transport_graph_sha256": digest(directory / "h8_transport_graph.py"),
        "base_freeze_sha256": digest(base_path),
        "baseline_binding_sha256": digest(output / "h8_feature_fragments/BINDING.json"),
        "protocol_sha256": digest(output / "TRAINING_PROTOCOL.json"),
        "admission_sha256": digest(admission_path),
        "bounded_raw_admission_sha256": digest(bounded_path),
        "bounded_raw_source_sha256": digest(directory / "h8_bounded_union.py"),
        "bounded_raw_retained_bytes_max": int(bounded["retained_bytes_max"]),
        "bounded_raw_rows": bounded["rows"],
        "bounded_raw_all_tensors_exact_bytes": True,
        "bounded_raw_oversized_union_cases": int(bounded["oversized_union_cases"]),
        "graph_cache_entries": 8,
        "graph_warmup_calls": 3,
        "precision": "float32",
        "inference_only": True,
        "optimizer_updates": 0,
        "canonical_functions_unchanged": True,
        "outputs_cloned_for_independent_lifetime": True,
        "owner_thread_and_stream_confined": True,
        "new_fragment_receipts_bind_transport_execution_freeze": True,
        "base_execution_binding_preserved": True,
        "admitted_transport_signatures": [
            {"shape": [32, 64, 32, 32], "radius": 1},
            {"shape": [32, 64, 16, 16], "radius": 2},
        ],
        "admitted_median_end_to_end_speed_ratio": admission["median_end_to_end_speed_ratio"],
        "speed_is_recorded_not_a_scientific_gate": True,
        "QA": qa,
    }
    destination = output / "H8_TRANSPORT_FREEZE.json"
    if destination.exists() and read(destination) != freeze:
        raise ValueError("Preserve existing H8 transport freeze")
    atomic_json(destination, freeze)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/train40_system_20261005")
    args = parser.parse_args()
    run(args.output.resolve())
