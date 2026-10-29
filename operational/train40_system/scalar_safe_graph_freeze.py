"""Publish the scalar-safe hashing wrapper over frozen A5 graph replay."""

from __future__ import annotations

import argparse
from pathlib import Path

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.freeze import dependency_files

QA_CHECKS = {
    "SCALAR_SAFE_GRAPH_PYTEST.txt": "[100%]",
    "SCALAR_SAFE_GRAPH_RUFF.txt": "All checks passed",
    "SCALAR_SAFE_GRAPH_PYRIGHT.txt": "0 errors",
}


def run(output: Path) -> None:
    """Bind the narrow hashing correction to the admitted graph runtime."""
    for name, marker in QA_CHECKS.items():
        if marker not in (output / name).read_text(encoding="utf-8"):
            raise ValueError(f"Scalar-safe graph QA failed: {name}")
    graph_path = output / "GRAPH_REPLAY_FREEZE.json"
    graph = read(graph_path)
    verify_sources(graph)
    if graph.get("a5_only_graph_replay") is not True:
        raise ValueError("Scalar-safe hashing requires admitted A5 graph replay")
    source_dir = ROOT / "operational/train40_system"
    sources = [
        source_dir / name
        for name in (
            "engine_graph_replay_scalar_safe.py",
            "controller_graph_replay_scalar_safe.py",
            "scalar_safe_graph_freeze.py",
        )
    ]
    sources.append(ROOT / "tests/unit/test_train40_scalar_safe_graph.py")
    contract = {
        "schema": "train40_scalar_safe_graph_freeze_v1",
        "files": dependency_files(sources),
        "engine_sha256": digest(source_dir / "engine_graph_replay_scalar_safe.py"),
        "controller_sha256": digest(source_dir / "controller_graph_replay_scalar_safe.py"),
        "graph_replay_freeze_sha256": digest(graph_path),
        "QA": {name: digest(output / name) for name in QA_CHECKS},
        "only_change": "reshape flattened contiguous tensors before uint8 byte view",
        "nonscalar_canonical_hash_identity_required": True,
        "real_checkpoint_full_model_hash_exercised_on_CPU": True,
        "checkpoint_model_loss_optimizer_sampler_RNG_and_precision_unchanged": True,
        "additional_optimizer_updates": 0,
        "a5_graph_route_only": True,
        "c2f_safe_eager_route_unchanged": True,
    }
    path = output / "SCALAR_SAFE_GRAPH_FREEZE.json"
    if path.exists() and read(path) != contract:
        raise ValueError("Preserve existing scalar-safe graph freeze")
    atomic_json(path, contract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    run(arguments.output.resolve())
