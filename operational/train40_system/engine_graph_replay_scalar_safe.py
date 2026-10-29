"""Run admitted A5 graph replay with scalar-safe canonical module hashing."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from torch import nn

from operational.efficient_context.common import Lease, digest
from operational.train40_system import engine_graph_replay
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT


def scalar_safe_module_tensor_sha256(module: nn.Module) -> str:
    """Hash ordered tensors identically while permitting zero-dimensional tensors."""

    result = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        for value in (name, str(cpu.dtype), repr(tuple(cpu.shape))):
            encoded = value.encode("utf-8")
            result.update(len(encoded).to_bytes(8, "big"))
            result.update(encoded)
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        result.update(len(raw).to_bytes(8, "big"))
        result.update(raw)
    return result.hexdigest()


def run(output: Path, arm: str) -> None:
    """Patch only the canonical tensor hasher for one delegated A5 process."""
    if arm != "a5":
        raise ValueError("Scalar-safe graph replay is admitted only for A5")
    graph_path = output / "GRAPH_REPLAY_FREEZE.json"
    freeze_path = output / "SCALAR_SAFE_GRAPH_FREEZE.json"
    graph, freeze = read(graph_path), read(freeze_path)
    verify_sources(graph)
    verify_sources(freeze)
    if freeze["graph_replay_freeze_sha256"] != digest(graph_path):
        raise ValueError("Scalar-safe hash admission differs from graph replay")
    if freeze["engine_sha256"] != digest(Path(__file__)):
        raise ValueError("Scalar-safe graph engine differs from its freeze")

    import e_jepa_ttc.training.causal_scale_eap as training

    original = training._module_tensor_sha256
    training._module_tensor_sha256 = scalar_safe_module_tensor_sha256
    try:
        engine_graph_replay.run(output, arm)
    finally:
        training._module_tensor_sha256 = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--arm", choices=("a5",), required=True)
    arguments = parser.parse_args()
    with Lease(arguments.output.resolve()):
        run(arguments.output.resolve(), arguments.arm)
