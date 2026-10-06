"""CPU-only evidence for the additive scalar-safe graph hash wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.training import causal_scale_eap as training
from operational.efficient_context.common import digest
from operational.train40_system import (
    controller_graph_replay_scalar_safe as controller,
)
from operational.train40_system import (
    engine_graph_replay_scalar_safe as engine,
)
from operational.train40_system import (
    freeze as freeze_module,
)
from operational.train40_system import (
    scalar_safe_graph_freeze,
)
from operational.train40_system.contracts import read
from operational.train40_system.durable_io import atomic_json


class _NonScalarModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("bool", torch.tensor([True, False]))
        self.register_buffer("bf16", torch.tensor([1.0, -2.0], dtype=torch.bfloat16))
        self.register_buffer("f32", torch.tensor([[1.25, -3.5]], dtype=torch.float32))
        self.register_buffer("i64", torch.tensor([1, 2, 3], dtype=torch.int64))


def test_scalar_safe_hash_matches_original_for_every_nonscalar_dtype() -> None:
    module = _NonScalarModule()
    assert engine.scalar_safe_module_tensor_sha256(module) == training._module_tensor_sha256(
        module
    )


def test_scalar_safe_hash_covers_real_checkpoint_full_model_on_cpu() -> None:
    output = Path("artifacts/train40_system_20261005")
    protected = [
        output / "fits/a5_seed7/checkpoint_last.pt",
        output / "fits/a5_seed7/CHECKPOINT_RECEIPT.json",
        output / "fits/a5_seed7/UPDATE_JOURNAL.json",
    ]
    before = {path: digest(path) for path in protected}
    protocol = read(output / "TRAINING_PROTOCOL.json")
    checkpoint = torch.load(
        protected[0], map_location="cpu", weights_only=False
    )
    model = CausalScaleTTC(
        CausalScaleTTCConfig(**protocol["producers"]["a5"]["model_config"])
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    state = model.state_dict()
    assert len(state) > 59
    assert any(tensor.ndim == 0 for tensor in state.values())
    first = engine.scalar_safe_module_tensor_sha256(model)
    assert first == engine.scalar_safe_module_tensor_sha256(model)
    assert engine.scalar_safe_module_tensor_sha256(model.encoder) == training._module_tensor_sha256(
        model.encoder
    )
    assert len(first) == 64
    with pytest.raises(RuntimeError, match="dim.*0"):
        training._module_tensor_sha256(model)
    assert {path: digest(path) for path in protected} == before


def test_engine_restores_original_hasher_when_delegate_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph = tmp_path / "GRAPH_REPLAY_FREEZE.json"
    atomic_json(graph, {"files": []})
    atomic_json(
        tmp_path / "SCALAR_SAFE_GRAPH_FREEZE.json",
        {
            "files": [],
            "graph_replay_freeze_sha256": digest(graph),
            "engine_sha256": digest(Path(engine.__file__)),
        },
    )
    original = training._module_tensor_sha256
    monkeypatch.setattr(
        engine.engine_graph_replay,
        "run",
        lambda *_: (_ for _ in ()).throw(RuntimeError("delegated failure")),
    )
    with pytest.raises(RuntimeError, match="delegated failure"):
        engine.run(tmp_path, "a5")
    assert training._module_tensor_sha256 is original


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        ("a5", "operational.train40_system.engine_graph_replay_scalar_safe"),
        ("c2f", "operational.train40_system.engine_fast_scan_safe"),
    ],
)
def test_controller_routes_only_exact_producers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arm: str, expected: str
) -> None:
    calls = []
    monkeypatch.setattr(controller.controller_overlap, "_launch", lambda *args: calls.append(args))
    controller.launch(tmp_path, arm, "operational.train40_system.engine", ["--arm", arm])
    assert calls == [(tmp_path, arm, expected, ["--arm", arm])]


def test_freeze_binds_graph_freeze_and_qa(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    source_dir = root / "operational/train40_system"
    test_dir = root / "tests/unit"
    source_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    monkeypatch.setattr(scalar_safe_graph_freeze, "ROOT", root)
    monkeypatch.setattr(freeze_module, "ROOT", root)
    output = tmp_path / "output"
    output.mkdir()
    atomic_json(output / "GRAPH_REPLAY_FREEZE.json", {"files": [], "a5_only_graph_replay": True})
    for name, marker in scalar_safe_graph_freeze.QA_CHECKS.items():
        (output / name).write_text(marker, encoding="utf-8")
    for name in (
        "engine_graph_replay_scalar_safe.py",
        "controller_graph_replay_scalar_safe.py",
        "scalar_safe_graph_freeze.py",
    ):
        (source_dir / name).write_text(name, encoding="utf-8")
    (test_dir / "test_train40_scalar_safe_graph.py").write_text("test", encoding="utf-8")
    scalar_safe_graph_freeze.run(output)
    frozen = read(output / "SCALAR_SAFE_GRAPH_FREEZE.json")
    assert frozen["additional_optimizer_updates"] == 0
    assert frozen["graph_replay_freeze_sha256"] == digest(
        output / "GRAPH_REPLAY_FREEZE.json"
    )
