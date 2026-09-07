"""Whole-phase compact export composition without training or model inference."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import compact_phase as module
from e_jepa_ttc.simplex_t.phase_manifest import fit_key
from e_jepa_ttc.simplex_t.registry import registered_graph


@pytest.mark.parametrize("stage", ["T2", "T3", "T4", "T5"])
def test_all_heads_and_resume(tmp_path, monkeypatch, stage):
    seal = tmp_path / "seal.json"
    seal.write_text("synthetic fixture")
    flags = dict(
        d1=False, density=False, t3=True, latent=True, replicate_scalar=True, replicate_latent=True
    )
    graph = registered_graph(**flags)
    selected = [spec for spec in graph if spec.stage == stage]
    record = {
        "source_contract": {"availability": flags},
        "source_identities": {fit_key(spec): {"inner_oof": "a" * 64} for spec in selected},
    }
    endpoints = {
        fit_key(spec): {
            "train_source_sha256": "a" * 64,
            "resolved_checkpoint": seal,
            "checkpoint_sha256": sha256(seal),
        }
        for spec in selected
    }
    monkeypatch.setattr(module, "read_scientific_freeze", lambda *args, **kwargs: record)
    monkeypatch.setattr(module, "validated_phase", lambda *args, **kwargs: (graph, endpoints))
    exports = []

    def export(checkpoint, output, **kwargs):
        exports.append(kwargs["spec"])
        kwargs["validate_prerequisites"]()
        output.mkdir()
        manifest = output / "WEIGHTS.json"
        manifest.write_text(
            json.dumps(
                {
                    "fit": asdict(kwargs["spec"]),
                    "scientific_freeze_sha256": kwargs["freeze_sha256"],
                    "train_source_sha256": kwargs["train_source_sha256"],
                    "source_endpoint_sha256": kwargs["endpoint_sha256"],
                }
            )
        )
        return sha256(manifest)

    def model(*args, **kwargs):
        return SimpleNamespace(state_dict=lambda: {"weight": torch.tensor([1.0])})

    monkeypatch.setattr(module, "export_compact_endpoint", export)
    monkeypatch.setattr(module, "load_compact_endpoint", model)
    monkeypatch.setattr(module, "load_endpoint", model)
    output = tmp_path / "compact"
    kwargs = dict(
        endpoints=seal,
        endpoints_sha256=sha256(seal),
        checkpoint_root=tmp_path,
        freeze=seal,
        freeze_sha256=sha256(seal),
        roots={"work": tmp_path},
        stage=stage,
        availability=flags,
        validate_authority_and_qa=lambda: None,
        resource_ok=lambda: True,
    )
    result = module.export_compact_phase(output, **kwargs, resume=False)
    assert set(result["fits"]) == {fit_key(spec) for spec in selected}
    assert exports == selected
    assert module.export_compact_phase(output, **kwargs, resume=True) == result
    assert exports == selected
    monkeypatch.setattr(
        module,
        "load_compact_endpoint",
        lambda *args, **kwargs: SimpleNamespace(state_dict=lambda: {"weight": torch.tensor([2.0])}),
    )
    with pytest.raises(ValueError, match="roundtrip"):
        module.export_compact_phase(output, **kwargs, resume=True)
    partial = tmp_path / "partial"
    first = partial / fit_key(selected[0])
    first.mkdir(parents=True)
    with pytest.raises(ValueError, match="unmanifested"):
        module.export_compact_phase(partial, **kwargs, resume=True)
    assert first.is_dir() and not (first / "WEIGHTS.json").exists()
    paused = tmp_path / "paused"
    with pytest.raises(InterruptedError):
        module.export_compact_phase(
            paused, **{**kwargs, "resource_ok": lambda: False}, resume=False
        )
    assert not paused.exists()
