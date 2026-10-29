"""Safe-array roundtrips with untrained models and explicitly mocked endpoint validation."""

import json

import numpy as np
import pytest
import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import compact_weights
from e_jepa_ttc.simplex_t.arms import resolve_arm
from e_jepa_ttc.simplex_t.model import TemporalRefiner
from e_jepa_ttc.simplex_t.registry import registered_graph


def export_fixture(tmp_path, monkeypatch, name="TPR-D0-H8-C160"):
    graph = registered_graph(
        d1=False,
        density=False,
        t3=True,
        latent=True,
        replicate_scalar=False,
        replicate_latent=False,
    )
    spec = next(s for s in graph if s.name == name and s.fold == 0)
    model = TemporalRefiner(resolve_arm(spec, graph).model).float().eval()
    monkeypatch.setattr(compact_weights, "load_endpoint", lambda *a, **kw: model)
    output = tmp_path / "compact"
    digest = compact_weights.export_compact_endpoint(
        tmp_path / "mock_checkpoint",
        output,
        spec=spec,
        graph=graph,
        freeze_sha256="a" * 64,
        train_source_sha256="b" * 64,
        endpoint_sha256="c" * 64,
        validate_prerequisites=lambda: None,
        resource_check=lambda: None,
    )
    return output, digest, model


@pytest.mark.parametrize("name", ["TPR-D0-H8-C160", "LATENT-D0-H8-C160", "TRANSFORMER-D0-H8-C128"])
def test_every_parameter_roundtrips_without_pickle(tmp_path, monkeypatch, name):
    output, digest, model = export_fixture(tmp_path, monkeypatch, name)
    restored = compact_weights.load_compact_endpoint(output, manifest_sha256=digest)
    assert not restored.training
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[key])
    with np.load(output / "weights.npz", allow_pickle=False) as archive:
        assert all(archive[k].dtype == np.float32 for k in archive.files)
    assert not json.loads((output / "WEIGHTS.json").read_text())["optimizer_state_included"]


@pytest.mark.parametrize(
    "corruption", ["manifest", "weights", "schema", "parameter_set", "control"]
)
def test_changed_compact_artifacts_are_rejected(tmp_path, monkeypatch, corruption):
    output, digest, _ = export_fixture(tmp_path, monkeypatch)
    path = output / "WEIGHTS.json"
    manifest = json.loads(path.read_text())
    if corruption == "manifest":
        path.write_text("{}", encoding="utf-8")
    elif corruption == "weights":
        (output / "weights.npz").write_bytes(b"changed")
    elif corruption == "schema":
        manifest["optimizer_state_included"] = True
        path.write_text(json.dumps(manifest), encoding="utf-8")
        digest = sha256(path)
    elif corruption == "control":
        manifest["control"] = "REPEAT_CURRENT"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        digest = sha256(path)
    else:
        np.savez_compressed(output / "weights.npz", unwanted=np.ones(1, np.float32))
        manifest["weights_sha256"] = sha256(output / "weights.npz")
        path.write_text(json.dumps(manifest), encoding="utf-8")
        digest = sha256(path)
    with pytest.raises(ValueError):
        compact_weights.load_compact_endpoint(output, manifest_sha256=digest)
