"""Portable fusion must retain H transforms and dynamic exact fallback."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from e_jepa_ttc.rgb_port.normalization import fit_normalizer
from operational.joint_measurement.fusion_export import NormalizedFusion, export_head
from operational.joint_measurement.fusion_features import FeatureIdentity
from operational.joint_measurement.fusion_model import RGBAnchorComplement


def fixtures(width=3):
    identities = [
        FeatureIdentity(modality, "a" * 64, "b" * 64, "c" * 64) for modality in ("rgb", "event")
    ]
    rng = np.random.default_rng(20261014)
    norms = [
        fit_normalizer(
            rng.normal(2, 3, size=(8, width)).astype(np.float32),
            [f"q{i}" for i in range(8)],
            fit_role="H",
            expected_role="H",
            producer_sha256=identity.normalizer_producer_sha256(),
            schema_sha256=identity.schema_sha256,
            modality=identity.modality,
        )
        for identity in identities
    ]
    examples = []
    for mask in ([True, True, True, True], [False, True, False], [False, False]):
        n = len(mask)
        rgb = torch.from_numpy(rng.normal(size=(n, width)).astype(np.float32))
        event = torch.from_numpy(rng.normal(size=(n, width)).astype(np.float32))
        available = torch.tensor(mask)
        anchor = torch.tensor([6.0, -3.0, 2.0, 9.0][:n])
        fallback = torch.tensor([-2.0, 4.0, -5.0, 8.0][:n])
        rgb[~available] = float("nan")
        anchor[~available] = float("nan")
        examples.append((rgb, event, anchor, fallback, available))
    return norms, identities, examples


@pytest.mark.parametrize("mode", ["TRUE", "ZERO"])
def test_export_reloads_nontrivial_normalizers_and_dynamic_fallback(tmp_path, mode):
    norms, identities, examples = fixtures()
    model = RGBAnchorComplement(3, 3, hidden=8, mode=mode)
    with torch.no_grad():
        model.network[-1].weight.fill_(0.01)
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    result = export_head(
        tmp_path, model, *norms, *identities, examples, binding={"purpose": "unit"}
    )
    assert result["status"] == "PASS" and result["fallback_bit_exact"]
    assert all(torch.equal(initial[key], value) for key, value in model.state_dict().items())
    assert model.training and all(p.requires_grad for p in model.parameters())
    assert (
        export_head(tmp_path, model, *norms, *identities, examples, binding={"purpose": "unit"})
        == result
    )
    original_mode = model.mode
    model.mode = "ZERO" if original_mode == "TRUE" else "TRUE"
    with pytest.raises(ValueError, match="Frozen identity changed"):
        export_head(tmp_path, model, *norms, *identities, examples, binding={"purpose": "unit"})
    model.mode = original_mode
    with torch.no_grad():
        model.network[-1].bias.add_(1.0)
    with pytest.raises(ValueError, match="Frozen identity changed"):
        export_head(tmp_path, model, *norms, *identities, examples, binding={"purpose": "unit"})
    model.load_state_dict(initial)
    (tmp_path / "model.torchscript.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact changed"):
        export_head(tmp_path, model, *norms, *identities, examples, binding={"purpose": "unit"})


def test_wrong_parent_normalizer_is_rejected():
    norms, identities, _ = fixtures()
    norms[0] = replace(norms[0], producer_sha256="f" * 64)
    with pytest.raises(ValueError, match="identity mismatch"):
        NormalizedFusion(RGBAnchorComplement(3, 3), *norms, *identities)


def test_missing_coverage_cases_cannot_claim_dynamic_export(tmp_path):
    norms, identities, examples = fixtures()
    with pytest.raises(ValueError, match="all, missing, mixed"):
        export_head(
            tmp_path,
            RGBAnchorComplement(3, 3),
            *norms,
            *identities,
            examples[:1],
            binding={"purpose": "unit"},
        )


@pytest.mark.parametrize("mode", ["TRUE", "ZERO"])
def test_bundle_replays_fusion_in_isolated_process_without_changing_artifacts(tmp_path, mode):
    from operational.joint_measurement.replay import file_hash, replay_export

    norms, identities, examples = fixtures()
    export_head(
        tmp_path,
        RGBAnchorComplement(3, 3, hidden=8, mode=mode),
        *norms,
        *identities,
        examples,
        binding={"purpose": "bundle replay regression"},
    )
    original = {path.name: file_hash(path) for path in tmp_path.iterdir()}
    result = replay_export(tmp_path)
    assert result["status"] == "PASS" and result["examples"] == 3
    assert result["fallback_bit_exact"] and result["project_python_modules_loaded"] == []
    assert not result["cuda_initialized"] and result["optimizer_updates"] == 0
    assert {path.name: file_hash(path) for path in tmp_path.iterdir()} == original
