import pytest
import torch

from operational.evttc_transfer_audit.common import file_sha, write_receipt
from operational.final_refit.training import PhaseOnlyFinal
from operational.phase_adapt_bundle import HeadExport, verified_paths


@pytest.mark.parametrize("wrong", ["seed", "parent"])
def test_valid_checkpoint_cannot_misstate_seed_or_parent(wrong):
    from operational.phase_adapt_bundle import validate_head_binding

    payload = {
        "status": "COMPLETE",
        "model_class": "PhaseOnlyFinal",
        "binding": {"parent": {"id": "original"}, "data_receipt_sha256": "data"},
        "recipe": {"seed": 7},
        "selection": "last_endpoint_fixed_budget_no_test_selection",
    }
    manifest = {
        "parent_binding": {"id": "original"},
        "new_head_seed": 7,
        "data_receipt_sha256": "data",
        "adapted_recipe": {"seed": 7},
    }
    validate_head_binding(payload, manifest)
    if wrong == "seed":
        payload["recipe"]["seed"] = 13
    else:
        payload["binding"]["parent"] = {"id": "another"}
    with pytest.raises(ValueError, match="binding"):
        validate_head_binding(payload, manifest)


def test_export_matches_all_point_predictions_and_quantiles():
    model = PhaseOnlyFinal().eval()
    features = torch.zeros(3, 340)
    features[:, 160] = torch.tensor([0.03, -0.01, 0.0])
    widths = torch.zeros(3, 2)
    exported = torch.jit.trace(HeadExport(model), (features, widths))
    actual = exported(features, widths)
    expected = model(features, widths)
    for index, name in enumerate(("ttc", "point_phase", "q10", "q90")):
        assert torch.equal(actual[:, index], expected[name])


def test_bundle_rejects_modified_and_escaping_components(tmp_path):
    (tmp_path / "weights").write_bytes(b"original")
    document = {
        "schema": "phase_adapt_bundle_v1",
        "files": {"head": {"path": "weights", "sha256": file_sha(tmp_path / "weights")}},
    }
    write_receipt(tmp_path / "BUNDLE.json", document)
    assert verified_paths(tmp_path)["head"] == (tmp_path / "weights").resolve()
    (tmp_path / "weights").write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash"):
        verified_paths(tmp_path)
