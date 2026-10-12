"""A reused E control must be exactly the registered F0 control."""

import torch

from operational.range_transfer.qa import same
from operational.representation_audit.geometry import initialize as initialize_e
from operational.representation_audit.temporal_geometry import initialize as initialize_f


def test_f0_reuses_exact_e_control_initialization_optimizer_and_sampler():
    e, ec, eo, es, eg = initialize_e(7, "Q1", device="cpu")
    f, fc, fo, fs, fg = initialize_f(7, "F0", device="cpu")
    assert same(e.state_dict(), f.state_dict())
    assert ec == fc
    assert same(eo.state_dict(), fo.state_dict())
    assert same(es.state_dict(), fs.state_dict())
    assert same(eg.get_state(), fg.get_state())
    assert e.model.residual.scorer.training and f.model.residual.scorer.training


def test_f1_changes_only_calibration_coefficients_before_optimizer_updates():
    control, *_ = initialize_f(7, "F0", device="cpu")
    changed, *_ = initialize_f(7, "F1", device="cpu")
    a, b = control.state_dict(), changed.state_dict()
    differing = [key for key in a if not torch.equal(a[key], b[key])]
    assert differing == ["height_coefficient"]
    assert all(
        name.startswith("model.residual.scorer.")
        for name, p in changed.named_parameters()
        if p.requires_grad
    )
