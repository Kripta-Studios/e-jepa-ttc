"""Research contracts that prevent plausible but physically or statistically invalid results."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from operational.release_research.auxiliaries import (
    CorrectionGate,
    TimeConditionedPredictor,
    anchored_predictive_loss,
    attenuate_correction,
    sensor_increment,
    sensor_interval_loss,
)
from operational.release_research.evaluation import compare
from operational.release_research.photometric import recover_q_and_light, simulate_patch
from operational.release_research.physics import (
    ClockConfig,
    Measurement,
    affine_design,
    estimate_expansion,
    observe,
    q_to_raw_phase,
    raw_phase_to_q,
    transport,
)
from operational.release_research.protocol import (
    Admission,
    ResearchRun,
    promotion_decision,
    sha256,
    validate_group_roles,
    verify_artifacts,
)
from operational.release_research.visibility import (
    composition_residual,
    sensor_log_scale_change,
    sensor_velocity,
    surface_log_expansion,
)


@pytest.mark.parametrize("q", [-5.0, -0.1, 0.0, 0.01, 1.0, 9.0])
def test_signed_raw_roundtrip_without_floor(q):
    assert raw_phase_to_q(q_to_raw_phase(q)) == pytest.approx(q)


def test_contact_domain_and_stationarity():
    assert transport(0.0, 0.6) == (0.0, 1.0)
    assert transport(-1.0, 0.5)[0] == pytest.approx(-2 / 3)
    with pytest.raises(ValueError, match="contact"):
        transport(2.0, 0.5)
    with pytest.raises(ValueError):
        q_to_raw_phase(10.0)


def test_transport_composition_and_time_equivariance():
    for q in [-0.8, 0.1, 1.0]:
        one = transport(q, 0.3)[0]
        assert transport(transport(q, 0.1)[0], 0.2)[0] == pytest.approx(one)
        assert 1.25 * transport(q / 1.25, 0.3 * 1.25)[0] == pytest.approx(one)


def test_clock_causal_cutoff_and_roi():
    history = [Measurement(0.6, 0.6, 0.5, 0.01, "roi")]
    base = observe(history, 1.0, "roi", 0.1, "transport")
    future = Measurement(1.2, 1.2, -5.0, 0.01, "other")
    old = Measurement(0.1, 0.1, 5.0, 0.01, "other")
    assert observe([old, *history, future], 1.0, "roi", 0.1, "transport") == base
    with pytest.raises(ValueError, match="ROI"):
        observe([replace(history[0], frame="other")], 1.0, "roi", 0.1, "observer")
    with pytest.raises(ValueError, match="Duplicate"):
        observe(history * 2, 1.0, "roi", 0.1, "observer")
    assert observe([], 1.0, "roi", 0.1, "observer").reason == "no_causal_history"
    with pytest.raises(ValueError):
        ClockConfig(history_s=1.0)


def test_clock_singularity_and_braking_innovation():
    singular = observe([Measurement(0.5, 0.5, 3.0, 0.01, "r")], 1.0, "r", 0.1, "observer")
    assert singular.q == 0.1 and singular.reason == "contact_fallback"
    braking = [Measurement(0.5, 0.5, 1.0, 0.01, "r"), Measurement(0.8, 0.8, 0.0, 0.01, "r")]
    result = observe(braking, 0.8, "r", 0.1, "observer")
    assert result.innovations[0] < 0
    assert 0 < result.q < transport(1.0, 0.3)[0]


def test_q_estimable_even_when_full_affine_motion_is_not():
    design = np.zeros((2, 6))
    design[0, 0], design[1, 3] = 1, 1
    result = estimate_expansion(design, np.array([0.2, 0.6]))
    assert result.estimable and result.rank == 2 and result.q == pytest.approx(0.4)
    assert result.sensitivity == pytest.approx(np.sqrt(0.5))


def test_pseudoinverse_ambiguity_and_background_bias_are_not_confidence():
    aperture = np.zeros((3, 6))
    aperture[:, 0] = 1
    result = estimate_expansion(aperture, np.ones(3))
    assert not result.estimable and result.q is None and result.null_leakage > 0.5
    zero = estimate_expansion(np.zeros((4, 6)), np.zeros(4))
    assert zero.rank == 0 and not zero.estimable
    # Full-rank background has excellent numerical support but q is the background's.
    background = estimate_expansion(np.eye(6), np.zeros(6))
    assert background.estimable and background.q == 0


def test_affine_normal_flow_units():
    rng = np.random.default_rng(7)
    xy, normals = rng.normal(size=(30, 2)), rng.normal(size=(30, 2))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    h = affine_design(xy, normals)
    beta = np.array([0.3, 0.1, -0.2, 0.5, 1.0, -1.0])
    assert estimate_expansion(h, h @ beta).q == pytest.approx(0.4)


def test_crop_zoom_and_translation_removed_in_same_units():
    scales = np.array([0.75, 1.0, 1.25])
    np.testing.assert_allclose(
        sensor_log_scale_change(0.2 + np.log(scales), np.ones(3), scales), 0.2
    )
    xy = np.array([[1.0, 2.0]])
    velocity = np.array([[3.0, 4.0]])
    origin = np.array([0.5, -0.5])
    crop_velocity = 2 * (velocity - origin) + 0.1 * xy
    np.testing.assert_allclose(sensor_velocity(xy, crop_velocity, 2.0, 0.2, origin), velocity)


def test_surface_visibility_empty_and_independent_direct_path():
    exp, valid = surface_log_expansion(np.stack([2 * np.eye(2), np.diag([1.0, -1.0])]))
    assert exp[0] == pytest.approx(np.log(2)) and valid.tolist() == [True, False]
    a = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="independent"):
        composition_residual(a, a, 2 * a, np.ones(2, bool), independent_direct_estimate=False)
    report = composition_residual(a, a, 2 * a, np.zeros(2, bool), independent_direct_estimate=True)
    assert report["accepted"] == 0 and report["mean_absolute_residual"] is None


def test_attenuation_identity_rgb_endpoint_and_uncalibrated_widths():
    rgb, r0 = torch.tensor([-0.7, 0.01, 9.0]), torch.tensor([-0.69, 0.03, 8.0])
    widths = torch.ones(3, 2) * 0.02
    assert torch.equal(attenuate_correction(rgb, r0, widths, torch.ones(3))["raw_location"], r0)
    assert torch.equal(attenuate_correction(rgb, r0, widths, torch.zeros(3))["raw_location"], rgb)
    middle = attenuate_correction(rgb, r0, widths, torch.ones(3) * 0.5)
    assert torch.equal(middle["widths_uncalibrated"], widths)
    with pytest.raises(ValueError):
        attenuate_correction(rgb, r0, widths, torch.ones(3) * 1.1)


def test_gate_missingness_no_nan_and_gradient():
    gate = CorrectionGate()
    values = torch.ones(3, 6)
    values[:, 0] = float("nan")
    mask = torch.isfinite(values)
    output = gate(values, mask)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert gate.linear.weight.grad is not None
    assert torch.isfinite(gate.linear.weight.grad).all()


def test_sensor_q_gradient_dropout_and_unknown_reference():
    q = torch.tensor([0.3], requires_grad=True)
    xy = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
    prediction = sensor_increment(
        q, xy, -torch.ones_like(xy), torch.ones(1, 2) * 0.1, torch.zeros(1, 2), torch.zeros(1)
    )
    yes = torch.ones(1, 2, dtype=torch.bool)
    args = (
        prediction,
        torch.ones(1, 2),
        torch.zeros(1, 2),
        0.2,
        0.15,
        torch.zeros(1, 2),
        yes,
        yes,
        yes,
    )
    loss, metrics = sensor_interval_loss(*args)
    loss.backward()
    assert q.grad is not None
    assert q.grad.abs().item() > 0 and metrics["admitted"] == 2
    with pytest.raises(ValueError, match="coverage"):
        sensor_interval_loss(*args[:-2], ~yes, yes)
    with pytest.raises(ValueError, match="coverage"):
        sensor_interval_loss(*args[:-1], ~yes)


def test_jepa_time_conditioning_detached_target_and_anchor():
    torch.manual_seed(7)
    predictor = TimeConditionedPredictor(4)
    context = torch.ones(3, 4, requires_grad=True)
    target = torch.randn(3, 4, requires_grad=True)
    result = predictor(context, torch.tensor([0.1, 0.2, 0.3]))
    assert not torch.equal(result[0], result[1])
    physical = torch.ones(3, requires_grad=True)
    loss = anchored_predictive_loss(
        result, target, physical, torch.zeros(3), torch.ones(3, dtype=torch.bool)
    )
    loss.backward()
    assert target.grad is None and context.grad is not None and physical.grad is not None


def test_protocol_denies_unsupported_jepa_and_split_leakage(tmp_path):
    receipt = tmp_path / "evidence.json"
    receipt.write_text(
        json.dumps(
            {"family": "jepa", "decision": "admit", "conditions": ["observed_q_recoverable"]}
        )
    )
    admission = Admission("jepa", 1000, 4, (7,), "development", {receipt.name: sha256(receipt)})
    with pytest.raises(ValueError, match="Missing family"):
        admission.validate(tmp_path)
    with pytest.raises(ValueError, match="leakage"):
        validate_group_roles({"a"}, {"b"}, {"a"})
    with pytest.raises(ValueError, match="Screen cap"):
        replace(admission, updates=1001).validate(tmp_path)


def test_promotion_requires_independent_native_statistical_evidence():
    result = promotion_decision(
        1.0, 0.95, 0.2, 0.19, 0, False, confirmation=True, delta_ci_upper=-0.01, latency_ratio=1.05
    )
    assert not result["passed"]
    assert isinstance(result["reasons"], list)
    assert "native_noninferiority_not_demonstrated" in result["reasons"]
    result = promotion_decision(
        1.0,
        0.95,
        0.2,
        0.19,
        0,
        False,
        confirmation=True,
        delta_ci_upper=-0.01,
        latency_ratio=1.05,
        independent_acquisitions=True,
        native_simultaneous_ci_upper={"overall": 0.0, "crucial": -0.1, "negative": 0.0},
    )
    assert result["passed"] and not result["release_modified"]


def test_run_freeze_detects_mutation_and_refuses_overwrite(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("original")
    run = ResearchRun(tmp_path / "run", {"source": source}, {})
    source.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        run.finish()
    assert not (run.output / "COMPLETE.json").exists()
    with pytest.raises(ValueError, match="fresh"):
        ResearchRun(run.output, {}, {})
    with pytest.raises(ValueError):
        verify_artifacts(run.output, {"../source.txt": sha256(source)})
    clean = ResearchRun(tmp_path / "completed", {"source": source}, {})
    clean.finish()
    with pytest.raises(ValueError, match="terminal"):
        clean.finish()
    with pytest.raises(ValueError, match="terminal"):
        clean.fail(ValueError("late error"))


def test_acquisition_bootstrap_does_not_weight_long_tracks_as_independent():
    target = np.ones(11)
    baseline = np.array([2.0] * 10 + [4.0])
    candidate = np.array([1.0] * 10 + [3.0])
    report = compare(target, baseline, candidate, np.array(["a"] * 10 + ["b"]))
    assert report["baseline"]["macro_mae"] == 2.0
    assert report["candidate"]["macro_mae"] == 1.0
    assert report["paired_macro_delta_ci95"] == [-1.0, -1.0]
    candidate[-1] = np.nan
    report = compare(target, baseline, candidate, np.array(["a"] * 10 + ["b"]))
    assert report["newly_missing"] == 1 and report["candidate"]["macro_mae"] is None


def test_photometric_crossings_recover_motion_separately_from_light():
    xy = np.column_stack((np.linspace(-2.0, 2.0, 64), np.zeros(64)))
    gradient = np.array([1.0, 0.0])
    times = np.linspace(0.0, 0.5, 101)
    for q in (-0.5, 0.5):
        sim = simulate_patch(xy, gradient, times, q, 0.3, 0.005, 0.004)
        estimated = recover_q_and_light(sim, xy, gradient)
        assert estimated["q"] == pytest.approx(q, abs=0.01)
        assert estimated["light"] == pytest.approx(0.3, abs=0.01)
        assert np.all(np.diff(sim.events[:, 1]) >= 0)
        assert sim.events[:, 1].min() >= 0 and sim.events[:, 1].max() <= 0.5 + 1e-12
    holes = np.zeros((len(times) - 1, len(xy)), bool)
    sim = simulate_patch(xy, gradient, times, 0.5, 0.0, 0.01, 0.01, covered=holes)
    assert len(sim.events) == 0
    assert recover_q_and_light(sim, xy, gradient)["q"] is None
