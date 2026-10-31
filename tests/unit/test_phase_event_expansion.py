"""Independent physical identities and input contracts for causal expansion."""

import numpy as np
import pytest

from operational.phase_event_expansion.model import (
    SolverConfig,
    bilinear_image,
    estimate_expansion,
    select_events,
    warp_endpoint,
)
from operational.phase_event_expansion.synthetic import make_fixture


def test_endpoint_warp_matches_hand_calculated_translation_and_expansion():
    # 0.2 + (2*0.2 - 0.1)*0.04 = 0.212; -0.1 + (-0.2+0.2)*.04=-.1.
    actual = warp_endpoint(np.array([[0.2, -0.1]]), np.array([0.04]), [2, 0.1, -0.2])
    np.testing.assert_allclose(actual, [[0.212, -0.1]], atol=1e-15)


def test_reference_shift_obeys_constant_velocity_ttc_law():
    # Scene point is X/Z, Z=2-t and X=.4-.1*t. Its TTC at t=.2 is 1.8s.
    t = np.array([-0.04, -0.02, 0.0])
    points = np.column_stack(((0.4 - 0.1 * t) / (2 - t), np.zeros(3)))
    actual = warp_endpoint(points, 0.2 - t, [1 / 1.8, 0.1 / 1.8, 0])
    np.testing.assert_allclose(actual[:, 0], np.repeat(0.38 / 1.8, 3), atol=1e-15)


def test_bilinear_voting_conserves_both_polarities_and_reports_loss():
    points = np.array([[0.123, 0.234], [-0.2, 0.1], [3, 3]])
    image, lost = bilinear_image(points, np.array([1, -1, 1]), 32, 1.0)
    assert image.shape == (2, 32, 32)
    assert image.sum() == pytest.approx(2)
    assert image[0].sum() == pytest.approx(1)
    assert image[1].sum() == pytest.approx(1)
    assert lost == pytest.approx(1 / 3)


def test_future_events_rejected_not_silently_dropped():
    events = np.array([[10, 20, 101, 1]], dtype=float)
    with pytest.raises(ValueError, match="future"):
        estimate_expansion(events, [0, 0, 32, 32], 100)


def test_event_budget_sampling_is_seeded_sorted_and_does_not_mutate():
    events = np.column_stack((np.ones(100), np.ones(100), np.arange(100), np.ones(100)))
    original = events.copy()
    first = select_events(events, 20, 7)
    np.testing.assert_array_equal(first, select_events(events, 20, 7))
    assert not np.array_equal(first, select_events(events, 20, 13))
    assert len(first) == 20
    assert np.all(np.diff(first[:, 2]) >= 0)
    np.testing.assert_array_equal(events, original)


def test_empty_and_single_edge_abstain_without_optimizer():
    empty = estimate_expansion(np.empty((0, 4)), [0, 0, 256, 256], 100_000)
    assert not empty.valid and empty.reason == "insufficient_events"
    events = np.column_stack((np.arange(256), np.ones(256), np.arange(256) * 100, np.ones(256)))
    edge = estimate_expansion(events, [0, 0, 256, 256], 100_000)
    assert not edge.valid and edge.reason == "insufficient_spatial_support"
    assert edge.diagnostics["objective_evaluations"] == 0


def test_single_bbox_fixture_is_causal_and_matches_independent_motion_equation():
    fixture = make_fixture(7, "expansion", count=512, jitter=0)
    events = fixture.events
    assert events.shape == (512, 4)
    assert np.max(events[:, 2]) <= fixture.query_us
    center = (fixture.roi[:2] + fixture.roi[2:]) / 2
    scale = np.max(fixture.roi[2:] - fixture.roi[:2])
    points = (events[:, :2] - center) / scale
    dt = (fixture.query_us - events[:, 2]) / 1e6
    endpoint = points * (1 + fixture.q * dt[:, None]) - fixture.translation * dt[:, None]
    # Endpoint contours are the four rectangle sides; no bbox history is supplied.
    on_edge = np.minimum(abs(abs(endpoint[:, 0]) - 0.35), abs(abs(endpoint[:, 1]) - 0.22))
    assert np.max(on_edge) < 1e-12


def test_absolute_clock_shift_leaves_estimation_unchanged():
    fixture = make_fixture(7, "expansion", count=512)
    config = SolverConfig()
    first = estimate_expansion(fixture.events, fixture.roi, fixture.query_us, config)
    shifted = fixture.events.copy()
    shifted[:, 2] += 10_000_000
    second = estimate_expansion(shifted, fixture.roi, fixture.query_us + 10_000_000, config)
    assert first.q == second.q
    assert first.valid == second.valid
    assert first.reason == second.reason
    np.testing.assert_array_equal(first.translation, second.translation)


@pytest.mark.parametrize("bad", [0, 20001])
def test_fixed_event_budget_cannot_be_exceeded(bad):
    with pytest.raises(ValueError, match="budget"):
        SolverConfig(event_budget=bad)


def test_synthetic_gate_counts_abstention_as_motion_failure():
    from operational.phase_event_expansion.model import ExpansionResult
    from scripts.benchmark_phase_event_expansion_synthetic import score_fixture

    fixture = make_fixture(7, "expansion")
    missing = ExpansionResult(None, None, None, False, "flat_objective", 0.01)
    assert not score_fixture(fixture, missing)["passed"]
    correct = ExpansionResult(
        fixture.q, tuple(fixture.translation), 1 / fixture.q, True, "available", 0.01
    )
    assert score_fixture(fixture, correct)["passed"]
    noise = make_fixture(7, "noise")
    assert score_fixture(noise, missing)["passed"]
    assert not score_fixture(noise, correct)["passed"]


def test_fractional_microseconds_and_invalid_roi_are_rejected():
    events = np.array([[10, 20, 99.5, 1]])
    with pytest.raises(ValueError, match="integer microseconds"):
        estimate_expansion(events, [0, 0, 32, 32], 100)
    with pytest.raises(ValueError, match="ROI"):
        estimate_expansion(events, [0, 0, 0, 32], 100)


def test_midpoint_parameters_convert_exactly_to_query_parameters():
    from operational.phase_event_expansion.model import parameters_at_query, parameters_at_reference

    # q_mid=.5 -> TTCmid=2; after .2s TTCend=1.8; b scales by same 1/.9.
    endpoint = parameters_at_query(np.array([0.5, 0.1, -0.2]), 0.2)
    np.testing.assert_allclose(endpoint, [1 / 1.8, 1 / 9, -2 / 9], atol=1e-15)
    np.testing.assert_allclose(parameters_at_reference(endpoint, 0.2), [0.5, 0.1, -0.2])
    with pytest.raises(ValueError, match="reference"):
        parameters_at_query(np.array([5.0, 0.0, 0.0]), 0.2)


def test_temporal_agreement_does_not_reward_stationary_false_contraction():
    from operational.phase_event_expansion.model import temporal_agreement

    fixture = make_fixture(7, "stationary")
    points = (fixture.events[:, :2] - 128) / 256
    age = (fixture.query_us - fixture.events[:, 2]) / 1e6
    early = age >= (age.min() + age.max()) / 2
    stationary, _ = temporal_agreement(points, fixture.events[:, 3], early, 128, 1, 0.8)
    false = warp_endpoint(points, age, [-1.0341496217625645, 0.0037355348, 0.0022569678])
    contracted, _ = temporal_agreement(false, fixture.events[:, 3], early, 128, 1, 0.8)
    assert stationary > contracted + 0.01


def test_temporal_agreement_distinguishes_aligned_motion_from_raw_events():
    from operational.phase_event_expansion.model import temporal_agreement

    fixture = make_fixture(7, "expansion")
    points = (fixture.events[:, :2] - 128) / 256
    age = (fixture.query_us - fixture.events[:, 2]) / 1e6
    early = age >= (age.min() + age.max()) / 2
    raw, _ = temporal_agreement(points, fixture.events[:, 3], early, 128, 1, 0.8)
    aligned = warp_endpoint(points, age, [fixture.q, *fixture.translation])
    score, _ = temporal_agreement(aligned, fixture.events[:, 3], early, 128, 1, 0.8)
    assert score > raw + 0.02
