"""Deterministic resource timing without sleeping or launching background monitors."""

import pytest

from e_jepa_ttc.simplex_t.resource_cadence import ResourceCadence


def test_reuses_only_within_bounded_age_and_forces_fresh_allocation_check():
    now, calls = [10.0], []
    guard = ResourceCadence(
        lambda: calls.append(True) or True, maximum_age_seconds=0.5, clock=lambda: now[0]
    )
    assert guard()
    now[0] = 10.49
    assert guard() and len(calls) == 1
    assert guard(force=True) and len(calls) == 2
    now[0] = 10.99
    assert guard() and len(calls) == 3
    assert (guard.checks, guard.reuses) == (3, 1)


@pytest.mark.parametrize("age", [0, -1, 1.01, float("nan"), float("inf"), True])
def test_invalid_or_excessive_cadence_rejected(age):
    with pytest.raises(ValueError):
        ResourceCadence(lambda: True, maximum_age_seconds=age)


def test_denial_is_rechecked_not_promoted_or_cached_as_success():
    responses = iter([False, True])
    guard = ResourceCadence(lambda: next(responses), maximum_age_seconds=1, clock=lambda: 0.0)
    assert not guard()
    assert guard()
    assert guard.checks == 2


@pytest.mark.parametrize("failure", [RuntimeError("counter unavailable"), "not boolean"])
def test_failed_probe_revokes_previous_positive_decision(failure):
    responses = iter([True, failure, False])

    def probe():
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    guard = ResourceCadence(probe, maximum_age_seconds=1, clock=lambda: 0.0)
    assert guard()
    with pytest.raises((RuntimeError, ValueError)):
        guard(force=True)
    assert not guard()
    assert guard.checks == 3


def test_slow_probe_does_not_extend_sample_lifetime():
    now = [0.0]

    def probe():
        now[0] += 2
        return True

    guard = ResourceCadence(probe, maximum_age_seconds=1, clock=lambda: now[0])
    assert guard() and guard()
    assert guard.checks == 2 and guard.reuses == 0


def test_clock_rollback_requires_new_sample():
    now = [10.0]
    guard = ResourceCadence(lambda: True, maximum_age_seconds=1, clock=lambda: now[0])
    assert guard()
    now[0] = 9.0
    assert guard() and guard.checks == 2


def test_nonfinite_clock_revokes_previous_admission():
    now = [0.0]
    guard = ResourceCadence(lambda: True, maximum_age_seconds=1, clock=lambda: now[0])
    assert guard()
    now[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        guard()
    now[0] = 0.0
    assert guard() and guard.checks == 2
