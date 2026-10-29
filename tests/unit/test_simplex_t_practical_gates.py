"""Practical gates require complete evidence; confidence intervals do not gate replication."""

from dataclasses import replace

import pytest

from e_jepa_ttc.simplex_t.evaluation import may_explore_context
from e_jepa_ttc.simplex_t.gates import Evidence, development_accepted, may_replicate


def context(**overrides):
    arguments = dict(
        delta_h1=-1.0,
        finite=True,
        sign_delta=0.005,
        crucial_delta=3.0,
        fraction_train_h8=(0.5, 0.5, 0.5),
        integrity=True,
    )
    arguments.update(overrides)
    return may_explore_context(**arguments)


@pytest.mark.parametrize("coverage", [(), (0.5,), (0.5, 0.5), (0.5,) * 4, (0.5, 0.49, 0.5)])
def test_t3_requires_coverage_for_exactly_three_folds(coverage):
    assert not context(fraction_train_h8=coverage)


def test_t3_practical_boundary_does_not_require_ci():
    assert context()
    assert not context(delta_h1=-0.999)
    assert not context(finite=False)
    assert not context(integrity=False)


def evidence():
    return Evidence(-3.0, -1.0, 6, 9, 1.0, 0.005, 3.0, True)


@pytest.mark.parametrize("wins", [-1, 5, 10, 6.5, True])
def test_t5_rejects_insufficient_or_impossible_sequence_evidence(wins):
    assert not may_replicate(replace(evidence(), sequence_wins=wins))


def test_missing_integrity_cannot_authorize_replication():
    assert not may_replicate(Evidence(-3.0, -1.0, 6, 9, 1.0, 0.005, 3.0))


def test_replication_and_development_acceptance_remain_separate():
    assert may_replicate(evidence())
    assert may_replicate(replace(evidence(), sequence_wins=9))
    assert not development_accepted(evidence(), 1.0, 1.0)
    assert development_accepted(evidence(), -0.1, -0.1)


@pytest.mark.parametrize(
    "field",
    [
        "delta_risk17",
        "delta_current_control",
        "finite_fraction",
        "sign_error_delta",
        "crucial_delta",
    ],
)
def test_nonfinite_evidence_never_opens_replication(field):
    assert not may_replicate(replace(evidence(), **{field: float("nan")}))
