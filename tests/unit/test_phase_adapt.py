import numpy as np
import pytest

from operational.phase_adapt import (
    balanced_mass,
    compatible_target,
    epoch_count,
    validate_population,
)


def test_all_anchors_mode_preserves_original_default():
    from scripts.final_train40_transfer_inputs import uniform_indices

    assert len(uniform_indices(35)) == 16
    assert uniform_indices(35, None) == list(range(35))


def test_balancing_preserves_equal_domain_and_recording_mass():
    domains = np.array(["eap", "eap", "evttc", "evttc", "evttc"])
    groups = np.array(["a", "b", "c", "c", "d"])
    weights = balanced_mass(domains, groups)
    assert weights.sum() == pytest.approx(1)
    assert weights[:2].sum() == pytest.approx(0.5)
    assert weights[2:4].sum() == pytest.approx(weights[4])


def test_targets_do_not_extrapolate_or_enter_invalid_phase_domain():
    table = np.array([[0, 1.0, 0, 0, 2.0], [1, 1.01, 0, 0, 1.0]])
    assert compatible_target(table, 1_005_000) == pytest.approx(1.5)
    assert compatible_target(table, 999_999) is None
    table[:, 4] = 0.05
    assert compatible_target(table, 1_000_000) is None
    table[:, 4] = -1.0
    assert compatible_target(table, 1_000_000) == -1.0


def test_duplicate_or_nonmonotone_label_clock_rejected():
    table = np.array([[0, 1.0, 0, 0, 2.0], [1, 1.0, 0, 0, 1.0]])
    with pytest.raises(ValueError, match="clock"):
        compatible_target(table, 1_000_000)


def test_population_rejects_duplicates_and_wrong_role():
    with pytest.raises(ValueError, match="unique"):
        validate_population(["a", "a"], ["all_compatible_local_refit"] * 2)
    with pytest.raises(ValueError, match="role"):
        validate_population(["a"], ["TEST"])


def test_budget_uses_complete_epochs_covering_all_rows():
    assert epoch_count(10_000, batch_size=128, max_updates=2000, max_epochs=30) == 25
    with pytest.raises(ValueError, match="complete epoch"):
        epoch_count(300_000, batch_size=128, max_updates=2000, max_epochs=30)
