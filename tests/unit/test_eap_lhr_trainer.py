from __future__ import annotations

import pytest

from e_jepa_ttc.training.eap_lhr_jepa_ttc import (
    EAPLHRTrainerConfig,
    _deterministic_indices,
)


def test_bounded_indices_are_reproducible_and_cover_dataset_span() -> None:
    indices = _deterministic_indices(4096, 96)
    assert indices is not None
    assert len(indices) == 96
    assert indices[0] == 0
    assert indices[-1] == 95
    assert indices == _deterministic_indices(4096, 96)
    assert _deterministic_indices(4096, None) is None
    assert _deterministic_indices(96, 96) is None


def test_bounded_trainer_controls_reject_non_positive_values() -> None:
    with pytest.raises(ValueError, match="max_train_samples"):
        EAPLHRTrainerConfig(max_train_samples=0)
    with pytest.raises(ValueError, match="max_validation_samples"):
        EAPLHRTrainerConfig(max_validation_samples=-1)
