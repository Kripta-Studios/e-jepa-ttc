"""Training-only current normalization and closed table admission."""

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.simplex_t.current_inputs import load_current_inputs, normalize_current_train


def test_dev_cannot_fit_normalizer():
    with pytest.raises(ValueError, match="TRAIN"):
        normalize_current_train({"role": "outer_dev"})


def test_current_train_normalizer_and_population_mass():
    table = {
        "role": "inner_oof",
        "arrays": {"features17": np.arange(68, dtype=np.float64).reshape(4, 17)},
        "metadata": pd.DataFrame(
            {"target_ttc": [1.0, 1.0, 2.0, 2.0], "sequence_id": ["a", "a", "b", "b"]}
        ),
    }
    normalizer, mass = normalize_current_train(table)
    np.testing.assert_array_equal(normalizer.mean, table["arrays"]["features17"].mean(0))
    assert mass.sum() == pytest.approx(1)
    assert mass[:2].sum() == pytest.approx(0.5)


def test_current_loader_refuses_unregistered_role_before_io(tmp_path):
    with pytest.raises(ValueError, match="unregistered"):
        load_current_inputs(
            tmp_path, 0, "confirmation", ancestry_sha256="x", allowed_sequences=set()
        )
