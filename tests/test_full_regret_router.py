from __future__ import annotations

import numpy as np

from e_jepa_ttc.models.full_regret_router import FullRegretRidge


def test_full_regret_retains_catastrophic_third_expert_cost() -> None:
    features = np.zeros((4, 2))
    mass = np.ones(4) / 4
    ordinary = FullRegretRidge.fit(features, np.tile([10.0, 11.0, 12.0], (4, 1)), mass)
    catastrophic = FullRegretRidge.fit(features, np.tile([10.0, 11.0, 1000.0], (4, 1)), mass)
    assert catastrophic.predict_regret(features)[0, 2] > ordinary.predict_regret(features)[0, 2] + 9


def test_full_regret_recovers_informative_feature_and_tie_order() -> None:
    feature = np.linspace(-1, 1, 200)[:, None]
    losses = np.column_stack((np.full(200, 20.0), 20 + 10 * feature[:, 0], np.full(200, 40.0)))
    fit = FullRegretRidge.fit(feature, losses, np.ones(200) / 200)
    assert fit.select(np.array([[-0.8], [0.8]])).tolist() == [1, 0]
    tied = FullRegretRidge.fit(np.zeros((4, 1)), np.ones((4, 3)), np.ones(4) / 4)
    assert tied.select(np.zeros((2, 1))).tolist() == [0, 0]
