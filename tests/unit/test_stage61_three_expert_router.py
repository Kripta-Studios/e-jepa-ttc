from __future__ import annotations

import numpy as np
import pandas as pd

from e_jepa_ttc.models.three_expert_router import (
    BASE8_FEATURES,
    fit_three_expert_router,
    predict_expert_index,
)


def test_three_expert_router_is_inner_only_and_uses_fixed_schema() -> None:
    rows = 90
    rng = np.random.default_rng(7)
    features = pd.DataFrame(rng.normal(size=(rows, 8)), columns=BASE8_FEATURES)
    target = np.tile(np.asarray([-2.0, 2.0, 5.0]), 30)
    prediction = np.column_stack(
        (
            target + np.tile([0.01, 0.5, 0.5], 30),
            target + np.tile([0.5, 0.01, 0.5], 30),
            target + np.tile([0.5, 0.5, 0.01], 30),
        )
    )
    fit = fit_three_expert_router(
        features,
        phase_features=False,
        target=target,
        predictions=prediction,
        base_weights=np.ones(rows),
        sample_tokens=tuple(f"t{index}" for index in range(rows)),
        seed=7,
    )
    assert fit.signature["inner_oof_only"] is True
    assert set(predict_expert_index(fit, features)) <= {0, 1, 2}
    assert not ({"target", "sequence", "track", "fold", "bucket"} & set(BASE8_FEATURES))
