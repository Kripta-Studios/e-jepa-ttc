from __future__ import annotations

import inspect

from e_jepa_ttc.data.stage61_pair_feature_cache import LocalTemporalFieldBatch, PairFeatureBatch
from e_jepa_ttc.training.stage61_pair_head import train_pair_head
from e_jepa_ttc.training.stage62_local_field import train_local_field


def test_model_input_contracts_cannot_carry_identity_or_targets() -> None:
    assert set(PairFeatureBatch.__dataclass_fields__) == {"features"}
    assert set(LocalTemporalFieldBatch.__dataclass_fields__) == {
        "patch_features",
        "patch_valid",
        "a5_phase",
    }


def test_fixed_budget_trainers_accept_no_dev_arrays() -> None:
    for function in (train_pair_head, train_local_field):
        names = set(inspect.signature(function).parameters)
        assert not any(
            (name.startswith("dev") and name != "device")
            or "outer_dev" in name
            or "validation" in name
            for name in names
        )
