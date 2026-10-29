from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from e_jepa_ttc.data.stage61_pair_feature_cache import (
    PairFeatureBatch,
    load_feature_cache,
    save_feature_cache,
)
from e_jepa_ttc.training.stage61_pair_head import CachedPairDirectPhase


def test_pair_cache_roundtrip_and_supervision_separation(tmp_path) -> None:
    path = tmp_path / "features.npz"
    metadata = pd.DataFrame(
        {"sample_token": ["a", "b"], "sequence_id": ["s", "s"], "track_id": ["x", "y"]}
    )
    values = np.arange(266, dtype=np.float32).reshape(2, 133)
    save_feature_cache(path, arrays={"pair_features": values}, metadata=metadata, identity={})
    loaded, loaded_metadata, manifest = load_feature_cache(path)
    assert np.array_equal(loaded["pair_features"], values)
    assert loaded_metadata["sample_token"].tolist() == ["a", "b"]
    assert manifest["contains_targets"] is False
    with pytest.raises(ValueError, match="supervision"):
        save_feature_cache(
            tmp_path / "bad.npz",
            arrays={"target_phase": np.ones(2)},
            metadata=metadata,
            identity={},
        )


def test_cached_pair_head_accepts_exactly_133_fields() -> None:
    model = CachedPairDirectPhase().eval()
    values = torch.randn(64, 133)
    with torch.no_grad():
        online_equivalent = model.network(values).squeeze(-1)
        cached = model.raw_phase(PairFeatureBatch(values))
    torch.testing.assert_close(cached, online_equivalent, atol=1e-7, rtol=1e-6)
    with pytest.raises(ValueError, match="133"):
        PairFeatureBatch(torch.randn(2, 132))
