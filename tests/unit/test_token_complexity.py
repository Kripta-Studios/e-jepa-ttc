from __future__ import annotations

import pytest

from e_jepa_ttc.evaluation.token_complexity import (
    global_attention_pairs,
    patch_grid_tokens,
    temporal_factorized_pairs,
)
from e_jepa_ttc.models.highres_factorized import TheoreticalOOMError, theoretical_oom_guard


def test_resolution_token_counts_are_ceil_padded() -> None:
    assert patch_grid_tokens(90, 160, 16) == 60
    assert patch_grid_tokens(192, 320, 8) == 960
    assert patch_grid_tokens(96, 160, 8) == 240


def test_factorized_temporal_cost_is_not_global_r4_cost() -> None:
    assert global_attention_pairs(5, 960) == 23_040_000
    assert temporal_factorized_pairs(5, 960) == 24_000
    with pytest.raises(TheoreticalOOMError, match="Global attention is forbidden"):
        theoretical_oom_guard(
            batch=1,
            steps=5,
            patches=960,
            heads=4,
            memory_budget_gb=12.0,
            global_attention=True,
        )
