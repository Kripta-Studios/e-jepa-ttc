from __future__ import annotations

from scripts import runpod_rgb_input_cache_reclaim_v1 as v1
from scripts import runpod_rgb_input_cache_reclaim_v2 as v2


def test_v2_binds_v1_and_changes_only_poll_policy() -> None:
    assert v2.digest(v1.Path(v1.__file__)) == v2.V1_SHA256
    assert v2.POLL_SECONDS == 1.0
    assert v1.TARGET_HEADROOM_BYTES == 8 * 1024**3
    assert v1.MEMORY_BUDGET_BYTES == 60_999_999_488
