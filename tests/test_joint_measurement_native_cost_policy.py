"""The inference exception must not relax subsequent training determinism."""

import pytest
import torch

from operational.joint_measurement.fusion_cost_native_v2 import native_median_policy


@pytest.mark.parametrize("enabled,warning", [(True, False), (True, True), (False, False)])
@pytest.mark.parametrize("failure", [False, True])
def test_native_cost_policy_restores_both_flags(
    enabled: bool, warning: bool, failure: bool
) -> None:
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=warning)
        try:
            with native_median_policy():
                assert torch.are_deterministic_algorithms_enabled()
                assert torch.is_deterministic_algorithms_warn_only_enabled()
                if failure:
                    raise RuntimeError("native inference failure")
        except RuntimeError as error:
            assert failure and str(error) == "native inference failure"
        assert torch.are_deterministic_algorithms_enabled() is enabled
        assert torch.is_deterministic_algorithms_warn_only_enabled() is warning
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
