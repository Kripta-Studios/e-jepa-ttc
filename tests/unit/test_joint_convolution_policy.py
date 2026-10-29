"""Backend diagnostics must restore process policy and never alter matmul precision."""

import pytest
import torch

from operational.joint_measurement.fusion_cost_backend_probe import convolution_policy


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_convolution_flag_is_scoped_and_restored(initial, raises):
    before = torch.backends.cudnn.allow_tf32
    matrix = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = initial
        try:
            with convolution_policy(not initial):
                assert torch.backends.cudnn.allow_tf32 is not initial
                assert torch.backends.cuda.matmul.allow_tf32 is matrix
                if raises:
                    raise RuntimeError("synthetic diagnostic failure")
        except RuntimeError:
            assert raises
        assert torch.backends.cudnn.allow_tf32 is initial
        assert torch.backends.cuda.matmul.allow_tf32 is matrix
    finally:
        torch.backends.cudnn.allow_tf32 = before
