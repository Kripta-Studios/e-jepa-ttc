"""Only gradient clipping differs from the frozen precision-U0 control."""

import pytest
import torch

from operational.range_transfer.qa import same
from operational.representation_audit.precision import initialize as base
from operational.representation_audit.unclipped import initialize, unclipped_norm


def test_unclipped_initialization_and_optimizer_are_exact_u0():
    a, ac, ao, al, ar = base(7, device="cpu")
    b, bc, bo, bl, br = initialize(7, "N0", device="cpu")
    assert same(a.state_dict(), b.state_dict()) and ac == bc
    assert same(ao.state_dict(), bo.state_dict())
    assert same(al.state_dict(), bl.state_dict()) and same(ar.get_state(), br.get_state())


def test_unclipped_norm_preserves_large_gradients_and_rejects_nonfinite():
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([30.0, 40.0])
    before = p.grad.clone()
    assert unclipped_norm([p]) == 50
    assert torch.equal(p.grad, before)
    p.grad = torch.tensor([float("inf"), 0.0])
    with pytest.raises(RuntimeError, match="non-finite"):
        unclipped_norm([p])
