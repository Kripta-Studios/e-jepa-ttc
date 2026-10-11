"""Numerical parity and narrowly scoped fusion recovery/routing."""

import itertools
from pathlib import Path

import pytest
import torch

from e_jepa_ttc.rgb_port.fusion import DualClockFusion, fusion_loss
from operational.rgb_port_fusion_fix.fusion import deterministic_forward
from operational.rgb_port_fusion_fix.queue import route
from operational.rgb_port_fusion_fix.recovery import retry_state


@pytest.mark.parametrize("mode", ["F_TRUE", "F_ZERO"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_values_loss_and_parameter_gradients(mode, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    enabled = torch.are_deterministic_algorithms_enabled()
    try:
        torch.manual_seed(7)
        model = DualClockFusion(mode).to(device)
        with torch.no_grad():
            model.location.weight.normal_(std=0.01)
            model.width.weight.normal_(std=0.01)
        x = torch.randn(6, 3, 17, device=device)
        timing = torch.randn(6, 3, 4, device=device)
        valid = torch.tensor([[False, True, True]] * 6, device=device)
        experts = torch.tensor(list(itertools.permutations([0.02, 0.03, 0.04])), device=device)
        args = (x, timing, valid, experts, x * 0.5, timing * 0.9, valid)
        truth = torch.full((6,), 0.02, device=device)
        mass = torch.full((6,), 1 / 6, device=device)
        torch.use_deterministic_algorithms(False)  # Historical reference only.
        old = model(*args)
        old_loss = fusion_loss(old, truth, mass, 6)
        old_loss.backward()
        grads = {k: p.grad.clone() for k, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        torch.use_deterministic_algorithms(True)
        new = deterministic_forward(model, *args)
        loss = fusion_loss(new, truth, mass, 6)
        loss.backward()
        for key in old:
            torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)
        torch.testing.assert_close(loss, old_loss, rtol=0, atol=0)
        for key, p in model.named_parameters():
            torch.testing.assert_close(p.grad, grads[key], rtol=0, atol=0)
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(enabled)


def test_anchor_ties_and_extremes():
    values = torch.tensor(list(itertools.product([-1e20, -0.0, 0.02, 1e20], repeat=3)))
    torch.testing.assert_close(
        values.sort(dim=-1, stable=True).values[:, 1], values.median(-1).values, rtol=0, atol=0
    )


def test_route_preserves_completed_head_and_inference_repair():
    base = ["python", "-m", "operational.rgb_port.train_heads", "--fit-id", "R_CTX"]
    assert route(base, Path("run")) == base
    fusion = [*base[:-1], "F_TRUE"]
    result = route(fusion, Path("run"))
    assert result[2:5] == ["operational.rgb_port_fusion_fix.runtime", "run", base[2]]
    assert result[5:] == fusion[3:]
    for target in ("predict_heads", "transfer", "profile", "cost_routes"):
        assert route(["python", "-m", f"operational.rgb_port.{target}"], Path("run"))[2].endswith(
            ".runtime"
        )
    other = ["python", "-m", "operational.rgb_port_signature_fix.infer"]
    assert route(other, Path("run")) == other


def test_retry_is_selective():
    state = {
        "tasks": {
            "F_TRUE": {"status": "FAILED", "completed_updates": 0, "attempts": 1},
            "F_ZERO": {"status": "FAILED", "completed_updates": 0, "attempts": 1},
            "R_CTX": {"status": "COMPLETE", "completed_updates": 2500},
            "V": {"status": "SKIPPED_DEPENDENCY"},
            "OTHER": {"status": "FAILED"},
        }
    }
    result, changed = retry_state(state, [{"id": "V", "depends": ["F_TRUE"]}])
    assert set(changed) == {"F_TRUE", "F_ZERO", "V"}
    assert result["tasks"]["R_CTX"] == state["tasks"]["R_CTX"]
    assert result["tasks"]["OTHER"] == state["tasks"]["OTHER"]
    assert result["tasks"]["F_TRUE"]["attempts"] == 1
    assert state["tasks"]["F_TRUE"]["status"] == "FAILED"
    state["tasks"]["R_CTX"]["status"] = "RUNNING"
    with pytest.raises(ValueError, match="running"):
        retry_state(state, [])
