"""Zero-update CPU admission of predictor width and capture tensor arithmetic."""

from __future__ import annotations

import copy

import torch

from e_jepa_ttc.joint_measurement.model import PairMeasurement
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from operational.range_transfer.common import sha, write

from .capacity_model import PairTensorCore, WidePairMeasurement
from .common import ROOT, RUN
from .qa import exact
from .train import initialize


def run() -> dict:
    """Check native ancestry, width-one identity and exact wide forward/backward."""
    torch.set_num_threads(1)
    torch.manual_seed(7)
    parent = initialize("A5-P2-FROZEN")
    if not isinstance(parent, CausalScaleTTC):
        raise TypeError("Native parent required")
    original = copy.deepcopy(parent.state_dict())
    torch.manual_seed(7)
    reference = PairMeasurement.from_native(parent)
    torch.manual_seed(7)
    width_one = WidePairMeasurement(parent, width=1)
    torch.manual_seed(7)
    wide = WidePairMeasurement(parent)
    captured = copy.deepcopy(wide)
    core = PairTensorCore(captured)
    inputs = torch.randn(1, 2, 12, 128, 128)
    delta = torch.tensor([0.05])
    rng = torch.get_rng_state()
    expected = wide(inputs, delta)["raw_phase"]
    expected.sum().backward()
    end_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    actual, valid = core(inputs, delta)
    actual.sum().backward()
    actual_rng = torch.get_rng_state()
    gradients = [parameter.grad for parameter in wide.encoder.parameters()]
    checks = {
        "width_one_initialization_exact": exact(reference.state_dict(), width_one.state_dict()),
        "native_encoder_preserved": exact(
            parent.encoder.features.state_dict(), wide.encoder.state_dict()
        ),
        "parent_unchanged": exact(original, parent.state_dict()),
        "private_encoder": all(
            id(p) != id(q)
            for p, q in zip(
                parent.encoder.features.parameters(), wide.encoder.parameters(), strict=True
            )
        ),
        "wide_output_exact": torch.equal(expected, actual),
        "wide_gradients_exact": all(
            p.grad is not None and q.grad is not None and torch.equal(p.grad, q.grad)
            for p, q in zip(wide.parameters(), captured.parameters(), strict=True)
        ),
        "encoder_gradients_finite_nonzero": all(
            g is not None and torch.isfinite(g).all() for g in gradients
        )
        and any(g is not None and bool(g.abs().sum() > 0) for g in gradients),
        "native_time_guard": bool(valid) and not bool(core(inputs, -delta)[1]),
        "rng_exact": torch.equal(end_rng, actual_rng),
        "cuda_not_initialized": not torch.cuda.is_initialized(),
    }
    # The invalid-time forward is diagnostic only, never differentiated or emitted.
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": {name: bool(value) for name, value in checks.items()},
        "model_source_sha256": sha(ROOT / "operational/joint_measurement/capacity_model.py"),
        "parameters": {
            "control": sum(p.numel() for p in reference.parameters()),
            "wide": sum(p.numel() for p in wide.parameters()),
            "encoder_shared_architecture": sum(p.numel() for p in wide.encoder.parameters()),
        },
        "optimizer_updates": 0,
        "scope": (
            "CPU native-parent initialization and synthetic pixel forward/backward; "
            "no TTC accuracy claim"
        ),
    }
    write(RUN / "adaptive/D1_predictor_width/prototype/CPU.json", result)
    if result["status"] != "PASS":
        raise ValueError("D1 CPU prototype failed")
    return result


if __name__ == "__main__":
    print(run())
