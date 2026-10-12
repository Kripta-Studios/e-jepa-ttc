"""The second intervention changes NLL precision only, with a closed-form gradient."""

import torch

from operational.range_readout.cache_readout import loss as original_loss
from operational.representation_audit.precision import (
    PRECISION,
    Vectors,
    homogeneous_nll,
    initialize,
    objective,
)


def test_homogeneous_nll_gradient():
    prediction = torch.tensor([0.01, -0.04, 0.2], dtype=torch.float64, requires_grad=True)
    truth = torch.tensor([0.02, -0.02, 0.1], dtype=torch.float64)
    logvar = torch.tensor([-10.0, -9.0, -6.0], dtype=torch.float64)
    value = homogeneous_nll(prediction, truth, logvar)
    derivative = torch.autograd.grad(value, prediction)[0]
    torch.testing.assert_close(
        derivative, PRECISION * (prediction - truth) / 3, rtol=1e-14, atol=1e-14
    )


def test_u0_is_original_native_objective_and_gradients():
    model, config, *_ = initialize(7, device="cpu")
    data = Vectors("P", synthetic=True)
    values = {k: torch.from_numpy(v[:8].copy()) for k, v in data.values.items()}
    parameters = [p for p in model.parameters() if p.requires_grad]
    torch.manual_seed(42)
    original, _ = original_loss(model, values, config)
    gradients = torch.autograd.grad(original, parameters)
    torch.manual_seed(42)
    actual, _ = objective(model, values, config, "U0")
    actual_grad = torch.autograd.grad(actual, parameters)
    assert torch.equal(original, actual)
    for a, b in zip(gradients, actual_grad, strict=True):
        assert torch.equal(a, b)
    assert all(
        name.startswith("model.residual.scorer.")
        for name, p in model.named_parameters()
        if p.requires_grad
    )
