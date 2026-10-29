"""The deterministic transport sampler preserves both differentiable inputs."""

import pytest
import torch
from torch.nn import functional

from e_jepa_ttc.models.local_transport import deterministic_bilinear_sample


@pytest.mark.parametrize("height,width", [(3, 5), (8, 8), (1, 4), (4, 1)])
def test_bilinear_outputs_and_both_gradients_match(height, width):
    generator = torch.Generator().manual_seed(710)
    source = torch.randn(2, 3, height, width, generator=generator, dtype=torch.float64)
    grid = torch.rand(2, 7, 6, 2, generator=generator, dtype=torch.float64) * 3 - 1.5
    outputs, gradients = [], []
    for explicit in (False, True):
        values = source.clone().requires_grad_()
        coords = grid.clone().requires_grad_()
        out = (
            deterministic_bilinear_sample(values, coords)
            if explicit
            else functional.grid_sample(values, coords, align_corners=True)
        )
        grad = torch.autograd.grad(out.square().sum(), (values, coords))
        outputs.append(out)
        gradients.append(grad)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-12, atol=1e-12)
    for original, deterministic in zip(gradients[0], gradients[1], strict=True):
        torch.testing.assert_close(original, deterministic, rtol=1e-11, atol=1e-11)
