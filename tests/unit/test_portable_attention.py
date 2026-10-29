"""Independent interpolation and gradient checks for the portable backend."""

import pytest
import torch

from e_jepa_ttc.detection.portable_attention import (
    PortableMSDeformAttnFunction,
    deformable_attention_pytorch,
)


def test_known_pixel_centers_and_zero_padding() -> None:
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0]).reshape(1, 5, 1, 1)
    shapes = torch.tensor([[2, 2], [1, 1]])
    locations = torch.tensor([[[0.25, 0.25], [0.5, 0.5]], [[2.0, 2.0], [0.5, 0.5]]])
    locations = locations.reshape(1, 2, 1, 2, 1, 2)
    weights = torch.tensor([0.25, 0.75, 0.5, 0.5]).reshape(1, 2, 1, 2, 1)
    actual = deformable_attention_pytorch(values, shapes, locations, weights)
    torch.testing.assert_close(actual.flatten(), torch.tensor([7.75, 5.0]))


def test_known_bilinear_interior_and_partial_padding() -> None:
    values = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 4, 1, 1)
    locations = torch.tensor([[0.375, 0.625], [0.0, 0.5]]).reshape(1, 2, 1, 1, 1, 2)
    # Pixel coordinates are (0.25,0.75) and (-0.5,0.5), respectively.
    # First interpolation: .1875*1 + .0625*2 + .5625*3 + .1875*4 = 2.75.
    # Second interpolation: half zero padding and half the mean of pixels 1,3.
    actual = deformable_attention_pytorch(
        values, torch.tensor([[2, 2]]), locations, torch.ones(1, 2, 1, 1, 1)
    )
    torch.testing.assert_close(actual.flatten(), torch.tensor([2.75, 1.0]))


def test_double_precision_gradcheck() -> None:
    torch.manual_seed(7)
    shapes = torch.tensor([[2, 3], [1, 2]])
    values = torch.randn(1, 8, 1, 2, dtype=torch.float64, requires_grad=True)
    locations = (torch.rand(1, 2, 1, 2, 2, 2, dtype=torch.float64) * 0.8 + 0.1).requires_grad_()
    weights = torch.rand(1, 2, 1, 2, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value, loc, weight: deformable_attention_pytorch(value, shapes, loc, weight),
        (values, locations, weights),
        atol=1e-5,
        rtol=1e-4,
    )


def test_wrong_level_start_is_rejected() -> None:
    with pytest.raises(ValueError, match="level_start_index"):
        PortableMSDeformAttnFunction.apply(
            torch.ones(1, 5, 1, 1),
            torch.tensor([[2, 2], [1, 1]]),
            torch.tensor([0, 3]),
            torch.full((1, 1, 1, 2, 1, 2), 0.5),
            torch.ones(1, 1, 1, 2, 1),
            64,
        )


def test_nonfinite_grid_is_not_reinterpreted_as_padding() -> None:
    with pytest.raises(ValueError, match="finite before grid_sample"):
        deformable_attention_pytorch(
            torch.ones(1, 1, 1, 1),
            torch.tensor([[1, 1]]),
            torch.full((1, 1, 1, 1, 1, 2), float("nan")),
            torch.ones(1, 1, 1, 1, 1),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_cuda_forward_and_first_gradients(dtype: torch.dtype) -> None:
    torch.manual_seed(19)
    shapes = torch.tensor([[3, 4], [2, 2]])
    tensors = [
        torch.randn(2, 16, 2, 3, dtype=dtype),
        torch.rand(2, 5, 2, 2, 3, 2, dtype=dtype) * 1.4 - 0.2,
        torch.rand(2, 5, 2, 2, 3, dtype=dtype),
    ]
    cpu = [tensor.requires_grad_() for tensor in tensors]
    cuda = [tensor.detach().cuda().requires_grad_() for tensor in tensors]
    a = deformable_attention_pytorch(cpu[0], shapes, cpu[1], cpu[2])
    b = deformable_attention_pytorch(cuda[0], shapes.cuda(), cuda[1], cuda[2])
    grad_a = torch.autograd.grad(a.square().mean(), cpu)
    grad_b = torch.autograd.grad(b.square().mean(), cuda)
    torch.testing.assert_close(a, b.cpu(), atol=2e-5, rtol=2e-5)
    for left, right in zip(grad_a, grad_b, strict=True):
        torch.testing.assert_close(left, right.cpu(), atol=2e-5, rtol=2e-5)
