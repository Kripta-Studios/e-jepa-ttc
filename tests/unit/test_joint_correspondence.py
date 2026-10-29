"""Known-coordinate supervision and label independence, without optimizer updates."""

import torch
from torch import nn

from operational.joint_measurement.correspondence_model import (
    TRANSFORMS,
    CorrespondenceCore,
    correspondence_target,
    warped_texture,
)


def test_warp_has_analytic_inverse_coordinates():
    axis = torch.arange(128).float() + 0.5 - 64
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    pixels = torch.stack((xx, yy))[None].repeat(4, 1, 1, 1)
    actual = warped_texture(pixels)
    for i, (scale, tx, ty) in enumerate(TRANSFORMS):
        expected = torch.stack(((xx - tx) / scale, (yy - ty) / scale))
        torch.testing.assert_close(
            actual[i, :, 32:96, 32:96], expected[:, 32:96, 32:96], rtol=0, atol=2e-5
        )


def test_bilinear_targets_encode_both_directions_and_resolutions():
    oy, ox = torch.meshgrid(torch.arange(-2, 3), torch.arange(-2, 3), indexing="ij")
    for size in (16, 32):
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        stride = 128 / size
        first_center = 0.5 if size == 32 else 2.5
        xx = (xx.float() * stride + first_center - 64) / stride
        yy = (yy.float() * stride + first_center - 64) / stride
        for reverse in (False, True):
            target, support = correspondence_target(size, reverse=reverse)
            assert (target >= 0).all() and support.sum() == 4 * (size // 2) ** 2
            torch.testing.assert_close(
                target.sum(1)[support], torch.ones_like(xx)[None].expand(4, -1, -1)[support]
            )
            dx = (target * ox.flatten()[None, :, None, None]).sum(1)
            dy = (target * oy.flatten()[None, :, None, None]).sum(1)
            for i, (scale, tx, ty) in enumerate(TRANSFORMS):
                tx, ty = tx * size / 128, ty * size / 128
                ex = (xx - tx) / scale - xx if reverse else (scale - 1) * xx + tx
                ey = (yy - ty) / scale - yy if reverse else (scale - 1) * yy + ty
                torch.testing.assert_close(dx[i][support[i]], ex[support[i]], rtol=0, atol=2e-7)
                torch.testing.assert_close(dy[i][support[i]], ey[support[i]], rtol=0, atol=2e-7)


def test_correspondence_trains_encoder_without_targets_or_readout():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    encoder = nn.Sequential(nn.Conv2d(12, 64, 3, stride=4, padding=1), nn.SiLU())
    core = CorrespondenceCore(encoder)
    pixels = torch.randn(4, 12, 128, 128)
    before = {k: v.clone() for k, v in encoder.state_dict().items()}
    loss = core(pixels)
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert any(p.grad.norm() > 0 for p in encoder.parameters())
    assert all(torch.equal(v, before[k]) for k, v in encoder.state_dict().items())
    assert pixels.grad is None
