"""Checks for the new TTC coordinate, causal padding and balanced gradients."""

import torch

from operational.ttc_revision.head import DirectTTCHead, symmetric_ttc_loss


def test_loss_pushes_over_and_under_predictions_toward_target():
    prediction = torch.tensor([-12.0, -1.0, -0.1, 0.1, 1.0, 12.0], requires_grad=True)
    target = torch.tensor([-8.0, -2.0, -0.2, 0.2, 2.0, 8.0])
    symmetric_ttc_loss(prediction, target).sum().backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.equal(prediction.grad.sign(), (prediction.detach() - target).sign())
    assert torch.allclose(
        symmetric_ttc_loss(prediction, target), symmetric_ttc_loss(-prediction, -target)
    )


def test_padding_and_unavailable_metadata_cannot_change_prediction():
    torch.manual_seed(7)
    model = DirectTTCHead().eval()
    features = torch.randn(2, 8, 17)
    times = torch.randn(2, 8, 4)
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[:, :3] = False
    expected = model(features, times, valid)
    features[:, :3] = 1e20
    times[:, :, (1, 3)] = 1e20
    assert torch.equal(model(features, times, valid), expected)
    assert torch.allclose(model(features[:, 3:], times[:, 3:], valid[:, 3:]), expected)


def test_output_crosses_zero_continuously_with_finite_gradient():
    model = DirectTTCHead()
    for param in model.parameters():
        param.data.zero_()
    x, t, mask = torch.zeros(1, 8, 17), torch.zeros(1, 8, 4), torch.ones(1, 8, dtype=torch.bool)
    y = model(x, t, mask)
    y.sum().backward()
    assert y.item() == 0
    assert model.output[-1].bias.grad.item() == 1
    model.output[-1].bias.data.fill_(-1e-6)
    assert -2e-6 < model(x, t, mask).item() < 0
