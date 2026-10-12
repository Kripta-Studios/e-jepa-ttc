import torch

from operational.multimodal_research.regret_loss import rgb_regret


def test_regret_preserves_initial_gradient_and_does_not_penalize_improvement():
    anchor = torch.tensor([0.04, -0.04], dtype=torch.float64)
    target = torch.tensor([0.05, -0.05], dtype=torch.float64)
    mass = torch.tensor([0.25, 0.75], dtype=torch.float64)
    initial = anchor.clone().requires_grad_()
    loss = rgb_regret(initial, target, anchor, mass, 2)
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(initial.grad, torch.zeros_like(initial))
    improved = torch.tensor([0.045, -0.045], dtype=torch.float64, requires_grad=True)
    loss = rgb_regret(improved, target, anchor, mass, 2)
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(improved.grad, torch.zeros_like(improved))


def test_regret_penalizes_harm_in_both_signs_using_global_mass():
    anchor = torch.tensor([0.04, -0.04], dtype=torch.float64)
    target = torch.tensor([0.05, -0.05], dtype=torch.float64)
    point = torch.tensor([0.02, -0.02], dtype=torch.float64, requires_grad=True)
    mass = torch.tensor([0.25, 0.75], dtype=torch.float64)
    loss = rgb_regret(point, target, anchor, mass, 2)
    torch.testing.assert_close(loss, torch.tensor(2 / 3, dtype=torch.float64))
    loss.backward()
    torch.testing.assert_close(point.grad, torch.tensor([-25 / 3, 25.0], dtype=torch.float64))
