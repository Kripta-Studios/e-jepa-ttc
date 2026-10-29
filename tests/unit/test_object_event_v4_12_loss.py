import torch

from e_jepa_ttc.training.object_event_v4_12 import reversal_balanced_sign_loss


def test_reversal_balanced_loss_prefers_opposite_direction_logits() -> None:
    target = torch.tensor([-0.02, 0.03, -0.01, 0.04])
    good_original = torch.tensor([3.0, -3.0, 2.0, -2.0])
    good_reverse = -good_original
    bad_original = -good_original
    bad_reverse = -good_reverse
    good = reversal_balanced_sign_loss(good_original, good_reverse, target)
    bad = reversal_balanced_sign_loss(bad_original, bad_reverse, target)
    assert good.total < bad.total
    assert torch.isfinite(good.total)


def test_exact_opposite_logits_have_zero_antisymmetry_penalty() -> None:
    target = torch.tensor([-0.02, 0.03, -0.01, 0.04])
    logits = torch.tensor([2.0, -2.0, 1.5, -1.5])
    output = reversal_balanced_sign_loss(logits, -logits, target)
    assert torch.allclose(output.components["antisymmetry"], torch.zeros(()))
