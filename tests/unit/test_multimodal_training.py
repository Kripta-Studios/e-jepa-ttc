from __future__ import annotations

import torch

from e_jepa_ttc.training.multimodal import _student_teacher_distillation_loss


def test_dinov3_distillation_stops_teacher_gradient() -> None:
    student = torch.randn(2, 1, 8, requires_grad=True)
    teacher = torch.randn(2, 1, 8, requires_grad=True)
    loss = _student_teacher_distillation_loss(
        student,
        teacher,
        torch.ones(2, 1, dtype=torch.bool),
    )
    loss.backward()

    assert student.grad is not None
    assert teacher.grad is None

