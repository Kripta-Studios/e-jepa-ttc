"""Student-conditioned reliability-gated multi-teacher distillation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ReliabilityGatedTeacherOutput:
    """Detached consensus target and transparent teacher weights."""

    target: torch.Tensor
    weights: torch.Tensor
    disagreement: torch.Tensor
    valid: torch.Tensor


def reliability_gated_teacher_target(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_reliability: torch.Tensor,
    *,
    teacher_valid: torch.Tensor | None = None,
    disagreement_temperature: float = 1.0,
) -> ReliabilityGatedTeacherOutput:
    """Combine ``[B,K,...]`` frozen teachers without allowing teacher gradients."""

    if teacher_logits.ndim != student_logits.ndim + 1:
        raise ValueError("teacher_logits must add a teacher axis after batch.")
    if teacher_logits.shape[0] != student_logits.shape[0]:
        raise ValueError("Teacher and student batch sizes differ.")
    if teacher_reliability.shape != teacher_logits.shape[:2]:
        raise ValueError("teacher_reliability must have shape [B,K].")
    if disagreement_temperature <= 0:
        raise ValueError("disagreement_temperature must be positive.")
    teachers = teacher_logits.detach()
    student = student_logits.detach().unsqueeze(1)
    reduce_dims = tuple(range(2, teachers.ndim))
    disagreement = (teachers.sigmoid() - student.sigmoid()).abs().mean(dim=reduce_dims)
    reliability = teacher_reliability.detach().clamp_min(0.0)
    if teacher_valid is not None:
        if teacher_valid.shape != reliability.shape:
            raise ValueError("teacher_valid must have shape [B,K].")
        reliability = reliability * teacher_valid.detach().to(reliability.dtype)
    logits = reliability.clamp_min(1e-8).log()
    logits = logits - disagreement / disagreement_temperature
    weights = torch.softmax(logits, dim=1)
    valid = reliability.sum(dim=1) > 0
    weights = weights * valid[:, None].to(weights.dtype)
    expanded_weights = weights.view(*weights.shape, *([1] * (teachers.ndim - 2)))
    target = (expanded_weights * teachers).sum(dim=1)
    return ReliabilityGatedTeacherOutput(
        target=target,
        weights=weights,
        disagreement=disagreement,
        valid=valid,
    )


__all__ = ["ReliabilityGatedTeacherOutput", "reliability_gated_teacher_target"]
