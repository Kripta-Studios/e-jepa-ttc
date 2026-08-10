"""Local relational distillation primitives for A4 DINOv3 dense supervision.

The teacher produces six local cosine-similarity maps per spatial position on a
32×32 grid.  These maps capture the *structure* of spatial relationships without
requiring channel-level alignment between the RGB-pretrained teacher and the
event-only student.

The offsets, loss function, and relation computation are frozen by the A4
experimental protocol and must not be modified without re-registering the
experiment.

Key invariants
--------------
- Each sample_token corresponds to a specific object (not a whole frame).
  The common-square crop is object-specific.
- No ``torch.roll`` — all offsets use explicit slicing to avoid wrap-around.
- All cosine computations happen in float32 for numerical stability.
- The teacher relation maps are pre-computed and cached; only the student
  relations are computed online during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

# ---------------------------------------------------------------------------
# Frozen A4 protocol offsets — do not add/remove/reorder
# ---------------------------------------------------------------------------

A4_RELATION_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 1),   # horizontal 1
    (1, 0),   # vertical 1
    (0, 2),   # horizontal 2
    (2, 0),   # vertical 2
    (1, 1),   # diagonal ↘
    (1, -1),  # diagonal ↙
)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LocalRelationMaps:
    """Cosine similarities at prescribed spatial offsets.

    ``values`` has shape ``[..., K, H, W]`` and ``valid`` is a boolean mask
    of the same shape indicating which positions have an in-bounds neighbour.
    """

    values: torch.Tensor
    valid: torch.Tensor


# ---------------------------------------------------------------------------
# Relation map computation
# ---------------------------------------------------------------------------

def local_cosine_relation_maps(
    features: torch.Tensor,
    *,
    offsets: tuple[tuple[int, int], ...] = A4_RELATION_OFFSETS,
    eps: float = 1.0e-6,
) -> LocalRelationMaps:
    """Compute local cosine similarities at fixed spatial offsets.

    Parameters
    ----------
    features:
        Dense feature tensor with shape ``[..., C, H, W]``.
    offsets:
        Spatial offsets ``(dy, dx)`` defining neighbour pairs.
    eps:
        Epsilon for L2 normalisation stability.

    Returns
    -------
    LocalRelationMaps with ``values`` and ``valid``, each ``[..., K, H, W]``.
    """

    if features.ndim < 3:
        raise ValueError("features must have at least 3 dimensions [..., C, H, W]")

    # Always compute in float32 for cosine stability
    normed = functional.normalize(features.float(), dim=-3, eps=eps)
    height, width = normed.shape[-2], normed.shape[-1]
    leading = normed.shape[:-3]

    k = len(offsets)
    values = normed.new_zeros((*leading, k, height, width))
    valid = torch.zeros((*leading, k, height, width), dtype=torch.bool, device=normed.device)

    for idx, (dy, dx) in enumerate(offsets):
        # Determine valid source/destination ranges — NO wrap-around
        # Source pixel (sy, sx) is related to destination (sy+dy, sx+dx)
        # We store the result at the source position.

        # Source ranges
        if dy >= 0:
            src_y_start, src_y_end = 0, height - dy
        else:
            src_y_start, src_y_end = -dy, height

        if dx >= 0:
            src_x_start, src_x_end = 0, width - dx
        else:
            src_x_start, src_x_end = -dx, width

        # Destination ranges
        dst_y_start = src_y_start + dy
        dst_y_end = src_y_end + dy
        dst_x_start = src_x_start + dx
        dst_x_end = src_x_end + dx

        if src_y_end <= src_y_start or src_x_end <= src_x_start:
            continue

        source = normed[..., :, src_y_start:src_y_end, src_x_start:src_x_end]
        destination = normed[..., :, dst_y_start:dst_y_end, dst_x_start:dst_x_end]

        # Cosine similarity = sum of element-wise product along channel dim
        cosine = (source * destination).sum(dim=-3)

        values[..., idx, src_y_start:src_y_end, src_x_start:src_x_end] = cosine
        valid[..., idx, src_y_start:src_y_end, src_x_start:src_x_end] = True

    return LocalRelationMaps(values=values, valid=valid)


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------

def local_relational_distillation_loss(
    student_features: torch.Tensor,
    teacher_relations: torch.Tensor,
    teacher_valid: torch.Tensor,
    *,
    offsets: tuple[tuple[int, int], ...] = A4_RELATION_OFFSETS,
) -> torch.Tensor:
    """L1 relational distillation loss between student and cached teacher.

    Parameters
    ----------
    student_features:
        Dense student feature tensor ``[B, 2, C_s, 32, 32]`` (t1/t2 endpoints).
    teacher_relations:
        Pre-computed teacher cosine relation maps ``[B, 2, K, 32, 32]``.
    teacher_valid:
        Boolean mask for teacher relation validity ``[B, 2, K, 32, 32]``.
    offsets:
        Must match the offsets used to compute teacher relations.

    Returns
    -------
    Scalar L1 loss over all valid positions (float32).
    """

    if student_features.ndim != 5:
        raise ValueError(
            f"student_features must have shape [B,2,C,H,W], got ndim={student_features.ndim}"
        )
    if student_features.shape[1] != 2:
        raise ValueError(
            f"student_features must have 2 endpoints (t1/t2), got {student_features.shape[1]}"
        )
    k = len(offsets)
    expected_relation_shape = (
        student_features.shape[0],
        2,
        k,
        student_features.shape[3],
        student_features.shape[4],
    )
    if teacher_relations.shape != expected_relation_shape:
        raise ValueError(
            f"teacher_relations shape {tuple(teacher_relations.shape)} "
            f"does not match expected {expected_relation_shape}"
        )
    if teacher_valid.shape != expected_relation_shape:
        raise ValueError(
            f"teacher_valid shape {tuple(teacher_valid.shape)} "
            f"does not match expected {expected_relation_shape}"
        )

    # Compute student relations online in float32
    student_relations = local_cosine_relation_maps(
        student_features, offsets=offsets,
    )

    # Combined validity
    combined_valid = teacher_valid.bool() & student_relations.valid

    if not combined_valid.any():
        return student_features.new_tensor(0.0, dtype=torch.float32)

    # Check teacher values are finite where valid
    teacher_float = teacher_relations.float()
    if not torch.isfinite(teacher_float[combined_valid]).all():
        raise ValueError("teacher relation values contain non-finite entries in valid region")

    # Check student values are finite
    if not torch.isfinite(student_relations.values[combined_valid]).all():
        raise ValueError("student relation values contain non-finite entries in valid region")

    # L1 loss over valid positions
    error = (student_relations.values[combined_valid] - teacher_float[combined_valid]).abs()
    return error.mean()


def local_relational_temporal_delta_loss(
    student_features: torch.Tensor,
    teacher_relations: torch.Tensor,
    teacher_valid: torch.Tensor,
    *,
    offsets: tuple[tuple[int, int], ...] = A4_RELATION_OFFSETS,
) -> torch.Tensor:
    """Match the *change* in local relations between t1 and t2.

    This is the post-A4 temporal probe.  The original A4 endpoint loss is left
    untouched; this additional objective supervises only

        (R_student(t2) - R_student(t1))
            ~= (R_teacher(t2) - R_teacher(t1)).

    Validity is the intersection of teacher/student validity at *both*
    endpoints.  The loss is an L1 mean in float32.  No bbox mask, TTC label,
    direct DINO feature alignment, or inference-time teacher input is used.
    """

    if student_features.ndim != 5:
        raise ValueError(
            f"student_features must have shape [B,2,C,H,W], got ndim={student_features.ndim}"
        )
    if student_features.shape[1] != 2:
        raise ValueError(
            f"student_features must have 2 endpoints (t1/t2), got {student_features.shape[1]}"
        )
    k = len(offsets)
    expected_relation_shape = (
        student_features.shape[0],
        2,
        k,
        student_features.shape[3],
        student_features.shape[4],
    )
    if teacher_relations.shape != expected_relation_shape:
        raise ValueError(
            f"teacher_relations shape {tuple(teacher_relations.shape)} "
            f"does not match expected {expected_relation_shape}"
        )
    if teacher_valid.shape != expected_relation_shape:
        raise ValueError(
            f"teacher_valid shape {tuple(teacher_valid.shape)} "
            f"does not match expected {expected_relation_shape}"
        )

    student_relations = local_cosine_relation_maps(
        student_features, offsets=offsets,
    )
    endpoint_valid = teacher_valid.bool() & student_relations.valid
    pair_valid = endpoint_valid[:, 0] & endpoint_valid[:, 1]
    if not pair_valid.any():
        return student_features.new_tensor(0.0, dtype=torch.float32)

    teacher_float = teacher_relations.float()
    teacher_t1 = teacher_float[:, 0]
    teacher_t2 = teacher_float[:, 1]
    student_t1 = student_relations.values[:, 0]
    student_t2 = student_relations.values[:, 1]

    if not torch.isfinite(teacher_t1[pair_valid]).all() or not torch.isfinite(
        teacher_t2[pair_valid]
    ).all():
        raise ValueError(
            "teacher relation values contain non-finite entries in temporal valid region"
        )
    if not torch.isfinite(student_t1[pair_valid]).all() or not torch.isfinite(
        student_t2[pair_valid]
    ).all():
        raise ValueError(
            "student relation values contain non-finite entries in temporal valid region"
        )

    teacher_delta = teacher_t2 - teacher_t1
    student_delta = student_t2 - student_t1
    return (student_delta[pair_valid] - teacher_delta[pair_valid]).abs().mean()


__all__ = [
    "A4_RELATION_OFFSETS",
    "LocalRelationMaps",
    "local_cosine_relation_maps",
    "local_relational_distillation_loss",
    "local_relational_temporal_delta_loss",
]
