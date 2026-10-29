from __future__ import annotations

import torch
from torch.nn import functional

from e_jepa_ttc.losses.level_dynamics_jepa import (
    build_horizon_positive_weights,
    build_temporal_residual_target,
    residual_visreg_loss,
    validate_nce_preflight,
    within_track_nce_loss,
)


def test_target_only_temporal_residual_uses_layer_norm_and_intersection_mask() -> None:
    reference = torch.tensor([[[1.0, 2.0, 3.0], [8.0, 9.0, 10.0]]], requires_grad=True)
    future = torch.tensor([[[[2.0, 4.0, 6.0], [10.0, 11.0, 12.0]]]], requires_grad=True)
    reference_valid = torch.tensor([[True, False]])
    future_valid = torch.tensor([[[True, True]]])

    target = build_temporal_residual_target(reference, future, reference_valid, future_valid)
    expected = functional.layer_norm(future, (3,)) - functional.layer_norm(
        reference, (3,)
    ).unsqueeze(1)

    assert not target.tokens.requires_grad
    assert torch.allclose(target.tokens[:, :, :1], expected[:, :, :1])
    assert torch.equal(target.valid_mask, torch.tensor([[[True, False]]]))
    assert target.tokens[:, :, 1].eq(0.0).all()
    assert target.raw_norm.requires_grad is False


def test_nce_horizon_labels_preserve_positives_and_reject_cross_track_fallback() -> None:
    anchors = torch.eye(3, requires_grad=True)
    candidates = torch.eye(3)
    timestamps = torch.tensor([0.1, 0.2, 0.3])
    positive = build_horizon_positive_weights(
        torch.zeros(3),
        timestamps,
        timestamps,
        ("seq", "seq", "seq"),
        ("seq", "seq", "seq"),
        ("track", "track", "track"),
        ("track", "track", "track"),
        tolerance_s=1e-6,
    )
    # Deliberately exclude every candidate through the external mask. Positives
    # survive this mask even though the anchor becomes invalid for lack of negatives.
    preserved = within_track_nce_loss(
        anchors,
        candidates,
        positive,
        ("seq", "seq", "seq"),
        ("seq", "seq", "seq"),
        ("track", "track", "track"),
        ("track", "track", "track"),
        timestamps,
        timestamps,
        candidate_mask=torch.zeros(3, 3, dtype=torch.bool),
        exclusion_window_s=0.01,
        temperature=0.2,
        min_negatives=2,
    )
    assert preserved.candidate_mask.diagonal().all()
    assert preserved.negatives_per_anchor.tolist() == [0, 0, 0]

    # With candidates available, the two other same-track horizons remain negatives.
    result = within_track_nce_loss(
        anchors,
        candidates,
        positive,
        ("seq", "seq", "seq"),
        ("seq", "seq", "seq"),
        ("track", "track", "track"),
        ("track", "track", "track"),
        timestamps,
        timestamps,
        candidate_mask=torch.ones(3, 3, dtype=torch.bool),
        exclusion_window_s=0.01,
        temperature=0.2,
        min_negatives=2,
    )

    assert torch.equal(result.positive_weights > 0.0, torch.eye(3, dtype=torch.bool))
    assert result.candidate_mask.diagonal().all()
    assert result.negatives_per_anchor.tolist() == [2, 2, 2]
    assert result.valid_anchor_fraction.item() == 1.0
    validate_nce_preflight(result)
    result.loss.backward()
    assert anchors.grad is not None

    try:
        within_track_nce_loss(
            anchors.detach().clone().requires_grad_(True),
            candidates,
            positive,
            ("seq", "seq", "seq"),
            ("seq", "seq", "seq"),
            ("track-a", "track-a", "track-a"),
            ("track-b", "track-b", "track-b"),
            timestamps,
            timestamps,
            exclusion_window_s=0.01,
            temperature=0.2,
            min_negatives=2,
        )
    except ValueError as exc:
        assert "share both sequence and track" in str(exc)
    else:  # pragma: no cover - protects the no-cross-track fallback rule
        raise AssertionError("cross-track candidates must never be converted into positives")


def test_nce_exclusion_is_centered_on_each_desired_future_timestamp() -> None:
    predicted = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    candidates = torch.eye(3)
    desired_future = torch.tensor([10.0])
    candidate_timestamps = torch.tensor([10.0, 10.005, 10.1])
    positive = torch.tensor([[1.0, 0.0, 0.0]])

    result = within_track_nce_loss(
        predicted,
        candidates,
        positive,
        ("sequence",),
        ("sequence", "sequence", "sequence"),
        ("track",),
        ("track", "track", "track"),
        desired_future,
        candidate_timestamps,
        exclusion_window_s=0.01,
        temperature=0.2,
        min_negatives=1,
    )

    # Candidate 1 is near the requested 10.0 s future and must not become a
    # negative.  Candidate 2 remains a same-track far negative.  Positive mass
    # survives independently of the exclusion rule.
    assert result.candidate_mask[0, 0]
    assert not result.negative_mask[0, 1]
    assert result.negative_mask[0, 2]
    assert result.negatives_per_anchor.tolist() == [1]
    assert result.valid_anchor_mask.tolist() == [True]


def test_residual_visreg_is_deterministic_finite_and_differentiable() -> None:
    tokens = torch.randn(2, 3, 4, 6, requires_grad=True)
    mask = torch.tensor(
        [
            [[True, True, True, True], [True, True, False, False], [True, True, True, False]],
            [[True, True, True, False], [True, True, True, True], [False, True, True, True]],
        ]
    )
    first = residual_visreg_loss(
        tokens,
        mask,
        generator=torch.Generator().manual_seed(91),
        projections=5,
        temperature=0.12,
    )
    second = residual_visreg_loss(
        tokens,
        mask,
        generator=torch.Generator().manual_seed(91),
        projections=5,
        temperature=0.12,
    )

    assert torch.isfinite(first.loss)
    assert torch.equal(first.loss, second.loss)
    first.loss.backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
