import inspect
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from e_jepa_ttc.models.object_event_v4_29 import (
    LocalCorrelation,
    ObjectEventTTCV429,
    ObjectEventV429Config,
    compose_affines,
    normalized_coordinate_grid,
)
from e_jepa_ttc.training.object_event_v4_29 import (
    ObjectEventV429LossConfig,
    common_roi_box_invariants,
    oof_gates,
    seed_dominance,
)
from scripts.analyze_object_event_v4_29_local_affine import (
    _extra_metrics,
    _frame,
    aggregate_seed_results,
    calibration_slopes,
    choose_champion,
    factorial_effect_table,
)
from scripts.preflight_object_event_v4_29 import validate_checkpoints, validate_config
from scripts.train_e_jepa_object_event_v4_6 import MaterializedV46Split


def _bare() -> ObjectEventTTCV429:
    model = object.__new__(ObjectEventTTCV429)
    model.config = ObjectEventV429Config(min_effective_mass=0.1)
    return model


def _correlation(
    displacement: torch.Tensor, weight: torch.Tensor | None = None
) -> LocalCorrelation:
    h, w = displacement.shape[:2]
    weight = torch.ones(1, h, w) if weight is None else weight
    return LocalCorrelation(
        displacement[None], torch.zeros(1, h, w), torch.ones(1, h, w), torch.zeros(1, h, w), weight
    )


def test_identity_translation_anisotropy_rotation_and_composition():
    model = _bare()
    grid = normalized_coordinate_grid(5, 5, device=torch.device("cpu"), dtype=torch.float32)
    identity, _ = model._fit_affine(_correlation(torch.zeros_like(grid)))
    torch.testing.assert_close(identity.matrix[0], torch.eye(2), atol=0.01, rtol=0.01)
    translated, _ = model._fit_affine(_correlation(torch.full_like(grid, 0.1)))
    torch.testing.assert_close(
        translated.translation[0], torch.tensor([0.1, 0.1]), atol=0.015, rtol=0.0
    )
    target = torch.einsum("hwi,ji->hwj", grid, torch.tensor([[1.0, 0.0], [0.0, 1.5]]))
    vertical, _ = model._fit_affine(_correlation(target - grid))
    assert vertical.matrix[0, 1, 1] > 1.35
    assert (
        torch.log(torch.linalg.vector_norm(vertical.matrix[0] @ torch.tensor([1.0, 0.0]))).abs()
        < 0.03
    )
    assert torch.log(
        torch.linalg.vector_norm(vertical.matrix[0] @ torch.tensor([0.0, 1.0]))
    ).item() == pytest.approx(torch.log(torch.tensor(1.5)).item(), abs=0.03)
    theta = 0.25
    r = torch.tensor(
        [
            [torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta))],
            [torch.sin(torch.tensor(theta)), torch.cos(torch.tensor(theta))],
        ]
    )
    rotated, _ = model._fit_affine(_correlation(torch.einsum("hwi,ji->hwj", grid, r) - grid))
    assert rotated.determinant[0] > 0.9
    composed_a, composed_t = compose_affines(identity, translated)
    torch.testing.assert_close(composed_a, translated.matrix)
    torch.testing.assert_close(composed_t, translated.translation)


def test_reflection_collinear_zero_activity_fail_closed():
    model = _bare()
    grid = normalized_coordinate_grid(5, 5, device=torch.device("cpu"), dtype=torch.float32)
    reflected = grid.clone()
    reflected[..., 0] *= -1
    reflected_fit, _ = model._fit_affine(_correlation(reflected - grid))
    assert not bool(reflected_fit.valid[0])
    zero, zero_penalty = model._fit_affine(
        _correlation(torch.zeros_like(grid), torch.zeros(1, 5, 5))
    )
    assert not bool(zero.valid[0])
    assert zero_penalty.item() > 0.0
    collinear_weight = torch.zeros(1, 5, 5)
    collinear_weight[:, 2] = 1
    collinear, _ = model._fit_affine(_correlation(torch.zeros_like(grid), collinear_weight))
    assert not bool(collinear.valid[0])


def test_local_boundary_entropy_and_finite_backward():
    model = _bare()
    previous = torch.randn(1, 4, 5, 5, requires_grad=True)
    current = torch.randn_like(previous, requires_grad=True)
    local = model._local_correlation(previous, current, torch.ones(1, 5, 5), 1)
    assert torch.isfinite(local.entropy).all() and torch.isfinite(local.boundary_probability).all()
    assert ((local.boundary_probability >= 0) & (local.boundary_probability <= 1)).all()
    fit, validity_penalty = model._fit_affine(local)
    assert validity_penalty.requires_grad
    (fit.matrix.square().sum() + fit.translation.square().sum() + validity_penalty.sum()).backward()
    assert previous.grad is not None
    assert current.grad is not None
    assert torch.isfinite(previous.grad).all() and torch.isfinite(current.grad).all()


def test_teacher_uses_exact_t1_t2_only_and_forward_contract_is_event_only():
    boxes = torch.tensor(
        [[[0.0, 0.0, 100.0, 100.0], [10.0, 20.0, 50.0, 60.0], [12.0, 24.0, 92.0, 104.0]]]
    )
    target = common_roi_box_invariants(boxes, height=128, width=128)
    assert target["box_log_height_ratio_t1_t2"].item() == pytest.approx(
        torch.log(torch.tensor(0.5)).item()
    )
    changed = boxes.clone()
    changed[:, 0] = 999.0
    again = common_roi_box_invariants(changed, height=128, width=128)
    torch.testing.assert_close(target["center_t1"], again["center_t1"])
    assert list(inspect.signature(ObjectEventTTCV429.forward).parameters) == ["self", "events"]
    assert ObjectEventV429LossConfig().arm == "local_affine_lhr"


def test_rng_dominance_and_gate_fail_closed():
    assert (
        seed_dominance({7: 0.0, 13: 0.04, 23: 0.01}, {7: 0.0, 13: 0.01, 23: 0.0})
        == "matcher_dominant"
    )
    assert (
        seed_dominance({7: 0.0, 13: 0.01, 23: 0.0}, {7: 0.0, 13: 0.01, 23: 0.0})
        == "neither_inconclusive"
    )
    assert (
        seed_dominance({7: 0.0, 13: 0.01, 23: 0.0}, {7: 0.0, 13: 0.05, 23: 0.0})
        == "backbone_dominant"
    )
    assert oof_gates({}, {}, {}) == {"complete_finite": False}


def test_pixel_center_displacement_calibration_and_lexical_tie():
    grid = normalized_coordinate_grid(2, 4, device=torch.device("cpu"), dtype=torch.float32)
    torch.testing.assert_close(grid[0, 0], torch.tensor([-0.75, -0.5]))
    torch.testing.assert_close(grid[-1, -1], torch.tensor([0.75, 0.5]))
    # prediction = 2 * target: orientation must report slope two, not one half.
    intercept, zero = calibration_slopes(
        torch.tensor([2.0, 4.0, 6.0]).numpy(), torch.tensor([1.0, 2.0, 3.0]).numpy()
    )
    assert intercept == pytest.approx(2.0) and zero == pytest.approx(2.0)
    tied = {
        "z_arm": {
            "oof_metrics": {
                "pearson": 0.7,
                "minimum_sequence_pearson": 0.5,
                "negative_accuracy": 0.6,
            }
        },
        "a_arm": {
            "oof_metrics": {
                "pearson": 0.7,
                "minimum_sequence_pearson": 0.5,
                "negative_accuracy": 0.6,
            }
        },
    }
    assert choose_champion(tied) == "a_arm"


def test_endpoint_reversal_mapping_and_frame_schema():
    model = _bare()
    grid = normalized_coordinate_grid(5, 5, device=torch.device("cpu"), dtype=torch.float32)
    forward, _ = model._fit_affine(_correlation(torch.full_like(grid, 0.1)))
    reverse, _ = model._fit_affine(_correlation(torch.full_like(grid, -0.1)))
    composed_a, composed_t = compose_affines(reverse, forward)
    torch.testing.assert_close(composed_a, torch.eye(2)[None], atol=0.03, rtol=0.0)
    assert torch.linalg.vector_norm(composed_t).item() < 0.03

    class Split:
        delta_t_s = torch.tensor([0.1])
        target_ttc_s = torch.tensor([1.0])
        sequence_ids = ["s"]
        sample_tokens = ["x"]
        track_ids = ["t"]

    split = cast(MaterializedV46Split, Split())
    assert {"delta_t_s", "target_ttc_s", "target_expansion"}.issubset(_frame(split).columns)


def test_fixed_effect_table_reports_crossover():
    cells = []
    for b in (7, 13, 23):
        for m in (7, 13, 23):
            cells.append(
                {
                    "backbone_checkpoint_seed": b,
                    "matcher_init_seed": m,
                    "cell_metrics": {"pearson": 0.1 if (b == m) else -0.1},
                }
            )
    table = factorial_effect_table(cells, "pearson")
    assert table["metric"] == "pearson" and table["interaction_range"] > 0


def test_entropy_uses_valid_candidate_count_and_identity_anisotropy():
    model = _bare()
    previous = torch.ones(1, 2, 5, 5)
    current = torch.ones_like(previous)
    local = model._local_correlation(previous, current, torch.ones(1, 5, 5), 1)
    # Uniform logits have maximal entropy at both a four-candidate corner and nine-candidate centre.
    assert local.entropy[0, 0, 0].item() == pytest.approx(1.0, abs=1e-5)
    assert local.entropy[0, 2, 2].item() == pytest.approx(1.0, abs=1e-5)
    grid = normalized_coordinate_grid(5, 5, device=torch.device("cpu"), dtype=torch.float32)
    fit, _ = model._fit_affine(_correlation(torch.zeros_like(grid)))
    assert torch.log(
        torch.linalg.svdvals(fit.matrix)[0, 0] / torch.linalg.svdvals(fit.matrix)[0, 1]
    ).item() == pytest.approx(0.0, abs=0.01)


def test_robust_mass_and_nonfinite_affine_fail_closed():
    model = _bare()
    grid = normalized_coordinate_grid(5, 5, device=torch.device("cpu"), dtype=torch.float32)
    corr = _correlation(torch.zeros_like(grid))
    corr.weight[:, 0, 0] = 100.0
    corr.displacement[:, 0, 0] = 10.0
    fit, _ = model._fit_affine(corr)
    assert fit.effective_weight_mass.item() < corr.weight.sum().item()
    bad = _correlation(torch.zeros_like(grid))
    bad.displacement[0, 0, 0, 0] = float("nan")
    failed, _ = model._fit_affine(bad)
    assert not bool(failed.valid[0]) and torch.isfinite(failed.matrix).all()


def test_seed_aggregation_propagates_any_invalid_seed_and_worst_reason():
    base = {
        "prediction": np.array([0.1]),
        "log_eta": np.array([-0.1]),
        "horizontal": np.array([-0.1]),
        "area": np.array([-0.1]),
        "valid": np.array([1.0]),
        "det": np.array([1.0]),
        "condition": np.array([10.0]),
        "mass": np.array([8.0]),
        "residual": np.array([0.01]),
        "boundary": np.array([0.1]),
    }
    bad = {key: value.copy() for key, value in base.items()}
    bad["valid"][0] = 0.0
    bad["prediction"][0] = np.nan
    bad["condition"][0] = 120.0
    aggregate = aggregate_seed_results([base, bad, base])
    assert aggregate["valid"][0] == 0.0
    assert np.isnan(aggregate["prediction"][0])
    assert aggregate["condition"][0] == 120.0


def test_protocol_validator_rejects_extra_keys():
    raw = yaml.safe_load(
        Path("configs/experiment/e_jepa_garl_object_event_local_affine_v4_29.yaml").read_text()
    )
    validate_config(raw)
    raw["selection"]["unlocked"] = True
    with pytest.raises(ValueError, match="selection keys"):
        validate_config(raw)


def test_checkpoint_validator_exercises_strict_loader(tmp_path, monkeypatch):
    checkpoint = tmp_path / "adapted_seed_7.pt"
    torch.save(
        {
            "artifact_type": "object_event_v4_22_adapted_v48",
            "seed": 7,
            "source_checkpoint": "artifacts/debug/x/screen-seed-7/best_gate_passing.pt",
            "model_state_dict": {"bad": torch.zeros(1)},
        },
        checkpoint,
    )

    def reject_strict_load(**_kwargs):
        raise ValueError("strict state mismatch")

    monkeypatch.setattr("scripts.preflight_object_event_v4_29._load_backbone", reject_strict_load)
    with pytest.raises(ValueError, match="strict state mismatch"):
        validate_checkpoints({7: checkpoint}, Path("unused.yaml"))


def test_invalid_metrics_preserve_horizontal_and_area_nonfinite_reasons():
    result = {
        key: np.ones(2, dtype=np.float64)
        for key in (
            "prediction",
            "log_eta",
            "horizontal",
            "area",
            "valid",
            "det",
            "condition",
            "mass",
            "residual",
        )
    }
    result["horizontal"][0] = np.nan
    result["area"][1] = np.nan
    metrics = _extra_metrics(
        pd.DataFrame({"target_expansion": [0.0, 0.0]}),
        cast(MaterializedV46Split, object()),
        result,
    )
    reasons = metrics["invalid_reason_counts"]
    assert reasons["nonfinite_horizontal"] == 1
    assert reasons["nonfinite_area"] == 1
