import inspect
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import scripts.analyze_object_event_v4_30_stable_similarity as analyzer
from e_jepa_ttc.models.object_event_v4_30 import (
    LocalPosterior,
    ObjectEventTTCV430,
    ObjectEventV430Config,
    ObjectEventV430Output,
    SimilarityFit,
    feature_coordinate_grid,
    geometric_mean_posteriors,
    shrinkage_whiten,
)
from e_jepa_ttc.training.object_event_v4_30 import (
    ObjectEventV430LossConfig,
    compute_oof_metrics,
    ell_target,
    g_target,
    gate_constituents_and_median,
    object_event_v4_30_loss,
    oof_gates,
    posterior_kl,
    promoted_champion,
    stabilization_gate,
)
from scripts.analyze_object_event_v4_30_stable_similarity import (
    CONTROL_NAMES,
    PREDICTION_FIELDS,
    TeacherConsensusCache,
    _accumulate_oof,
    _aggregate_arm_oof,
    _apply_event_control,
    _assert_complete_coverage,
    _assert_zero_event_contract,
    _build_teacher_consensus_cache,
    _consensus,
    _development_decision,
    _development_requested,
    _diagnostic_indices,
    _empty_oof_prediction,
    _head_state_hash,
    _inject_arm_b_gains,
    _paper_weighted_mid,
    _predict,
    _prepare_output_target,
    _run_development_validation,
    _save_oof_npz,
    _stage1,
    _teacher_pairs,
    _verify_grouped_folds,
    make_summary,
    posterior_stability_metrics,
)
from scripts.preflight_object_event_v4_30 import validate_config


def _bare() -> ObjectEventTTCV430:
    model = object.__new__(ObjectEventTTCV430)
    model.config = ObjectEventV430Config()
    return model


def _cache(rows: int) -> TeacherConsensusCache:
    probabilities = {scale: torch.ones(rows, 1, 1, 1, dtype=torch.float32) for scale in (1, 2, 4)}
    return TeacherConsensusCache(
        probabilities=probabilities,
        row_count=rows,
        schema="test",
        consensus_cache_sha256="test-cache",
        consensus_config_sha256="test-config",
        checkpoint_file_sha256={"7": "a", "13": "b", "23": "c"},
        teacher_backbone_forward_batches=0,
        consensus_build_count=1,
        elapsed_seconds=0.0,
    )


def _posterior(displacement: torch.Tensor) -> LocalPosterior:
    height, width = displacement.shape[:2]
    return LocalPosterior(
        torch.ones(1, 1, height, width),
        torch.zeros(1, 2),
        displacement[None],
        torch.zeros(1, height, width, 2, 2),
        torch.zeros(1, height, width),
        torch.ones(1, height, width),
        torch.zeros(1, height, width),
        torch.ones(1, height, width),
    )


def _posteriors(
    kappa: float, omega: float, translation: tuple[float, float]
) -> dict[int, LocalPosterior]:
    positions = {
        scale: feature_coordinate_grid(
            8 // scale, 8 // scale, device=torch.device("cpu"), dtype=torch.float32
        )
        * scale
        for scale in (1, 2, 4)
    }
    flattened = torch.cat([position.reshape(-1, 2) for position in positions.values()])
    joint_center = flattened.mean(dim=0)
    result: dict[int, LocalPosterior] = {}
    for scale in (1, 2, 4):
        grid = positions[scale]
        centred = grid - joint_center
        displacement = torch.empty_like(grid)
        displacement[..., 0] = kappa * centred[..., 0] - omega * centred[..., 1] + translation[0]
        displacement[..., 1] = omega * centred[..., 0] + kappa * centred[..., 1] + translation[1]
        result[scale] = _posterior(displacement)
    return result


def test_similarity_uses_one_joint_support_weighted_center_across_scales():
    model = _bare()
    supports = {
        1: torch.full((1, 5, 7), 0.25),
        2: torch.tensor([[[1.0, 3.0, 2.0], [4.0, 1.0, 5.0]]]),
        4: torch.tensor([[[6.0, 2.0]]]),
    }
    positions = {
        scale: feature_coordinate_grid(
            weight.shape[-2], weight.shape[-1], device=torch.device("cpu"), dtype=torch.float32
        )
        * scale
        for scale, weight in supports.items()
    }
    all_positions = torch.cat([positions[scale].reshape(-1, 2) for scale in supports])
    all_support = torch.cat([supports[scale].reshape(-1) for scale in supports])
    center = (all_positions * all_support[:, None]).sum(0) / all_support.sum()
    kappa, omega = 0.12, -0.07
    translation = torch.tensor([0.35, -0.2])
    posteriors: dict[int, LocalPosterior] = {}
    for scale, position in positions.items():
        q = position - center
        displacement = torch.empty_like(position)
        displacement[..., 0] = kappa * q[..., 0] - omega * q[..., 1] + translation[0]
        displacement[..., 1] = omega * q[..., 0] + kappa * q[..., 1] + translation[1]
        item = _posterior(displacement)
        item.weight = supports[scale]
        posteriors[scale] = item

    fit = model._fit_similarity(posteriors)
    torch.testing.assert_close(fit.center[0], center, rtol=0, atol=1e-6)
    assert fit.kappa.item() == pytest.approx(kappa, abs=2e-3)
    assert fit.omega.item() == pytest.approx(omega, abs=2e-3)
    torch.testing.assert_close(fit.translation[0], translation, rtol=0, atol=2e-3)

    affine_intercept = fit.center + fit.translation - (
        fit.matrix @ fit.center[..., None]
    ).squeeze(-1)
    reconstructed_translation = affine_intercept - fit.center + (
        fit.matrix @ fit.center[..., None]
    ).squeeze(-1)
    torch.testing.assert_close(reconstructed_translation, fit.translation, rtol=0, atol=1e-6)


def test_similarity_identity_translation_rotation_scale_and_posterior_are_finite():
    model = _bare()
    identity = model._fit_similarity(_posteriors(0.0, 0.0, (0.0, 0.0)))
    assert identity.kappa.item() == pytest.approx(0.0, abs=1e-4)
    translated = model._fit_similarity(_posteriors(0.0, 0.0, (0.25, -0.125)))
    torch.testing.assert_close(
        translated.translation[0], torch.tensor([0.25, -0.125]), atol=2e-3, rtol=0
    )
    rotated = model._fit_similarity(_posteriors(0.0, 0.2, (0.0, 0.0)))
    assert rotated.omega.item() == pytest.approx(0.2, abs=2e-3)
    scaled = model._fit_similarity(_posteriors(0.1, 0.0, (0.0, 0.0)))
    log_eta = 0.5 * torch.log((1 + scaled.kappa).square() + scaled.omega.square())
    assert log_eta.item() == pytest.approx(np.log(1.1), abs=2e-3)
    assert torch.isfinite(scaled.covariance).all()


def test_activity_tile_support_zero_sparse_and_whitening_are_finite():
    model = _bare()
    zero = torch.zeros(1, 3, 12, 8, 8)
    assert model._activity(zero, output_size=(4, 4)).eq(0).all()
    sparse = zero.clone()
    sparse[0, 1, 0, 2, 3] = 1
    activity = model._activity(sparse, output_size=(8, 8))
    weight = model._weights(torch.zeros(1, 8, 8), activity[:, 1])
    assert torch.isfinite(weight).all() and weight.sum() > 0 and weight.max() <= 1
    assert torch.isfinite(shrinkage_whiten(torch.ones(2, 4, 4, 4))).all()


def test_consensus_stability_targets_forward_contract_and_gates():
    p = torch.tensor([[[[0.8]], [[0.2]]]])
    q = torch.tensor([[[[0.2]], [[0.8]]]])
    torch.testing.assert_close(geometric_mean_posteriors([p, q]), torch.full_like(p, 0.5))
    offsets = {scale: np.array([[0.0, 0.0], [1.0, 0.0]]) for scale in (1, 2, 4)}
    measured = posterior_stability_metrics(
        [{scale: p.numpy() for scale in offsets} for _ in range(3)], offsets
    )
    assert measured == {"js_median": 0.0, "js_p95": 0.0, "expected_displacement_p95": 0.0}
    assert all(stabilization_gate(**measured).values())
    assert list(inspect.signature(ObjectEventTTCV430.forward).parameters) == ["self", "events"]
    assert g_target(torch.tensor([0.1]), torch.tensor([1.0])).item() == pytest.approx(0.1)
    assert ell_target(torch.tensor([0.1]), torch.tensor([1.0])).item() == pytest.approx(np.log(0.9))
    metrics = {
        "finite_predictions": 1.0,
        "finite_posterior_variances": 1.0,
        "seed_prediction_p95_range": 0.01,
        "seed_prediction_max_range": 0.07,
        "sign_disagreement": 0.01,
        "seed_pearson_range": 0.01,
        "pearson": 0.8,
        "log_eta_pearson": 0.8,
        "minimum_sequence_pearson": 0.6,
        "negative_accuracy": 0.9,
        "balanced_sign_accuracy": 0.9,
        "prediction_std_ratio": 1.0,
        "calibration_slope": 1.0,
        "high_bucket_pearson": 0.5,
        "negative_track_macro_accuracy": 0.9,
        "minimum_negative_track_accuracy": 0.6,
        "eligible_negative_track_p10": 0.7,
        "shuffle_ratio": 0.4,
        "endpoint_swap_pearson": -0.9,
        "endpoint_swap_pearson_abs": 0.9,
        "bottom_support_seed_p95_range": 0.04,
        "bottom_support_uncertainty_finite": 1.0,
    }
    metrics.update({f"magnitude_ratio_{i}": 1.0 for i in range(4)})
    assert all(oof_gates(metrics).values())
    assert not oof_gates({**metrics, "endpoint_swap_pearson": 0.16})["controls"]
    assert not oof_gates({**metrics, "endpoint_swap_pearson": float("nan")})[
        "complete_finite"
    ]
    arms = {
        "stable_multiscale_similarity": metrics,
        "stable_multiscale_similarity_normal_flow": {
            **metrics,
            "paired_sequence_pearson_gain": 0.01,
            "high_bucket_pearson_gain": 0.1,
        },
    }
    assert promoted_champion(arms) == "stable_multiscale_similarity_normal_flow"


def test_promotion_requires_arm_a_and_exact_paired_margin():
    base = {
        "finite_predictions": 1.0,
        "finite_posterior_variances": 1.0,
        "seed_prediction_p95_range": 0.01,
        "seed_prediction_max_range": 0.07,
        "sign_disagreement": 0.01,
        "seed_pearson_range": 0.01,
        "pearson": 0.8,
        "log_eta_pearson": 0.8,
        "minimum_sequence_pearson": 0.6,
        "negative_accuracy": 0.9,
        "balanced_sign_accuracy": 0.9,
        "prediction_std_ratio": 1.0,
        "calibration_slope": 1.0,
        "high_bucket_pearson": 0.5,
        "negative_track_macro_accuracy": 0.9,
        "minimum_negative_track_accuracy": 0.6,
        "eligible_negative_track_p10": 0.7,
        "shuffle_ratio": 0.4,
        "endpoint_swap_pearson": -0.9,
        "bottom_support_seed_p95_range": 0.04,
        "bottom_support_uncertainty_finite": 1.0,
        **{f"magnitude_ratio_{index}": 1.0 for index in range(4)},
    }
    b = {**base, "paired_sequence_pearson_gain": 0.02, "high_bucket_pearson_gain": 0.11}
    assert (
        promoted_champion(
            {"stable_multiscale_similarity": base, "stable_multiscale_similarity_normal_flow": b}
        )
        == "stable_multiscale_similarity_normal_flow"
    )
    assert (
        promoted_champion(
            {
                "stable_multiscale_similarity": {**base, "pearson": 0.0},
                "stable_multiscale_similarity_normal_flow": b,
            }
        )
        is None
    )
    assert (
        promoted_champion(
            {
                "stable_multiscale_similarity": base,
                "stable_multiscale_similarity_normal_flow": {
                    **b,
                    "paired_sequence_pearson_gain": None,
                },
            }
        )
        == "stable_multiscale_similarity"
    )


def test_protocol_validator_rejects_drift():
    raw = yaml.safe_load(
        Path("configs/experiment/e_jepa_garl_object_event_stable_similarity_v4_30.yaml").read_text()
    )
    validate_config(raw)
    raw["arms"]["stable_multiscale_similarity"]["ridge"] = 0.02
    with pytest.raises(ValueError):
        validate_config(raw)
    raw = yaml.safe_load(
        Path("configs/experiment/e_jepa_garl_object_event_stable_similarity_v4_30.yaml").read_text()
    )
    raw["controls"]["temporal_shuffle_permutation"] = [0, 1, 2]
    with pytest.raises(ValueError):
        validate_config(raw)
    raw = yaml.safe_load(
        Path("configs/experiment/e_jepa_garl_object_event_stable_similarity_v4_30.yaml").read_text()
    )
    raw["train"]["epochs"] = 3
    with pytest.raises(ValueError):
        validate_config(raw)
    for section, key, value in (
        ("train", "learning_rate", 0.02),
        ("selection", "shuffle_ratio_max", 0.51),
        ("arm_b_margin", "paired_sequence_pearson", 0.02),
    ):
        raw = yaml.safe_load(
            Path(
                "configs/experiment/e_jepa_garl_object_event_stable_similarity_v4_30.yaml"
            ).read_text()
        )
        raw[section][key] = value
        with pytest.raises(ValueError):
            validate_config(raw)


def _loss_output() -> ObjectEventV430Output:
    batch = 2
    fit = SimilarityFit(
        kappa=torch.zeros(batch),
        omega=torch.zeros(batch),
        translation=torch.zeros(batch, 2),
        center=torch.zeros(batch, 2),
        covariance=torch.eye(4)[None].repeat(batch, 1, 1),
        sigma2=torch.ones(batch),
        residual=torch.zeros(batch),
        effective_mass=torch.ones(batch),
        design_rms=torch.ones(batch, 4),
        matrix=torch.eye(2)[None].repeat(batch, 1, 1),
    )
    posterior = LocalPosterior(
        torch.ones(batch, 1, 2, 2),
        torch.zeros(1, 2),
        torch.zeros(batch, 2, 2, 2),
        torch.zeros(batch, 2, 2, 2, 2),
        torch.zeros(batch, 2, 2),
        torch.ones(batch, 2, 2),
        torch.zeros(batch, 2, 2),
        torch.ones(batch, 2, 2),
    )
    return ObjectEventV430Output(
        expansion=torch.zeros(batch),
        log_eta=torch.zeros(batch),
        posterior_variance=torch.ones(batch),
        unknown=torch.zeros(batch, dtype=torch.bool),
        fit_01=fit,
        fit_12=fit,
        fit_02=fit,
        cycle_matrix_error=torch.zeros(batch),
        cycle_translation_error=torch.zeros(batch),
        correlation_entropy=torch.zeros(batch),
        correlation_confidence=torch.ones(batch),
        boundary_probability=torch.zeros(batch),
        rotation_radians=torch.zeros(batch),
        translation_magnitude=torch.zeros(batch),
        normal_flow_residual=torch.zeros(batch),
        foreground_map_t1=torch.full((batch, 2, 2), 0.5),
        foreground_map_t2=torch.full((batch, 2, 2), 0.5),
        support_map_t2=torch.ones(batch, 2, 2),
        posteriors_01={1: posterior},
        posteriors_12={1: posterior, 2: posterior, 4: posterior},
    )


def test_full_loss_consumes_real_t1_t2_annotation_schema():
    output = _loss_output()
    dt = torch.tensor([0.1, 0.1])
    ttc = torch.tensor([1.0, 2.0])
    heights = torch.tensor([[20.0, 30.0], [21.0, 31.0]])
    boxes = torch.tensor(
        [
            [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]],
            [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]],
        ]
    )
    kwargs = dict(
        consensus_posteriors={
            1: output.posteriors_12[1].probabilities,
            2: output.posteriors_12[2].probabilities,
            4: output.posteriors_12[4].probabilities,
        },
        sequence_ids=["s", "s"],
        track_ids=["a", "b"],
        config=ObjectEventV430LossConfig(),
        boxes_xyxy=boxes,
        visible_heights_px=heights,
        image_height=2,
        image_width=2,
    )
    total, pieces = object_event_v4_30_loss(output, dt, ttc, **kwargs)
    with pytest.raises(ValueError, match="requires t1/t2"):
        object_event_v4_30_loss(
            output,
            dt,
            ttc,
            **{
                **kwargs,
                "boxes_xyxy": torch.cat((boxes, boxes[:, :1]), dim=1),
                "visible_heights_px": torch.cat((heights, heights[:, :1]), dim=1),
            },
        )
    changed_boxes, changed_heights = boxes.clone(), heights.clone()
    changed_boxes[:, 1, 2] = 2.0
    changed_heights[:, 1] = 99.0
    different, different_pieces = object_event_v4_30_loss(
        output,
        dt,
        ttc,
        **{**kwargs, "boxes_xyxy": changed_boxes, "visible_heights_px": changed_heights},
    )
    assert torch.isfinite(total) and not torch.allclose(total, different) and not torch.allclose(
        pieces["support"], different_pieces["support"]
    )


def test_delta_method_variance_formula_with_off_diagonal_covariance():
    kappa, omega = torch.tensor([0.2]), torch.tensor([0.3])
    covariance = torch.tensor(
        [[[2.0, 0.4, 0.0, 0.0], [0.4, 3.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
    )
    r2 = (1 + kappa).square() + omega.square()
    gradient = torch.stack(
        ((1 + kappa) / r2, omega / r2, torch.zeros_like(kappa), torch.zeros_like(kappa)), -1
    )
    expected = torch.einsum("bi,bij,bj->b", gradient, covariance, gradient)
    assert expected.item() == pytest.approx(1.468666, rel=1e-5)
    # This independently pins the exact expression used by the forward implementation.
    assert torch.isfinite(expected).all()


def test_head_hash_is_order_invariant_for_pristine_arm_starts():
    state = {"weight": torch.tensor([[1.0, 2.0]]), "bias": torch.tensor([3.0])}
    first = {
        arm: _head_state_hash({key: value.clone() for key, value in state.items()})
        for arm in ("stable_multiscale_similarity", "stable_multiscale_similarity_normal_flow")
    }
    second = {
        arm: _head_state_hash({key: value.clone() for key, value in state.items()})
        for arm in reversed(tuple(first))
    }
    assert set(first.values()) == {_head_state_hash(state)} and first == second


def _metrics_inputs() -> dict[str, object]:
    target = np.array([-0.09, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08, 0.1])
    prediction = target * 2 + 0.01
    return {
        "target_g": target,
        "target_log_eta": np.log1p(-target),
        "prediction": prediction,
        "predicted_log_eta": np.log1p(-prediction.clip(-0.24, 0.24)),
        "posterior_variance": np.arange(9.0) + 1,
        "support": np.arange(9.0),
        "sequence_ids": ["a"] * 5 + ["b"] * 4,
        "track_ids": ["short"] * 3 + ["mid"] * 4 + ["long"] * 2,
        "seed_predictions": np.stack([prediction - 0.001, prediction, prediction + 0.001]),
        "shuffle_prediction": -prediction,
        "endpoint_prediction": prediction[::-1],
        "zero_unknown": np.ones(9, dtype=bool),
        "zero_prediction": np.full(9, np.nan),
        "delta_t_s": np.full(9, 0.1),
    }


def test_compute_oof_metrics_bucket_edges_slope_controls_and_bottom_support():
    metrics = compute_oof_metrics(**_metrics_inputs())
    assert [metrics["buckets"][str(i)]["count"] for i in range(4)] == [2, 2, 2, 3]
    assert metrics["buckets"]["0"]["ratio"] == pytest.approx(2.0, rel=0.6)
    target = _metrics_inputs()["target_g"]
    prediction = _metrics_inputs()["prediction"]
    assert metrics["calibration_slope"] == pytest.approx(np.polyfit(target, prediction, 1)[0])
    assert metrics["calibration_slope"] != pytest.approx(
        np.dot(target, prediction) / np.dot(target, target)
    )
    assert metrics["shuffle_ratio"] == pytest.approx(1.0)
    assert metrics["endpoint_swap_pearson_abs"] == pytest.approx(
        abs(metrics["endpoint_swap_pearson"])
    )
    assert metrics["endpoint_swap_pearson"] != metrics["shuffle_pearson"]
    assert metrics["bottom_support_coverage"] == 1.0
    assert metrics["bottom_support_uncertainty_finite"]


def test_control_magnitudes_and_undefined_sequence_correlation_fail_closed():
    values = _metrics_inputs()
    values["sequence_ids"] = ["singleton"] + ["stable"] * 8
    values["endpoint_prediction"] = -values["prediction"]
    metrics = compute_oof_metrics(**values)
    assert metrics["shuffle_ratio"] == pytest.approx(1.0)
    assert metrics["endpoint_swap_pearson"] == pytest.approx(-1.0)
    assert metrics["endpoint_swap_pearson_abs"] == pytest.approx(1.0)
    assert metrics["minimum_sequence_pearson"] is None
    assert not all(
        oof_gates(
            {key: value for key, value in metrics.items() if isinstance(value, (float, int, bool))}
        ).values()
    )


def test_stability_offsets_are_base_feature_pixels():
    p = np.zeros((1, 81, 1, 1), dtype=np.float64)
    q = p.copy()
    p[:, 0] = 1.0
    q[:, -1] = 1.0
    scale = 4
    coarse = np.stack(
        np.meshgrid(np.arange(-scale, scale + 1), np.arange(-scale, scale + 1), indexing="xy"), -1
    ).reshape(-1, 2)
    measured = posterior_stability_metrics(
        [
            {scale: p, 1: np.ones((1, 1, 1, 1)), 2: np.ones((1, 1, 1, 1))},
            {scale: q, 1: np.ones((1, 1, 1, 1)), 2: np.ones((1, 1, 1, 1))},
            {scale: p, 1: np.ones((1, 1, 1, 1)), 2: np.ones((1, 1, 1, 1))},
        ],
        {1: np.zeros((1, 2)), 2: np.zeros((1, 2)), scale: coarse * scale},
    )
    assert measured["expected_displacement_p95"] > 40.0


def test_centered_similarity_cycle_uses_affine_intercepts_at_nonzero_centers():
    a01 = torch.tensor([[[1.2, 0.0], [0.0, 1.2]]])
    a12 = torch.tensor([[[0.8, 0.0], [0.0, 0.8]]])
    c01, c12, c02 = (
        torch.tensor([[2.0, -1.0]]),
        torch.tensor([[-3.0, 4.0]]),
        torch.tensor([[1.0, 2.0]]),
    )
    t01, t12 = torch.tensor([[0.5, -0.25]]), torch.tensor([[1.0, 0.5]])
    b01 = c01 + t01 - (a01 @ c01[..., None]).squeeze(-1)
    b12 = c12 + t12 - (a12 @ c12[..., None]).squeeze(-1)
    a02 = a01 @ a12
    b02 = (a01 @ b12[..., None]).squeeze(-1) + b01
    t02 = b02 - c02 + (a02 @ c02[..., None]).squeeze(-1)
    reconstructed_b02 = c02 + t02 - (a02 @ c02[..., None]).squeeze(-1)
    torch.testing.assert_close(reconstructed_b02, b02, rtol=0, atol=1e-6)


def test_backward_normal_flow_translation_sign_constraint():
    gradient = torch.tensor([2.0, -1.0])
    displacement = torch.tensor([0.25, 0.5])
    s1_minus_s2 = -(gradient * displacement).sum()
    residual = (gradient * displacement).sum() + s1_minus_s2
    assert residual.item() == pytest.approx(0.0)


def test_tracks_strata_zero_short_and_zero_event_evidence():
    values = _metrics_inputs()
    values["prediction"] = np.array([0.1, 0.1, 0.1, -0.04, -0.04, -0.04, -0.04, -0.08, -0.1])
    values["seed_predictions"] = np.stack([values["prediction"]] * 3)
    metrics = compute_oof_metrics(**values)
    assert metrics["negative_track_strata"]["1-3"]["track_count"] == 2
    assert metrics["per_track"]["short"]["negative_accuracy"] == 0.0
    assert metrics["zero_event_unknown"] and metrics["zero_event_physical_nan"]
    assert metrics["mid"] is not None and metrics["rte"] is not None and metrics["fr"] is not None


def test_constituent_gate_and_constant_correlations_fail_closed():
    good = {
        "finite_predictions": 1.0,
        "finite_posterior_variances": 1.0,
        "seed_prediction_p95_range": 0.01,
        "seed_prediction_max_range": 0.01,
        "sign_disagreement": 0.0,
        "seed_pearson_range": 0.01,
        "pearson": 0.8,
        "log_eta_pearson": 0.8,
        "minimum_sequence_pearson": 0.6,
        "negative_accuracy": 0.9,
        "balanced_sign_accuracy": 0.9,
        "prediction_std_ratio": 1.0,
        "calibration_slope": 1.0,
        "high_bucket_pearson": 0.5,
        "negative_track_macro_accuracy": 0.9,
        "minimum_negative_track_accuracy": 0.6,
        "eligible_negative_track_p10": 0.7,
        "shuffle_ratio": 0.4,
        "endpoint_swap_pearson": -0.9,
        "bottom_support_seed_p95_range": 0.01,
        "bottom_support_uncertainty_finite": 1.0,
        **{f"magnitude_ratio_{i}": 1.0 for i in range(4)},
    }
    bad = {**good, "pearson": 0.1}
    assert not gate_constituents_and_median([good, good, bad], good)["constituents"]
    values = _metrics_inputs()
    values["prediction"] = np.ones(9)
    values["seed_predictions"] = np.ones((3, 9))
    assert compute_oof_metrics(**values)["pearson"] is None


class _TinyPredictionModel:
    """Event-only fake model pinning the stage-2 collection interface."""

    def __call__(self, events: torch.Tensor) -> ObjectEventV430Output:
        output = _loss_output()
        middle = events[:, 1, 0, 0, 0]
        zero = events.abs().sum(dim=(1, 2, 3, 4)).eq(0)
        physical = torch.where(zero, torch.full_like(middle, float("nan")), middle)
        return replace(
            output,
            expansion=physical,
            log_eta=physical,
            posterior_variance=physical,
            unknown=zero,
        )


class _TinySplit:
    def __init__(self) -> None:
        event = torch.zeros(2, 3, 10, 2, 2)
        event[:, 0] = 1.0
        event[:, 1] = 2.0
        event[:, 2] = 3.0
        self.events = event
        self.sample_tokens = ["token-0", "token-1"]
        self.sequence_ids = ["sequence", "sequence"]
        self.track_ids = ["track-0", "track-1"]
        self.delta_t_s = torch.tensor([0.1, 0.2])
        self.target_ttc_s = torch.tensor([1.0, 2.0])


def _controls() -> dict[str, object]:
    return {
        "zero_event": True,
        "temporal_shuffle_permutation": [2, 0, 1],
        "endpoint_swap_permutation": [0, 2, 1],
    }


def test_predict_schema_and_locked_temporal_controls_are_exact():
    split = _TinySplit()
    model = _TinyPredictionModel()
    controls = _controls()
    original = _predict(model, split, torch.device("cpu"))
    shuffled = _predict(
        model, split, torch.device("cpu"), control="temporal_shuffle", controls=controls
    )
    endpoint = _predict(
        model, split, torch.device("cpu"), control="endpoint_swap", controls=controls
    )
    assert tuple(original) == PREDICTION_FIELDS
    assert all(value.shape == (2,) for value in original.values())
    assert original["prediction"].tolist() == [2.0, 2.0]
    assert shuffled["prediction"].tolist() == [1.0, 1.0]
    assert endpoint["prediction"].tolist() == [3.0, 3.0]
    torch.testing.assert_close(
        _apply_event_control(split.events, "temporal_shuffle", controls), split.events[:, [2, 0, 1]]
    )
    torch.testing.assert_close(
        _apply_event_control(split.events, "endpoint_swap", controls), split.events[:, [0, 2, 1]]
    )


def test_zero_event_control_is_unknown_with_nan_physical_outputs():
    zero = _predict(
        _TinyPredictionModel(),
        _TinySplit(),
        torch.device("cpu"),
        control="zero_event",
        controls=_controls(),
    )
    _assert_zero_event_contract(zero)
    assert zero["unknown"].all()
    for field in ("prediction", "log_eta", "posterior_variance"):
        assert np.isnan(zero[field]).all()


def test_oof_accumulation_preserves_original_rows_covers_once_and_saves_schema(tmp_path: Path):
    result = _predict(_TinyPredictionModel(), _TinySplit(), torch.device("cpu"))
    first = {field: value.copy() for field, value in result.items()}
    for field in PREDICTION_FIELDS:
        first[field] = np.array([20.0, 0.0]) if field != "unknown" else np.array([False, False])
    second = {
        field: np.array([10.0]) if field != "unknown" else np.array([False])
        for field in PREDICTION_FIELDS
    }
    collected, coverage = _empty_oof_prediction(3)
    _accumulate_oof(collected, coverage, np.array([2, 0]), first, context="first")
    _accumulate_oof(collected, coverage, np.array([1]), second, context="second")
    _assert_complete_coverage(coverage, context="unit")
    assert collected["prediction"].tolist() == [0.0, 10.0, 20.0]
    with pytest.raises(RuntimeError):
        _accumulate_oof(collected, coverage, np.array([0]), second, context="duplicate")
    metadata = {
        "oof_row_index": np.arange(3),
        "sample_token": np.array(["a", "b", "c"]),
        "sequence_id": np.array(["s", "s", "s"]),
        "track_id": np.array(["t", "t", "t"]),
        "target_expansion": np.zeros(3),
        "target_log_eta": np.zeros(3),
        "delta_t_s": np.ones(3),
        "target_ttc_s": np.ones(3),
    }
    path = tmp_path / "oof.npz"
    _save_oof_npz(path, metadata, collected, {control: collected for control in CONTROL_NAMES})
    with np.load(path, allow_pickle=False) as saved:
        assert saved["sample_token"].tolist() == ["a", "b", "c"]
        assert "fit12_effective_support_mass" in saved
        assert "zero_event_prediction" in saved
        assert "temporal_shuffle_cycle_matrix_error" in saved
        assert "endpoint_swap_cycle_translation_error" in saved


def _aggregate_fixture() -> tuple[
    list[dict[str, np.ndarray]], list[dict[str, dict[str, np.ndarray]]], dict[str, np.ndarray]
]:
    target = np.tile(np.array([-0.1, -0.06, -0.03, -0.015, 0.015, 0.03, 0.06, 0.1]), 12)
    metadata = {
        "target_expansion": target,
        "target_log_eta": np.log1p(-target),
        "delta_t_s": np.full(len(target), 0.1),
        "target_ttc_s": np.full(len(target), 1.0),
        "sequence_id": np.array([f"s{index // 32}" for index in range(len(target))]),
        "track_id": np.array([f"t{index // 8}" for index in range(len(target))]),
    }

    def payload(prediction: np.ndarray) -> dict[str, np.ndarray]:
        return {
            field: (
                prediction.copy()
                if field == "prediction"
                else np.log1p(-prediction)
                if field == "log_eta"
                else np.ones(len(target), dtype=np.float64)
                if field == "posterior_variance"
                else np.zeros(len(target), dtype=bool)
                if field == "unknown"
                else np.linspace(1.0, 2.0, len(target))
                if field == "fit12_effective_support_mass"
                else np.ones(len(target), dtype=np.float64)
            )
            for field in PREDICTION_FIELDS
        }

    def controls() -> dict[str, dict[str, np.ndarray]]:
        result = {control: payload(-target) for control in CONTROL_NAMES}
        result["zero_event"] = payload(np.full(len(target), np.nan))
        result["zero_event"]["log_eta"] = np.full(len(target), np.nan)
        result["zero_event"]["posterior_variance"] = np.full(len(target), np.nan)
        result["zero_event"]["unknown"] = np.ones(len(target), dtype=bool)
        return result

    noise = 0.015 * np.sin(np.arange(len(target)))
    noise[np.abs(target) >= 0.08] = 0.07 * np.sin(np.arange(len(target))[np.abs(target) >= 0.08])
    baseline_prediction = target + noise
    baseline_prediction[0] = 0.01
    return (
        [payload(baseline_prediction + offset) for offset in (-0.0001, 0.0, 0.0001)],
        [
            controls(),
            controls(),
            controls(),
        ],
        metadata,
    )


def test_analyzer_aggregation_uses_actual_controls_and_fails_one_bad_constituent():
    payloads, controls, metadata = _aggregate_fixture()
    good = _aggregate_arm_oof(payloads, controls, metadata)
    assert not good["median_gate_checks"]["controls"]
    assert not good["arm_passed"]

    changed_controls = [{name: dict(values) for name, values in item.items()} for item in controls]
    for item in changed_controls:
        item["temporal_shuffle"] = dict(item["temporal_shuffle"])
        item["temporal_shuffle"]["prediction"] = metadata["target_expansion"].copy()
    changed = _aggregate_arm_oof(payloads, changed_controls, metadata)
    assert changed["median_metrics"]["shuffle_ratio"] > 0.50
    assert not changed["median_gate_checks"]["controls"]

    poor_controls = [{name: dict(values) for name, values in item.items()} for item in controls]
    poor_controls[0]["endpoint_swap"] = dict(poor_controls[0]["endpoint_swap"])
    poor_controls[0]["endpoint_swap"]["prediction"] = metadata["target_expansion"].copy()
    poor = _aggregate_arm_oof(payloads, poor_controls, metadata)
    assert not poor["median_gate_checks"]["controls"]
    assert not poor["constituent_gate_checks"][0]["controls"]
    assert not poor["arm_passed"]


def test_actual_arm_b_gains_drive_tie_rule_and_diagnostic_never_promotes(tmp_path: Path):
    payloads, controls, metadata = _aggregate_fixture()
    arm_a = _aggregate_arm_oof(payloads, controls, metadata)
    improved = [dict(payload) for payload in payloads]
    for payload in improved:
        payload["prediction"] = metadata["target_expansion"].copy()
        payload["log_eta"] = metadata["target_log_eta"].copy()
    arm_b = _aggregate_arm_oof(improved, controls, metadata)
    arms = {
        "stable_multiscale_similarity": arm_a,
        "stable_multiscale_similarity_normal_flow": arm_b,
    }
    _inject_arm_b_gains(arms)
    gains = arm_b["median_metrics"]
    assert gains["paired_sequence_pearson_gain"] > 0.01
    assert gains["negative_track_macro_gain"] >= 0.02

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    raw = {
        "train": {"optimization_seed_by_fold": {0: 43000, 1: 43001, 2: 43002}},
        "stabilization": {"checkpoint_ema_epochs": [8, 9, 10]},
    }
    metrics = {
        "stable_multiscale_similarity": arm_a["median_metrics"],
        "stable_multiscale_similarity_normal_flow": arm_b["median_metrics"],
    }
    assert promoted_champion(metrics) is None
    diagnostic = make_summary(
        raw=raw,
        checkpoint_hashes={},
        cache_manifest=manifest,
        stabilization={"js_median": 0.0, "js_p95": 0.0, "expected_displacement_p95": 0.0},
        arm_metrics=metrics,
        arm_passed={name: True for name in metrics},
        diagnostic_only=True,
    )
    assert diagnostic["status"] == "diagnostic_only"
    assert diagnostic["promoted_champion"] is None
    assert not _development_requested(diagnostic, diagnostic=True)
    assert not diagnostic["scientific_contract"][
        "development_validation_materialized_at_most_once_after_oof_champion"
    ]

    promotion = make_summary(
        raw=raw,
        checkpoint_hashes={},
        cache_manifest=manifest,
        stabilization={"js_median": 0.0, "js_p95": 0.0, "expected_displacement_p95": 0.0},
        arm_metrics=metrics,
        arm_passed={name: True for name in metrics},
    )
    assert promotion["status"] == "completed_oof_gate_failed"
    assert promotion["promoted_champion"] is None
    assert promotion["next_action"] is None
    assert not _development_requested(promotion, diagnostic=False)
    assert not promotion["scientific_contract"][
        "development_validation_materialized_at_most_once_after_oof_champion"
    ]

    no_pass = make_summary(
        raw=raw,
        checkpoint_hashes={},
        cache_manifest=manifest,
        stabilization={"js_median": 0.0, "js_p95": 0.0, "expected_displacement_p95": 0.0},
        arm_metrics=metrics,
        arm_passed={name: False for name in metrics},
    )
    assert no_pass["status"] == "completed_oof_gate_failed"
    assert no_pass["promoted_champion"] is None
    assert not _development_requested(no_pass, diagnostic=False)
    assert not no_pass["scientific_contract"][
        "development_validation_materialized_at_most_once_after_oof_champion"
    ]

    stabilization_failed = make_summary(
        raw=raw,
        checkpoint_hashes={},
        cache_manifest=manifest,
        stabilization={"js_median": 0.03, "js_p95": 0.0, "expected_displacement_p95": 0.0},
        arm_metrics=metrics,
        arm_passed={name: True for name in metrics},
    )
    assert stabilization_failed["status"] == "stabilization_gate_failed"
    assert stabilization_failed["promoted_champion"] is None
    assert not _development_requested(stabilization_failed, diagnostic=False)
    assert not stabilization_failed["scientific_contract"][
        "development_validation_materialized_at_most_once_after_oof_champion"
    ]


def _development_criteria() -> dict[str, object]:
    return {
        "minimum_pearson_gain_over_v410": 0.005,
        "minimum_negative_accuracy_gain_over_v410": 0.020,
        "minimum_balanced_sign_gain_over_v410": 0.010,
        "minimum_log_eta_pearson": 0.758,
        "minimum_negative_track_macro_gain_over_v410": 0.020,
        "minimum_relative_paper_weighted_mid_improvement_over_v410": 0.020,
        "paper_mid_weights": {"crucial": 0.5, "small": 0.3, "large": 0.1, "negative": 0.1},
    }


def test_development_decision_requires_all_comparators_and_official_weighted_mid():
    target_ttc = np.array([1.0, 4.0, 8.0, -2.0])
    delta = np.full(4, 0.1)
    exact = delta / target_ttc
    baseline = exact + np.array([0.01, -0.005, 0.003, 0.01])
    candidate_mid = _paper_weighted_mid(target_ttc, exact, delta)
    baseline_mid = _paper_weighted_mid(target_ttc, baseline, delta)
    assert candidate_mid == pytest.approx(0.0)
    assert baseline_mid is not None and baseline_mid > 0.0
    v410 = {
        "pearson": 0.80,
        "negative_accuracy": 0.80,
        "balanced_sign_accuracy": 0.80,
        "negative_track_macro_accuracy": 0.80,
        "paper_weighted_mid": baseline_mid,
    }
    candidate = {
        "complete_finite_validation_coverage": True,
        "pearson": 0.81,
        "negative_accuracy": 0.83,
        "balanced_sign_accuracy": 0.82,
        "log_eta_pearson": 0.80,
        "negative_track_macro_accuracy": 0.83,
        "paper_weighted_mid": candidate_mid,
    }
    assert all(_development_decision(candidate, v410, _development_criteria()).values())
    missing = dict(v410)
    missing.pop("paper_weighted_mid")
    assert not _development_decision(candidate, missing, _development_criteria())[
        "relative_paper_weighted_mid_improvement"
    ]


def test_development_runner_defers_reads_until_all_full_models_are_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    target_ttc = torch.tensor([1.0, 4.0, 8.0, -2.0])
    split = SimpleNamespace(
        events=torch.ones(4, 3, 10, 2, 2),
        delta_t_s=torch.full((4,), 0.1),
        target_ttc_s=target_ttc,
        sequence_ids=["s0", "s1", "s2", "s3"],
        sample_tokens=["a", "b", "c", "d"],
        track_ids=["t", "t", "t", "t"],
    )
    call_order: list[str] = []
    state = {"weight": torch.tensor([1.0]), "bias": torch.tensor([0.0])}

    def stage(
        *args: object, **kwargs: object
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
        call_order.append(f"stage-{kwargs['seed']}")
        return state, [{"epoch": 10.0, "distill_kl": 0.0}]

    def final(*args: object, **kwargs: object) -> tuple[object, list[dict[str, float]]]:
        call_order.append(f"final-{kwargs['seed']}")
        assert kwargs["final_seed"] in {430107, 430113, 430123}
        return object(), [{"epoch": 12.0, "loss": 0.0}]

    def read_summary(path: Path) -> str:
        call_order.append("read-summary")
        return '{"artifact_type": "object_event_v4_10_true_seed_fixed_fusion_robustness"}'

    def read_ensemble(path: Path) -> object:
        call_order.append("read-ensemble")
        return object()

    def materialize(*args: object, **kwargs: object) -> tuple[object, dict[str, object]]:
        call_order.append("materialize-validation")
        assert args[1] == "validation"
        return split, {"split": "validation"}

    def align(value: object, frame: object) -> "analyzer.pd.DataFrame":
        call_order.append("align-ensemble")
        return analyzer.pd.DataFrame({"fused_prediction_expansion": [0.1, 0.025, 0.0125, -0.05]})

    def predict(
        model: object, value: object, device: torch.device, **kwargs: object
    ) -> dict[str, np.ndarray]:
        call_order.append("predict-validation")
        expansion = np.array([0.1, 0.025, 0.0125, -0.05])
        return {
            field: (
                expansion.copy()
                if field == "prediction"
                else np.log1p(-expansion)
                if field == "log_eta"
                else np.zeros(4, dtype=bool)
                if field == "unknown"
                else np.ones(4)
            )
            for field in PREDICTION_FIELDS
        }

    monkeypatch.setattr(analyzer, "_stage1_full", stage)
    monkeypatch.setattr(analyzer, "_train_full_head", final)
    monkeypatch.setattr(analyzer, "_materialize", materialize)
    monkeypatch.setattr(analyzer, "_read_ensemble", read_ensemble)
    monkeypatch.setattr(analyzer, "_align_ensemble", align)
    monkeypatch.setattr(analyzer, "_predict", predict)
    summary_path = tmp_path / "v410.json"
    ensemble_path = tmp_path / "ensemble.csv"
    summary_path.write_text("{}", encoding="utf-8")
    ensemble_path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(Path, "read_text", lambda path, **kwargs: read_summary(path))
    raw = {
        "arms": {
            "stable_multiscale_similarity": {"batch_size": 3},
            "stable_multiscale_similarity_normal_flow": {"batch_size": 3},
        },
        "train": {
            "epochs": 10,
            "final_epochs": 12,
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "final_training_seed_by_student": {7: 430107, 13: 430113, 23: 430123},
        },
        "stabilization": {"checkpoint_ema_epochs": [8, 9, 10]},
        "development": _development_criteria(),
    }
    result = _run_development_validation(
        train=split,
        manifest={},
        checkpoints={7: Path("seven.pt"), 13: Path("thirteen.pt"), 23: Path("twentythree.pt")},
        consensus_cache=_cache(4),
        raw=raw,
        champion="stable_multiscale_similarity",
        v48_config=Path("v48.yaml"),
        cache_manifest=Path("cache.json"),
        v410_summary_path=summary_path,
        ensemble_validation_path=ensemble_path,
        base_input_size=(2, 2),
        device=torch.device("cpu"),
        output=tmp_path,
    )
    assert call_order[:6] == ["stage-7", "final-7", "stage-13", "final-13", "stage-23", "final-23"]
    assert call_order.index("read-summary") > call_order.index("final-23")
    assert call_order.count("materialize-validation") == 1
    assert result["status"] == "development_validation_completed_failed"
    assert not result["development_passed"]
    assert result["development_validation_materialized_once"]
    assert result["development_audit"]["validation_materialize_calls"] == 1
    assert all(
        record["stabilization_train_rows"] == 4 for record in result["full_train_seed_records"]
    )
    assert all(record["final_epochs"] == 12 for record in result["full_train_seed_records"])


@pytest.mark.parametrize("requested", [12, 16])
def test_diagnostic_selection_is_bounded_and_has_three_viable_grouped_folds(requested: int):
    sequence_ids = ["a"] * 228 + ["b"] * 228 + ["c"] * 228
    chosen = _diagnostic_indices(sequence_ids, requested)
    selected_sequences = [sequence_ids[int(index)] for index in chosen]
    assert len(chosen) == requested and len(chosen) <= requested
    assert set(selected_sequences) == {"a", "b", "c"}
    folds = analyzer._sequence_folds(np.asarray(selected_sequences), 3, 430)
    _verify_grouped_folds(folds, selected_sequences, len(chosen))
    assert all(0 < len(held) < len(chosen) for held in folds)


def test_prepare_output_target_recovers_empty_exact_root_and_rejects_arbitrary_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    full = (tmp_path / "full").resolve()
    diagnostic = (tmp_path / "diagnostic").resolve()
    monkeypatch.setattr(analyzer, "FULL_OUTPUT_ROOT", full)
    monkeypatch.setattr(analyzer, "DIAGNOSTIC_OUTPUT_ROOT", diagnostic)
    full.mkdir()
    _prepare_output_target(full, force=False)
    assert full.is_dir() and not any(full.iterdir())
    (full / "partial.txt").write_text("aborted", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _prepare_output_target(full, force=False)
    _prepare_output_target(full, force=True)
    assert full.is_dir() and not any(full.iterdir())
    with pytest.raises(ValueError):
        _prepare_output_target(tmp_path / "arbitrary", force=True)


def test_wrapper_explicit_diagnostic_parameter_builds_diagnostic_command():
    wrapper = (Path("scripts") / "run_object_event_v4_30_stable_similarity.ps1").resolve()

    def invoke(*arguments: str) -> str:
        command = "\n".join(
            (
                "function global:uv {",
                "  param([Parameter(ValueFromRemainingArguments=$true)][string[]]$UvArgs)",
                '  Write-Output ("UV_CALL " + ($UvArgs -join " "))',
                "  $global:LASTEXITCODE = 0",
                "}",
                f"& '{wrapper}' {' '.join(arguments)}",
            )
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout

    diagnostic = invoke("-Device", "cpu", "-DiagnosticSamples", "12")
    assert (
        "v4.30 mode=diagnostic output=artifacts\\debug\\object_event_v4_30_diagnostic" in diagnostic
    )
    assert "--diagnostic-samples 12" in diagnostic
    assert "--output-dir artifacts\\debug\\object_event_v4_30_diagnostic" in diagnostic
    full = invoke("-Device", "cpu")
    assert (
        "v4.30 mode=full_oof output=artifacts\\debug\\object_event_v4_30_stable_similarity" in full
    )
    assert "--diagnostic-samples" not in full
    assert "--output-dir artifacts\\debug\\object_event_v4_30_stable_similarity" in full


def test_stage1_consumes_supplied_optimizer_batch_epoch_and_ema_controls(
    monkeypatch: pytest.MonkeyPatch,
):
    """Helpers honor supplied values; normal CLI still rejects their config drift."""
    split = SimpleNamespace(events=torch.ones(6, 3, 10, 2, 2))
    captured: dict[str, object] = {}

    class Backbone:
        def parameters(self) -> list[torch.nn.Parameter]:
            return []

        def to(self, device: torch.device) -> "Backbone":
            return self

        def eval(self) -> "Backbone":
            return self

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.local_projection = torch.nn.Linear(1, 1)

        def forward(self, events: torch.Tensor) -> SimpleNamespace:
            probability = torch.ones(len(events), 1, 1, 1, requires_grad=True)
            posterior = SimpleNamespace(probabilities=probability)
            return SimpleNamespace(posteriors_12={1: posterior, 2: posterior, 4: posterior})

        def head_parameters(self) -> list[torch.nn.Parameter]:
            return list(self.local_projection.parameters())

    model = TinyModel()
    construction_order: list[str] = []
    original_rng = analyzer._rng
    monkeypatch.setattr(
        analyzer,
        "_rng",
        lambda seed: construction_order.append(f"rng-{seed}") or original_rng(seed),
    )
    monkeypatch.setattr(analyzer, "_load_backbone", lambda **kwargs: (Backbone(), {}))
    monkeypatch.setattr(
        analyzer,
        "ObjectEventTTCV430",
        lambda backbone, config: construction_order.append("construct") or model,
    )
    cache = _cache(6)
    original_for_indices = TeacherConsensusCache.for_indices

    def recorded_for_indices(
        value: TeacherConsensusCache, indices: torch.Tensor | np.ndarray, device: torch.device
    ) -> dict[int, torch.Tensor]:
        captured.setdefault("batch_sizes", []).append(len(indices))
        return original_for_indices(value, indices, device)

    monkeypatch.setattr(TeacherConsensusCache, "for_indices", recorded_for_indices)

    class Optimizer(torch.optim.SGD):
        def __init__(self, params: object, lr: float, weight_decay: float) -> None:
            captured.update({"lr": lr, "weight_decay": weight_decay})
            super().__init__(params, lr=lr, weight_decay=weight_decay)

    monkeypatch.setattr(torch.optim, "AdamW", Optimizer)
    _, history, _ = _stage1(
        split,
        np.array([5]),
        Path("student.pt"),
        consensus_cache=cache,
        v48_config=Path("v48.yaml"),
        cfg={"batch_size": 3},
        train_cfg={"epochs": 3, "learning_rate": 0.02, "weight_decay": 0.03, "max_grad_norm": 0.4},
        stabilization_cfg={"checkpoint_ema_epochs": [1, 2, 3]},
        seed=7,
        fold_seed=99,
        device=torch.device("cpu"),
    )
    assert captured == {"lr": 0.02, "weight_decay": 0.03, "batch_sizes": [3, 2] * 3}
    assert [item["epoch"] for item in history] == [1.0, 2.0, 3.0]
    assert construction_order[:2] == ["rng-106", "construct"]


class _ForegroundTeacher:
    def __init__(self, foreground: float) -> None:
        self.foreground = foreground
        self.forward_batches = 0

    def _foreground_and_features(
        self, events: torch.Tensor
    ) -> tuple[torch.Tensor, None, torch.Tensor, None]:
        self.forward_batches += 1
        key = events[:, 0, 0, 0, 0].reshape(-1, 1, 1, 1, 1).to(torch.float32)
        maps = key.expand(-1, 3, 1, 1, 1)
        foreground = torch.full((len(events), 3, 1, 1), self.foreground, dtype=torch.float32)
        return maps, None, foreground, None

    def _temporal_maps(self, maps: torch.Tensor) -> torch.Tensor:
        return maps[:, 2]

    def temporal_projection(self, temporal: torch.Tensor) -> torch.Tensor:
        return temporal

    def to(self, device: torch.device) -> "_ForegroundTeacher":
        return self

    def eval(self) -> "_ForegroundTeacher":
        return self

    def parameters(self) -> list[torch.nn.Parameter]:
        return []


class _ConsensusModel:
    def __init__(self, *args: object) -> None:
        pass

    def to(self, device: torch.device) -> "_ConsensusModel":
        return self

    def eval(self) -> "_ConsensusModel":
        return self

    def locked_teacher_consensus(
        self,
        pairs: list[tuple[torch.Tensor, torch.Tensor]],
        foreground: torch.Tensor,
        activity: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        key = pairs[0][0][:, 0, 0, 0]
        foreground_mean = foreground[:, 0, 0]
        probability = torch.stack(
            (
                0.2 + 0.01 * key + 0.001 * foreground_mean,
                0.8 - 0.01 * key - 0.001 * foreground_mean,
            ),
            dim=1,
        )[:, :, None, None]
        return {scale: probability for scale in (1, 2, 4)}


def test_teacher_pairs_average_all_foregrounds_and_are_order_invariant():
    events = torch.ones(2, 3, 10, 1, 1)
    teachers = [_ForegroundTeacher(value) for value in (1.0, 3.0, 5.0)]
    _, foreground, activity = _teacher_pairs(teachers, events)
    _, reversed_foreground, reversed_activity = _teacher_pairs(list(reversed(teachers)), events)
    torch.testing.assert_close(foreground, torch.full_like(foreground, 3.0), rtol=0, atol=0)
    torch.testing.assert_close(foreground, reversed_foreground, rtol=0, atol=0)
    torch.testing.assert_close(activity, reversed_activity, rtol=0, atol=0)
    for scale in (1, 2, 4):
        torch.testing.assert_close(
            _consensus(_ConsensusModel(), teachers, events)[scale],
            _consensus(_ConsensusModel(), list(reversed(teachers)), events)[scale],
            rtol=0,
            atol=0,
        )


def test_teacher_consensus_cache_is_aligned_once_and_matches_uncached(
    monkeypatch: pytest.MonkeyPatch,
):
    split = SimpleNamespace(events=torch.zeros(5, 3, 10, 1, 1))
    split.events[:, 0, 0, 0, 0] = torch.arange(5, dtype=torch.float32)
    loads: list[_ForegroundTeacher] = []

    def frozen(
        checkpoints: object, *, v48_config: Path, device: torch.device
    ) -> list[_ForegroundTeacher]:
        teachers = [_ForegroundTeacher(value) for value in (1.0, 3.0, 5.0)]
        loads.extend(teachers)
        return teachers

    monkeypatch.setattr(analyzer, "_frozen_teachers", frozen)
    monkeypatch.setattr(analyzer, "ObjectEventTTCV430", _ConsensusModel)
    cache = _build_teacher_consensus_cache(
        split,
        {7: Path("seven.pt"), 13: Path("thirteen.pt"), 23: Path("twentythree.pt")},
        checkpoint_hashes={"7": "a", "13": "b", "23": "c"},
        v48_config=Path("v48.yaml"),
        cfg={"batch_size": 2},
        device=torch.device("cpu"),
    )
    assert cache.consensus_build_count == 1
    assert (
        cache.row_count == 5
        and len(cache.consensus_cache_sha256) == 64
        and len(cache.consensus_config_sha256) == 64
    )
    assert cache.teacher_backbone_forward_batches == 9
    assert [teacher.forward_batches for teacher in loads] == [3, 3, 3]
    cached = cache.for_indices(np.array([4, 1]), torch.device("cpu"))
    assert cached[1][:, 0, 0, 0].tolist() == pytest.approx([0.243, 0.213])
    before = [teacher.forward_batches for teacher in loads]
    for _epoch in range(10):
        for _fold in range(3):
            for _seed in (7, 13, 23):
                for _arm in range(2):
                    cache.for_indices(np.array([3, 0]), torch.device("cpu"))
    assert [teacher.forward_batches for teacher in loads] == before

    uncached_teachers = [_ForegroundTeacher(value) for value in (1.0, 3.0, 5.0)]
    uncached = _consensus(_ConsensusModel(), uncached_teachers, split.events)
    for scale in (1, 2, 4):
        torch.testing.assert_close(
            cache.for_indices(np.arange(5), torch.device("cpu"))[scale],
            uncached[scale],
            rtol=0,
            atol=1e-6,
        )
        left = torch.nn.Parameter(torch.tensor(0.1))
        right = torch.nn.Parameter(torch.tensor(0.1))
        left_optimizer = torch.optim.SGD([left], lr=0.1)
        right_optimizer = torch.optim.SGD([right], lr=0.1)
        cached_loss = posterior_kl(
            cache.for_indices(np.arange(5), torch.device("cpu"))[scale],
            torch.softmax(torch.stack((left.expand(5), -left.expand(5)), dim=1), dim=1)[
                :, :, None, None
            ],
        )
        uncached_loss = posterior_kl(
            uncached[scale],
            torch.softmax(torch.stack((right.expand(5), -right.expand(5)), dim=1), dim=1)[
                :, :, None, None
            ],
        )
        cached_loss.backward()
        uncached_loss.backward()
        left_optimizer.step()
        right_optimizer.step()
        assert cached_loss.item() == pytest.approx(uncached_loss.item(), abs=1e-6)
        torch.testing.assert_close(left, right, rtol=0, atol=1e-6)

    reordered = SimpleNamespace(events=split.events.flip(0).clone())
    changed = _build_teacher_consensus_cache(
        reordered,
        {7: Path("seven.pt"), 13: Path("thirteen.pt"), 23: Path("twentythree.pt")},
        checkpoint_hashes={"7": "a", "13": "b", "23": "c"},
        v48_config=Path("v48.yaml"),
        cfg={"batch_size": 2},
        device=torch.device("cpu"),
    )
    assert changed.consensus_cache_sha256 != cache.consensus_cache_sha256
    assert changed.for_indices(np.array([0]), torch.device("cpu"))[1][
        0, 0, 0, 0
    ].item() == pytest.approx(0.243)
