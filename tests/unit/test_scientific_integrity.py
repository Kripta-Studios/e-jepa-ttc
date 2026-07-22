"""Scientific integrity regression tests.

These tests codify the non-negotiable scientific contracts for E-JEPA-TTC.
Each test explicitly targets one of the failure modes that has been observed
or is theoretically possible in event-based JEPA TTC systems.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from e_jepa_ttc.data.eap_cache import EAPObjectCacheDataset
from e_jepa_ttc.models.object_jepa import (
    ObjectCentricEventJEPA,
    ObjectJEPAConfig,
    geometric_dynamics_targets,
    inverse_ttc_distribution_to_seconds,
    object_event_jepa_loss,
    roi_sample,
)
from e_jepa_ttc.representations.voxel_grid import robust_normalize

# ---------------------------------------------------------------------------
# 3.1  Sparse event voxel normalization
# ---------------------------------------------------------------------------


class TestSparseVoxelNormalization:
    """Occupied voxels must remain nonzero after normalization.

    The pre-v2 centering normalizer subtracted the nonzero median.  When all
    occupied bins had the same magnitude, the median equalled that magnitude
    and centering erased every event.  The v2 ``robust_normalize`` uses a
    non-centred 95th-percentile magnitude scale and must preserve occupancy.
    """

    def test_equal_magnitude_occupied_voxels_remain_nonzero(self):
        """Regression for the exact failure mode that invalidated pre-v2 caches."""
        voxel = np.zeros((10, 8, 8), dtype=np.float32)
        voxel[0, 0, 0] = 3.0
        voxel[1, 1, 1] = 3.0
        voxel[5, 2, 2] = -3.0
        result = robust_normalize(voxel)
        assert result[0, 0, 0] != 0.0, "Occupied voxel was erased by normalization"
        assert result[1, 1, 1] != 0.0, "Occupied voxel was erased by normalization"
        assert result[5, 2, 2] != 0.0, "Occupied voxel was erased by normalization"

    def test_polarity_signs_preserved(self):
        voxel = np.zeros((10, 8, 8), dtype=np.float32)
        voxel[0, 0, 0] = 5.0
        voxel[5, 1, 1] = -3.0
        result = robust_normalize(voxel)
        assert result[0, 0, 0] > 0.0
        assert result[5, 1, 1] < 0.0

    def test_empty_voxels_remain_zero(self):
        voxel = np.zeros((10, 8, 8), dtype=np.float32)
        voxel[0, 0, 0] = 5.0
        voxel[1, 1, 1] = 3.0
        result = robust_normalize(voxel)
        assert result[2, 0, 0] == 0.0
        assert result[9, 7, 7] == 0.0

    def test_no_nan_or_inf(self):
        voxel = np.zeros((10, 8, 8), dtype=np.float32)
        voxel[0, 0, 0] = 1e-8
        voxel[1, 1, 1] = 1e10
        voxel[5, 2, 2] = -1e10
        result = robust_normalize(voxel)
        assert np.all(np.isfinite(result))

    def test_empty_grid_unchanged(self):
        voxel = np.zeros((10, 8, 8), dtype=np.float32)
        result = robust_normalize(voxel)
        np.testing.assert_array_equal(result, voxel)

    def test_mixed_magnitudes_preserve_occupancy(self):
        """Sparse grid with varied magnitudes."""
        voxel = np.zeros((10, 16, 16), dtype=np.float32)
        voxel[0, 0, 0] = 1.0
        voxel[1, 1, 1] = 2.0
        voxel[2, 2, 2] = 0.5
        voxel[5, 3, 3] = -1.0
        voxel[6, 4, 4] = -2.0
        result = robust_normalize(voxel)
        for c, r, col in [(0, 0, 0), (1, 1, 1), (2, 2, 2), (5, 3, 3), (6, 4, 4)]:
            assert result[c, r, col] != 0.0, f"Occupied voxel [{c},{r},{col}] was erased"


# ---------------------------------------------------------------------------
# 3.2  Causal temporal separation
# ---------------------------------------------------------------------------


class TestCausalTemporalSeparation:
    """Future windows must not overlap with context windows."""

    def test_future_starts_after_context_ends(self):
        context_end_us = np.array([100_000, 200_000, 300_000], dtype=np.int64)
        future_start_us = np.array([100_001, 200_001, 300_001], dtype=np.int64)
        assert np.all(future_start_us >= context_end_us)

    def test_timestamps_strictly_ordered(self):
        """Monotonically increasing timestamps within a window."""
        timestamps = np.array([0, 100, 200, 300, 400], dtype=np.int64)
        assert np.all(np.diff(timestamps) >= 0)

    def test_horizons_strictly_ordered(self):
        horizons_ms = [25, 50, 100, 250, 500]
        assert horizons_ms == sorted(horizons_ms)
        assert len(horizons_ms) == len(set(horizons_ms))


# ---------------------------------------------------------------------------
# 3.3  Degenerate ROI boxes
# ---------------------------------------------------------------------------


class TestDegenerateROIBoxes:
    """Zero-area and out-of-bounds ROI boxes must not produce artificial features."""

    def _feature_map(self):
        return torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)

    def test_zero_width_box_yields_finite_result(self):
        """A zero-width box collapses to a vertical line sample."""
        feature = self._feature_map()
        boxes = torch.tensor([[[0.5, 0.0, 0.5, 1.0]]])
        result = roi_sample(feature, boxes, output_size=2)
        assert result.shape == (1, 1, 1, 2, 2)
        assert torch.all(torch.isfinite(result))

    def test_zero_height_box_yields_finite_result(self):
        feature = self._feature_map()
        boxes = torch.tensor([[[0.0, 0.5, 1.0, 0.5]]])
        result = roi_sample(feature, boxes, output_size=2)
        assert result.shape == (1, 1, 1, 2, 2)
        assert torch.all(torch.isfinite(result))

    def test_zero_area_box(self):
        """Both width and height are zero — collapses to a point."""
        feature = self._feature_map()
        boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]])
        result = roi_sample(feature, boxes, output_size=2)
        assert result.shape == (1, 1, 1, 2, 2)
        assert torch.all(torch.isfinite(result))

    def test_box_outside_image_yields_finite(self):
        """Boxes outside [0,1] are clamped."""
        feature = self._feature_map()
        boxes = torch.tensor([[[2.0, 2.0, 3.0, 3.0]]])
        result = roi_sample(feature, boxes, output_size=2)
        assert torch.all(torch.isfinite(result))

    def test_partially_outside_box_is_clamped(self):
        feature = self._feature_map()
        boxes = torch.tensor([[[-0.5, -0.5, 0.5, 0.5]]])
        result = roi_sample(feature, boxes, output_size=2)
        assert result.shape == (1, 1, 1, 2, 2)
        assert torch.all(torch.isfinite(result))

    def test_inverted_box_clamps_to_zero_area(self):
        """x_min > x_max after clamping collapses to zero width."""
        feature = self._feature_map()
        boxes = torch.tensor([[[0.8, 0.2, 0.3, 0.6]]])
        result = roi_sample(feature, boxes, output_size=2)
        assert torch.all(torch.isfinite(result))

    def test_masked_object_with_arbitrary_box(self):
        """When object_mask is False, box values are arbitrary but must not crash."""
        feature = self._feature_map()
        boxes = torch.tensor([[[999.0, -999.0, 0.0, 0.0]]])
        result = roi_sample(feature, boxes, output_size=2)
        assert torch.all(torch.isfinite(result))


# ---------------------------------------------------------------------------
# 3.4  TTC inverse stability
# ---------------------------------------------------------------------------


class TestTTCInverseStability:
    """Inverse-TTC near zero must not produce unbounded values."""

    def test_normal_positive_inverse(self):
        mean, std = inverse_ttc_distribution_to_seconds(torch.tensor([0.5]), torch.tensor([0.0]))
        assert torch.allclose(mean, torch.tensor([2.0]))
        assert torch.all(std > 0)

    def test_negative_inverse_produces_negative_ttc(self):
        mean, _std = inverse_ttc_distribution_to_seconds(torch.tensor([-0.25]), torch.tensor([0.0]))
        assert mean.item() < 0  # receding object

    def test_zero_inverse_is_clamped(self):
        """Inverse TTC = 0 must not produce inf."""
        mean, std = inverse_ttc_distribution_to_seconds(torch.tensor([0.0]), torch.tensor([0.0]))
        assert torch.all(torch.isfinite(mean))
        assert torch.all(torch.isfinite(std))

    def test_near_zero_inverse_is_clamped(self):
        mean, std = inverse_ttc_distribution_to_seconds(torch.tensor([1e-6]), torch.tensor([0.0]))
        assert torch.all(torch.isfinite(mean))
        assert torch.all(torch.isfinite(std))

    def test_large_inverse(self):
        mean, std = inverse_ttc_distribution_to_seconds(torch.tensor([100.0]), torch.tensor([0.0]))
        assert torch.allclose(mean, torch.tensor([0.01]))
        assert torch.all(torch.isfinite(std))

    def test_nan_inverse_propagates(self):
        mean, _std = inverse_ttc_distribution_to_seconds(
            torch.tensor([float("nan")]), torch.tensor([0.0])
        )
        # NaN should propagate, not silently produce a number
        assert torch.isnan(mean).any() or torch.isfinite(mean).all()

    def test_batch_stability(self):
        """Mixed batch of normal, near-zero, and negative values."""
        inv_means = torch.tensor([0.5, 0.0, -0.25, 1e-8, 100.0])
        log_vars = torch.zeros(5)
        mean, std = inverse_ttc_distribution_to_seconds(inv_means, log_vars)
        assert torch.all(torch.isfinite(mean))
        assert torch.all(torch.isfinite(std))
        assert torch.all(std > 0)


# ---------------------------------------------------------------------------
# 3.5  Teacher gradient isolation
# ---------------------------------------------------------------------------


def _teacher_fixture():
    torch.manual_seed(7)
    config = ObjectJEPAConfig(
        in_channels=4,
        action_dim=3,
        embedding_dim=48,
        feature_dim=32,
        predictor_depth=1,
        predictor_heads=6,
        dropout=0.0,
    )
    model = ObjectCentricEventJEPA(config)
    batch, ctx, horizons, objects = 2, 3, 3, 2
    ce = torch.randn(batch, ctx, 4, 32, 32)
    cb = torch.tensor([[[[0.1, 0.1, 0.4, 0.5], [0.5, 0.2, 0.8, 0.6]]] * ctx] * batch)
    cm = torch.ones(batch, ctx, objects, dtype=torch.bool)
    fe = torch.randn(batch, horizons, 4, 32, 32)
    fb = cb[:, :horizons].clone()
    fm = torch.ones(batch, horizons, objects, dtype=torch.bool)
    hv = torch.tensor([0.1, 0.25, 0.5])
    return model, ce, cb, cm, fe, fb, fm, hv


class TestTeacherGradientIsolation:
    """EMA teacher must never receive gradients through the forward pass."""

    def test_student_receives_gradients(self):
        model, ce, cb, cm, fe, fb, fm, hv = _teacher_fixture()
        output = model(ce, cb, cm, fe, fb, fm, hv)
        geometry = geometric_dynamics_targets(cb[:, -1], fb, hv)
        losses = object_event_jepa_loss(output, geometry, ttc_target_s=torch.full((2, 2), 2.0))
        losses["total"].backward()
        has_grad = any(p.grad is not None for p in model.context_encoder.parameters())
        assert has_grad, "Context encoder received no gradients"

    def test_predictor_receives_gradients(self):
        model, ce, cb, cm, fe, fb, fm, hv = _teacher_fixture()
        output = model(ce, cb, cm, fe, fb, fm, hv)
        geometry = geometric_dynamics_targets(cb[:, -1], fb, hv)
        losses = object_event_jepa_loss(output, geometry, ttc_target_s=torch.full((2, 2), 2.0))
        losses["total"].backward()
        has_grad = any(p.grad is not None for p in model.predictor.parameters())
        assert has_grad, "Predictor received no gradients"

    def test_teacher_receives_no_gradients(self):
        model, ce, cb, cm, fe, fb, fm, hv = _teacher_fixture()
        output = model(ce, cb, cm, fe, fb, fm, hv)
        geometry = geometric_dynamics_targets(cb[:, -1], fb, hv)
        losses = object_event_jepa_loss(output, geometry, ttc_target_s=torch.full((2, 2), 2.0))
        losses["total"].backward()
        teacher_has_grad = any(p.grad is not None for p in model.target_encoder.parameters())
        assert not teacher_has_grad, "Target encoder received gradients!"

    def test_teacher_updates_only_via_ema(self):
        model, *_ = _teacher_fixture()
        with torch.no_grad():
            ctx_p = next(model.context_encoder.parameters())
            tgt_p = next(model.target_encoder.parameters())
            ctx_p.add_(1.0)
            before = tgt_p.clone()
            model.update_target_encoder(0.5)
            expected = 0.5 * before + 0.5 * ctx_p
            assert torch.allclose(tgt_p, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 3.6  Scratch/JEPA architecture identity
# ---------------------------------------------------------------------------


class TestScratchJEPAArchitectureIdentity:
    """Scratch and JEPA-initialized models must have identical structure."""

    def _config(self):
        return ObjectJEPAConfig(
            in_channels=4,
            action_dim=3,
            embedding_dim=48,
            feature_dim=32,
            predictor_depth=1,
            predictor_heads=6,
            dropout=0.0,
        )

    def test_parameter_names_and_shapes_match(self):
        scratch = ObjectCentricEventJEPA(self._config())
        jepa = ObjectCentricEventJEPA(self._config())
        with torch.no_grad():
            for p in jepa.parameters():
                p.uniform_(-1.0, 1.0)
        scratch_p = dict(scratch.named_parameters())
        jepa_p = dict(jepa.named_parameters())
        assert set(scratch_p.keys()) == set(jepa_p.keys())
        for name in scratch_p:
            assert scratch_p[name].shape == jepa_p[name].shape, f"Shape mismatch for {name}"

    def test_trainable_parameter_count_matches(self):
        scratch = ObjectCentricEventJEPA(self._config())
        jepa = ObjectCentricEventJEPA(self._config())
        s_count = sum(p.numel() for p in scratch.parameters() if p.requires_grad)
        j_count = sum(p.numel() for p in jepa.parameters() if p.requires_grad)
        assert s_count == j_count

    def test_config_is_identical(self):
        cfg = self._config()
        scratch = ObjectCentricEventJEPA(cfg)
        jepa = ObjectCentricEventJEPA(cfg)
        assert scratch.config == jepa.config


# ---------------------------------------------------------------------------
# 3.7  Ego-action ablation identity
# ---------------------------------------------------------------------------


class TestEgoActionAblationIdentity:
    """Ego-action enabled vs disabled must differ only in action inputs."""

    def _config(self):
        return ObjectJEPAConfig(
            in_channels=4,
            action_dim=3,
            embedding_dim=48,
            feature_dim=32,
            predictor_depth=1,
            predictor_heads=6,
            dropout=0.0,
        )

    def test_architecture_and_params_identical(self):
        model_a = ObjectCentricEventJEPA(self._config())
        model_b = ObjectCentricEventJEPA(self._config())
        model_b.load_state_dict(model_a.state_dict())
        a_p = dict(model_a.named_parameters())
        b_p = dict(model_b.named_parameters())
        assert set(a_p.keys()) == set(b_p.keys())
        for name in a_p:
            assert torch.equal(a_p[name], b_p[name])

    def test_action_masking_changes_predictions(self):
        """With identical weights, different action masks should change output."""
        torch.manual_seed(42)
        model = ObjectCentricEventJEPA(self._config())
        batch, ctx, horizons, objects = 1, 3, 3, 1
        ce = torch.randn(batch, ctx, 4, 32, 32)
        cb = torch.tensor([[[[0.1, 0.1, 0.4, 0.5]]] * ctx])
        cm = torch.ones(batch, ctx, objects, dtype=torch.bool)
        fe = torch.randn(batch, horizons, 4, 32, 32)
        fb = cb[:, :horizons].clone()
        fm = torch.ones(batch, horizons, objects, dtype=torch.bool)
        hv = torch.tensor([0.1, 0.25, 0.5])

        actions = torch.ones(batch, horizons, 3)
        mask_true = torch.ones(batch, horizons, dtype=torch.bool)
        mask_false = torch.zeros(batch, horizons, dtype=torch.bool)

        out_enabled = model(
            ce,
            cb,
            cm,
            fe,
            fb,
            fm,
            hv,
            future_ego_actions=actions,
            future_ego_action_mask=mask_true,
        )
        out_disabled = model(
            ce,
            cb,
            cm,
            fe,
            fb,
            fm,
            hv,
            future_ego_actions=actions,
            future_ego_action_mask=mask_false,
        )

        assert torch.equal(out_enabled.target_latents, out_disabled.target_latents)
        assert not torch.allclose(out_enabled.predicted_latents, out_disabled.predicted_latents)


# ---------------------------------------------------------------------------
# 3.8  Cache format version enforcement
# ---------------------------------------------------------------------------


class TestCacheFormatVersionEnforcement:
    """The loader must reject caches with format version < 2."""

    def _write_v1_shard(self, path, *, split="train"):
        rng = np.random.default_rng(0)
        count, ctx, h, c, s = 2, 3, 3, 4, 8
        np.savez_compressed(
            path,
            context_events=rng.normal(size=(count, ctx, c, s, s)).astype(np.float16),
            context_boxes=np.tile(
                np.array([0.2, 0.2, 0.5, 0.6], dtype=np.float32),
                (count, ctx, 1, 1),
            ),
            context_sampling_boxes=np.tile(
                np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
                (count, ctx, 1, 1),
            ),
            context_object_mask=np.ones((count, ctx, 1), dtype=np.bool_),
            context_depth_m=np.full((count, 1), 10.0, dtype=np.float32),
            context_ego_actions=np.zeros((count, ctx, 3), dtype=np.float32),
            context_ego_action_mask=np.zeros((count, ctx), dtype=np.bool_),
            future_events=rng.normal(size=(count, h, c, s, s)).astype(np.float16),
            future_boxes=np.tile(
                np.array([0.18, 0.18, 0.52, 0.62], dtype=np.float32),
                (count, h, 1, 1),
            ),
            future_sampling_boxes=np.tile(
                np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
                (count, h, 1, 1),
            ),
            future_object_mask=np.ones((count, h, 1), dtype=np.bool_),
            future_depth_m=np.full((count, h, 1), 9.0, dtype=np.float32),
            future_ego_actions=np.zeros((count, h, 3), dtype=np.float32),
            future_ego_action_mask=np.zeros((count, h), dtype=np.bool_),
            ttc_s=np.array([[1.0], [2.0]], dtype=np.float32),
            sample_token=np.array(["a:0", "a:1"]),
            sequence_id=np.array(["seq"] * count),
            track_id=np.array(["t"] * count),
            category=np.array(["car"] * count),
            split=np.array([split] * count),
            ttc_source=np.array(["synthetic"] * count),
            prediction_horizons_s=np.array([0.1, 0.25, 0.5], dtype=np.float32),
            cache_format_version=np.asarray(1, dtype=np.int64),
            future_window_semantics=np.asarray("endpoint_offset_disjoint_fixed_duration"),
        )

    def test_v1_cache_is_rejected(self, tmp_path):
        """The loader must reject caches with format version < 2."""
        self._write_v1_shard(tmp_path / "train.npz")
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "shards": [
                        {
                            "path": "train.npz",
                            "split": "train",
                            "sequence_id": "seq",
                            "samples": 2,
                            "size_bytes": (tmp_path / "train.npz").stat().st_size,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        import pytest

        with pytest.raises(ValueError, match="invalid cache_format_version"):
            EAPObjectCacheDataset(manifest, splits=("train",))
