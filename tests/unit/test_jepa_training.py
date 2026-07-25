from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.data.evttc import NAVIGATION_FEATURE_NAMES
from e_jepa_ttc.data.ml_cache import remap_cache_splits
from e_jepa_ttc.training.jepa import (
    _build_temporal_pairs,
    _neutralize_synthetic_navigation,
    _tubelet_masked_context,
    _without_future_navigation,
    pretrain_jepa,
)
from e_jepa_ttc.training.supervised import evaluate_supervised_checkpoint, train_tiny_cnn
from e_jepa_ttc.utils.io import write_structured


def _temporal_cache_fields(count: int) -> dict[str, np.ndarray]:
    timestamp_us = np.arange(count, dtype=np.int64) * 20_000
    return {
        "timestamp_us": timestamp_us,
        "context_start_us": timestamp_us - 20_000,
        "context_end_us": timestamp_us.copy(),
    }


def _write_cache(path: Path) -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(12, 6, 12, 16)).astype(np.float16)
    y_ttc = np.linspace(1.0, 6.0, num=12, dtype=np.float32)
    split = np.array(["train"] * 6 + ["validation"] * 3 + ["test"] * 3)
    np.savez(
        path,
        cache_format_version=np.array(2, dtype=np.int32),
        source_manifest_sha256=np.array("mock"),
        split_manifest_sha256=np.array("mock"),
        preprocessing_config_sha256=np.array("mock"),
        normalization=np.array("mock"),
        x=x,
        y_ttc=y_ttc,
        split=split,
        **_temporal_cache_fields(12),
        sequence_id=np.array(["fixture"] * 12),
        event_count=np.arange(12, dtype=np.int32),
        width=np.array(16, dtype=np.int32),
        height=np.array(12, dtype=np.int32),
        bins=np.array(3, dtype=np.int32),
    )


def _write_tubelet_cache(path: Path) -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(12, 12, 32, 32)).astype(np.float16)
    y_ttc = np.linspace(1.0, 6.0, num=12, dtype=np.float32)
    split = np.array(["train"] * 6 + ["validation"] * 3 + ["test"] * 3)
    np.savez(
        path,
        cache_format_version=np.array(2, dtype=np.int32),
        source_manifest_sha256=np.array("mock"),
        split_manifest_sha256=np.array("mock"),
        preprocessing_config_sha256=np.array("mock"),
        normalization=np.array("mock"),
        x=x,
        y_ttc=y_ttc,
        split=split,
        **_temporal_cache_fields(12),
        sequence_id=np.array(["fixture"] * 12),
        event_count=np.arange(12, dtype=np.int32),
        width=np.array(32, dtype=np.int32),
        height=np.array(32, dtype=np.int32),
        bins=np.array(5, dtype=np.int32),
    )


def _write_navigation_cache(path: Path) -> None:
    rng = np.random.default_rng(13)
    x = rng.normal(size=(12, 21, 32, 32)).astype(np.float16)
    x[:, 10:12] = 0.0
    for idx in range(len(NAVIGATION_FEATURE_NAMES)):
        x[:, 12 + idx] = np.linspace(0.1, 0.9, num=12, dtype=np.float16)[:, None, None]
    y_ttc = np.linspace(1.0, 6.0, num=12, dtype=np.float32)
    split = np.array(["train"] * 6 + ["validation"] * 3 + ["test"] * 3)
    np.savez(
        path,
        cache_format_version=np.array(2, dtype=np.int32),
        source_manifest_sha256=np.array("mock"),
        split_manifest_sha256=np.array("mock"),
        preprocessing_config_sha256=np.array("mock"),
        normalization=np.array("mock"),
        x=x,
        y_ttc=y_ttc,
        split=split,
        **_temporal_cache_fields(12),
        sequence_id=np.array(["fixture"] * 12),
        event_count=np.arange(12, dtype=np.int32),
        width=np.array(32, dtype=np.int32),
        height=np.array(32, dtype=np.int32),
        bins=np.array(5, dtype=np.int32),
        metadata_channels=np.array(True, dtype=np.bool_),
        navigation_channels=np.array(True, dtype=np.bool_),
        navigation_feature_names=np.array(NAVIGATION_FEATURE_NAMES),
    )


def _write_multival_cache(path: Path) -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(16, 6, 12, 16)).astype(np.float16)
    y_ttc = np.linspace(1.0, 6.0, num=16, dtype=np.float32)
    split = np.array(
        ["train"] * 8 + ["validation_car"] * 4 + ["validation_pedestrian"] * 2 + ["test"] * 2
    )
    sequence_id = np.array(["seq-train"] * 8 + ["seq-car"] * 4 + ["seq-ped"] * 2 + ["seq-test"] * 2)
    np.savez(
        path,
        cache_format_version=np.array(2, dtype=np.int32),
        source_manifest_sha256=np.array("mock"),
        split_manifest_sha256=np.array("mock"),
        preprocessing_config_sha256=np.array("mock"),
        normalization=np.array("mock"),
        x=x,
        y_ttc=y_ttc,
        split=split,
        sequence_id=sequence_id,
        **_temporal_cache_fields(16),
        event_count=np.arange(16, dtype=np.int32),
        width=np.array(16, dtype=np.int32),
        height=np.array(12, dtype=np.int32),
        bins=np.array(3, dtype=np.int32),
    )


def test_jepa_checkpoint_loads_into_supervised_trainer(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    _write_cache(cache_path)

    pretrain_summary = pretrain_jepa(
        cache_path=cache_path,
        output_dir=tmp_path / "jepa",
        epochs=1,
        batch_size=3,
        seed=5,
        device_name="cpu",
        pretrain_splits=("train",),
        validation_splits=("validation",),
    )
    assert pretrain_summary["best_epoch"] == 1
    assert pretrain_summary["pretrain_seed"] == 5
    assert pretrain_summary["checkpoint_selection"]["recommended_role"] == "best"
    assert len(pretrain_summary["cache_sha256"]) == 64
    assert len(pretrain_summary["run_fingerprint"]) == 64
    assert (
        pretrain_summary["run_fingerprint_payload"]["cache_sha256"]
        == (pretrain_summary["cache_sha256"])
    )
    assert pretrain_summary["git_commit"]
    assert pretrain_summary["objective"] == "dense_temporal_token_motion_multihorizon"
    assert pretrain_summary["dense_tokens"] is True
    assert pretrain_summary["motion_conditioning"] is True
    assert pretrain_summary["leakage_audit"]["uses_ttc_labels"] is False
    assert pretrain_summary["leakage_audit"]["motion_conditioning_uses_context_only"] is True
    assert pretrain_summary["train_pair_stats"]["target_pair_count"] > 0
    checkpoint = tmp_path / "jepa" / "jepa_encoder_best.pt"
    assert checkpoint.exists()
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint_payload["checkpoint_role"] == "best"
    assert checkpoint_payload["checkpoint_selected_by"] == "validation_loss"
    assert checkpoint_payload["cache_sha256"] == pretrain_summary["cache_sha256"]
    assert checkpoint_payload["run_fingerprint"] == pretrain_summary["run_fingerprint"]

    train_summary = train_tiny_cnn(
        cache_path=cache_path,
        output_dir=tmp_path / "finetune",
        epochs=1,
        batch_size=3,
        seed=5,
        device_name="cpu",
        pretrained_encoder_path=checkpoint,
        freeze_encoder=True,
        train_fraction=0.5,
    )
    assert train_summary["pretrained_encoder"]["source_model"] == "tiny_cnn_jepa"
    assert train_summary["pretrained_encoder"]["source_seed"] == 5
    assert train_summary["pretrained_encoder"]["checkpoint_role"] == "best"
    assert len(train_summary["pretrained_encoder"]["checkpoint_sha256"]) == 64
    assert (
        train_summary["run_fingerprint_payload"]["pretraining_checkpoint_sha256"]
        == (train_summary["pretrained_encoder"]["checkpoint_sha256"])
    )
    assert train_summary["pretrain_seed"] == 5
    assert train_summary["downstream_seed"] == 5
    assert train_summary["freeze_encoder"] is True
    assert train_summary["effective_train_count"] == 3
    scratch_fingerprint = train_tiny_cnn(
        cache_path=cache_path,
        output_dir=tmp_path / "scratch_fingerprint",
        epochs=1,
        batch_size=3,
        seed=5,
        device_name="cpu",
        freeze_encoder=True,
        train_fraction=0.5,
        dry_run_fingerprint=True,
    )
    assert scratch_fingerprint != train_summary["run_fingerprint"]


def test_supervised_training_can_skip_test_evaluation(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    _write_cache(cache_path)

    train_summary = train_tiny_cnn(
        cache_path=cache_path,
        output_dir=tmp_path / "validation_only_finetune",
        epochs=1,
        batch_size=3,
        seed=5,
        device_name="cpu",
        evaluation_splits=("train", "validation"),
    )

    assert train_summary["evaluation_splits"] == ["train", "validation"]
    assert sorted(train_summary["splits"]) == ["train", "validation"]
    predictions = np.load(tmp_path / "validation_only_finetune" / "predictions.npz")
    assert "test_pred" not in predictions.files
    assert "test_true" not in predictions.files


def test_supervised_training_accepts_named_validation_splits(tmp_path: Path) -> None:
    cache_path = tmp_path / "multival_cache.npz"
    _write_multival_cache(cache_path)

    train_summary = train_tiny_cnn(
        cache_path=cache_path,
        output_dir=tmp_path / "multival_finetune",
        epochs=1,
        batch_size=4,
        seed=5,
        device_name="cpu",
        train_splits=("train",),
        validation_splits=("validation_car", "validation_pedestrian"),
        evaluation_splits=("train", "validation_car", "validation_pedestrian"),
    )

    assert train_summary["train_splits"] == ["train"]
    assert train_summary["validation_splits"] == ["validation_car", "validation_pedestrian"]
    assert sorted(train_summary["splits"]) == [
        "train",
        "validation_car",
        "validation_pedestrian",
    ]
    assert train_summary["splits"]["validation_car"]["count"] == 4
    assert train_summary["splits"]["validation_pedestrian"]["count"] == 2


def test_cache_split_remap_uses_sequence_ids(tmp_path: Path) -> None:
    cache_path = tmp_path / "multival_cache.npz"
    _write_multival_cache(cache_path)
    split_path = tmp_path / "split.yaml"
    output_path = tmp_path / "remapped.npz"
    write_structured(
        split_path,
        {
            "splits": {
                "train": ["seq-train"],
                "validation_car": ["seq-car"],
                "validation_pedestrian": ["seq-ped"],
                "test": ["seq-test"],
            }
        },
    )

    summary = remap_cache_splits(
        cache_path=cache_path,
        split_path=split_path,
        output_path=output_path,
    )
    remapped = np.load(output_path)

    assert summary["new_split_counts"] == {
        "test": 2,
        "train": 8,
        "validation_car": 4,
        "validation_pedestrian": 2,
    }
    assert set(remapped["split"].astype(str)) == {
        "train",
        "validation_car",
        "validation_pedestrian",
        "test",
    }


def test_evaluate_supervised_checkpoint_without_retraining(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    _write_cache(cache_path)

    train_summary = train_tiny_cnn(
        cache_path=cache_path,
        output_dir=tmp_path / "validation_only_finetune",
        epochs=1,
        batch_size=3,
        seed=5,
        device_name="cpu",
        evaluation_splits=("train", "validation"),
    )
    checkpoint = Path(train_summary["best_checkpoint"])
    eval_summary = evaluate_supervised_checkpoint(
        cache_path=cache_path,
        checkpoint_path=checkpoint,
        output_path=tmp_path / "test_eval.json",
        batch_size=3,
        device_name="cpu",
        evaluation_splits=("test",),
        allow_final_test_evaluation=True,
    )

    assert eval_summary["evaluation_splits"] == ["test"]
    assert eval_summary["checkpoint_seed"] == 5
    assert eval_summary["checkpoint_epoch"] == train_summary["best_epoch"]
    assert sorted(eval_summary["splits"]) == ["test"]
    assert eval_summary["splits"]["test"]["count"] == 3
    predictions = np.load(tmp_path / "test_eval.predictions.npz")
    assert sorted(predictions.files) == ["test_pred", "test_true"]


def test_token_jepa_deep_supervision(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    _write_cache(cache_path)

    pretrain_summary = pretrain_jepa(
        cache_path=cache_path,
        output_dir=tmp_path / "token_jepa",
        epochs=1,
        batch_size=3,
        seed=5,
        device_name="cpu",
        pretrain_splits=("train",),
        validation_splits=("validation",),
        model_name="token-transformer",
        deep_supervision_layers=(1, 3),
    )

    assert pretrain_summary["objective"] == "deep_dense_temporal_token_motion_multihorizon"
    assert pretrain_summary["deep_supervision"] is True
    assert pretrain_summary["deep_supervision_layers"] == [1, 3]
    assert pretrain_summary["deep_supervision_layer_conditioning"] is True
    assert pretrain_summary["last"]["train"]["deep_supervision_layer_count"] == 2.0


def test_dense_transformer_jepa_predictor(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    _write_cache(cache_path)

    pretrain_summary = pretrain_jepa(
        cache_path=cache_path,
        output_dir=tmp_path / "transformer_predictor_jepa",
        epochs=1,
        batch_size=3,
        seed=5,
        device_name="cpu",
        pretrain_splits=("train",),
        validation_splits=("validation",),
        model_name="token-transformer",
        dense_predictor="transformer",
    )

    assert pretrain_summary["objective"] == ("transformer_dense_temporal_token_motion_multihorizon")
    assert pretrain_summary["dense_predictor"] == "transformer"
    assert pretrain_summary["dense_tokens"] is True
    assert (tmp_path / "transformer_predictor_jepa" / "jepa_encoder_best.pt").exists()


def test_dense_alltoken_jepa_context_loss(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    _write_cache(cache_path)

    pretrain_summary = pretrain_jepa(
        cache_path=cache_path,
        output_dir=tmp_path / "alltoken_jepa",
        epochs=1,
        batch_size=3,
        seed=5,
        device_name="cpu",
        pretrain_splits=("train",),
        validation_splits=("validation",),
        model_name="token-transformer",
        dense_predictor="transformer",
        context_token_weight=0.25,
    )

    assert pretrain_summary["objective"] == (
        "alltoken_transformer_dense_temporal_token_motion_multihorizon"
    )
    assert pretrain_summary["context_token_loss"] is True
    assert pretrain_summary["context_token_weight"] == 0.25
    assert pretrain_summary["last"]["train"]["context_token_loss"] > 0.0
    assert pretrain_summary["last"]["train"]["context_token_target_count"] > 0.0
    assert pretrain_summary["leakage_audit"]["context_token_loss_uses_current_context_only"] is True


def test_visreg_jepa_regularizer(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    _write_cache(cache_path)

    pretrain_summary = pretrain_jepa(
        cache_path=cache_path,
        output_dir=tmp_path / "visreg_jepa",
        epochs=1,
        batch_size=3,
        seed=5,
        device_name="cpu",
        pretrain_splits=("train",),
        validation_splits=("validation",),
        model_name="token-transformer",
        dense_predictor="transformer",
        regularizer="visreg",
        visreg_center_weight=0.3,
        visreg_sketch_weight=0.2,
        visreg_projection_count=8,
        temporal_straightening_weight=0.05,
    )

    assert pretrain_summary["objective"] == (
        "visreg_transformer_dense_temporal_token_motion_multihorizon"
    )
    assert pretrain_summary["regularizer"] == "visreg"
    assert pretrain_summary["visreg_center_weight"] == 0.3
    assert pretrain_summary["visreg_sketch_weight"] == 0.2
    assert pretrain_summary["visreg_projection_count"] == 8
    assert pretrain_summary["temporal_straightening_weight"] == 0.05
    assert pretrain_summary["last"]["train"]["visreg_center_loss"] >= 0.0
    assert pretrain_summary["last"]["train"]["visreg_sketch_loss"] > 0.0
    assert pretrain_summary["last"]["train"]["visreg_projection_count"] == 8.0
    assert pretrain_summary["last"]["train"]["temporal_straightening_loss"] >= 0.0
    assert pretrain_summary["leakage_audit"]["visreg_uses_batch_embeddings_only"] is True
    assert pretrain_summary["leakage_audit"]["visreg_uses_ttc_labels"] is False
    assert pretrain_summary["leakage_audit"]["temporal_straightening_uses_predictions_only"] is True


def test_event_tubelet_jepa_pretraining_smoke(tmp_path: Path) -> None:
    cache_path = tmp_path / "tubelet_cache.npz"
    _write_tubelet_cache(cache_path)

    pretrain_summary = pretrain_jepa(
        cache_path=cache_path,
        output_dir=tmp_path / "tubelet_jepa",
        epochs=1,
        batch_size=2,
        seed=5,
        device_name="cpu",
        pretrain_splits=("train",),
        validation_splits=("validation",),
        model_name="event-tubelet-transformer",
        deep_supervision_layers=(1, 5),
    )

    assert pretrain_summary["model_name"] == "event-tubelet-transformer"
    assert pretrain_summary["objective"] == "deep_dense_temporal_token_motion_multihorizon"
    assert pretrain_summary["dense_tokens"] is True
    assert pretrain_summary["deep_supervision"] is True
    assert (tmp_path / "tubelet_jepa" / "jepa_encoder_best.pt").exists()


def test_flowmimic_auxiliary_pretraining_is_synthetic_only(tmp_path: Path) -> None:
    cache_path = tmp_path / "tubelet_cache.npz"
    _write_tubelet_cache(cache_path)

    pretrain_summary = pretrain_jepa(
        cache_path=cache_path,
        output_dir=tmp_path / "flowmimic_jepa",
        epochs=1,
        batch_size=2,
        seed=5,
        device_name="cpu",
        pretrain_splits=("train",),
        validation_splits=("validation",),
        temporal_horizons_ms=(20,),
        flowmimic_alignment_weight=0.2,
        flowmimic_inverse_ttc_weight=0.1,
        flowmimic_minimum_ttc_s=0.8,
        flowmimic_maximum_ttc_s=1.6,
    )

    assert pretrain_summary["objective"].startswith("flowmimic_")
    assert pretrain_summary["flowmimic_enabled"] is True
    assert pretrain_summary["last"]["train"]["flowmimic_alignment_loss"] > 0.0
    assert pretrain_summary["last"]["train"]["flowmimic_inverse_ttc_loss"] > 0.0
    assert pretrain_summary["last"]["validation"]["flowmimic_alignment_loss"] == 0.0
    assert pretrain_summary["leakage_audit"]["flowmimic_uses_real_ttc_labels"] is False
    assert pretrain_summary["leakage_audit"]["flowmimic_uses_analytic_synthetic_ttc"] is True
    checkpoint = torch.load(
        tmp_path / "flowmimic_jepa" / "jepa_encoder_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["flowmimic_inverse_ttc_head_state_dict"] is not None


def test_event_tubelet_rope_jepa_pretraining_smoke(tmp_path: Path) -> None:
    cache_path = tmp_path / "tubelet_cache.npz"
    _write_tubelet_cache(cache_path)

    pretrain_summary = pretrain_jepa(
        cache_path=cache_path,
        output_dir=tmp_path / "tubelet_rope_jepa",
        epochs=1,
        batch_size=2,
        seed=5,
        device_name="cpu",
        pretrain_splits=("train",),
        validation_splits=("validation",),
        model_name="event-tubelet-rope-transformer",
        dense_predictor="transformer",
    )

    assert pretrain_summary["model_name"] == "event-tubelet-rope-transformer"
    assert pretrain_summary["objective"] == ("transformer_dense_temporal_token_motion_multihorizon")
    assert pretrain_summary["dense_predictor"] == "transformer"
    assert (tmp_path / "tubelet_rope_jepa" / "jepa_encoder_best.pt").exists()


def test_tubelet_mask_preserves_auxiliary_channels() -> None:
    x = torch.ones(2, 14, 16, 16)
    x[:, 10:] = 3.0

    masked = _tubelet_masked_context(x, mask_ratio=0.4, block_count=4, event_bins=5)

    assert torch.any(masked[:, :10] == 0.0)
    assert torch.all(masked[:, 10:] == 3.0)


def test_flowmimic_navigation_is_neutral_after_train_normalization() -> None:
    synthetic = torch.zeros(2, 21, 8, 8)
    action_mean = torch.tensor([0.0] * 6 + [8.0, -4.0, 2.0, -0.1, 0.04, -0.2, 0.01, 0.02, 1.0])

    _neutralize_synthetic_navigation(
        synthetic,
        bins=5,
        metadata_channels=True,
        navigation_feature_count=9,
        action_feature_mean=action_mean,
    )

    expected = action_mean[-9:].view(1, 9, 1, 1).expand(2, -1, 8, 8)
    torch.testing.assert_close(synthetic[:, 12:21], expected)
    normalized = (synthetic[:, 12:21].mean(dim=(2, 3)) - action_mean[-9:]) / torch.tensor(
        [4.0, 2.0, 7.0, 0.2, 0.9, 1.4, 0.4, 0.4, 1e-6]
    )
    assert float(normalized.abs().max()) < 1e-6


def test_event_tubelet_jepa_tubelet_mask_smoke(tmp_path: Path) -> None:
    cache_path = tmp_path / "tubelet_cache.npz"
    _write_tubelet_cache(cache_path)

    pretrain_summary = pretrain_jepa(
        cache_path=cache_path,
        output_dir=tmp_path / "tubelet_mask_jepa",
        epochs=1,
        batch_size=2,
        seed=5,
        device_name="cpu",
        pretrain_splits=("train",),
        validation_splits=("validation",),
        model_name="event-tubelet-transformer",
        dense_predictor="transformer",
        mask_mode="tubelet",
    )

    assert pretrain_summary["objective"] == (
        "tubeletmask_transformer_dense_temporal_token_motion_multihorizon"
    )
    assert pretrain_summary["mask_mode"] == "tubelet"
    assert (
        pretrain_summary["leakage_audit"]["tubelet_masking_uses_context_event_channels_only"]
        is True
    )
    assert pretrain_summary["leakage_audit"]["tubelet_masking_preserves_auxiliary_channels"] is True
    assert (tmp_path / "tubelet_mask_jepa" / "jepa_encoder_best.pt").exists()


def test_jepa_action_conditioning_uses_causal_navigation(tmp_path: Path) -> None:
    cache_path = tmp_path / "navigation_cache.npz"
    _write_navigation_cache(cache_path)

    pretrain_summary = pretrain_jepa(
        cache_path=cache_path,
        output_dir=tmp_path / "action_jepa",
        epochs=1,
        batch_size=3,
        seed=5,
        device_name="cpu",
        pretrain_splits=("train",),
        validation_splits=("validation",),
    )

    assert pretrain_summary["objective"] == "dense_temporal_token_action_multihorizon"
    assert pretrain_summary["motion_conditioning"] is True
    assert pretrain_summary["action_conditioning"] is True
    assert pretrain_summary["uses_navigation_action_conditioning"] is True
    assert pretrain_summary["action_feature_dim"] == 15
    assert pretrain_summary["motion_feature_dim"] == 15
    assert pretrain_summary["action_feature_normalization"] is True
    assert pretrain_summary["action_feature_normalization_source"] == (
        "pretrain_context_indices_train_only"
    )
    assert len(pretrain_summary["action_feature_mean"]) == 15
    assert len(pretrain_summary["action_feature_std"]) == 15
    assert pretrain_summary["navigation_feature_names"] == list(NAVIGATION_FEATURE_NAMES)
    assert pretrain_summary["action_feature_names"][-1] == "ego_navigation_valid"
    assert pretrain_summary["leakage_audit"]["action_conditioning_uses_context_only"] is True
    assert pretrain_summary["leakage_audit"]["action_feature_normalization_uses_train_only"] is True
    assert pretrain_summary["leakage_audit"]["uses_future_navigation"] is False
    assert (
        pretrain_summary["leakage_audit"]["future_navigation_channels_zeroed_before_target_encoder"]
        is True
    )
    assert pretrain_summary["leakage_audit"]["target_windows_are_disjoint"] is True
    assert pretrain_summary["leakage_audit"]["collapse_statistics_mix_token_positions"] is False


def test_temporal_pairs_use_disjoint_future_window_starts() -> None:
    timestamp_us = np.arange(12, dtype=np.int64) * 20_000 + 100_000
    context_start_us = timestamp_us - 100_000
    context_end_us = timestamp_us.copy()

    context_idx, target_idx, stats = _build_temporal_pairs(
        split=np.array(["train"] * 12),
        sequence_id=np.array(["fixture"] * 12),
        timestamp_us=timestamp_us,
        context_start_us=context_start_us,
        context_end_us=context_end_us,
        split_names=("train",),
        horizons_ms=(20,),
        max_target_slop_ms=0,
    )

    assert context_idx[0] == 0
    assert target_idx[0, 0] == 6
    assert context_start_us[target_idx[0, 0]] == context_end_us[context_idx[0]] + 20_000
    assert stats["target_windows_are_disjoint"] is True


def test_future_navigation_is_zeroed_before_target_encoding() -> None:
    future = torch.arange(2 * 21 * 4, dtype=torch.float32).reshape(1, 2, 21, 2, 2)

    target = _without_future_navigation(
        future,
        bins=5,
        metadata_channels=True,
        navigation_feature_count=9,
    )

    assert torch.equal(target[:, :, :12], future[:, :, :12])
    assert torch.all(target[:, :, 12:] == 0.0)
    assert torch.any(future[:, :, 12:] != 0.0)
