from pathlib import Path

import numpy as np

from e_jepa_ttc.data.evttc import NAVIGATION_FEATURE_NAMES
from e_jepa_ttc.data.ml_cache import remap_cache_splits
from e_jepa_ttc.training.jepa import pretrain_jepa
from e_jepa_ttc.training.supervised import evaluate_supervised_checkpoint, train_tiny_cnn
from e_jepa_ttc.utils.io import write_structured


def _write_cache(path: Path) -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(12, 6, 12, 16)).astype(np.float16)
    y_ttc = np.linspace(1.0, 6.0, num=12, dtype=np.float32)
    split = np.array(["train"] * 6 + ["validation"] * 3 + ["test"] * 3)
    np.savez(
        path,
        x=x,
        y_ttc=y_ttc,
        split=split,
        timestamp_us=np.arange(12, dtype=np.int64) * 20_000,
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
        x=x,
        y_ttc=y_ttc,
        split=split,
        timestamp_us=np.arange(12, dtype=np.int64) * 20_000,
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
        x=x,
        y_ttc=y_ttc,
        split=split,
        timestamp_us=np.arange(12, dtype=np.int64) * 20_000,
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
    sequence_id = np.array(
        ["seq-train"] * 8 + ["seq-car"] * 4 + ["seq-ped"] * 2 + ["seq-test"] * 2
    )
    np.savez(
        path,
        x=x,
        y_ttc=y_ttc,
        split=split,
        sequence_id=sequence_id,
        timestamp_us=np.arange(16, dtype=np.int64) * 20_000,
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
    assert pretrain_summary["objective"] == "dense_temporal_token_motion_multihorizon"
    assert pretrain_summary["dense_tokens"] is True
    assert pretrain_summary["motion_conditioning"] is True
    assert pretrain_summary["leakage_audit"]["uses_ttc_labels"] is False
    assert pretrain_summary["leakage_audit"]["motion_conditioning_uses_context_only"] is True
    assert pretrain_summary["train_pair_stats"]["target_pair_count"] > 0
    checkpoint = tmp_path / "jepa" / "jepa_encoder_best.pt"
    assert checkpoint.exists()

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
    assert train_summary["freeze_encoder"] is True
    assert train_summary["effective_train_count"] == 3


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
    )

    assert eval_summary["evaluation_splits"] == ["test"]
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
    assert (
        pretrain_summary["leakage_audit"]["action_feature_normalization_uses_train_only"]
        is True
    )
    assert pretrain_summary["leakage_audit"]["uses_future_navigation"] is False
