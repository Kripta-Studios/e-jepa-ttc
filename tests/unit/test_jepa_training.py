from pathlib import Path

import numpy as np

from e_jepa_ttc.training.jepa import pretrain_jepa
from e_jepa_ttc.training.supervised import train_tiny_cnn


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
