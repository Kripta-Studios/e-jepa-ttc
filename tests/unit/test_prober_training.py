from pathlib import Path

import numpy as np
import pytest
import torch

from e_jepa_ttc.training.jepa import pretrain_jepa
from e_jepa_ttc.training.prober import (
    _compose_rollout_features,
    evaluate_roi_latent_ttc_prober_checkpoint,
    evaluate_roi_rollout_ttc_prober_checkpoint,
    train_latent_ttc_prober,
)


def _write_cache(path: Path) -> None:
    rng = np.random.default_rng(23)
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


def test_latent_ttc_prober_trains_from_frozen_jepa_encoder(tmp_path: Path) -> None:
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

    prober_summary = train_latent_ttc_prober(
        cache_path=cache_path,
        encoder_checkpoint_path=pretrain_summary["best_checkpoint"],
        output_dir=tmp_path / "prober",
        epochs=1,
        batch_size=3,
        seed=5,
        device_name="cpu",
        train_splits=("train",),
        validation_splits=("validation",),
        evaluation_splits=("train", "validation"),
    )

    assert prober_summary["model"] == "latent_ttc_prober"
    assert prober_summary["encoder_checkpoint"]["source_model_name"] == "tiny-cnn"
    assert prober_summary["physics_prior"] == "ridge"
    assert prober_summary["latent_feature_dim"] > 0
    assert prober_summary["physics_feature_dim"] == 6
    assert sorted(prober_summary["splits"]) == ["train", "validation"]
    assert prober_summary["leakage_audit"]["encoder_frozen"] is True
    assert prober_summary["leakage_audit"]["uses_validation_or_test_ttc_for_prior_fit"] is False
    assert (tmp_path / "prober" / "latent_prober_best.pt").exists()


def test_roi_latent_prober_checkpoint_evaluation_rejects_wrong_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "wrong.pt"
    torch.save({"model": "latent_ttc_prober"}, checkpoint_path)

    with pytest.raises(ValueError, match="roi_latent_ttc_prober"):
        evaluate_roi_latent_ttc_prober_checkpoint(
            manifest_path=tmp_path / "missing_manifest.yaml",
            split_path=tmp_path / "missing_split.yaml",
            cache_path=tmp_path / "missing_cache.npz",
            prober_checkpoint_path=checkpoint_path,
            device_name="cpu",
        )


def test_compose_rollout_dynamics_features_includes_temporal_terms() -> None:
    context = torch.ones((2, 4), dtype=torch.float32)
    pred = torch.stack(
        [
            torch.full((2, 4), 2.0),
            torch.full((2, 4), 3.0),
            torch.full((2, 4), 5.0),
        ],
        dim=1,
    )

    flat = _compose_rollout_features(
        context_summary=context,
        pred_summary=pred,
        horizons_ms=(20, 60, 100),
        include_context_latent=True,
        feature_mode="flat",
    )
    dynamics = _compose_rollout_features(
        context_summary=context,
        pred_summary=pred,
        horizons_ms=(20, 60, 100),
        include_context_latent=True,
        feature_mode="dynamics",
    )

    assert flat.shape == (2, 16)
    assert dynamics.shape[0] == 2
    assert dynamics.shape[1] > flat.shape[1]


def test_roi_rollout_prober_checkpoint_evaluation_rejects_wrong_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "wrong_rollout.pt"
    torch.save({"model": "roi_latent_ttc_prober"}, checkpoint_path)

    with pytest.raises(ValueError, match="roi_rollout_ttc_prober"):
        evaluate_roi_rollout_ttc_prober_checkpoint(
            manifest_path=tmp_path / "missing_manifest.yaml",
            split_path=tmp_path / "missing_split.yaml",
            cache_path=tmp_path / "missing_cache.npz",
            prober_checkpoint_path=checkpoint_path,
            device_name="cpu",
        )
