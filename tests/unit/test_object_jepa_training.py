from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from e_jepa_ttc.training.object_jepa import (
    fine_tune_object_ttc,
    pretrain_object_event_jepa,
)


def _write_shard(path: Path, *, split: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    count, context_steps, horizons, channels, size = 4, 3, 3, 4, 16
    context_boxes = np.tile(
        np.asarray([0.2, 0.2, 0.5, 0.6], dtype=np.float32),
        (count, context_steps, 1, 1),
    )
    future_boxes = np.tile(
        np.asarray([0.18, 0.18, 0.52, 0.62], dtype=np.float32),
        (count, horizons, 1, 1),
    )
    np.savez_compressed(
        path,
        context_events=rng.normal(
            size=(count, context_steps, channels, size, size)
        ).astype(np.float16),
        context_boxes=context_boxes,
        context_sampling_boxes=np.tile(
            np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
            (count, context_steps, 1, 1),
        ),
        context_object_mask=np.ones((count, context_steps, 1), dtype=np.bool_),
        context_depth_m=np.full((count, 1), 10.0, dtype=np.float32),
        context_ego_actions=np.zeros((count, context_steps, 3), dtype=np.float32),
        context_ego_action_mask=np.zeros((count, context_steps), dtype=np.bool_),
        future_events=rng.normal(size=(count, horizons, channels, size, size)).astype(
            np.float16
        ),
        future_boxes=future_boxes,
        future_sampling_boxes=np.tile(
            np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
            (count, horizons, 1, 1),
        ),
        future_object_mask=np.ones((count, horizons, 1), dtype=np.bool_),
        future_depth_m=np.tile(
            np.asarray([9.0, 8.0, 7.0], dtype=np.float32)[None, :, None],
            (count, 1, 1),
        ),
        future_ego_actions=np.zeros((count, horizons, 3), dtype=np.float32),
        future_ego_action_mask=np.zeros((count, horizons), dtype=np.bool_),
        ttc_s=np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
        sample_token=np.asarray([f"{split}:{index}" for index in range(count)]),
        sequence_id=np.asarray([f"{split}_sequence"] * count),
        track_id=np.asarray(["track"] * count),
        category=np.asarray(["car"] * count),
        split=np.asarray([split] * count),
        ttc_source=np.asarray(["synthetic"] * count),
        prediction_horizons_s=np.asarray([0.1, 0.25, 0.5], dtype=np.float32),
        cache_format_version=np.asarray(1),
        future_window_semantics=np.asarray("endpoint_offset_disjoint_fixed_duration"),
    )


def _write_manifest(root: Path) -> Path:
    shards = []
    for seed, split in enumerate(("train", "validation", "calibration", "test")):
        path = root / f"{split}.npz"
        _write_shard(path, split=split, seed=seed)
        shards.append(
            {
                "path": path.name,
                "split": split,
                "sequence_id": f"{split}_sequence",
                "samples": 4,
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"shards": shards}), encoding="utf-8")
    return manifest


def test_object_jepa_pretrain_and_matched_finetune_smoke(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    pretrain = pretrain_object_event_jepa(
        cache_manifest_path=manifest,
        output_dir=tmp_path / "pretrain",
        epochs=1,
        batch_size=2,
        embedding_dim=32,
        feature_dim=32,
        predictor_depth=1,
        predictor_heads=4,
        device_name="cpu",
    )

    assert pretrain["uses_ttc_labels"] is False
    assert Path(pretrain["best_checkpoint"]).is_file()
    finetune = fine_tune_object_ttc(
        cache_manifest_path=manifest,
        output_dir=tmp_path / "finetune",
        pretrained_checkpoint_path=pretrain["best_checkpoint"],
        epochs=1,
        batch_size=2,
        label_fraction=0.5,
        device_name="cpu",
    )

    assert finetune["initialization"] == "jepa"
    assert finetune["effective_label_count"] < finetune["full_train_count"]
    assert finetune["test_evaluated_after_model_selection_and_calibration"] is True
    assert Path(finetune["best_checkpoint"]).is_file()
