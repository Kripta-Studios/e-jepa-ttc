from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.train_garl_matched_from_cache import (
    GarlMatchedTensorCache,
    ShardGroupedSampler,
    _collate,
    _prediction,
)

ROOT = Path(__file__).resolve().parents[2]


def _write_cache(tmp_path: Path) -> Path:
    shards: list[dict[str, object]] = []
    for split in ("train", "validation"):
        split_dir = tmp_path / split
        split_dir.mkdir()
        path = split_dir / "shard-00000.pt"
        torch.save(
            {
                "data": torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2),
                "target": torch.tensor([1.0, 2.0]),
                "visible_height": torch.tensor([[10.0, 11.0], [20.0, 22.0]]),
                "sample_tokens": (f"{split}-0", f"{split}-1"),
                "sequence_ids": ("seq-a", "seq-b"),
            },
            path,
        )
        shards.append({"split": split, "path": f"{split}/{path.name}", "rows": 2})
    manifest = {
        "artifact_type": "garl_official_event_only_matched_preprocessing_cache_v1",
        "split_counts": {"train": 2, "validation": 2},
        "shards": shards,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_cached_dataset_preserves_tokens_and_tensor_rows(tmp_path: Path) -> None:
    dataset = GarlMatchedTensorCache(_write_cache(tmp_path), "train")

    assert len(dataset) == 2
    assert dataset.shard_index_groups() == ((0, 1),)
    assert dataset[1]["sample_token"] == "train-1"
    batch = _collate([dataset[0], dataset[1]])
    assert batch.data.shape == (2, 2, 2, 2)
    assert batch.sequence_ids == ("seq-a", "seq-b")
    assert torch.equal(batch.visible_height, torch.tensor([[10.0, 11.0], [20.0, 22.0]]))


def test_shard_grouped_sampler_is_exact_and_seeded() -> None:
    groups = ((0, 1), (2, 3), (4, 5))
    first = list(ShardGroupedSampler(groups, torch.Generator().manual_seed(7)))
    second = list(ShardGroupedSampler(groups, torch.Generator().manual_seed(7)))

    assert first == second
    assert sorted(first) == list(range(6))
    assert all(abs(first.index(left) - first.index(right)) == 1 for left, right in groups)


def test_shard_grouped_sampler_rejects_non_partition() -> None:
    with pytest.raises(ValueError, match="partition"):
        ShardGroupedSampler(((0, 2),), torch.Generator().manual_seed(7))


def test_official_height_ratio_prediction_matches_physics() -> None:
    raw_height = torch.tensor([[9.0, 10.0], [11.0, 10.0]])

    prediction = _prediction(raw_height, delta_t_s=0.1)

    assert torch.allclose(prediction, torch.tensor([1.0, -1.0]))


def test_cached_training_script_is_directly_executable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/train_garl_matched_from_cache.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--cache-manifest" in completed.stdout
