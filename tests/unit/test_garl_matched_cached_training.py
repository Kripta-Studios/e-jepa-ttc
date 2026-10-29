from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.train_garl_matched_from_cache import (
    GarlMatchedTensorCache,
    GarlSequenceIndexedView,
    ShardGroupedSampler,
    _collate,
    _load_grouped_contract,
    _prediction,
    _resolve_device,
)

ROOT = Path(__file__).resolve().parents[2]
GROUPED_PROTOCOL = ROOT / "configs/protocol/scientific_recovery_v5_train_only_grouped_dev.json"


def _write_cache(tmp_path: Path) -> Path:
    shards: list[dict[str, object]] = []
    for split in ("train", "validation"):
        split_dir = tmp_path / split
        split_dir.mkdir(parents=True)
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
    assert batch.track_ids == ("", "")
    assert torch.equal(batch.visible_height, torch.tensor([[10.0, 11.0], [20.0, 22.0]]))


def test_grouped_view_preserves_exact_fold_identity(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.parquet"
    import pandas as pd

    pd.DataFrame(
        {
            "sample_token": ["train-0", "train-1"],
            "sequence_id": ["seq-a", "seq-b"],
            "track_id": ["track-a", "track-b"],
        }
    ).to_parquet(metadata, index=False)
    base = GarlMatchedTensorCache(_write_cache(tmp_path / "cache"), "train")

    view = GarlSequenceIndexedView(
        base,
        metadata_path=metadata,
        sequence_ids={"seq-b"},
    )

    assert len(view) == 1
    assert view[0]["sample_token"] == "train-1"
    assert view[0]["track_id"] == "track-b"
    assert _collate([view[0]]).track_ids == ("track-b",)
    assert view.identity_frame().to_dict("records") == [
        {
            "sample_token": "train-1",
            "sequence_id": "seq-b",
            "track_id": "track-b",
        }
    ]
    assert view.shard_index_groups() == ((0,),)


def test_garl_grouped_contract_uses_frozen_fold_without_validation() -> None:
    contract = _load_grouped_contract(GROUPED_PROTOCOL, 2)

    assert contract["fold_index"] == 2
    assert contract["fold"]["train_rows"] == 5462
    assert contract["fold"]["dev_rows"] == 2730
    assert contract["train_sequences"].isdisjoint(contract["dev_sequences"])
    assert contract["protocol"]["checks"]["public_validation_used_for_selection"] is False
    assert contract["protocol"]["checks"]["private_test_opened"] is False


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


def test_cuda_device_gets_explicit_default_index() -> None:
    assert _resolve_device("cuda") == torch.device("cuda:0")
    assert _resolve_device("cuda:1") == torch.device("cuda:1")
    assert _resolve_device("cpu") == torch.device("cpu")


def test_garl_runner_binds_repository_revision_around_training() -> None:
    source = Path("scripts/train_garl_matched_from_cache.py").read_text(encoding="utf-8")

    launch = source.index("launch_git_commit = _repository_commit()")
    training = source.index("for epoch in range(start_epoch, epochs + 1):")
    completion = source.index("if _repository_commit() != launch_git_commit:")

    assert launch < training < completion
    assert "refusing to publish artifacts" in source
