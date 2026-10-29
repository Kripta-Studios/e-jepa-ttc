from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

import scripts.build_garl_matched_preprocessing_cache as cache_builder


class _FixtureDataset(torch.utils.data.Dataset[dict[str, Any]]):
    def __init__(self, *, wrong_target: bool = False) -> None:
        self.rows = [
            {
                "data": torch.full((40, 128, 128), float(index) / 10.0),
                "target": torch.tensor(float(index + 1 + int(wrong_target))),
                "visible_height": torch.tensor([10.0 + index, 11.0 + index]),
                "sample_token": f"token-{index}",
            }
            for index in range(4)
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]

    def get_collate_fn(self):
        def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "data": torch.stack([row["data"] for row in rows]),
                "target": torch.stack([row["target"] for row in rows]),
                "visible_height": torch.stack([row["visible_height"] for row in rows]),
                "sample_token": [row["sample_token"] for row in rows],
            }

        return collate


def _expected() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_token": [f"token-{index}" for index in range(4)],
            "sequence_id": ["seq-a", "seq-a", "seq-b", "seq-b"],
            "ttc": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_materialize_split_writes_hashed_exact_tensor_shard(tmp_path: Path) -> None:
    shards, report = cache_builder._materialize_split(
        dataset=_FixtureDataset(),
        split="train",
        destination=tmp_path,
        expected=_expected(),
        batch_size=2,
        num_workers=0,
        shard_size=4,
    )

    assert report["rows"] == 4
    assert report["sequence_count"] == 2
    assert report["maximum_target_absolute_error_s"] == 0.0
    assert len(shards) == 1
    assert shards[0]["torch_load_verified"] is True
    payload = torch.load(tmp_path / shards[0]["path"], weights_only=False)
    assert payload["data"].shape == (4, 40, 128, 128)
    assert payload["sample_tokens"] == tuple(_expected()["sample_token"])


def test_materialize_split_rejects_target_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="target mismatch"):
        cache_builder._materialize_split(
            dataset=_FixtureDataset(wrong_target=True),
            split="train",
            destination=tmp_path,
            expected=_expected(),
            batch_size=2,
            num_workers=0,
            shard_size=4,
        )
