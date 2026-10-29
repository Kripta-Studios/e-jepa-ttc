from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from e_jepa_ttc.training.eap_lhr_jepa_ttc import _balanced_sampler


class _FakeShardedDataset(Dataset[dict[str, Any]]):
    def __init__(self) -> None:
        self.getitem_calls = 0
        self.entries = [
            (Path("shard-a.pt.gz"), 0),
            (Path("shard-a.pt.gz"), 1),
            (Path("shard-a.pt.gz"), 2),
            (Path("shard-b.pt.gz"), 0),
            (Path("shard-b.pt.gz"), 1),
            (Path("shard-b.pt.gz"), 2),
        ]
        self.rows = [
            {"sampling_group": "near", "sequence_id": "a", "track_id": "1"},
            {"sampling_group": "near", "sequence_id": "a", "track_id": "1"},
            {"sampling_group": "far", "sequence_id": "a", "track_id": "2"},
            {"sampling_group": "near", "sequence_id": "b", "track_id": "3"},
            {"sampling_group": "far", "sequence_id": "b", "track_id": "4"},
            {"sampling_group": "far", "sequence_id": "b", "track_id": "4"},
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        self.getitem_calls += 1
        return dict(self.rows[index])


def test_balanced_batches_are_deterministic_and_shard_local() -> None:
    dataset = _FakeShardedDataset()
    first = list(_balanced_sampler(dataset, batch_size=2, seed=17))
    second = list(_balanced_sampler(dataset, batch_size=2, seed=17))

    assert first == second
    assert dataset.getitem_calls == len(dataset)
    assert sum(len(batch) for batch in first) == len(dataset)
    for batch in first:
        paths = {dataset.entries[index][0] for index in batch}
        assert len(paths) == 1
