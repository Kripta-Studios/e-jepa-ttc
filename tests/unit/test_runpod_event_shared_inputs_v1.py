from __future__ import annotations

import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import pytest

from scripts.runpod_event_shared_inputs_v1 import EXPECTED_LIMIT_BYTES, share_fold_inputs


class Inputs:
    def __init__(self, root: Path, values: dict[int, str]) -> None:
        self.output = root
        self.directory = root / "producer_inputs"
        self.metadata = [{"sample_token": f"token-{index}"} for index in sorted(values)]
        self.cache: OrderedDict[int, str] = OrderedDict()
        self.cache_bytes = 0
        self.limit = EXPECTED_LIMIT_BYTES
        self.lock = Lock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.reads = 0
        self.hits = 0
        self.values = values
        self.closed = 0

    def batch(self, indices: list[int]) -> list[str]:
        return self.pool.submit(self._batch, indices).result()

    def _batch(self, indices: list[int]) -> list[str]:
        output = []
        for index in indices:
            with self.lock:
                if index in self.cache:
                    self.hits += 1
                    value = self.cache.pop(index)
                else:
                    self.reads += 1
                    value = self.values[index]
                self.cache[index] = value
                while len(self.cache) > 2:
                    self.cache.popitem(last=False)
                self.cache_bytes = len(self.cache)
            output.append(value)
        return output

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        self.cache.clear()
        self.cache_bytes = 0
        self.closed += 1


class RoleSource:
    def __init__(self, role: str, root: Path, values: dict[int, str]) -> None:
        self._inputs = Inputs(root, values)
        self._tokens = [f"token-{index}" for index in sorted(values)]
        self._event_indices = list(sorted(values))
        self.identity = {
            "role": role,
            "event_source_root": str(root),
            "prepare_freeze_sha256": "prepare",
            "train40_rows_sha256": "rows",
            "train40_index_sha256": "index",
            "dino_teacher_freeze_sha256": "teacher-freeze",
            "dino_targets_sha256": "teacher-manifest",
            "decoded_event_cache_limit_bytes": EXPECTED_LIMIT_BYTES,
        }

    def batch(self, indices: list[int], modality: str) -> list[str]:
        assert modality == "event"
        return self._inputs.batch([self._event_indices[index] for index in indices])

    def close(self) -> None:
        self._inputs.close()


class Fold:
    def __init__(self, root: Path) -> None:
        values = {index: f"value-{index}" for index in range(4)}
        self.sources = {role: RoleSource(role, root, values) for role in "PHV"}
        self.identity = {
            "identity_sha256": "original-scientific-identity",
            "storage_source_identity": {
                role: dict(source.identity) for role, source in self.sources.items()
            },
        }

    def close(self) -> None:
        for source in self.sources.values():
            source.close()


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "event-root"
    (root / "producer_inputs").mkdir(parents=True)
    return root


def test_batches_match_for_miss_hit_and_eviction_without_filter_change(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    reference = Fold(root)
    expected = {
        role: [source.batch(indices, "event") for indices in ([0, 1], [1], [2, 3], [0])]
        for role, source in reference.sources.items()
    }
    reference.close()

    fold = Fold(root)
    identity = dict(fold.identity)
    tokens = {role: list(source._tokens) for role, source in fold.sources.items()}
    indices = {role: list(source._event_indices) for role, source in fold.sources.items()}
    shared = share_fold_inputs(fold, tmp_path / "policy.json")
    observed = {
        role: [source.batch(batch, "event") for batch in ([0, 1], [1], [2, 3], [0])]
        for role, source in shared.sources.items()
    }
    assert observed == expected
    assert shared.identity == identity
    assert {role: source._tokens for role, source in shared.sources.items()} == tokens
    assert {role: source._event_indices for role, source in shared.sources.items()} == indices
    assert len({id(source._inputs) for source in shared.sources.values()}) == 1
    policy = json.loads((tmp_path / "policy.json").read_text(encoding="utf-8"))
    assert policy["original_source_identity_sha256"] == identity["identity_sha256"]
    assert policy["scientific_source_identity_changed"] is False
    assert policy["data_filter_changed"] is False


def test_duplicate_pools_close_and_shared_close_is_idempotent(tmp_path: Path) -> None:
    fold = Fold(_root(tmp_path))
    original = [source._inputs for source in fold.sources.values()]
    share_fold_inputs(fold, tmp_path / "policy.json")
    assert sorted(inputs.closed for inputs in original) == [0, 1, 1]
    fold.close()
    fold.close()
    assert [inputs.closed for inputs in original] == [1, 1, 1]
    assert all(not thread.is_alive() for inputs in original for thread in inputs.pool._threads)


@pytest.mark.parametrize("difference", ["root", "teacher", "metadata"])
def test_distinct_storage_is_never_aliased(tmp_path: Path, difference: str) -> None:
    fold = Fold(_root(tmp_path))
    candidate = fold.sources["V"]
    if difference == "root":
        other = _root(tmp_path / "other")
        candidate._inputs.output = other
        candidate._inputs.directory = other / "producer_inputs"
        candidate.identity["event_source_root"] = str(other)
    elif difference == "teacher":
        candidate.identity["dino_targets_sha256"] = "different"
    else:
        candidate._inputs.metadata[0]["sample_token"] = "different"
    with pytest.raises(ValueError, match="same event/teacher storage"):
        share_fold_inputs(fold, tmp_path / "policy.json")
    assert len({id(source._inputs) for source in fold.sources.values()}) == 3
    fold.close()


def test_nonempty_cache_is_rejected_before_any_alias(tmp_path: Path) -> None:
    fold = Fold(_root(tmp_path))
    fold.sources["H"].batch([0], "event")
    with pytest.raises(ValueError, match="before the first cache read"):
        share_fold_inputs(fold, tmp_path / "policy.json")
    assert len({id(source._inputs) for source in fold.sources.values()}) == 3
    fold.close()
