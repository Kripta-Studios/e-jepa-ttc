"""Atomic shard rename must not crash directory disk accounting or alter task ownership."""

from operational.train40_system.prepare_safe import committed_bytes, selected_groups


def test_renamed_pending_shard_is_not_statted(tmp_path):
    (tmp_path / "shard_00000.npz").write_bytes(b"1234")
    (tmp_path / "shard_00001.pending.npz").write_bytes(b"pending")
    assert committed_bytes(tmp_path) == 4


def test_old_and_new_partitions_remain_disjoint_even_at_group_boundary():
    rows = [{"sequence_id": "old"} for _ in range(40)]
    rows += [{"sequence_id": "new"} for _ in range(40)]
    old = selected_groups(rows, {"old"}, "original")
    new = selected_groups(rows, {"old"}, "extra")
    assert {start for start, _ in old}.isdisjoint({start for start, _ in new})
    combined = sorted(old + new)
    assert combined == selected_groups(rows, {"old"}, "all")
    assert sum(len(group) for _, group in combined) == 80
