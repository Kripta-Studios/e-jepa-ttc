"""Additional preparation cannot race the active original-pass shard owners."""

from operational.train40_system.prepare_remaining import extra_groups


def test_extra_groups_are_disjoint_and_cover_skipped_boundary():
    rows = [{"sequence_id": "old"}] * 47 + [{"sequence_id": "new"}] * 36
    groups = extra_groups(rows, {"old"})
    assert [start for start, _ in groups] == [32, 64]
    assert sum(len(group) for _, group in groups) == 51
    original_groups = [
        start
        for start in range(0, len(rows), 32)
        if {r["sequence_id"] for r in rows[start : start + 32]} <= {"old"}
    ]
    assert not set(original_groups) & {start for start, _ in groups}
