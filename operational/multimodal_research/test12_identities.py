"""Resolve overlapping-clip aliases only when all native sensor/ROI input fields agree."""

from .test12_native_inputs import native_histories


def physical_histories(jobs: list[dict]) -> tuple[dict, dict, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    tokens = [row["sample_token"] for row in jobs]
    assert len(set(tokens)) == len(tokens), "Duplicate official token"
    for row in jobs:
        groups.setdefault((row["sequence_id"], row["track_id"], row["anchor_us"]), []).append(row)
    aliases, unique = {}, []
    for rows in groups.values():
        rows = sorted(rows, key=lambda row: row["sample_token"])
        canonical = rows[0]
        signature = {k: v for k, v in canonical.items() if k != "sample_token"}
        for row in rows:
            assert {k: v for k, v in row.items() if k != "sample_token"} == signature, (
                "Conflicting sensor/ROI inputs at the same physical object timestamp"
            )
            aliases[row["sample_token"]] = canonical["sample_token"]
        unique.append(canonical)
    histories = native_histories(unique)
    return {token: histories[aliases[token]] for token in tokens}, aliases, unique
