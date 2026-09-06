"""Resolve actual D1 query pools and fixed producer assignments, without fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.coordination import verified_ack
from e_jepa_ttc.simplex_t.pools import QueryIdentity, expanded_pool, expansion_producer


def main() -> None:
    """Preserve each OLD TRAIN set and assign expansion by registered SHA256 modulo3."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    root = Path(paths["worktree"])
    config = json.loads(
        (root / "configs/experiment/simplex_t_coordination.json").read_text(encoding="utf-8")
    )
    ack = verified_ack(
        Path(paths["shared_coordination"]) / config["ack_filename"], config["ack_sha256"]
    )
    roles = ack["interfaces"]["role_manifest"]["roles"]
    allowed = set(roles["expansion"])
    if allowed & set(roles["original"] + roles["protected"] + roles["confirmation"]):
        raise ValueError("expansion conflicts with existing roles")
    expansion = Path(paths["stage70_worktree_read_only"]) / (
        "artifacts/stage70_76_architecture/expansion_usability/USABLE_METADATA.parquet"
    )
    if sha256(expansion) != "e8514492952a3abd86e45e3b00c07e398d2e6def5b73062eb5b5f8f13a932064":
        raise ValueError("audited expansion metadata changed")
    columns = ["sample_token", "sequence_id", "track_id"]
    metadata = pd.read_parquet(expansion, columns=columns)
    candidates = [
        QueryIdentity(str(token), str(sequence), str(track), str(sequence))
        for token, sequence, track in metadata.itertuples(index=False, name=None)
    ]
    if len(candidates) != 27307 or {row.sequence for row in candidates} != allowed:
        raise ValueError("expansion inventory mismatch")
    ancestry = ack["producers"]["authoritative_historical_manifest"]
    historical = Path(ancestry["path"]).parent
    table_index = json.loads(
        (historical / "FROZEN_EXPERT_TABLE_INDEX.json").read_text(encoding="utf-8")
    )
    index_path = root / "artifacts/simplex_t/T1/query_context_index/INDEX_MANIFEST.json"
    dedup = json.loads(
        (root / "artifacts/simplex_t/T1/query_context_dedup/DEDUP_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    if sha256(index_path) != dedup["input_manifest_sha256"]:
        raise ValueError("existing D0 producer family manifest changed")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index["ancestry"] != ancestry:
        raise ValueError("producer ancestry reference mismatch")
    folds = {}
    for outer in range(3):
        ref = next(
            row for row in table_index if row["outer_fold"] == outer and row["role"] == "inner_oof"
        )
        table = historical / "tables" / f"outer{outer}_inner_oof.csv"
        if sha256(table) != ref["metadata_sha256"]:
            raise ValueError("OLD TRAIN identity changed")
        original_frame = pd.read_csv(table, usecols=columns, dtype={"track_id": str})
        original = tuple(
            QueryIdentity(str(token), str(sequence), str(track), str(sequence))
            for token, sequence, track in original_frame[columns].itertuples(index=False, name=None)
        )
        if not {row.sequence for row in original} <= set(roles["original"]):
            raise ValueError("OLD TRAIN includes closed role")
        pool = expanded_pool(
            original,
            candidates,
            original_train_groups={row.sequence for row in original},
            approved_expansion_groups=allowed,
        )
        families = tuple(
            next(
                family["family_sha256"]
                for family in index["families"]
                if family["outer_fold"] == outer and family["role"] == role
            )
            for role in ("inner0", "inner1", "inner2")
        )
        assignments = {
            sequence: expansion_producer(sequence, (families[0], families[1], families[2]))
            for sequence in sorted(allowed)
        }
        folds[str(outer)] = {
            "D0_train_metadata_sha256": ref["metadata_sha256"],
            "D0_train_queries": len(original),
            "D1_train_queries": len(pool.queries),
            "additional_groups": list(pool.additional_groups),
            "query_count_gate_passed": pool.available,
            "original_train_tokens": [row.token for row in original],
            "additional_train_tokens": [row.token for row in pool.queries[len(original) :]],
            "sequence_family_sha256": assignments,
            "inner_family_order": ["inner0", "inner1", "inner2"],
        }
    write_new_json(
        args.output,
        {
            "status": "INPUT_ONLY_D1_POOL_PLAN_NOT_EXTRACTION_OR_FIT_AUTHORIZATION",
            "folds": folds,
            "expansion_metadata_sha256": sha256(expansion),
            "D0_index_manifest_sha256": sha256(index_path),
            "ancestry": ancestry,
            "grouping": "sequence_id; independent acquisition identity unverified",
            "targets_read": False,
            "optimizer_updates": 0,
            "remaining": [
                "expansion time acknowledgment",
                "full transitive lineage checks",
                "additional history index and frozen expert cache",
                "production D1 loader",
            ],
        },
    )
    print(
        json.dumps(
            {
                outer: {key: row[key] for key in ("D0_train_queries", "D1_train_queries")}
                for outer, row in folds.items()
            }
        )
    )


if __name__ == "__main__":
    main()
