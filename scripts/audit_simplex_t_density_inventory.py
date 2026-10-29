"""Bound matched-density availability using only pinned, permitted query identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.coordination import verified_ack
from e_jepa_ttc.simplex_t.pools import density_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    root = Path(paths["worktree"])
    configuration = json.loads(
        (root / "configs/experiment/simplex_t_coordination.json").read_text(encoding="utf-8")
    )
    ack = verified_ack(
        Path(paths["shared_coordination"]) / configuration["ack_filename"],
        configuration["ack_sha256"],
    )
    old_groups = ack["interfaces"]["role_manifest"]["roles"]["original"]
    metadata_hash = "03dd3022db4b5f43bb10244fc8778476d74351e764f73a90c8566af949c17fd6"
    if sha256(args.train_metadata) != metadata_hash:
        raise ValueError("audited local TRAIN metadata changed")
    pool_path = root / "artifacts/simplex_t/T0/EXPANSION_POOL_PLAN.json"
    pool_hash = "2c2a36f42c93f3d5304c524e04bcb84c31c5e8a756715d1ab955288abc2d1849"
    if sha256(pool_path) != pool_hash:
        raise ValueError("registered D0/D1 pools changed")
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    rows = pd.read_parquet(
        args.train_metadata,
        columns=["sample_token", "sequence_id", "track_id"],
        filters=[("sequence_id", "in", old_groups)],
    )
    if rows.sample_token.duplicated().any() or set(rows.sequence_id) != set(old_groups):
        raise ValueError("original query identity inventory mismatch")
    table = rows.set_index("sample_token")
    result = []
    for outer in range(3):
        fold = pool["folds"][str(outer)]
        original = table.loc[fold["original_train_tokens"]]
        sequences = sorted(set(original.sequence_id))
        if len(sequences) != 6 or len(original) != fold["D0_train_queries"]:
            raise ValueError("expected six original TRAIN groups per fold")
        dense_count = int(rows.sequence_id.isin(sequences).sum())
        common = density_size(
            d0_count=len(original),
            dense_old_available=dense_count,
            diverse_available=fold["D1_train_queries"],
        )
        result.append(
            {
                "outer": outer,
                "train_sequences": sequences,
                "D0_queries": len(original),
                "DENSE_OLD_unique_query_upper_bound": dense_count,
                "minimum_required": 2 * len(original),
                "prospective_common_count": common,
                "status": "UNAVAILABLE_IN_PINNED_LOCAL_QUERY_RELEASE"
                if common is None
                else "COUNT_PASSES_REQUIRES_TIME_ROI_PRODUCER_AND_CACHE_AUDIT",
            }
        )
    output = {
        "status": "INPUT_ONLY_MATCHED_DENSITY_INVENTORY",
        "train_metadata_sha256": metadata_hash,
        "pool_sha256": pool_hash,
        "folds": result,
        "targets_read": False,
        "raw_events_read": False,
        "optimizer_updates": 0,
        "scores_read": False,
        "scope": (
            "Upper bound from all unique original TRAIN pair identities in pinned local release; "
            "not unfiltered object history"
        ),
    }
    write_new_json(args.output, output)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
