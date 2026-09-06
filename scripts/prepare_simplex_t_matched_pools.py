"""Prepare nominal matched-count control identities; no timing or fit authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.coordination import verified_ack
from e_jepa_ttc.simplex_t.pools import QueryIdentity, matched_density_pools


def identities(frame: pd.DataFrame) -> list[QueryIdentity]:
    return [
        QueryIdentity(str(t), str(s), str(k), str(s))
        for t, s, k in frame[["sample_token", "sequence_id", "track_id"]].itertuples(
            index=False, name=None
        )
    ]


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
    config = json.loads(
        (root / "configs/experiment/simplex_t_coordination.json").read_text(encoding="utf-8")
    )
    ack = verified_ack(
        Path(paths["shared_coordination"]) / config["ack_filename"], config["ack_sha256"]
    )
    roles = ack["interfaces"]["role_manifest"]["roles"]
    metadata_hash = "03dd3022db4b5f43bb10244fc8778476d74351e764f73a90c8566af949c17fd6"
    if sha256(args.train_metadata) != metadata_hash:
        raise ValueError("audited metadata changed")
    pool_path = root / "artifacts/simplex_t/T0/EXPANSION_POOL_PLAN.json"
    pool_hash = "2c2a36f42c93f3d5304c524e04bcb84c31c5e8a756715d1ab955288abc2d1849"
    if sha256(pool_path) != pool_hash:
        raise ValueError("D1 pool changed")
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    old_index_path = root / "artifacts/simplex_t/T1/query_context_index/INDEX_MANIFEST.json"
    if sha256(old_index_path) != pool["D0_index_manifest_sha256"]:
        raise ValueError("historical producer family manifest changed")
    old_index = json.loads(old_index_path.read_text(encoding="utf-8"))
    historical = Path(ack["producers"]["authoritative_historical_manifest"]["path"]).parent
    metadata = pd.read_parquet(
        args.train_metadata,
        columns=["sample_token", "sequence_id", "track_id"],
        filters=[("sequence_id", "in", roles["original"] + roles["expansion"])],
    )
    if not metadata.sample_token.is_unique:
        raise ValueError("duplicate query token")
    lookup = metadata.set_index("sample_token", drop=False)
    outputs = []
    for outer in range(3):
        fold = pool["folds"][str(outer)]
        old_table_path = historical / "tables" / f"outer{outer}_inner_oof.csv"
        if sha256(old_table_path) != fold["D0_train_metadata_sha256"]:
            raise ValueError("historical TRAIN identities changed")
        old_table = pd.read_csv(
            old_table_path,
            usecols=lambda name: name in {"sample_token", "sequence_id", "inner_fold"},
        )
        original_frame = lookup.loc[fold["original_train_tokens"]]
        if not isinstance(original_frame, pd.DataFrame):
            raise ValueError("original pool lookup must retain its query axis")
        groups = set(original_frame.sequence_id)
        if len(groups) != 6 or not groups <= set(roles["original"]):
            raise ValueError("invalid original TRAIN group set")
        dense_frame = metadata.loc[metadata.sequence_id.isin(groups)]
        diverse_frame = lookup.loc[fold["original_train_tokens"] + fold["additional_train_tokens"]]
        if not isinstance(dense_frame, pd.DataFrame) or not isinstance(diverse_frame, pd.DataFrame):
            raise ValueError("matched pool lookup must retain its query axis")
        matched = matched_density_pools(
            tuple(identities(original_frame)),
            identities(dense_frame),
            identities(diverse_frame),
            original_groups=groups,
            expansion_groups=set(roles["expansion"]),
        )
        if matched is None:
            raise ValueError("previous count gate no longer passes")
        assignments = dict(fold["sequence_family_sha256"])
        for sequence in groups:
            inner = old_table.loc[old_table.sequence_id == sequence, "inner_fold"].unique()
            if len(inner) != 1:
                raise ValueError("original sequence lacks one correct inner producer")
            family = next(
                f
                for f in old_index["families"]
                if f["outer_fold"] == outer and f["role"] == f"inner{int(inner[0])}"
            )
            assignments[sequence] = family["family_sha256"]
        outputs.append(
            {
                "outer": outer,
                "nominal_common_count": len(matched[0]),
                "pools": {
                    name: {
                        "tokens": [q.token for q in queries],
                        "sequences": sorted({q.sequence for q in queries}),
                        "sequence_family_sha256": {
                            s: assignments[s] for s in sorted({q.sequence for q in queries})
                        },
                    }
                    for name, queries in zip(("DENSE_OLD", "DIVERSE_MATCHED"), matched, strict=True)
                },
            }
        )
    result = {
        "status": "NOMINAL_MATCHED_POOLS_PENDING_INPUT_USABILITY_AND_REPLAY",
        "metadata_sha256": metadata_hash,
        "D1_pool_sha256": pool_hash,
        "D0_index_manifest_sha256": sha256(old_index_path),
        "folds": outputs,
        "targets_read": False,
        "optimizer_updates": 0,
        "availability_resolved": False,
        "remaining": (
            "Validate timing/ROI/ancestry/cache; if usable count changes, "
            "recompute both pools before scores"
        ),
    }
    write_new_json(args.output, result)
    print(
        json.dumps(
            {"status": result["status"], "counts": [r["nominal_common_count"] for r in outputs]}
        )
    )


if __name__ == "__main__":
    main()
