"""Prepare nominal matched-count control identities; no timing or fit authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.coordination import verified_ack
from e_jepa_ttc.simplex_t.density_selection import bound_selection_rows
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
    parser.add_argument("--d1-selection", type=Path)
    parser.add_argument("--d1-selection-sha256")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    root = Path(paths["worktree"])
    if (args.d1_selection is None) != (args.d1_selection_sha256 is None):
        raise ValueError("density amendment requires selection path and hash")
    selection_binding = None
    selected_by_outer = None
    if args.d1_selection is not None:
        selection_binding = {
            "path": str(args.d1_selection.resolve()),
            "sha256": args.d1_selection_sha256,
        }
        index_root = root / "artifacts/simplex_t/T1/expansion_query_context_index"
        manifest_path = index_root / "INDEX_MANIFEST.json"
        if (
            sha256(manifest_path)
            != "46a0749cb4a8c8394b14141e1a78e9181755a4718a44116fdd86b8aa82502cfe"
        ):
            raise ValueError("D1 index manifest changed")
        manifest = json.loads(manifest_path.read_text("utf-8"))
        array_path = index_root / "query_context_index.npz"
        if sha256(array_path) != manifest["index_sha256"]:
            raise ValueError("D1 index bytes changed")
        with np.load(array_path, allow_pickle=False) as archive:
            index = {
                name: archive[name]
                for name in ("tokens", "sequences", "anchor_us", "producer_family")
            }
        selected = bound_selection_rows(selection_binding, index, manifest["index_sha256"])
        selected_by_outer = [
            set(index["tokens"][selected[index["producer_family"][outer, selected] >= 0]])
            for outer in range(3)
        ]
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
        additions = fold["additional_train_tokens"]
        if selected_by_outer is not None:
            additions = [token for token in additions if token in selected_by_outer[outer]]
            if set(additions) != selected_by_outer[outer]:
                raise ValueError("selected D1 queries absent from authorized fold")
        diverse_frame = lookup.loc[fold["original_train_tokens"] + additions]
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
    if selection_binding is not None:
        parent_path = root / "artifacts/simplex_t/T0/MATCHED_CONTROL_POOL_PLAN.json"
        parent_hash = "b0685050b799058e6090d6e3b7c47b653f2939ca7db2d75a93526ab236ec9e3c"
        if sha256(parent_path) != parent_hash:
            raise ValueError("parent matched pool changed")
        parent = json.loads(parent_path.read_text("utf-8"))
        if any(
            current["pools"]["DENSE_OLD"] != previous["pools"]["DENSE_OLD"]
            or current["nominal_common_count"] != previous["nominal_common_count"]
            for current, previous in zip(outputs, parent["folds"], strict=True)
        ):
            raise ValueError("DENSE changed: cannot reuse its acknowledged index")
        result["query_selection"] = selection_binding
        result["parent_matched_pool"] = {"path": str(parent_path), "sha256": parent_hash}
        result["dense_membership_unchanged"] = True
    write_new_json(args.output, result)
    print(
        json.dumps(
            {"status": result["status"], "counts": [r["nominal_common_count"] for r in outputs]}
        )
    )


if __name__ == "__main__":
    main()
