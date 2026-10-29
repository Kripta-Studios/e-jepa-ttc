"""Bind registered dense TRAIN labels after immutable input-only pool selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.coordination import verified_ack
from e_jepa_ttc.simplex_t.selected_targets import load_selected_train_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous supervision evidence")
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    base = Path(paths["worktree"]) / "artifacts/simplex_t"
    ack = verified_ack(
        Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json",
        "3e55ab3c6e9a57eecd862ad05e999627ea90957e58e329b2eb3652120e953318",
    )
    allowed = set(ack["interfaces"]["role_manifest"]["roles"]["original"])
    pool_path = base / "T0/MATCHED_CONTROL_POOL_PLAN.json"
    index_root = base / "T1/dense_query_context_index"
    pins = {
        pool_path: "b0685050b799058e6090d6e3b7c47b653f2939ca7db2d75a93526ab236ec9e3c",
        index_root
        / "INDEX_MANIFEST.json": "cb9e51715a71e25128923ccb20c5ec7abf6efc7095de34b001c649ebc12a0850",
        args.metadata: "03dd3022db4b5f43bb10244fc8778476d74351e764f73a90c8566af949c17fd6",
    }
    for path, digest in pins.items():
        if sha256(path) != digest:
            raise ValueError("dense input-only selection pin changed")
    plan = json.loads(pool_path.read_text(encoding="utf-8"))
    manifest = json.loads((index_root / "INDEX_MANIFEST.json").read_text(encoding="utf-8"))
    if sha256(index_root / "query_context_index.npz") != manifest["index_sha256"]:
        raise ValueError("dense query arrays changed")
    with np.load(index_root / "query_context_index.npz", allow_pickle=False) as archive:
        index = {name: archive[name] for name in ("tokens", "sequences", "producer_family")}
    positions = {str(token): row for row, token in enumerate(index["tokens"])}
    results = []
    label_sha = "64e9a81232643ecce1d2f66e906c7cff6ea1e9e6958f81588b95e112693b11ad"
    for fold in plan["folds"]:
        outer, selection = fold["outer"], fold["pools"]["DENSE_OLD"]
        groups = set(selection["sequences"])
        if not groups <= allowed or len(groups) != 6:
            raise ValueError("dense pool changes original TRAIN groups")
        tokens = np.asarray(selection["tokens"])
        rows = np.asarray([positions[str(token)] for token in tokens], np.int64)
        sequences = index["sequences"][rows]
        for row, sequence in zip(rows, sequences, strict=True):
            family_id = int(index["producer_family"][outer, row])
            if family_id < 0 or family_id // 4 != outer or family_id % 4 == 3:
                raise ValueError("dense query not assigned an inner TRAIN producer")
            family = manifest["families"][family_id]
            if family["family_sha256"] != selection["sequence_family_sha256"][str(sequence)]:
                raise ValueError("dense producer differs from registered pool")
        metadata = pd.read_parquet(
            args.metadata,
            columns=["sample_token", "sequence_id", "timestamp_us"],
            filters=[
                ("sample_token", "in", tokens.tolist()),
                ("sequence_id", "in", sorted(groups)),
            ],
        )
        if len(metadata) != len(tokens) or not metadata.sample_token.is_unique:
            raise ValueError("selected metadata missing or duplicated")
        metadata = metadata.set_index("sample_token").loc[tokens.tolist()]
        if not np.array_equal(metadata.sequence_id.to_numpy(), sequences):
            raise ValueError("selected sequence metadata mismatch")
        targets = load_selected_train_targets(
            args.labels,
            expected_sha256=label_sha,
            tokens=tokens,
            sequences=sequences,
            timestamps_us=metadata.timestamp_us.to_numpy(),
            allowed_sequences=groups,
            pool="DENSE_OLD",
        )
        results.append(
            {
                "outer": outer,
                "queries": len(tokens),
                "groups": sorted(groups),
                "supervision_sha256": targets.identity_sha256,
                "queries_dropped": 0,
            }
        )
    write_new_json(
        args.output,
        {
            "status": "DENSE_SELECTED_TRAIN_TARGET_ALIGNMENT_PASSED_NOT_FIT_AUTHORIZATION",
            "label_file_sha256": label_sha,
            "pool_sha256": pins[pool_path],
            "folds": results,
            "TRAIN_ttc_values_read": True,
            "columns": ["sample_token", "sequence_id", "timestamp_us", "ttc"],
            "selection": (
                "query AND sequence projection; internal Parquet pages may contain other rows"
            ),
            "optimizer_updates": 0,
            "scores_read": False,
            "time_ack": (
                "Original ACK does not establish new-query time/ROI authority; "
                "supplementary acknowledgement remains pending"
            ),
            "auditor_sha256": sha256(Path(__file__)),
        },
    )
    print(json.dumps({"folds": results, "output_sha256": sha256(args.output)}))


if __name__ == "__main__":
    main()
