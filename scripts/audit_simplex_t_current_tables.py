"""Verify compact historical H1 inputs without fitting or reading architecture scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.coordination import verified_ack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    config = json.loads(Path("configs/experiment/simplex_t_coordination.json").read_text())
    ack = verified_ack(
        Path(paths["shared_coordination"]) / config["ack_filename"], config["ack_sha256"]
    )
    ancestry_record = ack["producers"]["authoritative_historical_manifest"]
    root = Path(ancestry_record["path"]).parent
    index_path = root / "FROZEN_EXPERT_TABLE_INDEX.json"
    index = json.loads(index_path.read_text())
    roles = ack["interfaces"]["role_manifest"]["roles"]
    allowed = set(roles["original"])
    results = []
    identities = {}
    for fold in range(3):
        for role in ("inner_oof", "outer_dev"):
            matches = [r for r in index if r["outer_fold"] == fold and r["role"] == role]
            if len(matches) != 1:
                raise ValueError("missing or duplicated historical table")
            expected = matches[0]
            stem = root / "tables" / f"outer{fold}_{role}"
            for suffix, key in ((".csv", "metadata_sha256"), (".npz", "arrays_sha256")):
                if sha256(stem.with_suffix(suffix)) != expected[key]:
                    raise ValueError("historical input byte mismatch")
            if expected["ancestry_sha256"] != ancestry_record["sha256"]:
                raise ValueError("table references another ancestry")
            metadata = pd.read_csv(
                stem.with_suffix(".csv"),
                usecols=lambda column, table_role=role: (
                    column
                    in [
                        "sample_token",
                        "sequence_id",
                        "track_id",
                        "producer_outer_fold",
                    ]
                    + (["inner_fold"] if table_role == "inner_oof" else [])
                ),
            )
            if not set(metadata.sequence_id) <= allowed or not metadata.sample_token.is_unique:
                raise ValueError("role violation or duplicate current query")
            if not (metadata.producer_outer_fold == fold).all():
                raise ValueError("mixed outer producer families")
            with np.load(stem.with_suffix(".npz"), allow_pickle=False) as arrays:
                checked = {}
                for name in ("features17", "expert_ttc", "expert_phase", "target_phase"):
                    value = arrays[name]
                    schema = expected["arrays"][name]
                    if list(value.shape) != schema["shape"] or str(value.dtype) != schema["dtype"]:
                        raise ValueError("array schema mismatch")
                    if len(value) != len(metadata) or not np.isfinite(value).all():
                        raise ValueError("invalid current feature/target array")
                    checked[name] = {"shape": list(value.shape), "finite": True}
            identities[(fold, role)] = set(metadata.sample_token)
            results.append(
                {
                    "outer_fold": fold,
                    "role": role,
                    "rows": len(metadata),
                    "metadata_sha256": expected["metadata_sha256"],
                    "arrays_sha256": expected["arrays_sha256"],
                    "checked": checked,
                    "sequence_ids": sorted(set(metadata.sequence_id)),
                    "inner_fold_ids": sorted(metadata.inner_fold.dropna().unique().tolist())
                    if role == "inner_oof"
                    else [],
                }
            )
    for fold in range(3):
        train, dev = identities[(fold, "inner_oof")], identities[(fold, "outer_dev")]
        if train & dev or len(train | dev) != 8192:
            raise ValueError("D0 cohort overlap or incompleteness")
    if len(set.union(*(identities[(fold, "outer_dev")] for fold in range(3)))) != 8192:
        raise ValueError("outer development cohort incomplete")
    write_new_json(
        args.output,
        {
            "status": "CURRENT_TABLE_BYTES_AND_COHORT_VERIFIED_NOT_REPLAY_PARITY",
            "tables": results,
            "table_index_sha256": sha256(index_path),
            "ancestry": ancestry_record,
            "scientific_updates": 0,
            "prediction_scores_computed": False,
            "new_architecture_scores_read": False,
            "current_arrays_inspected": True,
            "history_created_from_tables": False,
            "remaining": [
                "row-level full producer/normalizer lineage integration",
                "real replay parity",
                "production loader integration and freeze before fit",
            ],
        },
    )
    print(json.dumps({"tables_verified": len(results), "original_cohort": 8192}))


if __name__ == "__main__":
    main()
