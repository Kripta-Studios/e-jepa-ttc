"""Audit actual TRAIN target attachment after the frozen input-only selection proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.expansion_targets import load_expansion_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    base = Path(paths["worktree"]) / "artifacts/simplex_t/T0"
    proof = base / "EXPANSION_SELECTION_QA.json"
    if sha256(proof) != "7fb259f66e0cec361f852db2d83305f6584a49cb261d3f6ad65cee1630675569":
        raise ValueError("completed input-only selection proof changed")
    selection = json.loads(proof.read_text(encoding="utf-8"))
    pool_path = base / "EXPANSION_POOL_PLAN.json"
    if sha256(pool_path) != selection["pins"]["T0/EXPANSION_POOL_PLAN.json"]:
        raise ValueError("frozen input-only pool changed")
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    data = Path(paths["stage70_worktree_read_only"]) / (
        "artifacts/stage70_76_architecture/expansion_usability"
    )
    metadata_path = data / "USABLE_METADATA.parquet"
    if sha256(metadata_path) != pool["expansion_metadata_sha256"]:
        raise ValueError("audited label-reference metadata changed")
    metadata = pd.read_parquet(
        metadata_path, columns=["sample_token", "sequence_id", "timestamp_us"]
    )
    if not metadata.sample_token.is_unique:
        raise ValueError("duplicate input metadata token")
    metadata = metadata.set_index("sample_token")
    label_pin = "3ae7332991db4c393ff0005d1d6f5a23388b07b182c5ea8016ff0124db2ebe2b"
    rows = []
    for outer in range(3):
        tokens = np.asarray(pool["folds"][str(outer)]["additional_train_tokens"])
        frame = metadata.loc[tokens.tolist()]
        target = load_expansion_targets(
            data / "USABLE_TRAIN_LABELS.parquet",
            expected_sha256=label_pin,
            tokens=tokens,
            sequences=frame.sequence_id.to_numpy(),
            timestamps_us=frame.timestamp_us.to_numpy(),
        )
        rows.append(
            {
                "outer": outer,
                "queries": len(tokens),
                "supervision_identity_sha256": target.identity_sha256,
            }
        )
    result = {
        "status": "REAL_TRAIN_TARGET_ALIGNMENT_PASSED_NOT_SCIENTIFIC_FREEZE",
        "input_selection_sha256": sha256(proof),
        "labels_sha256": label_pin,
        "metadata_sha256": sha256(metadata_path),
        "folds": rows,
        "targets_read": True,
        "target_values_exported": False,
        "queries_dropped": 0,
        "optimizer_updates": 0,
        "stage70_scores_read": False,
    }
    write_new_json(args.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
