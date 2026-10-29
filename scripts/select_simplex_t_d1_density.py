"""Apply the authorized D1 time-density amendment to existing TRAIN metadata only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.density_selection import select_time_density


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = Path(__file__).resolve().parent.parent
    manifest = work / "artifacts/simplex_t/T1/expansion_query_context_index/INDEX_MANIFEST.json"
    if sha256(manifest) != "46a0749cb4a8c8394b14141e1a78e9181755a4718a44116fdd86b8aa82502cfe":
        raise ValueError("authorized D1 parent manifest changed")
    parent = json.loads(manifest.read_text("utf-8"))
    index = manifest.parent / "query_context_index.npz"
    if sha256(index) != parent["index_sha256"]:
        raise ValueError("D1 input index changed")
    with np.load(index, allow_pickle=False) as data:
        sequences = data["sequences"]
        rows = select_time_density(sequences, data["anchor_us"], data["tokens"], per_sequence=512)
        assignments = data["producer_family"][:, rows]
        if len(np.unique(sequences[rows])) != 22 or len(rows) != 11264:
            raise ValueError("unexpected D1 diversity or count")
        result = {
            "schema": "simplex_t_d1_density_selection_v1",
            "amendment": "SIMPLEX_T_THROUGHPUT_2026-09-08",
            "status": "SELECTED_INPUT_ONLY_PENDING_REPLAY_AND_CONTROL_BINDING",
            "parent_manifest": {"path": str(manifest), "sha256": sha256(manifest)},
            "parent_index_sha256": parent["index_sha256"],
            "selector_sha256": sha256(work / "src/e_jepa_ttc/simplex_t/density_selection.py"),
            "per_sequence_cap": 512,
            "original_queries": len(sequences),
            "selected_queries": len(rows),
            "selected_original_rows": rows.tolist(),
            "per_sequence": {
                str(seq): int((sequences[rows] == seq).sum()) for seq in np.unique(sequences)
            },
            "active_blocks_by_outer": (assignments >= 0).sum(axis=1).tolist(),
            "active_blocks_total": int((assignments >= 0).sum()),
            "removed_active_blocks": int((data["producer_family"] >= 0).sum())
            - int((assignments >= 0).sum()),
            "old_evaluation_queries": 8192,
            "old_evaluation_unchanged": True,
            "labels_opened": False,
            "optimizer_updates": 0,
            "roles_reassigned": False,
            "teachers_changed": False,
        }
    write_new_json(args.output, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "selected_queries",
                    "active_blocks_total",
                    "removed_active_blocks",
                    "status",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
