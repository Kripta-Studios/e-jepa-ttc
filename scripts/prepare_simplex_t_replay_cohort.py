"""Prepare, but do not claim execution of, the hash-bound TRAIN replay cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.simplex_t.coordination import verified_ack
from e_jepa_ttc.simplex_t.current_inputs import load_current_inputs
from e_jepa_ttc.simplex_t.replay_selection import select_replay_rows


def main() -> None:
    """Bind selection to acknowledged roles and historical table ancestry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    config = json.loads(Path("configs/experiment/simplex_t_coordination.json").read_text())
    ack = verified_ack(
        Path(paths["shared_coordination"]) / config["ack_filename"], config["ack_sha256"]
    )
    ancestry = ack["producers"]["authoritative_historical_manifest"]
    root = Path(ancestry["path"]).parent
    allowed = set(ack["interfaces"]["role_manifest"]["roles"]["original"])
    populations = {}
    references = []
    for outer in range(3):
        for role in ("inner_oof", "outer_dev"):
            table = load_current_inputs(
                root, outer, role, ancestry_sha256=ancestry["sha256"], allowed_sequences=allowed
            )
            references.append(table["reference"])
            metadata = table["metadata"]
            if role == "outer_dev":
                populations[f"outer{outer}/outer_dev"] = metadata.sample_token.tolist()
            else:
                for inner in range(3):
                    populations[f"outer{outer}/inner{inner}"] = metadata.loc[
                        metadata.inner_fold == inner, "sample_token"
                    ].tolist()
    selected = select_replay_rows(populations)
    write_new_json(
        args.output,
        {
            "status": "COHORT_PREPARED_REPLAY_NOT_EXECUTED",
            "selection": selected,
            "unique_rows": 64,
            "expert_evaluations_required": 192,
            "table_references": references,
            "ancestry": ancestry,
            "selection_uses_targets_or_scores": False,
            "optimizer_updates": 0,
            "scientific_freeze": False,
        },
    )
    print(json.dumps({"families": len(selected), "rows": 64, "replay_executed": False}))


if __name__ == "__main__":
    main()
