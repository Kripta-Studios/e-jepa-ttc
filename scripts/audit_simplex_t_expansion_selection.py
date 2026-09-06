"""Verify real D1 pool/index/producer binding without reading targets or raw events."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.coordination import verified_ack
from e_jepa_ttc.simplex_t.expansion_selection import selected_expansion_rows


def main() -> None:
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
    relative_pins = {
        "T0/EXPANSION_POOL_PLAN.json": (
            "2c2a36f42c93f3d5304c524e04bcb84c31c5e8a756715d1ab955288abc2d1849"
        ),
        "T1/expansion_query_context_index/INDEX_MANIFEST.json": (
            "46a0749cb4a8c8394b14141e1a78e9181755a4718a44116fdd86b8aa82502cfe"
        ),
        "T1/expansion_query_context_dedup/DEDUP_MANIFEST.json": (
            "9ac8280bb89644f4331287eeb6f1ef4e818d83e7191522383fab949378b93590"
        ),
    }
    base = root / "artifacts/simplex_t"
    for relative, expected in relative_pins.items():
        if sha256(base / relative) != expected:
            raise ValueError("audited D1 metadata manifest changed")
    pool, manifest, dedup = [
        json.loads((base / p).read_text(encoding="utf-8")) for p in relative_pins
    ]
    index_path = base / "T1/expansion_query_context_index/query_context_index.npz"
    if sha256(index_path) != manifest["index_sha256"]:
        raise ValueError("D1 input index changed")
    with np.load(index_path, allow_pickle=False) as archive:
        index = {key: archive[key] for key in ("tokens", "sequences", "producer_family", "valid")}
    outputs = []
    for outer in range(3):
        record = dedup["outputs"][outer]
        path = base / "T1/expansion_query_context_dedup" / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError("deduplicated history changed")
        with np.load(path, allow_pickle=False) as archive:
            history = archive["history"]
        rows = selected_expansion_rows(
            index,
            history,
            outer=outer,
            pool=pool,
            families=manifest["families"],
            allowed_expansion_sequences=set(
                ack["interfaces"]["role_manifest"]["roles"]["expansion"]
            ),
        )
        outputs.append(
            {
                "outer": outer,
                "selected_queries": len(rows),
                "pool_order_rows_sha256": hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest(),
            }
        )
    result = {
        "status": "REAL_D1_INPUT_SELECTION_PASSED_NOT_CACHE_OR_FIT_AUTHORIZATION",
        "pins": relative_pins,
        "folds": outputs,
        "optimizer_updates": 0,
        "targets_read": False,
        "raw_events_read": False,
    }
    write_new_json(args.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
