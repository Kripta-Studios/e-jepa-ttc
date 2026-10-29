"""Resolve every matched-control query against existing pinned input indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.control_references import SOURCE_NAMES, control_references


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = args.worktree / "artifacts/simplex_t"
    plan_path = base / "T0/MATCHED_CONTROL_POOL_PLAN.json"
    plan_hash = "b0685050b799058e6090d6e3b7c47b653f2939ca7db2d75a93526ab236ec9e3c"
    if sha256(plan_path) != plan_hash:
        raise ValueError("registered matched control plan changed")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    records = {
        "D0": ("query_context_index", plan["D0_index_manifest_sha256"]),
        "D1": (
            "expansion_query_context_index",
            "46a0749cb4a8c8394b14141e1a78e9181755a4718a44116fdd86b8aa82502cfe",
        ),
        "DENSE": (
            "dense_query_context_index",
            "cb9e51715a71e25128923ccb20c5ec7abf6efc7095de34b001c649ebc12a0850",
        ),
    }
    indices, manifests = {}, {}
    for name, (folder, expected) in records.items():
        root = base / "T1" / folder
        if sha256(root / "INDEX_MANIFEST.json") != expected:
            raise ValueError("audited control input manifest changed")
        manifest = json.loads((root / "INDEX_MANIFEST.json").read_text(encoding="utf-8"))
        if sha256(root / "query_context_index.npz") != manifest["index_sha256"]:
            raise ValueError("control input arrays changed")
        with np.load(root / "query_context_index.npz", allow_pickle=False) as archive:
            indices[name] = {key: archive[key] for key in archive.files}
        manifests[name] = manifest
    families = manifests["D0"]["families"]
    if any(manifest["families"] != families for manifest in manifests.values()):
        raise ValueError("input indices use different historical producer families")
    resolved = []
    for fold in plan["folds"]:
        for pool, selection in fold["pools"].items():
            arrays = control_references(
                pool=pool,
                selection=selection,
                outer=fold["outer"],
                indices=indices,
                families=families,
            )
            if len(arrays["tokens"]) != fold["nominal_common_count"]:
                raise ValueError("matched count changed")
            resolved.append((fold["outer"], pool, arrays))
    args.output.mkdir(parents=True)
    outputs = []
    for outer, pool, arrays in resolved:
        path = args.output / f"outer{outer}_{pool}.npz"
        np.savez_compressed(
            path,
            tokens=arrays["tokens"],
            source=arrays["source"],
            query_row=arrays["query_row"],
            producer_family=arrays["producer_family"],
        )
        outputs.append(
            {
                "outer": outer,
                "pool": pool,
                "queries": len(arrays["tokens"]),
                "path": path.name,
                "sha256": sha256(path),
                "input_sources": {
                    name: int((arrays["source"] == i).sum()) for i, name in enumerate(SOURCE_NAMES)
                },
            }
        )
    result = {
        "status": "MATCHED_CONTROL_INPUT_REFERENCES_VERIFIED_NOT_FEATURE_COMPLETENESS",
        "pool_manifest_sha256": plan_hash,
        "index_manifest_pins": records,
        "source_code_order": SOURCE_NAMES,
        "outputs": outputs,
        "targets_read": False,
        "optimizer_updates": 0,
    }
    write_new_json(args.output / "REFERENCES.json", result)
    print(json.dumps(outputs))


if __name__ == "__main__":
    main()
