"""Audit input-selected D0 TRAIN blocks for exact reuse in dense control indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.cache_reuse import load_reused_block


def main() -> None:
    """Verify first/last eight queries per inner family, without target or score selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve reuse QA evidence")
    base = args.worktree / "artifacts/simplex_t/T1"
    cache = base / "context_features_fp32"
    if sha256(cache / "IDENTITY.json") != (
        "00097c6e78bff173f9fe1497aae49da93f6ad7e86e678fe714da4269219955b1"
    ):
        raise ValueError("frozen D0 cache identity changed")
    roots = (base / "query_context_index", base / "dense_query_context_index")
    pins = (
        "93a4f62e5025c5046fc82fcb1428a428f8a8df869b34486b92d5c753b0f68a3b",
        "cb9e51715a71e25128923ccb20c5ec7abf6efc7095de34b001c649ebc12a0850",
    )
    indices = []
    for root, digest in zip(roots, pins, strict=True):
        if sha256(root / "INDEX_MANIFEST.json") != digest:
            raise ValueError("input manifest changed")
        manifest = json.loads((root / "INDEX_MANIFEST.json").read_text(encoding="utf-8"))
        if sha256(root / "query_context_index.npz") != manifest["index_sha256"]:
            raise ValueError("query arrays changed")
        with np.load(root / "query_context_index.npz", allow_pickle=False) as archive:
            indices.append({name: archive[name] for name in archive.files})
    old, dense = indices
    key_sets, histories = [], []
    for root, pin in (
        (base / "query_context_dedup", None),
        (
            base / "dense_query_context_dedup",
            "3a715f1d1f933f0e7d31a5078a919206c81dd6d776b667bf432db6229e3ef615",
        ),
    ):
        manifest_path = root / "DEDUP_MANIFEST.json"
        if pin is not None and sha256(manifest_path) != pin:
            raise ValueError("dense deduplication seal changed")
        record = json.loads(manifest_path.read_text(encoding="utf-8"))["outputs"][0]
        path = root / "outer0.npz"
        expected = record["sha256"]
        if root.name == "query_context_dedup":
            # Compiled D0 manifest pins the actual dedup payload used by heads.
            compiled = base / "compiled_context/outer0/COMPILED.json"
            if sha256(compiled) != (
                "498e5bb23d93a23f7513c6f268620138da084ed39e73c9164e958554d8b13cb4"
            ):
                raise ValueError("original compiled D0 manifest changed")
            expected = json.loads(compiled.read_text(encoding="utf-8"))["dedup_sha256"]
        if sha256(path) != expected:
            raise ValueError("content keys changed")
        with np.load(path, allow_pickle=False) as archive:
            key_sets.append(archive["keys"])
            histories.append(archive["history"])
    positions = {str(token): i for i, token in enumerate(dense["tokens"])}
    records = []
    for family in range(3):
        candidates = np.flatnonzero(old["producer_family"][0] == family)
        selected = np.unique(np.concatenate((candidates[:8], candidates[-8:])))
        for row in selected:
            target = positions[str(old["tokens"][row])]
            if dense["producer_family"][0, target] != family:
                raise ValueError("reuse changes producer family")
            path = cache / f"family{family:02d}_query{row:05d}.npz"
            receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if receipt["query"] != row or receipt["family"] != family:
                raise ValueError("block identity changed")
            if sha256(path) != receipt["sha256"]:
                raise ValueError("cached block bytes changed")
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            mask = dense["valid"][target]
            if not np.array_equal(arrays["observation_ids"], histories[0][row, mask]):
                raise ValueError("cached source observation IDs changed")
            result = load_reused_block(
                cache,
                identity_sha256=sha256(cache / "IDENTITY.json"),
                receipt_sha256=sha256(path.with_suffix(".json")),
                query=int(row),
                family=family,
                source_ids=histories[0][row, mask],
                source_keys=key_sets[0],
                destination_keys=key_sets[1],
                destination_ids=histories[1][target, mask],
                anchors_us=dense["anchor_us"][target] - dense["lag_us"][mask],
                available_us=int(dense["roi_available_us"][target]),
            )
            for name in arrays:
                if name != "observation_ids" and result[name].tobytes() != arrays[name].tobytes():
                    raise ValueError("reuse changed frozen output bytes")
            records.append(
                {
                    "family": family,
                    "D0_query": int(row),
                    "dense_query": target,
                    "block_sha256": receipt["sha256"],
                    "observations": int(mask.sum()),
                }
            )
    write_new_json(
        args.output,
        {
            "status": "REAL_D0_DENSE_BLOCK_REUSE_EXACT",
            "outer": 0,
            "queries": len(records),
            "records": records,
            "scope": "first/last eight input-index queries per inner family; not all cached blocks",
            "cache_blocks_written": 0,
            "targets_read": False,
            "scores_read": False,
            "expert_inference_runs": 0,
            "optimizer_updates": 0,
            "auditor_sha256": sha256(Path(__file__)),
            "rebind_sha256": sha256(args.worktree / "src/e_jepa_ttc/simplex_t/cache_reuse.py"),
        },
    )
    print(json.dumps({"queries": len(records), "output_sha256": sha256(args.output)}))


if __name__ == "__main__":
    main()
