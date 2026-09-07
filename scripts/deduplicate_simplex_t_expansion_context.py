"""Deduplicate only fold-selected D1 observations, retaining inactive queries as masked."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import cast

import numpy as np

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.simplex_t.context_dedup import ContextSource, deduplicate_contexts
from e_jepa_ttc.simplex_t.lifecycle import admitted
from e_jepa_ttc.simplex_t.query_context import QueryContextInput


def main() -> None:
    """Prepare reusable indices only; do not grant an inference slot or fit authorization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--raw-manifest", required=True, type=Path)
    parser.add_argument("--preprocessing-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists() != args.resume:
        raise ValueError("new output or explicit resume required")
    if (args.output / "DEDUP_MANIFEST.json").exists():
        raise FileExistsError("completed deduplication must not be repeated")
    started = time.perf_counter()
    manifest_path = args.index / "INDEX_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    array_path = args.index / "query_context_index.npz"
    if (
        compute_file_hash(str(array_path)) != manifest["index_sha256"]
        or manifest["queries"] != 27307
    ):
        raise ValueError("D1 index mismatch")
    raw_hash = compute_file_hash(str(args.raw_manifest))
    if raw_hash != "4b7bfab2dc9ff31a6f3c936f487c1456bec1f8fd739d28165acb12e92a9047ba":
        raise ValueError("audited expansion source manifest changed")
    prep_hash = compute_file_hash(str(args.preprocessing_manifest))
    if prep_hash != "063980fdae5fda0b2836befc662fdd1cd5659bf06f10d9760dfc0d566fac8e39":
        raise ValueError("historical preprocessing changed")
    raw = json.loads(args.raw_manifest.read_text(encoding="utf-8"))["identity"]["sources"]
    identity = {
        "index_manifest_sha256": compute_file_hash(str(manifest_path)),
        "raw_manifest_sha256": raw_hash,
        "preprocessing_sha256": prep_hash,
        "builder_sha256": compute_file_hash(__file__),
        "deduplicator_sha256": compute_file_hash("src/e_jepa_ttc/simplex_t/context_dedup.py"),
    }
    if not admitted([args.output.parent])["has_headroom"]:
        raise RuntimeError("RESOURCE_PAUSE")
    if args.resume:
        if json.loads((args.output / "IDENTITY.json").read_text(encoding="utf-8")) != identity:
            raise ValueError("deduplication resume identity changed")
    else:
        args.output.mkdir()
        write_new_json(args.output / "IDENTITY.json", identity)
    outputs = []
    with np.load(array_path, allow_pickle=False) as archive:
        index = {name: archive[name] for name in archive.files}
    for outer in range(3):
        output = args.output / f"outer{outer}.npz"
        receipt = args.output / f"outer{outer}.json"
        if receipt.exists():
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            if compute_file_hash(str(output)) != saved["sha256"]:
                raise ValueError("completed deduplication bytes changed")
            outputs.append(saved)
            continue
        if output.exists():
            raise ValueError("unreceipted output retained; explicit recovery required")
        selected = np.flatnonzero(index["producer_family"][outer] >= 0)
        sources = []
        for i in selected:
            if i % 1024 == 0 and not admitted([args.output])["has_headroom"]:
                raise RuntimeError("RESOURCE_PAUSE before metadata construction")
            sequence = str(index["sequences"][i])
            family = manifest["families"][int(index["producer_family"][outer, i])]
            if family["outer_fold"] != outer or family["role"] == "outer_dev":
                raise ValueError("D1 training query assigned to wrong producer role")
            current = QueryContextInput(
                str(index["tokens"][i]),
                family["family_sha256"],
                int(index["anchor_us"][i]),
                int(index["roi_available_us"][i]),
                int(index["roi_available_us"][i]),
                manifest["stream_bounds_us"][sequence][0],
                cast(
                    tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
                    tuple(tuple(int(v) for v in window) for window in index["base_windows_us"][i]),
                ),
                cast(
                    tuple[float, float, float, float],
                    tuple(float(v) for v in index["square_xyxy"][i]),
                ),
            )
            sources.append(ContextSource(sequence, raw[sequence]["sha256"], prep_hash, current))
        keys, local_history = deduplicate_contexts(sources)
        history = np.full(index["valid"].shape, -1, dtype=np.int64)
        history[selected] = local_history
        expected_mask = index["valid"] & (index["producer_family"][outer, :, None] >= 0)
        if not np.array_equal(history >= 0, expected_mask):
            raise ValueError("deduplicated D1 coverage differs from selected source support")
        if not admitted([args.output])["has_headroom"]:
            raise RuntimeError("RESOURCE_PAUSE before deduplication write")
        np.savez_compressed(output, keys=np.asarray(keys), history=history)
        saved = {
            "path": output.name,
            "sha256": compute_file_hash(str(output)),
            "unique_observations": len(keys),
            "consumed_slots": int((history >= 0).sum()),
            "selected_queries": len(selected),
            "inactive_queries": len(history) - len(selected),
        }
        write_new_json(receipt, saved)
        outputs.append(saved)
        print(json.dumps(saved), flush=True)
        del keys, local_history, history, sources
        gc.collect()
    write_new_json(
        args.output / "DEDUP_MANIFEST.json",
        {
            "status": "D1_CONTENT_INDEX_READY_NOT_FEATURE_CACHE_OR_REPLAY_AUTHORIZATION",
            "identity": identity,
            "outputs": outputs,
            "optimizer_updates": 0,
            "targets_read": False,
            "raw_payload_read": False,
            "seconds_this_invocation": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
