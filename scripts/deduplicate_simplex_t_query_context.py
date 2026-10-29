"""Build producer-specific observation keys from the verified input-only index."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.simplex_t.context_dedup import ContextSource, deduplicate_contexts
from e_jepa_ttc.simplex_t.lifecycle import admitted
from e_jepa_ttc.simplex_t.query_context import QueryContextInput


def main() -> None:
    """Deduplicate without loading labels, sensor payloads or expert weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--preprocessing-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve existing deduplication evidence")
    started = time.perf_counter()
    manifest = json.loads((args.index / "INDEX_MANIFEST.json").read_text(encoding="utf-8"))
    array_path = args.index / "query_context_index.npz"
    if compute_file_hash(str(array_path)) != manifest["index_sha256"]:
        raise ValueError("index hash mismatch")
    binding_sha = compute_file_hash(str(args.binding))
    if binding_sha != "47b3ee83d61654cf716399a4f564043d2f2289b6785d8c26c456b941e25e70f1":
        raise ValueError("raw binding hash mismatch")
    preprocessing_sha = compute_file_hash(str(args.preprocessing_manifest))
    if preprocessing_sha != "063980fdae5fda0b2836befc662fdd1cd5659bf06f10d9760dfc0d566fac8e39":
        raise ValueError("frozen preprocessing manifest mismatch")
    raw = pd.read_csv(args.binding, usecols=pd.Index(["sequence_id", "h5_file_sha256"]))
    raw_hashes = {}
    for sequence, rows in raw.groupby("sequence_id"):
        hashes = rows.h5_file_sha256.unique()
        if len(hashes) != 1:
            raise ValueError("ambiguous raw content identity")
        raw_hashes[str(sequence)] = str(hashes[0])
    resources = admitted([args.output.parent])
    if not resources["has_headroom"]:
        raise RuntimeError("RESOURCE_PAUSE")
    args.output.mkdir()
    outputs = []
    with np.load(array_path, allow_pickle=False) as archive:
        index = {name: archive[name] for name in archive.files}
        for outer in range(3):
            sources = []
            for i, token in enumerate(index["tokens"]):
                sequence = str(index["sequences"][i])
                family = manifest["families"][int(index["producer_family"][outer, i])]
                current = QueryContextInput(
                    str(token),
                    family["family_sha256"],
                    int(index["anchor_us"][i]),
                    int(index["roi_available_us"][i]),
                    int(index["roi_available_us"][i]),
                    manifest["stream_bounds_us"][sequence][0],
                    cast(
                        tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
                        tuple(
                            tuple(int(v) for v in window) for window in index["base_windows_us"][i]
                        ),
                    ),
                    cast(
                        tuple[float, float, float, float],
                        tuple(float(v) for v in index["square_xyxy"][i]),
                    ),
                )
                sources.append(
                    ContextSource(sequence, raw_hashes[sequence], preprocessing_sha, current)
                )
            keys, history = deduplicate_contexts(sources)
            if not np.array_equal(history >= 0, index["valid"]):
                raise ValueError("source support masks differ from audited index")
            output = args.output / f"outer{outer}.npz"
            np.savez_compressed(output, keys=np.asarray(keys), history=history)
            outputs.append(
                {
                    "path": output.name,
                    "sha256": compute_file_hash(str(output)),
                    "unique_observations": len(keys),
                    "consumed_slots": int((history >= 0).sum()),
                }
            )
            print(json.dumps(outputs[-1]), flush=True)
    write_new_json(
        args.output / "DEDUP_MANIFEST.json",
        {
            "status": "CONTENT_INDEX_READY_NOT_FEATURE_CACHE",
            "namespace": "SIMPLEX_T_QUERY_CONTEXT_AMENDMENT",
            "input_manifest_sha256": compute_file_hash(str(args.index / "INDEX_MANIFEST.json")),
            "binding_sha256": binding_sha,
            "preprocessing_sha256": preprocessing_sha,
            "outputs": outputs,
            "optimizer_updates": 0,
            "target_fields_read": False,
            "raw_payload_rehashed": False,
            "raw_hash_authority": "verified historical binding",
            "seconds": time.perf_counter() - started,
            "resources": resources,
        },
    )


if __name__ == "__main__":
    main()
