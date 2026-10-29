"""Deduplicate one DENSE_OLD fold and prove reusable D0 content identities."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.context_dedup import ContextSource, deduplicate_contexts
from e_jepa_ttc.simplex_t.coordination import shared_write_admission
from e_jepa_ttc.simplex_t.lifecycle import ExclusiveLease, admitted
from e_jepa_ttc.simplex_t.query_context import QueryContextInput


def main() -> None:
    """Read input metadata only; retain a complete fold receipt before the next fold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--outer", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    args = parser.parse_args()
    if args.other_reserved_bytes < 0:
        raise ValueError("negative output reservation")
    base = args.worktree / "artifacts/simplex_t/T1"
    dense_root, old_root = base / "dense_query_context_index", base / "query_context_index"
    output = base / "dense_query_context_dedup"
    pins = {
        dense_root
        / "INDEX_MANIFEST.json": "cb9e51715a71e25128923ccb20c5ec7abf6efc7095de34b001c649ebc12a0850",
        old_root
        / "INDEX_MANIFEST.json": "93a4f62e5025c5046fc82fcb1428a428f8a8df869b34486b92d5c753b0f68a3b",
        args.binding: "47b3ee83d61654cf716399a4f564043d2f2289b6785d8c26c456b941e25e70f1",
    }
    for path, expected in pins.items():
        if sha256(path) != expected:
            raise ValueError("audited input pin changed")
    indices, manifests = [], []
    for root in (dense_root, old_root):
        manifest = json.loads((root / "INDEX_MANIFEST.json").read_text(encoding="utf-8"))
        if sha256(root / "query_context_index.npz") != manifest["index_sha256"]:
            raise ValueError("input arrays changed")
        with np.load(root / "query_context_index.npz", allow_pickle=False) as archive:
            indices.append({key: archive[key] for key in archive.files})
        manifests.append(manifest)
    dense, old = indices
    manifest, old_manifest = manifests
    if manifest["families"] != old_manifest["families"]:
        raise ValueError("historical producer families changed")
    old_dedup = base / "query_context_dedup"
    reference = json.loads((old_dedup / "DEDUP_MANIFEST.json").read_text(encoding="utf-8"))
    if reference["input_manifest_sha256"] != pins[old_root / "INDEX_MANIFEST.json"]:
        raise ValueError("D0 deduplication refers to another index")
    prep = reference["preprocessing_sha256"]
    if prep != "063980fdae5fda0b2836befc662fdd1cd5659bf06f10d9760dfc0d566fac8e39":
        raise ValueError("historical preprocessing changed")
    record = reference["outputs"][args.outer]
    old_path = old_dedup / record["path"]
    if sha256(old_path) != record["sha256"]:
        raise ValueError("D0 deduplicated keys changed")
    raw = pd.read_csv(args.binding, usecols=pd.Index(["sequence_id", "h5_file_sha256"]))
    raw_hashes = {}
    for sequence, group in raw.groupby("sequence_id"):
        values = group.h5_file_sha256.unique()
        if len(values) != 1:
            raise ValueError("ambiguous raw identity")
        raw_hashes[str(sequence)] = str(values[0])

    def resource_check() -> None:
        status = admitted([base])
        if not status["has_headroom"] or not shared_write_admission(
            status["written_volume_free_bytes"][0], args.other_reserved_bytes + 500_000_000
        ):
            raise RuntimeError("RESOURCE_PAUSE; completed fold receipts remain reusable")

    resource_check()
    identity = {
        "dense_index_sha256": pins[dense_root / "INDEX_MANIFEST.json"],
        "D0_index_sha256": pins[old_root / "INDEX_MANIFEST.json"],
        "raw_binding_sha256": pins[args.binding],
        "preprocessing_sha256": prep,
        "builder_sha256": sha256(Path(__file__)),
        "deduplicator_sha256": sha256(args.worktree / "src/e_jepa_ttc/simplex_t/context_dedup.py"),
    }
    with ExclusiveLease(base / "DENSE_DEDUP.lock"):
        output.mkdir(exist_ok=True)
        identity_path = output / "IDENTITY.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
                raise ValueError("deduplication implementation or input identity changed")
        else:
            write_new_json(identity_path, identity)
        target, receipt = output / f"outer{args.outer}.npz", output / f"outer{args.outer}.json"
        if target.exists() or receipt.exists():
            raise FileExistsError("fold output already exists; do not overwrite evidence")
        started = time.perf_counter()
        selected = np.flatnonzero(dense["producer_family"][args.outer] >= 0)
        sources = []
        for position in selected:
            if len(sources) % 1024 == 0:
                resource_check()
            sequence = str(dense["sequences"][position])
            family = manifest["families"][int(dense["producer_family"][args.outer, position])]
            if family["outer_fold"] != args.outer or family["role"] == "outer_dev":
                raise ValueError("dense TRAIN query has invalid producer role")
            windows = dense["base_windows_us"][position]
            box = dense["square_xyxy"][position]
            current = QueryContextInput(
                str(dense["tokens"][position]),
                family["family_sha256"],
                int(dense["anchor_us"][position]),
                int(dense["roi_available_us"][position]),
                int(dense["roi_available_us"][position]),
                manifest["stream_bounds_us"][sequence][0],
                (
                    (int(windows[0, 0]), int(windows[0, 1])),
                    (int(windows[1, 0]), int(windows[1, 1])),
                    (int(windows[2, 0]), int(windows[2, 1])),
                ),
                (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            )
            sources.append(ContextSource(sequence, raw_hashes[sequence], prep, current))
        keys, selected_history = deduplicate_contexts(sources)
        history = np.full(dense["valid"].shape, -1, dtype=np.int64)
        history[selected] = selected_history
        if not np.array_equal(
            history >= 0, dense["valid"] & (dense["producer_family"][args.outer, :, None] >= 0)
        ):
            raise ValueError("dense source support changed during deduplication")
        # Reuse requires exact content identity, not just matching query tokens.
        old_positions = {str(token): i for i, token in enumerate(old["tokens"])}
        overlap = 0
        with np.load(old_path, allow_pickle=False) as archive:
            old_keys, old_history = archive["keys"], archive["history"]
        key_array = np.asarray(keys)
        for position in selected:
            previous = old_positions.get(str(dense["tokens"][position]))
            if previous is None:
                continue
            mask = history[position] >= 0
            if not np.array_equal(mask, old_history[previous] >= 0) or not np.array_equal(
                key_array[history[position, mask]], old_keys[old_history[previous, mask]]
            ):
                raise ValueError("DENSE/D0 overlap does not have identical content keys")
            overlap += 1
        if overlap != (5461, 5461, 5462)[args.outer]:
            raise ValueError("D0 TRAIN overlap count changed")
        resource_check()
        np.savez_compressed(target, keys=key_array, history=history)
        result = {
            "status": "DENSE_FOLD_KEYS_READY_NOT_REPLAY_AUTHORIZATION",
            "outer": args.outer,
            "path": target.name,
            "sha256": sha256(target),
            "selected_queries": len(selected),
            "unique_observations": len(keys),
            "D0_reusable_queries_exact": overlap,
            "additional_queries": len(selected) - overlap,
            "optimizer_updates": 0,
            "targets_read": False,
            "raw_payload_read": False,
            "seconds": time.perf_counter() - started,
        }
        write_new_json(receipt, result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
