"""Prepare input-only dense-old context and prove exact overlap with the D0 index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.expansion_context import expansion_context_arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    base = Path(paths["worktree"]) / "artifacts/simplex_t"
    plan_path = base / "T0/MATCHED_CONTROL_POOL_PLAN.json"
    timing_path = base / "T0/DENSE_OLD_EXPOSURE_TIMING.json"
    pins = {
        plan_path: "b0685050b799058e6090d6e3b7c47b653f2939ca7db2d75a93526ab236ec9e3c",
        timing_path: "11fda3e6666c23dd306b667c1f2e0920104ffaf363e34023b1f013d93d96bfcb",
        args.train_metadata: "03dd3022db4b5f43bb10244fc8778476d74351e764f73a90c8566af949c17fd6",
    }
    for path, expected in pins.items():
        if sha256(path) != expected:
            raise ValueError("audited dense input changed")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    old_manifest_path = base / "T1/query_context_index/INDEX_MANIFEST.json"
    if sha256(old_manifest_path) != plan["D0_index_manifest_sha256"]:
        raise ValueError("D0 index lineage changed")
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    old_array_path = old_manifest_path.parent / "query_context_index.npz"
    if sha256(old_array_path) != old_manifest["index_sha256"]:
        raise ValueError("D0 index bytes changed")
    with np.load(old_array_path, allow_pickle=False) as archive:
        old = {key: archive[key] for key in archive.files}
    tokens = set().union(*(set(f["pools"]["DENSE_OLD"]["tokens"]) for f in plan["folds"]))
    metadata = pd.read_parquet(
        args.train_metadata,
        columns=["sample_token", "sequence_id", "event_windows_us", "boxes_xyxy"],
        filters=[("sample_token", "in", sorted(tokens))],
    )
    if len(metadata) != len(tokens) or set(metadata.sample_token) != tokens:
        raise ValueError("dense query metadata mismatch")
    exposures = {row["sample_token"]: row for row in timing["rows"]}
    if set(exposures) != tokens:
        raise ValueError("dense exposure inventory mismatch")
    bounds = {key: tuple(value) for key, value in old_manifest["stream_bounds_us"].items()}
    permitted, rejected = [], []
    for row in metadata.to_dict("records"):
        token = row["sample_token"]
        try:
            expansion_context_arrays([row], {token: exposures[token]}, bounds)
        except ValueError as error:
            rejected.append({"sample_token": token, "input_reason": str(error)})
        else:
            permitted.append(row)
    retained = {row["sample_token"] for row in permitted}
    arrays = expansion_context_arrays(permitted, {t: exposures[t] for t in retained}, bounds)
    positions = {str(t): i for i, t in enumerate(arrays["tokens"])}
    if not set(old["tokens"]) <= retained:
        raise ValueError("existing D0 queries lost input support")
    overlap = np.array([positions[str(t)] for t in old["tokens"]], np.int64)
    for field in (
        "sequences",
        "base_windows_us",
        "square_xyxy",
        "anchor_us",
        "roi_available_us",
        "valid",
    ):
        if not np.array_equal(arrays[field][overlap], old[field]):
            raise ValueError(f"D0 exact input parity failed: {field}")
    if not np.array_equal(arrays["lag_us"], old["lag_us"]):
        raise ValueError("D0 lag schedule changed")
    assignments = np.full((3, len(retained)), -1, np.int16)
    counts = []
    for fold in plan["folds"]:
        outer = fold["outer"]
        selected = fold["pools"]["DENSE_OLD"]
        for token in set(selected["tokens"]) & retained:
            i = positions[token]
            digest = selected["sequence_family_sha256"][str(arrays["sequences"][i])]
            matches = [
                j
                for j, family in enumerate(old_manifest["families"])
                if family["outer_fold"] == outer
                and family["role"] != "outer_dev"
                and family["family_sha256"] == digest
            ]
            if len(matches) != 1:
                raise ValueError("dense query lacks correct historical producer")
            assignments[outer, i] = matches[0]
        counts.append(int((assignments[outer] >= 0).sum()))
    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "query_context_index.npz",
        tokens=arrays["tokens"],
        sequences=arrays["sequences"],
        base_windows_us=arrays["base_windows_us"],
        square_xyxy=arrays["square_xyxy"],
        anchor_us=arrays["anchor_us"],
        roi_available_us=arrays["roi_available_us"],
        lag_us=arrays["lag_us"],
        valid=arrays["valid"],
        producer_family=assignments,
    )
    proof = {
        "status": "DENSE_INPUT_INDEX_PREPARED_PENDING_TIME_ACK_AND_REPLAY",
        "queries": len(retained),
        "rejected": rejected,
        "usable_per_fold": counts,
        "D0_overlap_queries": len(overlap),
        "D0_input_parity_exact": True,
        "families": old_manifest["families"],
        "stream_bounds_us": bounds,
        "stream_bounds_source": str(old_manifest_path),
        "stream_bounds_remeasured": False,
        "index_sha256": sha256(args.output / "query_context_index.npz"),
        "input_pins": {str(path): digest for path, digest in pins.items()},
        "h8_available_queries": int(arrays["valid"][:, -8:].all(1).sum()),
        "h16_available_queries": int(arrays["valid"].all(1).sum()),
        "targets_read": False,
        "raw_event_payload_read": False,
        "optimizer_updates": 0,
    }
    write_new_json(args.output / "INDEX_MANIFEST.json", proof)
    print(
        json.dumps(
            {
                key: proof[key]
                for key in (
                    "queries",
                    "usable_per_fold",
                    "D0_overlap_queries",
                    "D0_input_parity_exact",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
