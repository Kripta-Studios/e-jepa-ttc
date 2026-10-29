"""Compare historical original and recovered producer inputs on the64-row cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.simplex_t.lifecycle import admitted


def main() -> None:
    """Read selected signed TRAIN shards only, with one shard resident at a time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--recovered-root", type=Path, required=True)
    parser.add_argument("--global-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior diagnostic")
    manifest_path = args.recovered_root / "manifest.json"
    if (
        compute_file_hash(str(manifest_path))
        != "76d90668334fde1a85c53edf8b036e9811527493f65bad5c089cb7b494ca26e2"
    ):
        raise ValueError("recovered producer manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = {record["path"]: record for record in manifest["shards"]}
    metadata_manifest = json.loads(
        args.global_metadata.with_name("outer0_final.manifest.json").read_text()
    )
    if compute_file_hash(str(args.global_metadata)) != metadata_manifest["metadata"]["sha256"]:
        raise ValueError("original row order metadata changed")
    metadata = pd.read_csv(args.global_metadata)
    indices = {token: index for index, token in enumerate(metadata.sample_token)}
    receipt = json.loads((args.replay / "INPUTS_READY.json").read_text(encoding="utf-8"))
    if compute_file_hash(str(args.replay / "inputs.pt")) != receipt["sha256"]:
        raise ValueError("saved input changed")
    saved = torch.load(args.replay / "inputs.pt", map_location="cpu", weights_only=True)
    grouped = {}
    for query, token in enumerate(saved["tokens"]):
        index = indices[token]
        grouped.setdefault(f"train/shard-{index // 32:05d}.pt", []).append(
            (query, index % 32, token)
        )
    results = []
    for relative, requested in grouped.items():
        if not admitted([args.output.parent])["has_headroom"]:
            raise RuntimeError("RESOURCE_PAUSE")
        path = args.recovered_root / relative
        if compute_file_hash(str(path)) != shards[relative]["sha256"]:
            raise ValueError("recovered shard changed")
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        for query, position, token in requested:
            row = loaded[position]
            if row["sample_token"] != token:
                raise ValueError(
                    "recovered shard ordering differs; explicit identity lookup required"
                )
            actual = np.asarray(row["event_v4_common_roi"], dtype=np.float32)
            expected = saved["inputs"][query].numpy()
            results.append(
                {
                    "token": token,
                    "shard": relative,
                    "bit_identical": bool(np.array_equal(actual, expected)),
                    "max_abs_by_window": np.abs(actual - expected).reshape(3, -1).max(1).tolist(),
                    "recovered_delta": float(np.asarray(row["garl_delta_t_s"])),
                    "original_delta": saved["delta"][query].tolist(),
                }
            )
        del loaded
        print(json.dumps({"shard": relative, "queries_checked": len(requested)}), flush=True)
    write_new_json(
        args.output,
        {
            "results": results,
            "optimizer_updates": 0,
            "all_input_tensors_bit_identical": all(row["bit_identical"] for row in results),
        },
    )


if __name__ == "__main__":
    main()
