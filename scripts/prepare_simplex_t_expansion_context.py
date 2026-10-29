"""Prepare D1 retrospective input indices; no expert replay or scientific authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.data.raw_event_binding import _reject_external_links
from e_jepa_ttc.simplex_t.expansion_context import expansion_context_arrays
from e_jepa_ttc.simplex_t.lifecycle import admitted


def main() -> None:
    """Read only permitted metadata and two timestamp endpoints per TRAIN stream."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    root = Path(paths["worktree"])
    stage = Path(paths["stage70_worktree_read_only"]) / "artifacts/stage70_76_architecture"
    pins = {
        stage / "expansion_usability/USABLE_METADATA.parquet": (
            "e8514492952a3abd86e45e3b00c07e398d2e6def5b73062eb5b5f8f13a932064"
        ),
        stage / "expansion_raw_bindings/RAW_BINDING_MANIFEST.json": (
            "4b7bfab2dc9ff31a6f3c936f487c1456bec1f8fd739d28165acb12e92a9047ba"
        ),
        root / "artifacts/simplex_t/T0/EXPANSION_EXPOSURE_TIMING.json": (
            "8061b30e6acead8249e099e3c488558d773cdef7b5d724adaead3417bdc7e63a"
        ),
        root / "artifacts/simplex_t/T0/EXPANSION_POOL_PLAN.json": (
            "2c2a36f42c93f3d5304c524e04bcb84c31c5e8a756715d1ab955288abc2d1849"
        ),
        root / "artifacts/simplex_t/T0/EXPANSION_ANCESTRY_RECHECK.json": (
            "f81b2874925ae584a6a96452e1a451364a3f3bff7d95982e559ea0a906021bcc"
        ),
    }
    for path, expected in pins.items():
        if sha256(path) != expected:
            raise ValueError(f"audited input changed: {path.name}")
    metadata = pd.read_parquet(
        stage / "expansion_usability/USABLE_METADATA.parquet",
        columns=["sample_token", "sequence_id", "event_windows_us", "boxes_xyxy"],
    )
    binding = json.loads(
        (stage / "expansion_raw_bindings/RAW_BINDING_MANIFEST.json").read_text(encoding="utf-8")
    )
    timing = json.loads(
        (root / "artifacts/simplex_t/T0/EXPANSION_EXPOSURE_TIMING.json").read_text(encoding="utf-8")
    )
    pool = json.loads(
        (root / "artifacts/simplex_t/T0/EXPANSION_POOL_PLAN.json").read_text(encoding="utf-8")
    )
    index_path = root / "artifacts/simplex_t/T1/query_context_index/INDEX_MANIFEST.json"
    if sha256(index_path) != pool["D0_index_manifest_sha256"]:
        raise ValueError("D0 family index changed")
    d0 = json.loads(index_path.read_text(encoding="utf-8"))
    allowed = set(pool["folds"]["0"]["additional_groups"])
    if set(metadata.sequence_id) != allowed or set(binding["identity"]["sources"]) != allowed:
        raise ValueError("D1 sequence identity mismatch")
    raw_root = (Path(paths["eap_root"]) / "data/train").resolve(strict=True)
    bounds = {}
    for sequence in sorted(allowed):
        if not admitted([root])["has_headroom"]:
            raise RuntimeError("RESOURCE_PAUSE before timestamp metadata read")
        source = binding["identity"]["sources"][sequence]
        path = (raw_root / sequence / "events.h5").resolve(strict=True)
        if not path.is_relative_to(raw_root) or path != Path(source["path"]).resolve(strict=True):
            raise ValueError("source path outside registered expansion TRAIN")
        stamp = path.stat()
        if (stamp.st_size, stamp.st_mtime_ns) != (source["bytes"], source["mtime_ns"]):
            raise ValueError("raw source stat differs from owner's full hash record")
        with h5py.File(path, "r") as handle:
            _reject_external_links(handle, path)
            times = handle["events/t"]
            if (
                not isinstance(times, h5py.Dataset)
                or times.dtype.kind not in "iu"
                or not len(times)
            ):
                raise ValueError("unsupported raw timestamp dataset")
            if times.is_virtual or times.external:
                raise ValueError("external raw timestamp storage")
            bounds[sequence] = (int(times[0]), int(times[-1]))
        if path.stat().st_mtime_ns != stamp.st_mtime_ns:
            raise ValueError("raw source changed during endpoint read")
    arrays = expansion_context_arrays(
        metadata.to_dict("records"), {r["sample_token"]: r for r in timing["rows"]}, bounds
    )
    assignments = np.full((3, len(metadata)), -1, dtype=np.int16)
    for outer in range(3):
        fold = pool["folds"][str(outer)]
        selected = set(fold["additional_train_tokens"])
        for index, (token, sequence) in enumerate(
            zip(arrays["tokens"], arrays["sequences"], strict=True)
        ):
            if token not in selected:
                continue
            digest = fold["sequence_family_sha256"][str(sequence)]
            matches = [
                i
                for i, family in enumerate(d0["families"])
                if family["outer_fold"] == outer
                and family["role"] != "outer_dev"
                and family["family_sha256"] == digest
            ]
            if len(matches) != 1:
                raise ValueError("ambiguous expansion producer family")
            assignments[outer, index] = matches[0]
        if int((assignments[outer] >= 0).sum()) != len(selected):
            raise ValueError("selected expansion query absent from context index")
    arrays["producer_family"] = assignments
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
        producer_family=arrays["producer_family"],
    )
    write_new_json(
        args.output / "INDEX_MANIFEST.json",
        {
            "status": "D1_INPUT_INDEX_PREPARED_PENDING_OWNER_TIME_ACK_AND_REPLAY",
            "queries": len(metadata),
            "families": d0["families"],
            "stream_bounds_us": bounds,
            "sources": {str(path): expected for path, expected in pins.items()},
            "index_sha256": sha256(args.output / "query_context_index.npz"),
            "h8_available_queries": int(arrays["valid"][:, -8:].all(1).sum()),
            "h16_available_queries": int(arrays["valid"].all(1).sum()),
            "current_queries_valid": int(arrays["valid"][:, -1].sum()),
            "raw_access": "first/last timestamp per permitted TRAIN stream; no event windows",
            "full_raw_hashes_repeated": False,
            "target_fields_read": False,
            "optimizer_updates": 0,
            "context_semantics": "RETROSPECTIVE_CURRENT_QUERY_ROI_NOT_OBJECT_TRACKING",
        },
    )
    print(json.dumps({"queries": len(metadata), "h8": int(arrays["valid"][:, -8:].all(1).sum())}))


if __name__ == "__main__":
    main()
