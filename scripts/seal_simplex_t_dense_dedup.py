"""Verify all three dense-control fold receipts before sealing their metadata index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.lifecycle import ExclusiveLease


def main() -> None:
    """Seal content keys only; no claim of replay, source authority or scientific freeze."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--identity-sha256", required=True)
    args = parser.parse_args()
    base = args.worktree / "artifacts/simplex_t/T1"
    root = base / "dense_query_context_dedup"
    identity_path = root / "IDENTITY.json"
    if sha256(identity_path) != args.identity_sha256:
        raise ValueError("dense producer/input identity changed")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    index_root = base / "dense_query_context_index"
    if sha256(index_root / "INDEX_MANIFEST.json") != identity["dense_index_sha256"]:
        raise ValueError("dense source manifest changed")
    index_manifest = json.loads((index_root / "INDEX_MANIFEST.json").read_text(encoding="utf-8"))
    if sha256(index_root / "query_context_index.npz") != index_manifest["index_sha256"]:
        raise ValueError("dense source arrays changed")
    with np.load(index_root / "query_context_index.npz", allow_pickle=False) as archive:
        valid, families = archive["valid"], archive["producer_family"]
    with ExclusiveLease(base / "DENSE_DEDUP.lock"):
        destination = root / "DEDUP_MANIFEST.json"
        if destination.exists():
            raise FileExistsError("completed dense metadata seal exists")
        records = []
        for outer in range(3):
            record = json.loads((root / f"outer{outer}.json").read_text(encoding="utf-8"))
            array_path = root / f"outer{outer}.npz"
            if record["outer"] != outer or record["path"] != array_path.name:
                raise ValueError("dense fold receipt identity mismatch")
            if sha256(array_path) != record["sha256"]:
                raise ValueError("dense content keys changed")
            with np.load(array_path, allow_pickle=False) as archive:
                keys, history = archive["keys"], archive["history"]
            mask = valid & (families[outer, :, None] >= 0)
            if (
                history.dtype != np.int64
                or history.shape != mask.shape
                or (history < -1).any()
                or not np.array_equal(history >= 0, mask)
                or len(keys) != record["unique_observations"]
                or len(np.unique(keys)) != len(keys)
                or not np.array_equal(np.unique(history[mask]), np.arange(len(keys)))
            ):
                raise ValueError("dense history coverage or observation bounds changed")
            selected = int((families[outer] >= 0).sum())
            if selected != record["selected_queries"] or selected != (14520, 13473, 14949)[outer]:
                raise ValueError("registered dense pool count changed")
            records.append(
                {
                    **record,
                    "consumed_slots": int(mask.sum()),
                    "inactive_queries": len(history) - selected,
                    "receipt_sha256": sha256(root / f"outer{outer}.json"),
                }
            )
        write_new_json(
            destination,
            {
                "status": "DENSE_CONTENT_INDEX_READY_PENDING_TIME_ACK_AND_FEATURE_REPLAY",
                "identity": identity,
                "identity_sha256": args.identity_sha256,
                "input_manifest_sha256": identity["dense_index_sha256"],
                "outputs": records,
                "sealer_sha256": sha256(Path(__file__)),
                "optimizer_updates": 0,
                "targets_read": False,
                "raw_payload_read": False,
            },
        )
        print(json.dumps({"path": str(destination), "sha256": sha256(destination)}))


if __name__ == "__main__":
    main()
