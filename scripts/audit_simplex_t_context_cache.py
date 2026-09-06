"""Audit completed temporal feature blocks without scoring or opening targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.simplex_t.expert_phase import expert_phase_from_ttc


def main() -> None:
    """Check content, schema, chronology and frozen observation assignments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--dedup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identity = json.loads((args.cache / "IDENTITY.json").read_text(encoding="utf-8"))
    index_path = args.index / "query_context_index.npz"
    if compute_file_hash(str(index_path)) != identity["index_sha256"]:
        raise ValueError("wrong cache index")
    with np.load(index_path, allow_pickle=False) as archive:
        index = {name: archive[name] for name in archive.files}
    history = []
    dedup_manifest = json.loads((args.dedup / "DEDUP_MANIFEST.json").read_text(encoding="utf-8"))
    for record in dedup_manifest["outputs"]:
        path = args.dedup / record["path"]
        if compute_file_hash(str(path)) != record["sha256"]:
            raise ValueError("dedup hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            history.append(archive["history"])
    blocks = []
    infinite_expert_points = []
    seen: set[tuple[int, int]] = set()
    for receipt in sorted(args.cache.glob("family*_query*.json")):
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        path = receipt.with_suffix(".npz")
        if compute_file_hash(str(path)) != saved["sha256"]:
            raise ValueError("cache block hash mismatch")
        qi, family = saved["query"], saved["family"]
        outer = family // 4
        if index["producer_family"][outer, qi] != family:
            raise ValueError("incorrect producer family")
        mask = index["valid"][qi]
        count = int(mask.sum())
        with np.load(path, allow_pickle=False) as arrays:
            features = arrays["features145"]
            expert = arrays["expert_ttc"]
            if features.shape != (count, 145) or features.dtype != np.float32:
                raise ValueError("feature schema drift")
            if expert.shape != (count, 3) or not np.isfinite(features).all():
                raise ValueError("invalid expert content")
            if not np.isfinite(expert[:, :2]).all():
                raise ValueError("nonfinite A5/C2F points")
            expected_phase = expert_phase_from_ttc(expert.astype(np.float64)).astype(np.float32)
            for row, expert_id in np.argwhere(np.isinf(expert)):
                infinite_expert_points.append(
                    {
                        "query": qi,
                        "family": family,
                        "row": int(row),
                        "expert": int(expert_id),
                        "observation_id": int(arrays["observation_ids"][row]),
                        "ttc_repr": repr(float(expert[row, expert_id])),
                        "phase": float(expected_phase[row, expert_id]),
                    }
                )
            if not np.array_equal(features[:, 8:11], expected_phase):
                raise ValueError("phase/point inconsistency")
            if not np.array_equal(arrays["observation_ids"], history[outer][qi, mask]):
                raise ValueError("observation identity mismatch")
            if not np.array_equal(
                arrays["anchor_us"], index["anchor_us"][qi] - index["lag_us"][mask]
            ):
                raise ValueError("sensor anchors changed")
            if not (arrays["available_us"] == index["roi_available_us"][qi]).all():
                raise ValueError("current crop availability backdated")
            for observation in arrays["observation_ids"]:
                key = (outer, int(observation))
                if key in seen:
                    raise ValueError("duplicate consumed observation")
                seen.add(key)
        blocks.append({"path": path.name, "sha256": saved["sha256"], "rows": count})
    write_new_json(
        args.output,
        {
            "status": "PARTIAL_TEMPORAL_CACHE_CONTENT_VERIFIED",
            "completed_query_blocks": len(blocks),
            "required_query_blocks": 3 * 8192,
            "observations": len(seen),
            "infinite_expert_points": infinite_expert_points,
            "blocks": blocks,
            "cache_identity_sha256": compute_file_hash(str(args.cache / "IDENTITY.json")),
            "targets_read": False,
            "optimizer_updates": 0,
        },
    )
    print(json.dumps({"blocks": len(blocks), "observations": len(seen)}))


if __name__ == "__main__":
    main()
