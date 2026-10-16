"""Recover historical producer batch shapes from hash-bound fragments, without inference."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from operational.range_transfer.common import freeze, read, sha
from operational.rgb_port.recipe import canonical_sha256

from .common import ROOT, RUN, owner

OUT = RUN / "fusion/cost/historical_batch_inventory"


def rgb_frame_counts(rows: list[dict]) -> dict[str, int]:
    """Reproduce label-free observation identities and T2/T3 counts only."""
    result = {}
    for row in rows:
        ids = list(row["observation_ids"])
        if ids:
            members = list(row["observation_members"])
            if len(ids) != len(members):
                raise ValueError("Observation metadata not aligned")
            entries = [(str(key), len(value)) for key, value in zip(ids, members, strict=True)]
        elif int(row["producer_frame_count"]) == 2:
            key = hashlib.sha256(
                f"{row['track_id']}\0cold_t2\0{int(row['rgb_anchor_us'])}".encode()
            ).hexdigest()
            entries = [(key, 2)]
        else:
            entries = []
        for key, frames in entries:
            if frames not in (2, 3) or (key in result and result[key] != frames):
                raise ValueError("Incompatible producer frame counts")
            result[key] = frames
    return result


def batch_shapes(ids: list[str], counts: dict[str, int]) -> dict[str, dict]:
    """A stored fragment is split by T; its length is not necessarily forward batch size."""
    if len(ids) != len(set(ids)) or any(key not in counts for key in ids):
        raise ValueError("Duplicate or unknown fragment observation")
    sizes = Counter(counts[key] for key in ids)
    return {
        key: {
            "fragment_rows": len(ids),
            "input_frames": counts[key],
            "forward_batch": sizes[counts[key]],
        }
        for key in ids
    }


def run() -> dict:
    """Audit original four diagnostic queries and every fragment membership, read-only."""
    diagnostic_path = RUN / "fusion/cost/parity_diagnostic_v1/REGISTRATION.json"
    diagnostic = read(diagnostic_path)
    manifest_path = owner() / "V_MANIFEST.json"
    manifest = read(manifest_path)
    rows_path = Path(manifest["rows_path"])
    if sha(rows_path) != manifest["rows_sha256"]:
        raise ValueError("Frozen V metadata changed")
    columns = [
        "sample_token",
        "observation_ids",
        "observation_members",
        "producer_frame_count",
        "track_id",
        "rgb_anchor_us",
    ]
    rows = pd.read_parquet(rows_path, columns=columns)
    if len(rows) != manifest["population_size"] or rows.sample_token.duplicated().any():
        raise ValueError("V population changed")
    rgb_counts = rgb_frame_counts(rows.to_dict("records"))
    modalities = {}
    for modality in ("E", "R"):
        folder = owner() / f"features/V/{modality}_RAW"
        receipt = read(folder / "REAL_EXPERT_RECEIPT.json")
        for name, expected in receipt["files"].items():
            if sha(folder / name) != expected:
                raise ValueError("Historical aggregate changed")
        with np.load(folder / "blocks.npz", allow_pickle=False) as archive:
            ordered = archive["observation_id"].astype(str).tolist()
        counts = dict.fromkeys(ordered, 3) if modality == "E" else rgb_counts
        if len(ordered) != len(set(ordered)) or set(ordered) != set(counts):
            raise ValueError("Observation population differs from frozen metadata")
        with np.load(folder / "history.npz", allow_pickle=False) as archive:
            queries = archive["query_id"].astype(str)
            history = archive["observation_id"].astype(str)
            valid = archive["valid"].astype(bool)
        if queries.tolist() != rows.sample_token.tolist():
            raise ValueError("History query order changed")
        wanted = {key for index in diagnostic["indices"] for key in history[index][valid[index]]}
        parts = folder / "parts"
        binding = read(parts / "BINDING.json")
        binding_hash = canonical_sha256(binding)
        for path, expected in binding["executed_source_sha256"].items():
            if sha(Path(path)) != expected:
                raise ValueError(f"Historical source changed: {path}")
        concatenated, selected, fragment_counts, forward_counts = [], {}, Counter(), Counter()
        fragments = []
        for path in sorted(parts.glob("part_*.npz")):
            part_receipt = read(path.with_suffix(".json"))
            if (
                part_receipt["binding_sha256"] != binding_hash
                or sha(path) != part_receipt["sha256"]
            ):
                raise ValueError("Historical fragment failed admission")
            with np.load(path, allow_pickle=False) as archive:
                ids = archive["observation_id"].astype(str).tolist()
            shapes = batch_shapes(ids, counts)
            concatenated.extend(ids)
            fragment_counts[len(ids)] += 1
            forward_counts.update(
                (value["input_frames"], value["forward_batch"]) for value in shapes.values()
            )
            for key in wanted.intersection(ids):
                if key in selected:
                    raise ValueError("Observation repeated across fragments")
                selected[key] = {
                    **shapes[key],
                    "fragment": path.name,
                    "fragment_sha256": part_receipt["sha256"],
                }
            fragments.append(
                {"file": path.name, "sha256": part_receipt["sha256"], "rows": len(ids)}
            )
        if concatenated != ordered or set(selected) != wanted:
            raise ValueError("Fragment population/order or required history differs")
        modalities[modality] = {
            "receipt_sha256": sha(folder / "REAL_EXPERT_RECEIPT.json"),
            "fragment_binding_sha256": sha(parts / "BINDING.json"),
            "fragments": fragments,
            "fragment_size_counts": dict(fragment_counts),
            "observation_weighted_forward_shape_counts": {
                f"T{frames}_B{batch}": count
                for (frames, batch), count in sorted(forward_counts.items())
            },
            "queries": [
                {
                    "index": index,
                    "sample_token": str(queries[index]),
                    "observations": [
                        {"id": key, **selected[key]} for key in history[index][valid[index]]
                    ],
                }
                for index in diagnostic["indices"]
            ],
        }
    result = {
        "status": "COMPLETE_METADATA_AUDIT",
        "sources": {
            str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
        },
        "prior_diagnostic_registration_sha256": sha(diagnostic_path),
        "V_manifest_sha256": sha(manifest_path),
        "V_rows_sha256": sha(rows_path),
        "indices": diagnostic["indices"],
        "modalities": modalities,
        "labels_read": False,
        "optimizer_updates": 0,
        "model_inference": False,
        "interpretation": (
            "Historical batch shapes reconstructed from executed-source policy "
            "and verified fragments; "
            "not evidence that batch shape alone explains numerical discrepancies"
        ),
        "cost_admitted": False,
        "tolerance_changes": False,
    }
    freeze(OUT / "RESULT.json", result)
    return result


if __name__ == "__main__":
    result = run()
    print({"status": result["status"], "indices": result["indices"]})
