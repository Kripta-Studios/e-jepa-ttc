"""Audit an eAP object cache before any experiment consumes its test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_cache(manifest_path: Path, *, hash_shards: bool = False) -> dict[str, Any]:
    """Validate shapes, chronology, split isolation and numerical integrity."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = manifest.get("shards", [])
    _require(bool(shards), "Cache manifest contains no shards.")
    expected_horizons = np.asarray(
        manifest["config"]["prediction_horizons_ms"],
        dtype=np.float64,
    ) / 1000.0
    split_sequences: dict[str, set[str]] = defaultdict(set)
    sequence_splits: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    ttc_values: list[np.ndarray] = []
    shard_records: list[dict[str, Any]] = []
    total_bytes = 0
    required = {
        "context_events",
        "context_boxes",
        "context_object_mask",
        "future_events",
        "future_boxes",
        "future_object_mask",
        "ttc_s",
        "context_window_start_us",
        "context_window_end_us",
        "future_window_start_us",
        "future_window_end_us",
        "sequence_id",
        "split",
        "prediction_horizons_s",
    }
    for shard in shards:
        path = manifest_path.parent / shard["path"]
        _require(path.is_file(), f"Missing cache shard: {path}")
        total_bytes += path.stat().st_size
        with np.load(path, allow_pickle=False) as arrays:
            missing = sorted(required.difference(arrays.files))
            _require(not missing, f"{path} lacks arrays: {missing}")
            sample_count = int(arrays["context_events"].shape[0])
            _require(sample_count == int(shard["samples"]), f"Bad sample count in {path}")
            for key in required.difference({"prediction_horizons_s"}):
                _require(arrays[key].shape[0] == sample_count, f"Bad {key} length in {path}")
            context_start = arrays["context_window_start_us"]
            context_end = arrays["context_window_end_us"]
            future_start = arrays["future_window_start_us"]
            future_end = arrays["future_window_end_us"]
            context_valid = arrays["context_object_mask"].any(axis=-1)
            future_valid = arrays["future_object_mask"].any(axis=-1)
            _require(context_start.ndim == 2, f"Bad context time rank in {path}")
            _require(future_start.shape[1] == expected_horizons.size, f"Bad horizons in {path}")
            _require(
                np.all(context_start[context_valid] < context_end[context_valid]),
                f"Empty valid context interval in {path}",
            )
            _require(
                np.all(future_start[future_valid] < future_end[future_valid]),
                f"Empty valid future interval in {path}",
            )
            _require(
                np.all(
                    (context_end[:, :-1] <= context_start[:, 1:])
                    | ~(context_valid[:, :-1] & context_valid[:, 1:])
                ),
                f"Overlapping context windows in {path}",
            )
            _require(
                np.all(
                    (context_end[:, -1, None] <= future_start)
                    | ~future_valid
                ),
                f"Future window overlaps context in {path}",
            )
            _require(
                np.all(
                    (future_end[:, :-1] <= future_start[:, 1:])
                    | ~(future_valid[:, :-1] & future_valid[:, 1:])
                ),
                f"Future target windows overlap in {path}",
            )
            horizons = arrays["prediction_horizons_s"]
            _require(
                horizons.shape == expected_horizons.shape
                and np.allclose(horizons, expected_horizons),
                f"Horizon metadata mismatch in {path}",
            )
            _require(np.isfinite(arrays["context_events"]).all(), f"Non-finite events in {path}")
            _require(np.isfinite(arrays["future_events"]).all(), f"Non-finite future in {path}")
            _require(
                np.all(arrays["context_object_mask"][:, -1]),
                f"A sample ends without a valid tracked object in {path}",
            )
            sequences = arrays["sequence_id"].astype(str)
            splits = arrays["split"].astype(str)
            _require(np.all(sequences == str(shard["sequence_id"])), f"Sequence mismatch in {path}")
            _require(np.all(splits == str(shard["split"])), f"Split mismatch in {path}")
            split_sequences[str(shard["split"])].add(str(shard["sequence_id"]))
            sequence_splits[str(shard["sequence_id"])].add(str(shard["split"]))
            counts[str(shard["split"])] += sample_count
            ttc = arrays["ttc_s"].reshape(sample_count, -1)[:, 0].astype(np.float64)
            _require(np.isfinite(ttc).all(), f"Non-finite TTC targets in {path}")
            _require(np.all(np.abs(ttc) >= 0.1), f"Near-zero TTC target in {path}")
            ttc_values.append(ttc)
        record: dict[str, Any] = {
            "path": shard["path"],
            "samples": sample_count,
            "size_bytes": path.stat().st_size,
        }
        if hash_shards:
            record["sha256"] = _sha256(path)
        shard_records.append(record)
    leaking = {key: sorted(value) for key, value in sequence_splits.items() if len(value) != 1}
    _require(not leaking, f"Sequences assigned to multiple splits: {leaking}")
    all_ttc = np.concatenate(ttc_values)
    reported_samples = int(manifest.get("total_samples", all_ttc.size))
    _require(all_ttc.size == reported_samples, "Manifest total_samples does not match shards.")
    return {
        "status": "valid",
        "manifest": manifest_path.as_posix(),
        "format": manifest.get("format"),
        "manifest_sha256": _sha256(manifest_path),
        "shard_count": len(shards),
        "total_samples": int(all_ttc.size),
        "total_size_bytes": total_bytes,
        "samples_by_split": dict(sorted(counts.items())),
        "sequences_by_split": {
            key: sorted(value) for key, value in sorted(split_sequences.items())
        },
        "sequence_split_overlap": leaking,
        "temporal_overlap_count": 0,
        "ttc_s": {
            "minimum": float(np.min(all_ttc)),
            "median": float(np.median(all_ttc)),
            "maximum": float(np.max(all_ttc)),
            "negative_count": int(np.sum(all_ttc < 0.0)),
            "positive_count": int(np.sum(all_ttc > 0.0)),
        },
        "shards": shard_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hash-shards", action="store_true")
    args = parser.parse_args()
    result = audit_cache(args.manifest, hash_shards=args.hash_shards)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
