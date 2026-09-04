"""Read-only X3 event-native feasibility audit; never trains a model."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (repo / "configs/protocol/scientific_recovery_v9_stage61_pair_router_x2.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_path = args.cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    forbidden = ("public", "private", "test", "codabench")
    text = json.dumps(manifest).lower()
    raw_binding_keys = {
        key
        for key in manifest
        if "raw" in key.lower() and ("path" in key.lower() or "manifest" in key.lower())
    }
    shards = sorted((args.cache_root / "train").glob("*.pt"))
    rows: list[dict[str, Any]] = []
    bytes_read = 0
    began = time.perf_counter()
    for shard_path in shards:
        loaded = torch.load(shard_path, map_location="cpu", weights_only=False)
        bytes_read += shard_path.stat().st_size
        for row in loaded:
            rows.append(row)
            if len(rows) == 64:
                break
        if len(rows) == 64:
            break
    elapsed = time.perf_counter() - began
    tokens = [str(row.get("sample_token", "")) for row in rows]
    raw_available = all(
        any(key in row for key in ("raw_events", "raw_event_path", "events_xytp")) for row in rows
    )
    timestamps_available = all("raw_event_timestamps_us" in row for row in rows)
    polarity_available = all("raw_event_polarity" in row for row in rows)
    decision = (
        "X3_DATA_READY"
        if raw_available and timestamps_available and polarity_available and raw_binding_keys
        else "X3_RAW_EVENTS_MISSING"
    )
    result = sign_artifact(
        {
            "artifact_type": "scientific_recovery_v9_x3_feasibility_v1",
            "schema_version": "1.0",
            "evidence_type": "read_only_feasibility",
            "code_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "protocol_version": "stage61_stage62_v1",
            "protocol_sha256": protocol["artifact_sha256"],
            "created_at": datetime.now(UTC).isoformat(),
            "status": "completed_read_only",
            "training_executed": False,
            "decision": decision,
            "cache_manifest": {
                "path": str(manifest_path),
                "sha256": compute_file_hash(str(manifest_path)),
            },
            "raw_token_binding_manifest_keys": sorted(raw_binding_keys),
            "audited_tokens": len(tokens),
            "audited_tokens_unique": len(set(tokens)),
            "raw_events_available": raw_available,
            "timestamps_available": timestamps_available,
            "polarity_available": polarity_available,
            "manifest_mentions_forbidden_paths": {item: item in text for item in forbidden},
            "forbidden_paths_opened": False,
            "event_counts_per_token_available": raw_available,
            "raw_intervals_per_token_available": timestamps_available,
            "read_probe": {
                "tokens": len(rows),
                "seconds": elapsed,
                "bytes": bytes_read,
                "tokens_per_second": len(rows) / max(elapsed, 1e-12),
            },
            "future_cache_proposal": {
                "microbin_us": 1000,
                "snapshot_interval_us": 5000,
                "fields": ["x", "y", "timestamp_us", "polarity", "sample_token"],
                "estimated_bytes": None,
                "estimate_reason": "raw event counts unavailable"
                if not raw_available
                else "pending full count",
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
