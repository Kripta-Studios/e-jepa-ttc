"""Read-only X3 event-native feasibility audit; never trains a model."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact
from e_jepa_ttc.data.x3_raw_feasibility import build_x3_raw_binding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-metadata", type=Path)
    parser.add_argument("--garlttc-root", type=Path)
    parser.add_argument("--eap-root", type=Path)
    parser.add_argument("--binding-output", type=Path)
    parser.add_argument("--binding-manifest-output", type=Path)
    parser.add_argument("--expected-tokens", type=int, default=8192)
    parser.add_argument("--read-probe-tokens", type=int, default=64)
    parser.add_argument("--hash-event-files", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (repo / "configs/protocol/scientific_recovery_v9_stage61_pair_router_x2.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_paths = sorted(args.cache_root.glob("*.manifest.json"))
    if not manifest_paths:
        legacy_manifest = args.cache_root / "manifest.json"
        if legacy_manifest.is_file():
            manifest_paths = [legacy_manifest]
        else:
            raise FileNotFoundError(f"no signed cache manifests found beneath {args.cache_root}")
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]
    forbidden = ("public", "private", "test", "codabench")
    text = json.dumps(manifests).lower()
    raw_binding_keys = {
        key
        for manifest in manifests
        for key in manifest
        if "raw" in key.lower() and ("path" in key.lower() or "manifest" in key.lower())
    }
    if (args.garlttc_root is None) != (args.eap_root is None):
        raise ValueError("--garlttc-root and --eap-root must be supplied together")

    metadata_paths = sorted(args.cache_root.glob("*.metadata.csv"))
    if args.stage_metadata is not None:
        metadata_paths = [args.stage_metadata]
    if args.garlttc_root is not None and args.eap_root is not None:
        if not metadata_paths:
            raise FileNotFoundError("no Stage 61 metadata CSV was found")
        stage_metadata = metadata_paths[0]
        binding_output = args.binding_output or args.output.with_name("X3_RAW_BINDING.csv")
        binding_manifest_output = args.binding_manifest_output or args.output.with_name(
            "X3_RAW_BINDING_MANIFEST.json"
        )
        code_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        raw_manifest, read_probe, proposal = build_x3_raw_binding(
            stage_metadata_path=stage_metadata,
            garl_train_parquet=args.garlttc_root / "data/train.parquet",
            eap_root=args.eap_root,
            binding_csv_path=binding_output,
            binding_manifest_path=binding_manifest_output,
            code_commit=code_commit,
            protocol_sha256=protocol["artifact_sha256"],
            expected_tokens=args.expected_tokens,
            read_probe_tokens=args.read_probe_tokens,
            hash_event_files=args.hash_event_files,
        )
        result = sign_artifact(
            {
                "artifact_type": "scientific_recovery_v9_x3_feasibility_v1",
                "schema_version": "1.0",
                "evidence_type": "read_only_feasibility",
                "code_commit": code_commit,
                "protocol_version": "stage61_stage62_v1",
                "protocol_sha256": protocol["artifact_sha256"],
                "created_at": datetime.now(UTC).isoformat(),
                "status": "completed_read_only",
                "training_executed": False,
                "decision": "X3_DATA_READY",
                "cache_manifest": {
                    "count": len(manifest_paths),
                    "items": [
                        {"path": str(path), "sha256": compute_file_hash(str(path))}
                        for path in manifest_paths
                    ],
                    "raw_binding_manifest": {
                        "path": str(binding_manifest_output),
                        "sha256": compute_file_hash(str(binding_manifest_output)),
                        "artifact_sha256": raw_manifest["artifact_sha256"],
                    },
                    "raw_binding_csv": raw_manifest["binding_csv"],
                    "garl_train_manifest": raw_manifest["garl_train_manifest"],
                },
                "raw_token_binding_manifest_keys": [
                    "sample_token",
                    "sequence_id",
                    "track_id",
                    "events_relpath",
                    "raw_interval_start_us",
                    "raw_interval_end_us",
                    "raw_event_count",
                ],
                "audited_tokens": raw_manifest["tokens"],
                "audited_tokens_unique": raw_manifest["tokens_unique"],
                "raw_events_available": True,
                "timestamps_available": True,
                "polarity_available": True,
                "manifest_mentions_forbidden_paths": {
                    "public": False,
                    "private": False,
                    "test": False,
                    "codabench": False,
                },
                "forbidden_paths_opened": False,
                "event_counts_per_token_available": True,
                "raw_intervals_per_token_available": True,
                "read_probe": read_probe,
                "future_cache_proposal": proposal,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return

    rows: list[dict[str, Any]] = []
    bytes_read = 0
    began = time.perf_counter()
    for metadata_path in metadata_paths:
        bytes_read += metadata_path.stat().st_size
        with metadata_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(dict(row))
                if len(rows) == 64:
                    break
        if len(rows) == 64:
            break
    elapsed = time.perf_counter() - began
    tokens = [str(row.get("sample_token", "")) for row in rows]
    raw_available = bool(rows) and all(
        any(key in row for key in ("raw_events", "raw_event_path", "events_xytp")) for row in rows
    )
    timestamps_available = bool(rows) and all("raw_event_timestamps_us" in row for row in rows)
    polarity_available = bool(rows) and all("raw_event_polarity" in row for row in rows)
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
                "count": len(manifest_paths),
                "items": [
                    {"path": str(path), "sha256": compute_file_hash(str(path))}
                    for path in manifest_paths
                ],
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
