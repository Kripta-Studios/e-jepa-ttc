"""Plan a lossless, sequence-disjoint rotating GarlTTC cache.

This command does not materialize media.  It derives the exact official-label
row selection used by the v4 cache builder, partitions it into deterministic
shards, verifies coverage/no-overlap, and estimates retained versus scratch
storage from an observed pilot.  A later materializer may delete a scratch
shard only after its per-shard hash has been recorded in the aggregate plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.data.garlttc_eap import (  # noqa: E402
    GARLTTC_JOIN_KEYS,
    load_garlttc_train_index,
    validate_garlttc_train_index,
)
from e_jepa_ttc.data.garlttc_sampling import select_balanced_cache_rows  # noqa: E402
from e_jepa_ttc.utils.io import read_structured, write_structured  # noqa: E402


def row_identity(row: Mapping[str, object]) -> tuple[str, ...]:
    """Return the immutable five-key identity used for shard coverage."""

    return tuple(str(row.get(key, "")) for key in GARLTTC_JOIN_KEYS)


def identity_hash(identities: Sequence[tuple[str, ...]]) -> str:
    """Hash ordered row identities without serializing media or labels."""

    payload = "\n".join(
        json.dumps(list(identity), ensure_ascii=False, separators=(",", ":"))
        for identity in identities
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _shard_rows(
    selected: pd.DataFrame,
    sequence_roles: Mapping[str, str],
    *,
    shard_size: int,
    bytes_per_sample: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    identities = [row_identity(row) for row in selected.to_dict(orient="records")]
    if len(set(identities)) != len(identities):
        raise ValueError("Selected rows contain duplicate five-key identities")

    shards: list[dict[str, Any]] = []
    role_counts = {"train": 0, "validation": 0}
    for role in ("train", "validation"):
        role_frame = selected[selected["sequence_id"].astype(str).map(sequence_roles.get) == role]
        role_identities = [row_identity(row) for row in role_frame.to_dict(orient="records")]
        role_counts[role] = len(role_identities)
        for shard_index, start in enumerate(range(0, len(role_identities), shard_size)):
            chunk = role_identities[start : start + shard_size]
            sequence_ids = sorted({identity[0] for identity in chunk})
            shards.append(
                {
                    "role": role,
                    "shard_index": shard_index,
                    "row_count": len(chunk),
                    "sequence_ids": sequence_ids,
                    "first_identity": list(chunk[0]),
                    "last_identity": list(chunk[-1]),
                    "row_identity_sha256": identity_hash(chunk),
                    "estimated_bytes": int(round(len(chunk) * bytes_per_sample)),
                    "scratch_deletable_after_hash": True,
                }
            )
    return shards, role_counts


def _pilot_bytes(pilot_root: Path) -> tuple[int, int]:
    manifest_path = pilot_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pilot manifest does not exist: {manifest_path}")
    manifest = read_structured(manifest_path)
    sample_count = sum(int(value) for value in manifest.get("split_counts", {}).values())
    if sample_count <= 0:
        raise ValueError("Pilot manifest has no materialized samples")
    total_bytes = 0
    for path in pilot_root.rglob("*"):
        if (
            path.is_file()
            and path.name not in {"manifest.json"}
            and not path.name.endswith(".meta.json")
        ):
            total_bytes += path.stat().st_size
    if total_bytes <= 0:
        raise ValueError("Pilot cache has no materialized shard bytes")
    return total_bytes, sample_count


def build_plan(
    *,
    garlttc_root: Path,
    split_path: Path,
    pilot_root: Path,
    shard_size: int,
    expected_train_rows: int,
    scratch_paths: Sequence[Path],
) -> dict[str, Any]:
    split = read_structured(split_path)
    assignments = split.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("Split artifact has no assignments mapping")
    sequence_roles = {
        str(sequence): role
        for role in ("train", "validation")
        for sequence in assignments.get(role, [])
    }
    if set(sequence_roles.values()) != {"train", "validation"}:
        raise ValueError("Both train and validation assignments are required")

    index = load_garlttc_train_index(garlttc_root, sorted(sequence_roles))
    validate_garlttc_train_index(
        index,
        expected_rows=expected_train_rows,
        allow_version_change=False,
    )
    rows = index.merged.sort_values(
        ["sequence_id", "timestamp_us", "track_id", "sample_token"],
        kind="mergesort",
    )
    selected, selection_report = select_balanced_cache_rows(
        rows,
        sequence_roles,
        seed=7,
        max_samples_per_split=None,
    )
    pilot_bytes, pilot_samples = _pilot_bytes(pilot_root)
    bytes_per_sample = pilot_bytes / pilot_samples
    shards, role_counts = _shard_rows(
        selected,
        sequence_roles,
        shard_size=shard_size,
        bytes_per_sample=bytes_per_sample,
    )
    if sum(role_counts.values()) != len(selected):
        raise RuntimeError("Shard role counts do not cover selected rows")
    if role_counts != {
        "train": len(
            selected[selected["sequence_id"].astype(str).map(sequence_roles.get) == "train"]
        ),
        "validation": len(
            selected[selected["sequence_id"].astype(str).map(sequence_roles.get) == "validation"]
        ),
    }:
        raise RuntimeError("Shard role coverage is inconsistent")

    free_bytes = {
        path.as_posix(): shutil.disk_usage(path).free for path in scratch_paths if path.exists()
    }
    peak_bytes = max((int(shard["estimated_bytes"]) for shard in shards), default=0)
    retained_bytes = sum(int(shard["estimated_bytes"]) for shard in shards)
    plan = {
        "artifact_type": "garlttc_rotating_cache_plan_v1",
        "status": "pass",
        "created_by": "scripts/plan_garlttc_rotating_cache.py",
        "garlttc_root": garlttc_root.as_posix(),
        "split_path": split_path.as_posix(),
        "split_sha256": split.get("artifact_sha256"),
        "garlttc_data_sha256": index.data_sha256,
        "garlttc_annotations_sha256": index.annotations_sha256,
        "garlttc_join_keys_sha256": index.join_keys_sha256,
        "selection_seed": 7,
        "selection_report": selection_report,
        "source_merged_row_count": index.source_merged_row_count,
        "selected_row_count": len(selected),
        "split_counts": role_counts,
        "shard_size": shard_size,
        "shard_count": len(shards),
        "shards": shards,
        "pilot_observed": {
            "root": pilot_root.as_posix(),
            "bytes": pilot_bytes,
            "samples": pilot_samples,
            "bytes_per_sample": bytes_per_sample,
        },
        "storage": {
            "estimated_retained_full_bytes": retained_bytes,
            "estimated_retained_full_gib": retained_bytes / 1024**3,
            "estimated_rotating_peak_bytes": peak_bytes,
            "estimated_rotating_peak_gib": peak_bytes / 1024**3,
            "free_bytes_by_path": free_bytes,
            "rotating_peak_fits": any(value >= peak_bytes for value in free_bytes.values()),
            "retained_full_fits": any(value >= retained_bytes for value in free_bytes.values()),
        },
        "contract": {
            "lossless_row_identity": list(GARLTTC_JOIN_KEYS),
            "delete_only_after_shard_hash": True,
            "global_coverage_required_before_training_claim": True,
            "validation_must_cover_all_shards": True,
            "uses_ttc_labels_for_cache_supervision": True,
            "uses_evttc_for_selection": False,
        },
        "materialization_started": False,
        "next_exact_step": (
            "Materialize each listed shard into scratch, verify its manifest and row hash, "
            "consume it, append the terminal shard metadata, then delete only the scratch "
            "tensor files."
        ),
    }
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--garlttc-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--expected-train-rows", type=int, default=88_744)
    parser.add_argument("--scratch-path", type=Path, action="append", default=[])
    args = parser.parse_args()
    plan = build_plan(
        garlttc_root=args.garlttc_root.resolve(),
        split_path=args.split.resolve(),
        pilot_root=args.pilot_root.resolve(),
        shard_size=args.shard_size,
        expected_train_rows=args.expected_train_rows,
        scratch_paths=[path.resolve() for path in args.scratch_path],
    )
    write_structured(args.output, plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
