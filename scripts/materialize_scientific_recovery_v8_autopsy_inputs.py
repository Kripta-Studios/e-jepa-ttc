#!/usr/bin/env python
"""Materialize exact A5/C2F V8-A replay payloads from the historical 12-channel V4 cache.

The full 8,192-row V4 tensor is intentionally never held in RAM.  Production
payloads are written as small signed fold shards (default eight rows each) so the
subsequent replay can stay within the memory envelope of a 12 GiB GPU.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.data.garlttc_lhr_cache import (  # noqa: E402
    GarlTTCLHRCacheConfig,
    materialize_garlttc_lhr_cache,
)
from e_jepa_ttc.data.object_event_v4 import GarlTTCObjectEventV4Dataset  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_signed(path: Path, value: dict[str, Any]) -> None:
    sign_artifact(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_artifact_hash(value):
        raise ValueError(f"unsigned artifact: {path}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _missing_split_shards(manifest_path: Path, *, split: str = "train") -> list[Path]:
    """Return missing cache shard paths without opening any tensor payload."""

    manifest = _read_json(manifest_path)
    root = manifest_path.resolve().parent
    missing: list[Path] = []
    for shard in manifest.get("shards", []):
        if str(shard.get("split")) != split:
            continue
        relative = shard.get("path")
        if not relative:
            raise ValueError(f"cache shard entry lacks path in {manifest_path}")
        path = root / str(relative)
        if not path.is_file():
            missing.append(path)
    if not any(str(shard.get("split")) == split for shard in manifest.get("shards", [])):
        raise ValueError(f"cache manifest has no {split!r} shards: {manifest_path}")
    return missing


def _resolve_historical_split(source_manifest: dict[str, Any]) -> Path:
    """Resolve the historical split by hash, not by a machine-specific absolute path."""

    expected_sha = str(source_manifest.get("split_sha256", ""))
    candidates: list[Path] = []
    raw = source_manifest.get("split_path")
    if raw:
        candidates.append(Path(str(raw)))
    candidates.extend(
        [
            ROOT / "data/splits/eap_train40_v1.json",
            ROOT / "data/splits/eap_train40_v1.yaml",
        ]
    )
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        if expected_sha and sha(resolved) != expected_sha:
            continue
        return resolved
    raise FileNotFoundError(
        "Unable to resolve the historical cache split with its frozen SHA-256; "
        f"expected split_sha256={expected_sha!r}."
    )


def _resolve_source_root(
    *,
    explicit: Path | None,
    manifest_value: object,
    label: str,
) -> Path:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(f"{label} root does not exist: {candidate}")
        return candidate
    if manifest_value:
        candidate = Path(str(manifest_value)).expanduser()
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"{label} root is required to recover missing historical V4 shards. "
        "Pass the corresponding CLI flag."
    )


def _recovery_config(source_manifest: dict[str, Any], *, shard_size: int) -> GarlTTCLHRCacheConfig:
    raw = source_manifest.get("config")
    if not isinstance(raw, dict):
        raise ValueError("historical V4 cache manifest lacks its exact config")
    normalized = dict(raw)
    if "materialize_splits" in normalized:
        normalized["materialize_splits"] = tuple(normalized["materialize_splits"])
    original = GarlTTCLHRCacheConfig(**normalized)
    if not original.store_event_v4_common_roi:
        raise ValueError("historical cache config does not materialize event_v4_common_roi")
    if original.event_v4_bins_per_polarity != 5 or original.roi_size != 128:
        raise ValueError(
            "historical cache does not match the frozen A5/C2F [3,12,128,128] input contract"
        )
    # These fields affect only storage/build scheduling, never the per-row tensor values.
    return replace(
        original,
        materialize_splits=("train",),
        shard_size=shard_size,
        workers=max(1, min(int(original.workers), 4)),
        compression="none",
    )


def _recovery_cache_healthy(manifest_path: Path) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        return not _missing_split_shards(manifest_path, split="train")
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _recover_missing_v4_cache(
    *,
    source_manifest_path: Path,
    protocol: dict[str, Any],
    eap_root: Path | None,
    garlttc_root: Path | None,
    recovery_dir: Path,
    recovery_shard_size: int,
) -> tuple[Path, dict[str, Any]]:
    """Rebuild only the missing train cache from the historical manifest config.

    The original manifest is never overwritten.  Recovery changes only build/storage
    controls (train-only, small shards); every value-affecting preprocessing field is
    inherited from the historical manifest.
    """

    source_manifest = _read_json(source_manifest_path)
    if source_manifest.get("artifact_sha256") and not verify_artifact_hash(source_manifest):
        raise ValueError(f"historical cache manifest signature mismatch: {source_manifest_path}")
    source_file_sha = sha(source_manifest_path)
    split_path = _resolve_historical_split(source_manifest)
    resolved_eap = _resolve_source_root(
        explicit=eap_root, manifest_value=source_manifest.get("eap_root"), label="eAP"
    )
    resolved_garl = _resolve_source_root(
        explicit=garlttc_root,
        manifest_value=source_manifest.get("garlttc_root"),
        label="GarlTTC",
    )
    config = _recovery_config(source_manifest, shard_size=recovery_shard_size)
    manifest_path = recovery_dir / "manifest.json"
    if _recovery_cache_healthy(manifest_path):
        recovery_manifest = _read_json(manifest_path)
    else:
        recovery_dir.mkdir(parents=True, exist_ok=True)
        state_path = recovery_dir / "build_state.json"
        if manifest_path.is_file() and not _recovery_cache_healthy(manifest_path):
            raise RuntimeError(
                f"Recovered V4 cache is itself incomplete: {recovery_dir}. "
                "Delete only this V8 recovery directory and rerun."
            )
        resume = state_path.is_file()
        recovery_manifest = materialize_garlttc_lhr_cache(
            eap_root=resolved_eap,
            garlttc_root=resolved_garl,
            split_path=split_path,
            output_dir=recovery_dir,
            config=config,
            max_samples_per_split=int(protocol["sample_contract"]["rows"]),
            resume=resume,
        )
        if not _recovery_cache_healthy(manifest_path):
            raise RuntimeError("V4 recovery build completed without a healthy train shard set")

    recovery_record = {
        "artifact_type": "scientific_recovery_v8_autopsy_v4_cache_recovery_v1",
        "status": "completed",
        "protocol_artifact_sha256": protocol["artifact_sha256"],
        "historical_manifest": {
            "path": source_manifest_path.resolve().as_posix(),
            "file_sha256": source_file_sha,
            "artifact_sha256": source_manifest.get("artifact_sha256"),
        },
        "recovered_manifest": {
            "path": manifest_path.resolve().as_posix(),
            "file_sha256": sha(manifest_path),
            "artifact_sha256": recovery_manifest.get("artifact_sha256"),
        },
        "historical_split_sha256": source_manifest.get("split_sha256"),
        "recovered_split_sha256": recovery_manifest.get("split_sha256"),
        "semantic_preprocessing_inherited_from_historical_manifest": True,
        "storage_only_overrides": {
            "materialize_splits": ["train"],
            "shard_size": recovery_shard_size,
            "workers": config.workers,
            "compression": "none",
        },
        "expected_frozen_rows": int(protocol["sample_contract"]["rows"]),
        "sealed_splits_opened": False,
    }
    if recovery_record["historical_split_sha256"] != recovery_record["recovered_split_sha256"]:
        raise ValueError("recovered V4 cache split SHA-256 differs from the historical cache")
    write_signed(recovery_dir / "RECOVERY.json", recovery_record)
    return manifest_path, recovery_record


def _resolve_repo_path(raw: object) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _frozen_a5_rows(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load the exact 8,192 V7 OOF identities from the SHA-pinned baseline CSV."""

    source = protocol.get("sources", {}).get("a5_oof_predictions")
    if not isinstance(source, dict):
        raise ValueError("V8 protocol lacks the frozen A5 OOF source")
    path = _resolve_repo_path(source.get("path"))
    expected_sha = str(source.get("sha256", ""))
    if not path.is_file():
        raise FileNotFoundError(f"frozen A5 OOF CSV is missing: {path}")
    actual_sha = sha(path)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(
            f"frozen A5 OOF CSV SHA-256 mismatch: expected={expected_sha} actual={actual_sha}"
        )
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_token", "sequence_id", "track_id", "target_ttc_s", "fold", "seed"}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"A5 OOF CSV lacks columns: {sorted(missing_columns)}")
        for raw in reader:
            token = str(raw["sample_token"])
            if token in rows:
                raise ValueError(f"duplicate token in frozen A5 OOF CSV: {token}")
            rows[token] = {
                "sample_token": token,
                "sequence_id": str(raw["sequence_id"]),
                "track_id": str(raw["track_id"]),
                "target_ttc_s": float(raw["target_ttc_s"]),
                "fold": int(raw["fold"]),
                "seed": int(raw["seed"]),
            }
    expected_rows = int(protocol["sample_contract"]["rows"])
    if len(rows) != expected_rows:
        raise ValueError(f"frozen A5 OOF row count mismatch: {len(rows)} != {expected_rows}")
    if any(item["seed"] != 7 for item in rows.values()):
        raise ValueError("frozen A5 OOF population contains a non-seed7 row")
    return rows


def bucket(value: Decimal) -> tuple[str, Decimal]:
    if Decimal("0") < value <= Decimal("3"):
        return "crucial", Decimal("0.5")
    if Decimal("3") < value <= Decimal("6"):
        return "small", Decimal("0.3")
    if Decimal("6") < value <= Decimal("10"):
        return "large", Decimal("0.1")
    if Decimal("-10") < value <= Decimal("0"):
        return "negative", Decimal("0.1")
    raise ValueError("TTC outside frozen MiD domain")


def sample_weight(protocol: dict[str, Any], *, sequence_id: str, target_ttc: float) -> float:
    label, coefficient = bucket(Decimal(str(target_ttc)))
    by_sequence_bucket = protocol["sample_contract"]["row_count_contract"]["by_sequence_bucket"]
    denominator = int(by_sequence_bucket[sequence_id][label])
    if denominator <= 0:
        raise ValueError(f"non-positive frozen sequence/bucket count for {sequence_id}/{label}")
    return float(coefficient / Decimal(9) / Decimal(denominator))


def _metadata_pass(
    dataset: GarlTTCObjectEventV4Dataset,
    *,
    frozen_rows: dict[str, dict[str, Any]],
    fold_by_sequence: dict[str, int],
) -> dict[int, list[dict[str, Any]]]:
    """Scan one cache shard at a time and retain metadata only, never event tensors."""

    grouped: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    seen: set[str] = set()
    for dataset_index in range(len(dataset)):
        row = dataset[dataset_index]
        token = str(row["sample_token"])
        if token not in frozen_rows:
            continue
        if token in seen:
            raise ValueError(f"duplicate frozen token in V4 cache: {token}")
        sequence = str(row["sequence_id"])
        if sequence not in fold_by_sequence:
            raise ValueError(f"frozen token belongs to unknown sequence: {sequence}")
        frozen = frozen_rows[token]
        track = str(row["track_id"])
        target = float(row["ttc_s"])
        expected_fold = fold_by_sequence[sequence]
        if sequence != frozen["sequence_id"]:
            raise ValueError(f"sequence mismatch for frozen token {token}")
        if track != frozen["track_id"]:
            raise ValueError(f"track mismatch for frozen token {token}")
        if expected_fold != int(frozen["fold"]):
            raise ValueError(f"fold mismatch for frozen token {token}")
        if abs(target - float(frozen["target_ttc_s"])) > 2.0e-5:
            raise ValueError(
                f"target TTC mismatch for frozen token {token}: "
                f"cache={target} baseline={frozen['target_ttc_s']}"
            )
        grouped[expected_fold].append(
            {
                "dataset_index": dataset_index,
                "sample_token": token,
                "sequence_id": sequence,
                "track_id": track,
                "ttc_s": target,
                "garl_delta_t_s": float(row["garl_delta_t_s"]),
            }
        )
        seen.add(token)
        # Do not retain `row`: its event tensor may share the backing shard memory.
        del row
    expected_tokens = set(frozen_rows)
    if seen != expected_tokens:
        missing = sorted(expected_tokens - seen)
        extra = sorted(seen - expected_tokens)
        raise ValueError(
            "V4 cache token identity differs from frozen V8 protocol "
            f"(missing={len(missing)}, extra={len(extra)})"
        )
    return grouped


def _chunk_payload(
    dataset: GarlTTCObjectEventV4Dataset,
    rows: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
    fold: int,
) -> dict[str, Any]:
    records = [dataset[int(item["dataset_index"])] for item in rows]
    # Keep cache payload compact on disk.  Replay validation promotes to float32
    # one shard at a time before GPU inference.
    events = torch.stack(
        [torch.as_tensor(record["event_v4_common_roi"]).detach().cpu().to(torch.float16) for record in records]
    )
    delta_scalar = torch.tensor([float(item["garl_delta_t_s"]) for item in rows], dtype=torch.float32)
    delta = delta_scalar[:, None].expand(-1, 2).clone()
    dt_us = torch.round(delta_scalar * 1_000_000.0).to(torch.int64).clamp_min(1)
    endpoints = torch.stack((torch.zeros_like(dt_us), dt_us, 2 * dt_us), dim=1)
    payload = {
        "events": events,
        "delta_t_s": delta,
        "target_ttc": torch.tensor([float(item["ttc_s"]) for item in rows], dtype=torch.float32),
        "sample_weight": torch.tensor(
            [
                sample_weight(
                    protocol,
                    sequence_id=str(item["sequence_id"]),
                    target_ttc=float(item["ttc_s"]),
                )
                for item in rows
            ],
            dtype=torch.float32,
        ),
        "token_id": [str(item["sample_token"]) for item in rows],
        "sequence_id": [str(item["sequence_id"]) for item in rows],
        "track_id": [str(item["track_id"]) for item in rows],
        "outer_fold": [fold] * len(rows),
        "seed": [7] * len(rows),
        "endpoint_us": endpoints,
    }
    del records
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--event-v4-manifest",
        type=Path,
        default=ROOT / "artifacts/cache/garl_object_event_common_roi_train8192_v1/manifest.json",
    )
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--eap-root", type=Path)
    p.add_argument("--garlttc-root", type=Path)
    p.add_argument(
        "--recovery-cache-dir",
        type=Path,
        default=ROOT / "artifacts/scientific_recovery_v8/cache/autopsy_v4_recovered_v1",
        help="V8-only recovery cache used when historical V4 shards were pruned.",
    )
    p.add_argument(
        "--recovery-shard-size",
        type=int,
        default=32,
        help="Small train-only recovery shards; changes storage only, not tensor values.",
    )
    p.add_argument(
        "--chunk-rows",
        type=int,
        default=8,
        help="Rows per signed replay shard. Eight is conservative for a 12 GiB GPU.",
    )
    args = p.parse_args()
    try:
        if args.chunk_rows < 1 or args.chunk_rows > 64:
            raise ValueError("--chunk-rows must lie in [1,64]")
        if args.recovery_shard_size < 1 or args.recovery_shard_size > 128:
            raise ValueError("--recovery-shard-size must lie in [1,128]")
        protocol = signed(args.protocol)
        source_manifest = args.event_v4_manifest.resolve()
        missing = _missing_split_shards(source_manifest, split="train")
        recovery_record: dict[str, Any] | None = None
        selected_manifest = source_manifest
        if missing:
            selected_manifest, recovery_record = _recover_missing_v4_cache(
                source_manifest_path=source_manifest,
                protocol=protocol,
                eap_root=args.eap_root,
                garlttc_root=args.garlttc_root,
                recovery_dir=args.recovery_cache_dir.resolve(),
                recovery_shard_size=args.recovery_shard_size,
            )
        dataset = GarlTTCObjectEventV4Dataset(str(selected_manifest), splits=("train",))
        frozen_rows = _frozen_a5_rows(protocol)
        fold_by_sequence = {
            str(sequence): int(item["fold"])
            for item in protocol["sample_contract"]["fold_definitions"]
            for sequence in item["dev_sequence_ids"]
        }
        grouped = _metadata_pass(
            dataset,
            frozen_rows=frozen_rows,
            fold_by_sequence=fold_by_sequence,
        )
        expected_counts = protocol["sample_contract"]["row_count_contract"]["by_outer_fold"]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Remove stale production parts from an interrupted prior materialization.
        for stale in args.output_dir.glob("fold*.part*.pt"):
            stale.unlink()

        for fold, rows in grouped.items():
            if len(rows) != int(expected_counts[str(fold)]):
                raise ValueError(f"autopsy fold {fold} row count mismatch")
            # Dataset-index order keeps the second pass shard-local and bounded in RAM.
            rows.sort(key=lambda item: int(item["dataset_index"]))
            parts: list[dict[str, Any]] = []
            for part_index, start in enumerate(range(0, len(rows), args.chunk_rows)):
                current = rows[start : start + args.chunk_rows]
                payload = _chunk_payload(dataset, current, protocol=protocol, fold=fold)
                out = args.output_dir / f"fold{fold}.part{part_index:04d}.pt"
                temporary = out.with_suffix(out.suffix + ".tmp")
                torch.save(payload, temporary)
                os.replace(temporary, out)
                parts.append(
                    {
                        "part": part_index,
                        "path": out.name,
                        "sha256": sha(out),
                        "rows": len(current),
                        "first_token": str(current[0]["sample_token"]),
                        "last_token": str(current[-1]["sample_token"]),
                    }
                )
                del payload
            if sum(int(part["rows"]) for part in parts) != len(rows):
                raise RuntimeError(f"autopsy fold {fold} shard row accounting mismatch")
            write_signed(
                args.output_dir / f"fold{fold}.manifest.json",
                {
                    "artifact_type": "scientific_recovery_v8_autopsy_replay_input_sharded_v2",
                    "protocol_artifact_sha256": protocol["artifact_sha256"],
                    "event_v4_manifest": {
                        "path": selected_manifest.as_posix(),
                        "sha256": sha(selected_manifest),
                    },
                    "historical_event_v4_manifest": {
                        "path": source_manifest.as_posix(),
                        "sha256": sha(source_manifest),
                    },
                    "cache_recovery": (
                        {
                            "used": True,
                            "artifact": (args.recovery_cache_dir.resolve() / "RECOVERY.json").as_posix(),
                            "artifact_sha256": recovery_record["artifact_sha256"],
                        }
                        if recovery_record is not None
                        else {"used": False}
                    ),
                    "outer_fold": fold,
                    "rows": len(rows),
                    "chunk_rows": args.chunk_rows,
                    "parts": parts,
                    "sealed_splits_opened": False,
                },
            )
    except Exception as error:  # fail closed and surface the exact production failure
        p.exit(
            2,
            "V8 autopsy input materialization failed closed: "
            f"{type(error).__name__}: {error}\n",
        )
    print(
        json.dumps(
            {
                "status": "completed_sharded",
                "output_dir": str(args.output_dir),
                "event_v4_manifest": str(selected_manifest),
                "cache_recovered": recovery_record is not None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
