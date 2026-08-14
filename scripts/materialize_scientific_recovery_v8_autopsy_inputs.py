#!/usr/bin/env python
"""Materialize exact A5/C2F V8-A replay payloads from the historical 12-channel V4 cache."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
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


def weights(records: list[dict[str, Any]]) -> dict[str, float]:
    counts = Counter((str(r["sequence_id"]), bucket(Decimal(str(r["ttc_s"])))[0]) for r in records)
    result: dict[str, float] = {}
    for row in records:
        target = Decimal(str(row["ttc_s"]))
        label, coefficient = bucket(target)
        value = coefficient / Decimal(9) / Decimal(counts[(str(row["sequence_id"]), label)])
        result[str(row["sample_token"])] = float(value)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--event-v4-manifest", type=Path,
        default=ROOT / "artifacts/cache/garl_object_event_common_roi_train8192_v1/manifest.json",
    )
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    try:
        protocol = signed(args.protocol)
        dataset = GarlTTCObjectEventV4Dataset(str(args.event_v4_manifest), splits=("train",))
        expected_tokens = set(protocol["sample_contract"]["token_order"])
        fold_by_sequence = {
            str(sequence): int(item["fold"])
            for item in protocol["sample_contract"]["fold_definitions"]
            for sequence in item["dev_sequence_ids"]
        }
        records = [dataset[i] for i in range(len(dataset))]
        selected = [r for r in records if str(r["sample_token"]) in expected_tokens]
        if len(selected) != int(protocol["sample_contract"]["rows"]):
            raise ValueError("V4 cache does not contain the exact frozen 8192-token population")
        if {str(r["sample_token"]) for r in selected} != expected_tokens:
            raise ValueError("V4 cache token identity differs from frozen V8 protocol")
        sample_weights = weights(selected)
        grouped: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
        for row in selected:
            grouped[fold_by_sequence[str(row["sequence_id"])]].append(row)
        expected_counts = protocol["sample_contract"]["row_count_contract"]["by_outer_fold"]
        for fold, rows in grouped.items():
            rows.sort(key=lambda r: str(r["sample_token"]))
            if len(rows) != int(expected_counts[str(fold)]):
                raise ValueError(f"autopsy fold {fold} row count mismatch")
            events = torch.stack([torch.as_tensor(r["event_v4_common_roi"], dtype=torch.float32) for r in rows])
            delta_scalar = torch.tensor([float(r["garl_delta_t_s"]) for r in rows], dtype=torch.float32)
            delta = delta_scalar[:, None].expand(-1, 2).clone()
            dt_us = torch.round(delta_scalar * 1_000_000.0).to(torch.int64).clamp_min(1)
            endpoints = torch.stack((torch.zeros_like(dt_us), dt_us, 2 * dt_us), dim=1)
            payload = {
                "events": events,
                "delta_t_s": delta,
                "target_ttc": torch.tensor([float(r["ttc_s"]) for r in rows], dtype=torch.float32),
                "sample_weight": torch.tensor([sample_weights[str(r["sample_token"])] for r in rows], dtype=torch.float32),
                "token_id": [str(r["sample_token"]) for r in rows],
                "sequence_id": [str(r["sequence_id"]) for r in rows],
                "track_id": [str(r["track_id"]) for r in rows],
                "outer_fold": [fold] * len(rows),
                "seed": [7] * len(rows),
                "endpoint_us": endpoints,
            }
            out = args.output_dir / f"fold{fold}.pt"
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, out)
            write_signed(
                args.output_dir / f"fold{fold}.manifest.json",
                {
                    "artifact_type": "scientific_recovery_v8_autopsy_replay_input_v1",
                    "protocol_artifact_sha256": protocol["artifact_sha256"],
                    "event_v4_manifest": {"path": args.event_v4_manifest.as_posix(), "sha256": sha(args.event_v4_manifest)},
                    "input_sha256": sha(out),
                    "outer_fold": fold,
                    "rows": len(rows),
                    "sealed_splits_opened": False,
                },
            )
    except (OSError, ValueError, KeyError) as error:
        p.exit(2, f"V8 autopsy input materialization failed closed: {type(error).__name__}: {error}\n")
    print(json.dumps({"status": "completed", "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
