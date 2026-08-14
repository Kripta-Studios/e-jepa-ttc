"""JEPA attribution data adapters for signed V8 caches and the frozen A5 V4 cache.

This module keeps the label-free pretraining boundary identical while allowing the
A5 fallback to reuse the historical 12-channel V4 cache without duplicating ~10 GiB
of tensors on disk.  Targets and MiD weights are exposed only by the downstream
collate path; pretraining consumes representations/identities only.
"""
from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash
from e_jepa_ttc.data.object_event_v4 import GarlTTCObjectEventV4Dataset
from e_jepa_ttc.data.scientific_recovery_v8_cache import ScientificRecoveryV8CacheDataset


def _bucket(value: Decimal) -> tuple[str, Decimal]:
    if Decimal("0") < value <= Decimal("3"):
        return "crucial", Decimal("0.5")
    if Decimal("3") < value <= Decimal("6"):
        return "small", Decimal("0.3")
    if Decimal("6") < value <= Decimal("10"):
        return "large", Decimal("0.1")
    if Decimal("-10") < value <= Decimal("0"):
        return "negative", Decimal("0.1")
    raise ValueError("TTC outside frozen MiD domain")


class HistoricalV4JEPAData(Dataset[dict[str, Any]]):
    """Expose the frozen 12-channel A5 V4 cache through the generic V8 row schema."""

    def __init__(self, manifest_path: Path, protocol_path: Path) -> None:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if not isinstance(protocol, dict) or not verify_artifact_hash(protocol):
            raise ValueError("V8 protocol must be a signed artifact")
        source = GarlTTCObjectEventV4Dataset(str(manifest_path), splits=("train",))
        token_order = tuple(str(x) for x in protocol["sample_contract"]["token_order"])
        token_set = set(token_order)
        fold_by_sequence = {
            str(sequence): int(item["fold"])
            for item in protocol["sample_contract"]["fold_definitions"]
            for sequence in item["dev_sequence_ids"]
        }
        selected: list[dict[str, Any]] = []
        for index in range(len(source)):
            row = source[index]
            if str(row["sample_token"]) in token_set:
                selected.append(row)
        if len(selected) != len(token_order) or {str(x["sample_token"]) for x in selected} != token_set:
            raise ValueError("historical V4 cache does not match the frozen V8 token universe")
        counts = Counter(
            (str(row["sequence_id"]), _bucket(Decimal(str(row["ttc_s"])))[0])
            for row in selected
        )
        weights: dict[str, float] = {}
        for row in selected:
            label, coefficient = _bucket(Decimal(str(row["ttc_s"])))
            weights[str(row["sample_token"])] = float(
                coefficient / Decimal(9) / Decimal(counts[(str(row["sequence_id"]), label)])
            )
        by_token = {str(row["sample_token"]): row for row in selected}
        self.rows = [by_token[token] for token in token_order]
        self.fold_by_sequence = fold_by_sequence
        self.weights = weights
        shape = source.manifest.get("object_lhr_extension", {}).get("event_v4_common_roi_shape")
        if not isinstance(shape, list) or len(shape) < 4:
            raise ValueError("historical V4 manifest lacks event_v4_common_roi_shape")
        self.shape = tuple(int(x) for x in shape[:4])
        self.manifest = {
            "artifact_type": "scientific_recovery_v8_historical_v4_adapter_v1",
            "train_only": True,
            "provenance": {"raw_materialization": True, "adapter_without_tensor_copy": True},
            "source_manifest": str(manifest_path),
            "protocol_artifact_sha256": protocol["artifact_sha256"],
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        dt_us = max(1, int(round(float(row["garl_delta_t_s"]) * 1_000_000.0)))
        token = str(row["sample_token"])
        sequence = str(row["sequence_id"])
        representation = torch.as_tensor(row["event_v4_common_roi"], dtype=torch.float32)
        return {
            "representation": representation,
            "endpoint_us": torch.tensor([0, dt_us, 2 * dt_us], dtype=torch.int64),
            "sample_token": token,
            "sequence_id": sequence,
            "track_id": str(row["track_id"]),
            "target_ttc": float(row["ttc_s"]),
            "sample_weight": self.weights[token],
            "outer_fold": self.fold_by_sequence[sequence],
            "row_identity": (
                token,
                sequence,
                str(row["track_id"]),
                str(self.fold_by_sequence[sequence]),
            ),
            "common_roi_xyxy": torch.as_tensor(
                row["event_v4_common_square_xyxy"], dtype=torch.float32
            ),
            "event_count": int(torch.count_nonzero(representation).item()),
        }


def open_jepa_dataset(
    *, cache_manifest: Path, protocol_path: Path | None = None, allow_fixture_cache: bool = False
) -> Dataset[dict[str, Any]]:
    """Open a production V8 cache or the historical A5 V4 cache through one contract."""
    try:
        manifest = json.loads(cache_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JEPA cache manifest: {cache_manifest}") from error
    if isinstance(manifest, dict) and verify_artifact_hash(manifest):
        artifact_type = str(manifest.get("artifact_type", ""))
        if artifact_type.startswith("scientific_recovery_v8_"):
            if manifest.get("train_only") is not True:
                raise ValueError("JEPA requires a signed train-only cache")
            if manifest.get("provenance", {}).get("raw_materialization") is not True and not allow_fixture_cache:
                raise ValueError("fixture V8 cache is forbidden outside explicit test execution")
            return ScientificRecoveryV8CacheDataset(cache_manifest)
    if protocol_path is None:
        raise ValueError("historical V4 JEPA adapter requires --protocol")
    return HistoricalV4JEPAData(cache_manifest, protocol_path)


__all__ = ["HistoricalV4JEPAData", "open_jepa_dataset"]
