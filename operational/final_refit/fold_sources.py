"""Manifest-bound producer sources for one nested OOF rotation.

The adapters expose only the rotation's ``producer`` rows.  Historical RGB and
event caches remain immutable storage backends; their full population is never a
training membership claim and no checkpoint weights are reused.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from operational.rgb_port.recipe import file_sha256
from operational.rgb_port.train_producers import EventProducerSource

from .recipe import canonical_sha256
from .rgb import AllTrainRGBSource, load_all_train_source

SOURCE_SCHEMA = "final_refit_fold_source_v1"
ROLE_NAMES = ("producer", "refiner", "generation")
ROW_COLUMNS = {
    "rotation",
    "rotation_role",
    "historical_local_ordinal",
    "source_ordinal",
    "sample_token",
    "group_id",
    "historical_role",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _ordered_sha(values: Sequence[object]) -> str:
    return canonical_sha256([str(value) for value in values])


@dataclass(frozen=True)
class FoldSourceInventory:
    """Exact producer membership and forbidden-role evidence for one rotation."""

    rotation: int
    fold_contract_path: str
    fold_contract_sha256: str
    rotation_manifest_path: str
    rotation_manifest_sha256: str
    producer_rows_path: str
    producer_rows_sha256: str
    producer_population: int
    producer_group_ids: tuple[str, ...]
    ordered_sample_token_sha256: str
    ordered_native_ordinal_sha256: str
    historical_role_counts: dict[str, int]
    forbidden: dict[str, dict[str, Any]]
    inventory_sha256: str

    def payload(self) -> dict[str, Any]:
        value = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "inventory_sha256"
        }
        value["producer_group_ids"] = list(self.producer_group_ids)
        return value


def load_fold_inventory(
    fold_contract_path: Path, rotation: int
) -> tuple[FoldSourceInventory, pd.DataFrame]:
    """Validate FOLD_CONTRACT and return producer rows in native ordinal order."""
    contract_path = fold_contract_path.resolve(strict=True)
    contract = _read(contract_path)
    rotations = contract.get("rotations")
    if (
        contract.get("schema") != "final_refit_oof_fold_contract_v1"
        or contract.get("status") != "PASS"
        or contract.get("population") != {"queries": 88_744, "groups": 40}
        or contract.get("generation_coverage") != "EACH_TRAIN40_TOKEN_EXACTLY_ONCE"
        or not isinstance(rotations, list)
        or len(rotations) != 5
        or rotation not in range(5)
    ):
        raise ValueError("FOLD_CONTRACT cannot authorize a producer source")
    manifest_ref = rotations[rotation]
    manifest_path = (contract_path.parent / str(manifest_ref["path"])).resolve(strict=True)
    if file_sha256(manifest_path) != manifest_ref.get("sha256"):
        raise ValueError("Rotation manifest changed")
    manifest = _read(manifest_path)
    if (
        manifest.get("schema") != "final_refit_rotation_roles_v1"
        or manifest.get("rotation") != rotation
        or manifest.get("status") != "VERIFIED_REFERENCE_MANIFESTS"
        or manifest.get("generation_ancestry_excludes_group") is not True
        or set(manifest.get("roles", {})) != set(ROLE_NAMES)
    ):
        raise ValueError("Rotation role manifest is not verified")

    frames: dict[str, pd.DataFrame] = {}
    paths: dict[str, Path] = {}
    for role in ROLE_NAMES:
        reference = manifest["roles"][role]
        path = (manifest_path.parent / str(reference["path"])).resolve(strict=True)
        if file_sha256(path) != reference.get("sha256"):
            raise ValueError(f"Rotation {rotation} {role} rows changed")
        frame = pd.read_parquet(path)
        if (
            not ROW_COLUMNS.issubset(frame.columns)
            or len(frame) != int(reference["queries"])
            or frame["sample_token"].astype(str).duplicated().any()
            or frame["source_ordinal"].astype(int).duplicated().any()
            or set(frame["rotation"].astype(int)) != {rotation}
            or set(frame["rotation_role"].astype(str)) != {role}
        ):
            raise ValueError(f"Rotation {rotation} {role} row identity differs")
        frame = frame.sort_values("source_ordinal", kind="stable").reset_index(drop=True)
        frames[role], paths[role] = frame, path

    group_sets = {
        role: set(frame["group_id"].astype(str).tolist()) for role, frame in frames.items()
    }
    token_sets = {
        role: set(frame["sample_token"].astype(str).tolist()) for role, frame in frames.items()
    }
    ordinal_sets = {
        role: set(frame["source_ordinal"].astype(int).tolist()) for role, frame in frames.items()
    }
    role_pairs = (
        ("producer", "refiner"),
        ("producer", "generation"),
        ("refiner", "generation"),
    )
    for left, right in role_pairs:
        if (
            group_sets[left] & group_sets[right]
            or token_sets[left] & token_sets[right]
            or ordinal_sets[left] & ordinal_sets[right]
        ):
            raise ValueError("Rotation roles overlap by group, token, or native ordinal")
    if tuple(len(group_sets[role]) for role in ROLE_NAMES) != (24, 8, 8):
        raise ValueError("Rotation roles are not the required 24/8/8 groups")
    if set().union(*ordinal_sets.values()) != set(range(88_744)):
        raise ValueError("Rotation roles do not cover native TRAIN40 ordinals exactly")

    producer = frames["producer"]
    payload: dict[str, Any] = {
        "rotation": rotation,
        "fold_contract_path": str(contract_path),
        "fold_contract_sha256": file_sha256(contract_path),
        "rotation_manifest_path": str(manifest_path),
        "rotation_manifest_sha256": file_sha256(manifest_path),
        "producer_rows_path": str(paths["producer"]),
        "producer_rows_sha256": file_sha256(paths["producer"]),
        "producer_population": len(producer),
        "producer_group_ids": sorted(group_sets["producer"]),
        "ordered_sample_token_sha256": _ordered_sha(producer["sample_token"].astype(str).tolist()),
        "ordered_native_ordinal_sha256": _ordered_sha(
            producer["source_ordinal"].astype(int).tolist()
        ),
        "historical_role_counts": {
            role: int((producer["historical_role"].astype(str) == role).sum())
            for role in ("P", "H", "V")
        },
        "forbidden": {
            role: {
                "rows_path": str(paths[role]),
                "rows_sha256": file_sha256(paths[role]),
                "population": len(frames[role]),
                "group_ids": sorted(group_sets[role]),
                "ordered_sample_token_sha256": _ordered_sha(
                    frames[role]["sample_token"].astype(str).tolist()
                ),
                "ordered_native_ordinal_sha256": _ordered_sha(
                    frames[role]["source_ordinal"].astype(int).tolist()
                ),
            }
            for role in ("refiner", "generation")
        },
    }
    inventory = FoldSourceInventory(
        **{
            **payload,
            "producer_group_ids": tuple(payload["producer_group_ids"]),
        },
        inventory_sha256=canonical_sha256(payload),
    )
    return inventory, producer


def _concat_batches(parts: Sequence[tuple[list[int], ObjectEventV4Batch]]) -> ObjectEventV4Batch:
    if not parts:
        raise ValueError("Cannot concatenate an empty fold batch")
    positions = [position for part_positions, _ in parts for position in part_positions]
    if sorted(positions) != list(range(len(positions))):
        raise ValueError("Fold batch parts do not cover requested positions exactly")
    order = torch.as_tensor(np.argsort(np.asarray(positions)), dtype=torch.long)
    batches = [batch for _, batch in parts]

    def tensor(name: str) -> torch.Tensor:
        return torch.cat([getattr(batch, name) for batch in batches], dim=0).index_select(0, order)

    def optional(name: str) -> torch.Tensor | None:
        values = [getattr(batch, name) for batch in batches]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"Fold batch mixes missing and present {name}")
        tensors = [value for value in values if value is not None]
        return torch.cat(tensors, dim=0).index_select(0, order)

    def strings(name: str) -> list[str]:
        values = [value for batch in batches for value in getattr(batch, name)]
        return [values[index] for index in order.tolist()]

    return ObjectEventV4Batch(
        events=tensor("events"),
        delta_t_s=tensor("delta_t_s"),
        observable_motion=tensor("observable_motion"),
        visible_heights_px=tensor("visible_heights_px"),
        target_ttc_s=tensor("target_ttc_s"),
        boxes_xyxy=tensor("boxes_xyxy"),
        common_square_xyxy=tensor("common_square_xyxy"),
        sequence_ids=strings("sequence_ids"),
        sample_tokens=strings("sample_tokens"),
        track_ids=strings("track_ids"),
        sam_teacher_masks=optional("sam_teacher_masks"),
        sam_teacher_mask_valid=optional("sam_teacher_mask_valid"),
        dinov3_relation_targets=optional("dinov3_relation_targets"),
        dinov3_relation_valid=optional("dinov3_relation_valid"),
    )


class FoldRGBProducerSource:
    """Producer-only index view over immutable all-TRAIN RGB storage."""

    population_size: int
    frame_counts: Sequence[int]
    input_span_us: Sequence[int]
    sampling_groups: Sequence[Sequence[int]]

    def __init__(
        self,
        base: AllTrainRGBSource,
        inventory: FoldSourceInventory,
        rows: pd.DataFrame,
    ) -> None:
        self.base, self.inventory = base, inventory
        self._native = rows["source_ordinal"].astype(int).tolist()
        self._tokens = rows["sample_token"].astype(str).tolist()
        for local, row in enumerate(rows.to_dict("records")):
            native = self._native[local]
            observed = base.row(native)
            if (
                str(observed["sample_token"]) != self._tokens[local]
                or str(observed["group_id"]) != str(row["group_id"])
                or str(observed["historical_role"]) != str(row["historical_role"])
                or int(observed["historical_role_ordinal"]) != int(row["historical_local_ordinal"])
            ):
                raise ValueError("Fold RGB row differs from native TRAIN40 lookup")
        self.population_size = len(self._native)
        self.frame_counts = tuple(base.frame_counts[index] for index in self._native)
        self.input_span_us = tuple(base.input_span_us[index] for index in self._native)
        self.batch_keys = tuple(base.batch_keys[index] for index in self._native)
        grouped: dict[tuple[str, int], list[int]] = {}
        for index, row in enumerate(rows.to_dict("records")):
            grouped.setdefault(
                (str(row["historical_role"]), int(row["historical_local_ordinal"]) // 256), []
            ).append(index)
        self.sampling_groups = tuple(tuple(group) for _, group in sorted(grouped.items()))
        identity = {
            "schema": SOURCE_SCHEMA,
            "modality": "rgb",
            "role": "producer_pair",
            "rotation": inventory.rotation,
            "inventory_sha256": inventory.inventory_sha256,
            "population_size": self.population_size,
            "group_ids": list(inventory.producer_group_ids),
            "ordered_sample_token_sha256": inventory.ordered_sample_token_sha256,
            "ordered_native_ordinal_sha256": inventory.ordered_native_ordinal_sha256,
            "teacher_lookup": "original_native_source_ordinal_plus_sample_token",
            "storage_scope": "immutable full TRAIN40 caches; producer membership is manifest-only",
            "base_source_identity_sha256": base.identity["identity_sha256"],
        }
        self.identity: Mapping[str, Any] = {
            **identity,
            "identity_sha256": canonical_sha256(identity),
        }

    def batch(self, indices: Sequence[int], modality: str) -> ObjectEventV4Batch:
        if modality != "rgb" or not indices:
            raise ValueError("Fold RGB source serves nonempty rgb producer batches only")
        groups: dict[tuple[str, int], list[tuple[int, int]]] = {}
        for position, local in enumerate(indices):
            index = int(local)
            if not 0 <= index < self.population_size:
                raise IndexError(index)
            groups.setdefault(self.batch_keys[index], []).append((position, self._native[index]))
        parts = [
            (
                [position for position, _ in selected],
                self.base.batch([native for _, native in selected], "rgb"),
            )
            for _, selected in sorted(groups.items())
        ]
        result = _concat_batches(parts)
        expected = [self._tokens[int(index)] for index in indices]
        if result.sample_tokens != expected:
            raise ValueError("Fold RGB batch escaped producer manifest order")
        return result

    def close(self) -> None:
        self.base.close()


class FoldEventProducerSource:
    """Producer-only view over raw native TRAIN40 event caches by historical role."""

    population_size: int
    frame_counts: Sequence[int]
    input_span_us: Sequence[int]
    sampling_groups: Sequence[Sequence[int]]

    def __init__(
        self,
        sources: Mapping[str, EventProducerSource],
        inventory: FoldSourceInventory,
        rows: pd.DataFrame,
    ) -> None:
        if not sources or set(sources) - {"P", "H", "V"}:
            raise ValueError("Fold event source requires historical P/H/V storage views")
        self.sources, self.inventory = dict(sources), inventory
        records = rows.to_dict("records")
        self._lookup = [
            (str(row["historical_role"]), int(row["historical_local_ordinal"])) for row in records
        ]
        self._tokens = [str(row["sample_token"]) for row in records]
        for index, (role, local) in enumerate(self._lookup):
            source = self.sources.get(role)
            if source is None or str(source._tokens[local]) != self._tokens[index]:
                raise ValueError("Fold event row differs from native historical storage")
        self.population_size = len(records)
        self.frame_counts = (3,) * self.population_size
        self.input_span_us = tuple(
            self.sources[role].input_span_us[local] for role, local in self._lookup
        )
        grouped: dict[tuple[str, int], list[int]] = {}
        for index, (role, local) in enumerate(self._lookup):
            grouped.setdefault((role, local // 256), []).append(index)
        self.sampling_groups = tuple(tuple(group) for _, group in sorted(grouped.items()))
        identity = {
            "schema": SOURCE_SCHEMA,
            "modality": "event",
            "role": "producer_pair",
            "rotation": inventory.rotation,
            "inventory_sha256": inventory.inventory_sha256,
            "population_size": self.population_size,
            "group_ids": list(inventory.producer_group_ids),
            "ordered_sample_token_sha256": inventory.ordered_sample_token_sha256,
            "ordered_native_ordinal_sha256": inventory.ordered_native_ordinal_sha256,
            "storage_scope": "raw native TRAIN40 cache only; no checkpoint weights reused",
            "storage_source_identity": {
                role: dict(source.identity) for role, source in sorted(self.sources.items())
            },
        }
        self.identity: Mapping[str, Any] = {
            **identity,
            "identity_sha256": canonical_sha256(identity),
        }

    def batch(self, indices: Sequence[int], modality: str) -> ObjectEventV4Batch:
        if modality != "event" or not indices:
            raise ValueError("Fold event source serves nonempty event producer batches only")
        groups: dict[str, list[tuple[int, int]]] = {}
        for position, local in enumerate(indices):
            index = int(local)
            if not 0 <= index < self.population_size:
                raise IndexError(index)
            role, historical = self._lookup[index]
            groups.setdefault(role, []).append((position, historical))
        parts = [
            (
                [position for position, _ in selected],
                self.sources[role].batch([local for _, local in selected], "event"),
            )
            for role, selected in sorted(groups.items())
        ]
        result = _concat_batches(parts)
        expected = [self._tokens[int(index)] for index in indices]
        if result.sample_tokens != expected:
            raise ValueError("Fold event batch escaped producer manifest order")
        return result

    def close(self) -> None:
        for source in self.sources.values():
            source.close()


def load_fold_rgb_source(
    source_descriptor: Path,
    teacher_manifest: Path,
    fold_contract: Path,
    rotation: int,
) -> FoldRGBProducerSource:
    """Construct an exact producer-fold RGB view with original teacher ordinals."""
    inventory, rows = load_fold_inventory(fold_contract, rotation)
    return FoldRGBProducerSource(
        load_all_train_source(source_descriptor, teacher_manifest), inventory, rows
    )


def load_fold_event_source(
    fold_contract: Path,
    rotation: int,
    event_source_root: Path,
) -> FoldEventProducerSource:
    """Construct an exact producer-fold event view over audited raw storage."""
    inventory, rows = load_fold_inventory(fold_contract, rotation)
    contract = _read(fold_contract.resolve(strict=True))
    sources = {
        role: EventProducerSource(
            Path(str(contract["historical_sources"][role]["manifest_path"])), event_source_root
        )
        for role in sorted(set(rows["historical_role"].astype(str)))
    }
    return FoldEventProducerSource(sources, inventory, rows)


__all__ = [
    "FoldEventProducerSource",
    "FoldRGBProducerSource",
    "FoldSourceInventory",
    "SOURCE_SCHEMA",
    "load_fold_event_source",
    "load_fold_inventory",
    "load_fold_rgb_source",
]
