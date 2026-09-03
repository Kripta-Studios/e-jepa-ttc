"""Read-only, protocol-bound adapter for the E-Clock train8192 cache."""

from __future__ import annotations

import gc
import json
import math
import time
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
import pandas as pd
import psutil
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash, verify_artifact_hash
from e_jepa_ttc.evaluation.collision_clock_protocol import canonical_records_hash
from e_jepa_ttc.evaluation.garl_ttc_protocol import BUCKETS, PAPER_MID_WEIGHTS

_EXPECTED_ROW_FIELDS = {
    "category",
    "category_index",
    "category_valid",
    "endpoint_delta_error_s",
    "endpoint_first_timestamp_us",
    "endpoint_second_timestamp_us",
    "event_v4_boxes_xyxy",
    "event_v4_common_roi",
    "event_v4_common_square_xyxy",
    "event_v4_precontext_source",
    "event_v4_precontext_valid",
    "event_v4_t0_box_is_proxy",
    "garl_delta_t_s",
    "garl_visible_heights_px",
    "geometry_v2_target",
    "geometry_v2_valid",
    "jepa_context_motion",
    "jepa_pair_valid",
    "observable_motion",
    "precontext_motion_valid",
    "public_track_id",
    "sample_token",
    "sampling_group",
    "sequence_id",
    "timestamp_us",
    "track_id",
    "ttc_label_index",
    "ttc_label_source",
    "ttc_label_timestamp_us",
    "ttc_s",
}

CacheMode = Literal["direct", "shard_lru", "fold_ram", "auto"]


@dataclass(frozen=True)
class CollisionClockSampleLocator:
    """A verified lightweight pointer to one immutable cache row."""

    shard_path: str
    row_index: int
    sample_token: str
    sequence_id: str
    track_id: str
    outer_fold: int
    target_ttc_s: float
    sample_weight: float


@dataclass(frozen=True)
class CollisionClockOuterTrainBatch:
    """Only batch type accepted by the fixed-budget trainer."""

    inputs: torch.Tensor
    delta_t_s: torch.Tensor
    target_ttc_seconds: torch.Tensor
    sample_tokens: tuple[str, ...]


@dataclass(frozen=True)
class CollisionClockOuterDevBatch:
    """Evaluation-only batch type carrying bookkeeping outside model inputs."""

    inputs: torch.Tensor
    delta_t_s: torch.Tensor
    target_ttc_seconds: torch.Tensor
    sample_tokens: tuple[str, ...]
    sequence_ids: tuple[str, ...]
    track_ids: tuple[str, ...]
    outer_fold: int
    sample_weights: torch.Tensor


@dataclass(frozen=True)
class CollisionClockOuterTrainView:
    fold: int
    locators: tuple[CollisionClockSampleLocator, ...]
    ordered_token_identity_sha256: str


@dataclass(frozen=True)
class CollisionClockOuterDevView:
    fold: int
    locators: tuple[CollisionClockSampleLocator, ...]
    ordered_token_identity_sha256: str


class CollisionClockOuterTrainSequence(Sequence[CollisionClockOuterTrainBatch]):
    """Lazy deterministic schedule that can materialize outer-train batches only."""

    def __init__(
        self,
        adapter: CollisionClockTrain8192Cache,
        view: CollisionClockOuterTrainView,
        *,
        batch_size: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._adapter = adapter
        self._view = view
        self._batch_size = batch_size

    def __len__(self) -> int:
        return math.ceil(len(self._view.locators) / self._batch_size)

    def __getitem__(self, index: int) -> CollisionClockOuterTrainBatch:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self._batch_size
        locators = self._view.locators[start : start + self._batch_size]
        inputs, delta, target = self._adapter._materialize(locators)
        return CollisionClockOuterTrainBatch(
            inputs, delta, target, tuple(item.sample_token for item in locators)
        )


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"cache shard path is not safely relative: {value}")
    return path


def _bucket_name(target_ttc_s: float) -> str:
    for name, lower, upper in BUCKETS:
        if lower < target_ttc_s <= upper:
            return name
    raise ValueError("cache target lies outside the signed TTC buckets")


def _bound_reference_path(
    reference: Mapping[str, Any],
    source_root: Path,
    *,
    family: str,
    semantic_identity: str,
) -> Path:
    families = reference.get("families")
    if not isinstance(families, Mapping) or not isinstance(families.get(family), Mapping):
        raise ValueError(f"canonical supervision family is missing: {family}")
    records = families[family].get("physical_references")
    if not isinstance(records, list):
        raise ValueError(f"canonical supervision references are missing: {family}")
    matches = [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("semantic_identity") == semantic_identity
    ]
    if len(matches) != 1:
        raise ValueError(f"canonical supervision reference is ambiguous: {semantic_identity}")
    record = matches[0]
    relative = _safe_relative_path(str(record["path"]))
    path = source_root.resolve(strict=True).joinpath(*relative.parts)
    if not path.is_file() or path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"canonical supervision file identity mismatch: {relative}")
    if compute_file_hash(str(path)) != record["file_sha256"]:
        raise ValueError(f"canonical supervision SHA mismatch: {relative}")
    return path


def load_canonical_supervision(reference: Mapping[str, Any], source_root: Path) -> pd.DataFrame:
    """Load only signed token, target and weight columns from authorized OOF evidence."""

    target_path = _bound_reference_path(
        reference,
        source_root,
        family="official_a5_oof",
        semantic_identity="official_a5_oof_csv",
    )
    weight_path = _bound_reference_path(
        reference,
        source_root,
        family="prospective_router_r",
        semantic_identity="v8_nested_router_seed7_oof_csv",
    )
    targets = pd.read_csv(target_path, usecols=["sample_token", "target_ttc_s"])
    weights = pd.read_csv(weight_path, usecols=["token_id", "sample_weight"]).rename(
        columns={"token_id": "sample_token"}
    )
    result = targets.merge(weights, on="sample_token", how="outer", validate="one_to_one")
    if (
        len(result) != 8192
        or result["sample_token"].duplicated().any()
        or not np.isfinite(
            result[["target_ttc_s", "sample_weight"]].to_numpy(dtype=np.float64)
        ).all()
    ):
        raise ValueError("canonical supervision row universe is invalid")
    official = reference["families"]["official_a5_oof"]
    if (
        canonical_records_hash(result, ("sample_token", "target_ttc_s"))
        != official["target_sha256"]
    ):
        raise ValueError("canonical supervision target identity mismatch")
    if (
        canonical_records_hash(result, ("sample_token", "sample_weight"))
        != official["sample_weight_sha256"]
    ):
        raise ValueError("canonical supervision weight identity mismatch")
    return result


class CollisionClockTrain8192Cache:
    """Verify and expose only the signed train cache through typed fold views."""

    def __init__(
        self,
        cache_root: Path,
        protocol: Mapping[str, Any],
        *,
        cache_mode: CacheMode = "direct",
        lru_capacity: int = 2,
        fold_ram_margin_bytes: int = 6 * 1024**3,
        canonical_supervision: pd.DataFrame | None = None,
    ) -> None:
        if cache_mode not in {"direct", "shard_lru", "fold_ram", "auto"}:
            raise ValueError(f"unsupported cache mode: {cache_mode}")
        if lru_capacity < 1 or lru_capacity > 4:
            raise ValueError("shard LRU capacity must be between 1 and 4")
        self.cache_root = cache_root.resolve(strict=True)
        self.protocol = protocol
        self.requested_cache_mode: CacheMode = cache_mode
        self.cache_mode: CacheMode = cache_mode
        self.lru_capacity = lru_capacity
        self.fold_ram_margin_bytes = fold_ram_margin_bytes
        self.canonical_supervision = (
            canonical_supervision.copy() if canonical_supervision is not None else None
        )
        self.manifest_path = self.cache_root / "manifest.json"
        binding = protocol.get("cache_binding")
        if not isinstance(binding, Mapping):
            raise ValueError("protocol cache binding is missing")
        self.binding = binding
        if not self.manifest_path.is_file():
            raise ValueError("cache manifest is missing")
        if self.manifest_path.stat().st_size != int(binding["bytes"]):
            raise ValueError("cache manifest byte count mismatch")
        if compute_file_hash(str(self.manifest_path)) != binding["file_sha256"]:
            raise ValueError("cache manifest physical SHA mismatch")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not verify_artifact_hash(manifest):
            raise ValueError("cache manifest artifact signature mismatch")
        if manifest.get("artifact_sha256") != binding["artifact_sha256"]:
            raise ValueError("cache manifest artifact SHA mismatch")
        if manifest.get("schema_version") != binding["schema_version"]:
            raise ValueError("cache schema version mismatch")
        if manifest.get("input_schema", {}).get("version") != binding["preprocessing_version"]:
            raise ValueError("cache preprocessing version mismatch")
        self.manifest = manifest
        signed_shards = binding.get("train_shards")
        if not isinstance(signed_shards, list) or len(signed_shards) != 32:
            raise ValueError("protocol must bind exactly 32 train shards")
        self.signed_shards = tuple(signed_shards)
        self._locators: tuple[CollisionClockSampleLocator, ...] | None = None
        self._lru: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._fold_inputs: torch.Tensor | None = None
        self._fold_delta: torch.Tensor | None = None
        self._fold_target: torch.Tensor | None = None
        self._fold_token_to_row: dict[str, int] = {}
        self._stats: dict[str, int | float | str] = {
            "requested_mode": cache_mode,
            "selected_mode": cache_mode,
            "shard_loads": 0,
            "shard_cache_hits": 0,
            "shard_cache_misses": 0,
            "shard_evictions": 0,
            "bytes_read": 0,
            "load_seconds": 0.0,
            "staging_seconds": 0.0,
            "staged_rows": 0,
            "staged_bytes": 0,
        }

    def engineering_stats(self) -> dict[str, int | float | str]:
        """Return a snapshot of engineering-only cache telemetry."""

        return dict(self._stats)

    def _load_shard(self, relative: str) -> list[dict[str, Any]]:
        if self.cache_mode == "shard_lru" and relative in self._lru:
            self._stats["shard_cache_hits"] = int(self._stats["shard_cache_hits"]) + 1
            shard = self._lru.pop(relative)
            self._lru[relative] = shard
            return shard
        self._stats["shard_cache_misses"] = int(self._stats["shard_cache_misses"]) + 1
        path = self.cache_root.joinpath(*PurePosixPath(relative).parts)
        started = time.perf_counter()
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        elapsed = time.perf_counter() - started
        if not isinstance(loaded, list):
            raise ValueError(f"cache shard payload is not a row list: {relative}")
        self._stats["shard_loads"] = int(self._stats["shard_loads"]) + 1
        self._stats["bytes_read"] = int(self._stats["bytes_read"]) + path.stat().st_size
        self._stats["load_seconds"] = float(self._stats["load_seconds"]) + elapsed
        if self.cache_mode == "shard_lru":
            self._lru[relative] = loaded
            while len(self._lru) > self.lru_capacity:
                self._lru.popitem(last=False)
                self._stats["shard_evictions"] = int(self._stats["shard_evictions"]) + 1
        return loaded

    def select_mode_for_train_view(self, view: CollisionClockOuterTrainView) -> CacheMode:
        """Select auto mode using only tensor bytes and available host memory."""

        if self.requested_cache_mode != "auto":
            self.cache_mode = self.requested_cache_mode
            self._stats["selected_mode"] = self.cache_mode
            return self.cache_mode
        shape = tuple(
            int(value)
            for value in self.manifest["object_lhr_extension"]["event_v4_common_roi_shape"]
        )
        row_bytes = math.prod(shape) * np.dtype(np.float32).itemsize
        projected = len(view.locators) * row_bytes + max(
            int(record["bytes"]) for record in self.signed_shards
        )
        available = int(psutil.virtual_memory().available)
        self.cache_mode = (
            "fold_ram" if available - projected >= self.fold_ram_margin_bytes else "shard_lru"
        )
        self._stats["selected_mode"] = self.cache_mode
        self._stats["auto_available_bytes"] = available
        self._stats["auto_projected_bytes"] = projected
        self._stats["auto_required_margin_bytes"] = self.fold_ram_margin_bytes
        return self.cache_mode

    def release_staged_view(self) -> None:
        """Release a staged fold before any train/dev role transition."""

        self._fold_inputs = None
        self._fold_delta = None
        self._fold_target = None
        self._fold_token_to_row.clear()
        self._lru.clear()
        gc.collect()

    def stage_view(self, view: CollisionClockOuterTrainView | CollisionClockOuterDevView) -> None:
        """Preallocate and fill one fold view without list+stack duplication."""

        self.release_staged_view()
        if self.cache_mode != "fold_ram":
            return
        shape = tuple(
            int(value)
            for value in self.manifest["object_lhr_extension"]["event_v4_common_roi_shape"]
        )
        count = len(view.locators)
        inputs = torch.empty((count, *shape), dtype=torch.float32)
        delta = torch.empty((count, 2), dtype=torch.float32)
        target = torch.empty((count,), dtype=torch.float64)
        by_shard: dict[str, list[tuple[int, CollisionClockSampleLocator]]] = {}
        for output_index, locator in enumerate(view.locators):
            by_shard.setdefault(locator.shard_path, []).append((output_index, locator))
        started = time.perf_counter()
        previous_mode = self.cache_mode
        try:
            self.cache_mode = "direct"
            for relative, requested in by_shard.items():
                shard = self._load_shard(relative)
                for output_index, locator in requested:
                    row = shard[locator.row_index]
                    validated = self._validate_row(
                        row, shard_path=locator.shard_path, row_index=locator.row_index
                    )
                    if validated["sample_token"] != locator.sample_token:
                        raise ValueError("cache row changed during fold RAM staging")
                    inputs[output_index].copy_(
                        torch.from_numpy(np.asarray(row["event_v4_common_roi"]))
                    )
                    dt = float(np.asarray(row["garl_delta_t_s"]).item())
                    delta[output_index] = dt
                    target[output_index] = locator.target_ttc_s
                del shard
                gc.collect()
        finally:
            self.cache_mode = previous_mode
        self._fold_inputs = inputs
        self._fold_delta = delta
        self._fold_target = target
        self._fold_token_to_row = {
            locator.sample_token: index for index, locator in enumerate(view.locators)
        }
        self._stats["staging_seconds"] = float(self._stats["staging_seconds"]) + (
            time.perf_counter() - started
        )
        self._stats["staged_rows"] = count
        self._stats["staged_bytes"] = (
            inputs.numel() * inputs.element_size()
            + delta.numel() * delta.element_size()
            + target.numel() * target.element_size()
        )

    def _verify_shard_file(self, record: Mapping[str, Any]) -> Path:
        relative = _safe_relative_path(str(record["path"]))
        path = self.cache_root.joinpath(*relative.parts)
        if not path.is_file():
            raise ValueError(f"cache shard is missing: {relative}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"cache shard byte count mismatch: {relative}")
        if compute_file_hash(str(path)) != record["file_sha256"]:
            raise ValueError(f"cache shard SHA mismatch: {relative}")
        return path

    def _validate_row(self, row: object, *, shard_path: str, row_index: int) -> dict[str, Any]:
        if not isinstance(row, dict) or set(row) != _EXPECTED_ROW_FIELDS:
            raise ValueError(f"cache row schema mismatch: {shard_path}:{row_index}")
        events = np.asarray(row["event_v4_common_roi"])
        declared_shape = tuple(
            int(value)
            for value in self.manifest["object_lhr_extension"]["event_v4_common_roi_shape"]
        )
        if events.shape != declared_shape or events.dtype != np.dtype(np.float32):
            raise ValueError("cache event tensor shape/dtype mismatch")
        if not np.isfinite(events).all():
            raise ValueError("cache event tensor is non-finite")
        first = int(np.asarray(row["endpoint_first_timestamp_us"]).item())
        second = int(np.asarray(row["endpoint_second_timestamp_us"]).item())
        label = int(np.asarray(row["ttc_label_timestamp_us"]).item())
        anchor = int(np.asarray(row["timestamp_us"]).item())
        interval = second - first
        t0 = first - interval
        if interval <= 0 or not t0 < first < second:
            raise ValueError("cache endpoints do not satisfy t0<t1<t2")
        if label != second or anchor != second:
            raise ValueError("cache TTC label is not anchored at t2")
        delta_t_s = float(np.asarray(row["garl_delta_t_s"], dtype=np.float64).item())
        expected_delta = interval / 1.0e6
        tolerance = float(self.manifest["config"]["delta_t_tolerance_s"])
        if not np.isfinite(delta_t_s) or abs(delta_t_s - expected_delta) > min(tolerance, 1e-6):
            raise ValueError("cache delta_t disagrees with endpoint timestamps")
        token = str(row["sample_token"])
        sequence = str(row["sequence_id"])
        track = str(row["track_id"])
        target = float(np.asarray(row["ttc_s"], dtype=np.float64).item())
        if not token or not sequence or not track or not np.isfinite(target):
            raise ValueError("cache identity/target field is invalid")
        if sequence not in self.protocol["canonical_sequence_to_fold"]:
            raise ValueError("cache row has a non-canonical sequence")
        return {
            "sample_token": token,
            "sequence_id": sequence,
            "track_id": track,
            "target_ttc_s": target,
            "outer_fold": int(self.protocol["canonical_sequence_to_fold"][sequence]),
        }

    def verify_and_index(self) -> tuple[CollisionClockSampleLocator, ...]:
        """Verify all physical shards and canonical rows without running a model."""

        if self._locators is not None:
            return self._locators
        expected_paths = {str(record["path"]) for record in self.signed_shards}
        physical_paths = {
            path.relative_to(self.cache_root).as_posix()
            for path in (self.cache_root / "train").glob("*.pt")
            if path.is_file()
        }
        if physical_paths != expected_paths:
            raise ValueError("physical train shard universe differs from signed protocol")
        raw_rows: list[dict[str, Any]] = []
        row_locations: list[tuple[str, int]] = []
        for record in self.signed_shards:
            path = self._verify_shard_file(record)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(loaded, list) or len(loaded) != int(record["row_count"]):
                raise ValueError(f"cache shard row count mismatch: {record['path']}")
            for index, row in enumerate(loaded):
                raw_rows.append(
                    self._validate_row(row, shard_path=str(record["path"]), row_index=index)
                )
                row_locations.append((str(record["path"]), index))
            del loaded
            gc.collect()
        frame = pd.DataFrame(raw_rows)
        expected_count = int(self.protocol["production_row_count"])
        if len(frame) != expected_count or frame["sample_token"].duplicated().any():
            raise ValueError("cache token count/uniqueness mismatch")
        counts = self.protocol["canonical_bucket_counts_by_sequence"]
        if self.canonical_supervision is None:
            weights = []
            targets = [float(row["target_ttc_s"]) for row in raw_rows]
            for row in raw_rows:
                bucket = _bucket_name(float(row["target_ttc_s"]))
                count = int(counts[row["sequence_id"]][bucket])
                weights.append(PAPER_MID_WEIGHTS[bucket] / len(counts) / count)
        else:
            supervision = self.canonical_supervision.set_index("sample_token")
            if set(supervision.index.astype(str)) != set(frame["sample_token"].astype(str)):
                raise ValueError("cache and canonical supervision token universes differ")
            ordered = supervision.loc[frame["sample_token"].astype(str)]
            targets = ordered["target_ttc_s"].to_numpy(dtype=np.float64).tolist()
            cached_targets = frame["target_ttc_s"].to_numpy(dtype=np.float64)
            if not np.allclose(cached_targets, targets, rtol=0, atol=1e-5):
                raise ValueError("cache target differs materially from canonical supervision")
            weights = ordered["sample_weight"].to_numpy(dtype=np.float64).tolist()
            frame["target_ttc_s"] = np.asarray(targets, dtype=np.float64)
        frame["sample_weight"] = np.asarray(weights, dtype=np.float64)
        observed = {
            "token_identity_sha256": canonical_records_hash(
                frame, ("sample_token", "sequence_id", "track_id")
            ),
            "target_sha256": canonical_records_hash(frame, ("sample_token", "target_ttc_s")),
            "fold_assignment_sha256": canonical_records_hash(
                frame, ("sample_token", "sequence_id", "outer_fold")
            ),
            "sample_weight_sha256": canonical_records_hash(
                frame, ("sample_token", "sample_weight")
            ),
        }
        if observed != self.protocol["canonical_hashes"]:
            raise ValueError("cache rows do not match the canonical signed identities")
        self._locators = tuple(
            CollisionClockSampleLocator(
                shard_path=location[0],
                row_index=location[1],
                sample_token=str(row["sample_token"]),
                sequence_id=str(row["sequence_id"]),
                track_id=str(row["track_id"]),
                outer_fold=int(row["outer_fold"]),
                target_ttc_s=float(target),
                sample_weight=float(weight),
            )
            for row, location, target, weight in zip(
                raw_rows, row_locations, targets, weights, strict=True
            )
        )
        return self._locators

    @staticmethod
    def _subset_sha(locators: Sequence[CollisionClockSampleLocator]) -> str:
        frame = pd.DataFrame(
            {
                "sample_token": [item.sample_token for item in locators],
                "sequence_id": [item.sequence_id for item in locators],
                "track_id": [item.track_id for item in locators],
            }
        )
        return canonical_records_hash(frame, ("sample_token", "sequence_id", "track_id"))

    def outer_views(
        self, fold: int
    ) -> tuple[CollisionClockOuterTrainView, CollisionClockOuterDevView]:
        """Build disjoint train/dev views solely from the canonical fold mapping."""

        if fold not in (0, 1, 2):
            raise ValueError("outer fold must be 0, 1 or 2")
        locators = self.verify_and_index()
        train = tuple(item for item in locators if item.outer_fold != fold)
        dev = tuple(item for item in locators if item.outer_fold == fold)
        if (
            not train
            or not dev
            or {item.sample_token for item in train} & {item.sample_token for item in dev}
        ):
            raise ValueError("outer-train/outer-dev split is empty or overlapping")
        return (
            CollisionClockOuterTrainView(fold, train, self._subset_sha(train)),
            CollisionClockOuterDevView(fold, dev, self._subset_sha(dev)),
        )

    def _materialize(
        self, locators: Sequence[CollisionClockSampleLocator]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cache_mode == "fold_ram":
            if self._fold_inputs is None or self._fold_delta is None or self._fold_target is None:
                raise RuntimeError("fold_ram materialization requires an explicitly staged view")
            try:
                indices = [self._fold_token_to_row[item.sample_token] for item in locators]
            except KeyError as error:
                raise ValueError("requested row is outside the staged fold view") from error
            selected = torch.tensor(indices, dtype=torch.int64)
            return (
                self._fold_inputs.index_select(0, selected),
                self._fold_delta.index_select(0, selected),
                self._fold_target.index_select(0, selected),
            )
        events: list[torch.Tensor] = []
        deltas: list[tuple[float, float]] = []
        targets: list[float] = []
        by_shard: dict[str, list[tuple[int, CollisionClockSampleLocator]]] = {}
        for output_index, locator in enumerate(locators):
            by_shard.setdefault(locator.shard_path, []).append((output_index, locator))
        output: list[tuple[torch.Tensor, tuple[float, float], float] | None] = [None] * len(
            locators
        )
        for relative, requested in by_shard.items():
            shard = self._load_shard(relative)
            for output_index, locator in requested:
                row = shard[locator.row_index]
                validated = self._validate_row(
                    row, shard_path=locator.shard_path, row_index=locator.row_index
                )
                if validated["sample_token"] != locator.sample_token:
                    raise ValueError("cache row changed after indexing")
                value = torch.from_numpy(np.asarray(row["event_v4_common_roi"])).clone()
                dt = float(np.asarray(row["garl_delta_t_s"]).item())
                output[output_index] = (value, (dt, dt), locator.target_ttc_s)
            if self.cache_mode == "direct":
                del shard
        if any(item is None for item in output):
            raise RuntimeError("cache materialization left an unresolved row")
        for item in output:
            assert item is not None
            events.append(item[0])
            deltas.append(item[1])
            targets.append(item[2])
        return (
            torch.stack(events),
            torch.tensor(deltas, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float64),
        )

    def iter_outer_train_batches(
        self, view: CollisionClockOuterTrainView, *, batch_size: int
    ) -> Iterator[CollisionClockOuterTrainBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(view.locators), batch_size):
            locators = view.locators[start : start + batch_size]
            inputs, delta, target = self._materialize(locators)
            yield CollisionClockOuterTrainBatch(
                inputs, delta, target, tuple(item.sample_token for item in locators)
            )

    def iter_outer_dev_batches(
        self, view: CollisionClockOuterDevView, *, batch_size: int
    ) -> Iterator[CollisionClockOuterDevBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(view.locators), batch_size):
            locators = view.locators[start : start + batch_size]
            inputs, delta, target = self._materialize(locators)
            yield CollisionClockOuterDevBatch(
                inputs=inputs,
                delta_t_s=delta,
                target_ttc_seconds=target,
                sample_tokens=tuple(item.sample_token for item in locators),
                sequence_ids=tuple(item.sequence_id for item in locators),
                track_ids=tuple(item.track_id for item in locators),
                outer_fold=view.fold,
                sample_weights=torch.tensor(
                    [item.sample_weight for item in locators], dtype=torch.float64
                ),
            )


__all__ = [
    "CacheMode",
    "CollisionClockOuterDevBatch",
    "CollisionClockOuterDevView",
    "CollisionClockOuterTrainBatch",
    "CollisionClockOuterTrainSequence",
    "CollisionClockOuterTrainView",
    "CollisionClockSampleLocator",
    "CollisionClockTrain8192Cache",
    "load_canonical_supervision",
]
