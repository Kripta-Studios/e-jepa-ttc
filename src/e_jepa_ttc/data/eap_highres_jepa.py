"""Raw/on-demand adapter for signed matched eAP label-free JEPA rows.

Only rows already present in a signed manifest are consumed.  This module never
opens a parquet file and never materializes a dense tensor cache.  A tiny reader
LRU bounds open HDF5 handles per process; each ``__getitem__`` retains only the
context and future volumes needed for one microbatch item.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset

from e_jepa_ttc.data.eap import EAPEventReader
from e_jepa_ttc.data.eap_representation import base_compatible_voxel, downsample_full_frame
from e_jepa_ttc.data.garlttc_eap import resolve_eap_events_path
from e_jepa_ttc.data.matched_eap_subset import validate_matched_manifest
from e_jepa_ttc.training.eap_highres_jepa import LabelFreeBatch


class ReaderLike(Protocol):
    def open(self) -> None: ...

    def close(self) -> None: ...

    def read_window(self, start_us: int, end_us: int) -> dict[str, np.ndarray]: ...


ReaderFactory: TypeAlias = Callable[[Path], ReaderLike]
_FORBIDDEN_ROW_KEY_FRAGMENTS = (
    "ttc",
    "depth",
    "3d",
    "category",
    "bbox",
    "box",
    "mask",
    "rgb",
    "label",
    "evttc",
)


def _reject_forbidden_row_fields(value: Mapping[str, Any], *, path: str) -> None:
    for key, child in value.items():
        lowered = str(key).lower()
        if any(fragment in lowered for fragment in _FORBIDDEN_ROW_KEY_FRAGMENTS):
            raise ValueError(f"Label-free manifest row contains prohibited field {path}.{key}.")
        if isinstance(child, Mapping):
            _reject_forbidden_row_fields(child, path=f"{path}.{key}")
        elif isinstance(child, Sequence) and not isinstance(child, (str, bytes)):
            for index, item in enumerate(child):
                if isinstance(item, Mapping):
                    _reject_forbidden_row_fields(item, path=f"{path}.{key}[{index}]")


def _as_python(value: object) -> object:
    if hasattr(value, "as_py"):
        value = value.as_py()  # type: ignore[union-attr]
    if isinstance(value, np.ndarray):
        return _as_python(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_as_python(item) for item in value]
    return value


def _window_bounds(value: object) -> tuple[int, int]:
    parsed = _as_python(value)
    if isinstance(parsed, str):
        import ast

        parsed = ast.literal_eval(parsed)
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 2:
        raise ValueError("event_windows_us must contain [start_us, end_us].")
    if all(isinstance(item, (int, float, np.number)) for item in parsed[:2]):
        start, end = int(parsed[0]), int(parsed[1])
    else:
        first = _as_python(parsed[0])
        last = _as_python(parsed[-1])
        if (
            not isinstance(first, (list, tuple))
            or len(first) < 2
            or not isinstance(last, (list, tuple))
            or len(last) < 2
        ):
            raise ValueError("event_windows_us has no valid interval pair.")
        start, end = int(first[0]), int(last[1])
    if end <= start:
        raise ValueError("event_windows_us interval must be positive.")
    return start, end


def _subdivide_window(bounds: tuple[int, int], steps: int) -> tuple[tuple[int, int], ...]:
    start, end = bounds
    if steps <= 0:
        raise ValueError("temporal_steps must be positive.")
    edges = np.linspace(start, end, steps + 1, dtype=np.int64)
    windows = tuple((int(edges[i]), int(edges[i + 1])) for i in range(steps))
    if any(right <= left for left, right in windows):
        raise ValueError("Temporal subdivision produced an empty interval.")
    return windows


def _row_future_endpoints(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    endpoints = row.get("future_endpoints")
    if not isinstance(endpoints, Sequence) or isinstance(endpoints, (str, bytes)):
        raise ValueError("Matched row lacks future_endpoints.")
    result = [endpoint for endpoint in endpoints if isinstance(endpoint, Mapping)]
    if not result:
        raise ValueError("Matched row has no valid future_endpoints.")
    return result


class EAPHighResLabelFreeDataset(Dataset[LabelFreeBatch]):
    """Decode one signed manifest row into full-frame ``[T,21,192,320]`` data."""

    def __init__(
        self,
        manifest: Mapping[str, Any] | str | Path,
        *,
        eap_root: str | Path,
        role: str,
        stage: str | int = "matched_256",
        temporal_steps: int = 5,
        width: int = 320,
        height: int = 192,
        bins: int = 5,
        reader_factory: ReaderFactory | None = None,
        max_open_readers: int = 2,
    ) -> None:
        if isinstance(manifest, (str, Path)):
            value = json.loads(Path(manifest).read_text(encoding="utf-8"))
        else:
            value = dict(manifest)
        if not isinstance(value, Mapping):
            raise ValueError("Matched manifest must be a mapping.")
        validate_matched_manifest(value)
        if role not in {"train", "validation"}:
            raise ValueError("Label-free adapter role must be exactly train or validation.")
        rows_by_id = {
            str(row["row_id"]): row
            for row in value.get("rows", [])
            if isinstance(row, Mapping) and "row_id" in row
        }
        stages = value.get("stages")
        if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
            raise ValueError("Matched manifest stages are required.")
        selected: Mapping[str, Any] | None = None
        if isinstance(stage, int):
            requested = int(stage)
            for candidate in stages:
                if int(candidate.get("nominal_row_count", -1)) == requested:
                    selected = candidate
                    break
        else:
            requested_text = str(stage)
            for candidate in stages:
                if str(candidate.get("stage")) == requested_text:
                    selected = candidate
                    break
        if selected is None:
            raise ValueError(f"Unknown matched manifest stage: {stage!r}")
        row_ids = selected.get("row_ids")
        if not isinstance(row_ids, Sequence) or isinstance(row_ids, (str, bytes)):
            raise ValueError("Matched manifest stage row_ids are required.")
        missing = [str(row_id) for row_id in row_ids if str(row_id) not in rows_by_id]
        if missing:
            raise ValueError(f"Matched manifest stage references unknown rows: {missing[:3]}")
        for row in rows_by_id.values():
            _reject_forbidden_row_fields(row, path="rows")
        block_by_row: dict[str, str] = {}
        for block in value.get("blocks", []):
            if not isinstance(block, Mapping):
                continue
            block_id = str(block.get("block_id", ""))
            for row_id in block.get("row_ids", []):
                block_by_row[str(row_id)] = block_id
        selected_rows = [
            rows_by_id[str(row_id)]
            for row_id in row_ids
            if str(rows_by_id[str(row_id)].get("role")) == role
        ]
        if not selected_rows:
            raise ValueError(f"Matched stage {stage!r} has no rows for role={role!r}.")
        selected_block_ids = tuple(
            str(row.get("block_id") or block_by_row.get(str(row.get("row_id")), ""))
            for row in selected_rows
        )
        if len(selected_rows) != len(selected_block_ids) or any(
            not block_id for block_id in selected_block_ids
        ):
            raise ValueError("Every role-filtered adapter row must have a non-empty block_id.")
        self.role = role
        self.rows: tuple[Mapping[str, Any], ...] = tuple(selected_rows)
        self.block_ids = selected_block_ids
        self.eap_root = Path(eap_root)
        self.temporal_steps = int(temporal_steps)
        self.width = int(width)
        self.height = int(height)
        self.bins = int(bins)
        self.max_open_readers = int(max_open_readers)
        if self.temporal_steps != 5 or (self.width, self.height) != (320, 192) or self.bins != 5:
            raise ValueError("Matched SSL adapter is frozen at T=5, 320x192, bins=5.")
        if self.max_open_readers <= 0:
            raise ValueError("max_open_readers must be positive.")
        self._reader_factory = reader_factory or EAPEventReader
        self._readers: OrderedDict[Path, ReaderLike] = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def row_ids(self) -> tuple[str, ...]:
        return tuple(str(row["row_id"]) for row in self.rows)

    def _reader(self, events_path: str) -> ReaderLike:
        resolved = resolve_eap_events_path(self.eap_root, events_path)
        reader = self._readers.pop(resolved, None)
        if reader is None:
            reader = self._reader_factory(resolved)
            open_method = getattr(reader, "open", None)
            if callable(open_method):
                open_method()
        self._readers[resolved] = reader
        while len(self._readers) > self.max_open_readers:
            _, evicted = self._readers.popitem(last=False)
            close_method = getattr(evicted, "close", None)
            if callable(close_method):
                close_method()
        return reader

    def _encode_interval(
        self,
        reader: ReaderLike,
        *,
        sequence_id: str,
        events_path: str,
        bounds: tuple[int, int],
    ) -> torch.Tensor:
        steps: list[torch.Tensor] = []
        for start_us, end_us in _subdivide_window(bounds, self.temporal_steps):
            raw = reader.read_window(start_us, end_us)
            frame = downsample_full_frame(
                raw,
                sequence_id=sequence_id,
                start_us=start_us,
                end_us=end_us,
                width=self.width,
                height=self.height,
            )
            steps.append(base_compatible_voxel(frame, bins=self.bins))
        return torch.stack(steps).contiguous()

    def __getitem__(self, index: int) -> LabelFreeBatch:
        row = self.rows[int(index)]
        sequence_id = str(row["sequence_id"])
        track_id = str(row["track_id"])
        events_path = str(row["events_path"])
        reader = self._reader(events_path)
        context = self._encode_interval(
            reader,
            sequence_id=sequence_id,
            events_path=events_path,
            bounds=_window_bounds(row["event_windows_us"]),
        )
        endpoints = _row_future_endpoints(row)
        futures: list[torch.Tensor] = []
        future_timestamps: list[float] = []
        for endpoint in endpoints:
            endpoint_path = str(endpoint.get("events_path", events_path))
            endpoint_reader = self._reader(endpoint_path)
            endpoint_sequence = str(endpoint.get("sequence_id", sequence_id))
            futures.append(
                self._encode_interval(
                    endpoint_reader,
                    sequence_id=endpoint_sequence,
                    events_path=endpoint_path,
                    bounds=_window_bounds(endpoint["event_windows_us"]),
                )
            )
            future_timestamps.append(float(endpoint["timestamp_us"]) / 1_000_000.0)
        reference_timestamp = float(row["timestamp_us"]) / 1_000_000.0
        future_tensor = torch.stack(futures).contiguous()
        future_ts = torch.tensor(future_timestamps, dtype=torch.float64)
        reference_ts = torch.tensor([reference_timestamp], dtype=torch.float64)
        horizon_delta = (future_ts - reference_ts[0]).reshape(1, -1)
        return LabelFreeBatch(
            context_events=context.unsqueeze(0),
            future_events=future_tensor.unsqueeze(0),
            horizon_delta_t_s=horizon_delta,
            sequence_ids=(sequence_id,),
            track_ids=(track_id,),
            reference_timestamps_s=reference_ts,
            future_timestamps_s=future_ts.reshape(1, -1),
            context_valid=torch.ones((1, self.temporal_steps), dtype=torch.bool),
            future_valid=torch.ones(
                (1, future_tensor.shape[0], self.temporal_steps), dtype=torch.bool
            ),
            nce_candidate_mask=torch.ones(
                (1, future_tensor.shape[0], 1, future_tensor.shape[0]), dtype=torch.bool
            ),
        )

    def close(self) -> None:
        while self._readers:
            _, reader = self._readers.popitem(last=False)
            close_method = getattr(reader, "close", None)
            if callable(close_method):
                close_method()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_readers"] = OrderedDict()
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# Short aliases make the adapter easy to discover without weakening the
# canonical class name used by the protocol.
LabelFreeEAPDataset = EAPHighResLabelFreeDataset
MatchedEAPDataset = EAPHighResLabelFreeDataset
LabelFreeDataset = EAPHighResLabelFreeDataset


def collate_label_free(
    batch: Sequence[LabelFreeBatch],
    *,
    exclusion_window_s: float = 0.02,
    positive_tolerance_s: float = 1e-6,
) -> LabelFreeBatch:
    """Collate items and construct the explicit ``[B,H,B,H]`` candidate mask."""

    if not batch:
        raise ValueError("Cannot collate an empty label-free batch.")
    contexts = torch.cat([item.context_events for item in batch], dim=0)
    futures = torch.cat([item.future_events for item in batch], dim=0)
    horizon_delta = torch.cat([item.horizon_delta_t_s for item in batch], dim=0)
    references = torch.cat([item.reference_timestamps_s for item in batch], dim=0)
    future_timestamps = torch.cat([item.future_timestamps_s for item in batch], dim=0)
    context_valid = torch.cat(
        [
            item.context_valid
            if item.context_valid is not None
            else torch.ones((1, contexts.shape[1]), dtype=torch.bool)
            for item in batch
        ],
        dim=0,
    )
    future_valid = torch.cat(
        [
            item.future_valid
            if item.future_valid is not None
            else torch.ones((1, futures.shape[1], futures.shape[2]), dtype=torch.bool)
            for item in batch
        ],
        dim=0,
    )
    batch_size, horizons = horizon_delta.shape
    candidate_mask = torch.zeros((batch_size, horizons, batch_size, horizons), dtype=torch.bool)
    for anchor in range(batch_size):
        for horizon in range(horizons):
            desired = float(references[anchor]) + float(horizon_delta[anchor, horizon])
            for candidate_batch in range(batch_size):
                if batch[anchor].sequence_ids[0] != batch[candidate_batch].sequence_ids[0]:
                    continue
                if batch[anchor].track_ids[0] != batch[candidate_batch].track_ids[0]:
                    continue
                for candidate_horizon in range(horizons):
                    candidate_time = float(future_timestamps[candidate_batch, candidate_horizon])
                    # Positives are always retained.  Other same-track candidates
                    # are excluded only inside the declared temporal exclusion.
                    is_positive = abs(candidate_time - desired) <= positive_tolerance_s
                    candidate_mask[anchor, horizon, candidate_batch, candidate_horizon] = (
                        is_positive or abs(candidate_time - desired) > exclusion_window_s
                    )
    return LabelFreeBatch(
        context_events=contexts,
        future_events=futures,
        horizon_delta_t_s=horizon_delta,
        sequence_ids=tuple(item for sample in batch for item in sample.sequence_ids),
        track_ids=tuple(item for sample in batch for item in sample.track_ids),
        reference_timestamps_s=references,
        future_timestamps_s=future_timestamps,
        context_valid=context_valid,
        future_valid=future_valid,
        nce_candidate_mask=candidate_mask,
    )


collate_eap_highres_label_free = collate_label_free


class BlockAwareBatchSampler(BatchSampler):
    """Deterministic contiguous batches preserving manifest block order."""

    def __init__(
        self,
        data_source: EAPHighResLabelFreeDataset | Sequence[Any],
        *,
        batch_size: int = 2,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0 or batch_size > 2:
            raise ValueError("Matched block-aware batch_size must lie in [1,2].")
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = bool(drop_last)
        self._indices = list(range(len(data_source)))
        super().__init__(self._indices, batch_size=batch_size, drop_last=drop_last)

    def __iter__(self) -> Iterator[list[int]]:
        if isinstance(self.data_source, EAPHighResLabelFreeDataset):
            block_ids = list(self.data_source.block_ids)
            start = 0
            while start < len(self._indices):
                block_id = block_ids[start]
                stop = start + 1
                while stop < len(self._indices) and block_ids[stop] == block_id:
                    stop += 1
                for offset in range(start, stop, self.batch_size):
                    batch = self._indices[offset : min(stop, offset + self.batch_size)]
                    if len(batch) < self.batch_size and self.drop_last:
                        continue
                    yield batch
                start = stop
            return
        for offset in range(0, len(self._indices), self.batch_size):
            batch = self._indices[offset : offset + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            yield batch

    def __len__(self) -> int:
        if isinstance(self.data_source, EAPHighResLabelFreeDataset):
            block_ids = list(self.data_source.block_ids)
            total = 0
            start = 0
            while start < len(block_ids):
                stop = start + 1
                while stop < len(block_ids) and block_ids[stop] == block_ids[start]:
                    stop += 1
                count = stop - start
                total += (
                    count // self.batch_size
                    if self.drop_last
                    else (count + self.batch_size - 1) // self.batch_size
                )
                start = stop
            return total
        if self.drop_last:
            return len(self._indices) // self.batch_size
        return (len(self._indices) + self.batch_size - 1) // self.batch_size


DeterministicBlockBatchSampler = BlockAwareBatchSampler


def make_label_free_loader(
    dataset: EAPHighResLabelFreeDataset,
    *,
    batch_size: int = 2,
    num_workers: int = 0,
    drop_last: bool = False,
    exclusion_window_s: float = 0.02,
    positive_tolerance_s: float = 1e-6,
) -> DataLoader[LabelFreeBatch]:
    """Construct the only supported matched-subset DataLoader policy."""

    if num_workers < 0 or num_workers > 8:
        raise ValueError("Matched raw-reader workers must lie in [0,8].")
    sampler = BlockAwareBatchSampler(dataset, batch_size=batch_size, drop_last=drop_last)
    collate = partial(
        collate_label_free,
        exclusion_window_s=float(exclusion_window_s),
        positive_tolerance_s=float(positive_tolerance_s),
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=num_workers,
        persistent_workers=False,
    )


__all__ = [
    "BlockAwareBatchSampler",
    "DeterministicBlockBatchSampler",
    "EAPHighResLabelFreeDataset",
    "LabelFreeEAPDataset",
    "LabelFreeDataset",
    "MatchedEAPDataset",
    "collate_eap_highres_label_free",
    "collate_label_free",
    "make_label_free_loader",
]
