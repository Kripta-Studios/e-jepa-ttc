"""Materialized object-ROI cache for public eAP media."""

from __future__ import annotations

import bisect
import random
import zlib
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from e_jepa_ttc.data.eap import (
    EAP_IMAGE_SIZE,
    EAPEventReader,
    EAPObjectState,
    EAPObjectWindow,
    EAPRGBReader,
    build_eap_object_windows,
    crop_events_to_roi,
    load_eap_media_table,
    load_eap_sequence_labels,
    reconstruct_eap_object_states,
)
from e_jepa_ttc.data.types import EventBatch
from e_jepa_ttc.representations.corruptions import (
    EventCorruptionSpec,
    corrupt_event_batch,
)
from e_jepa_ttc.representations.voxel_grid import encode_voxel_grid
from e_jepa_ttc.utils.io import write_structured


@dataclass(frozen=True)
class EAPObjectCacheConfig:
    """Preprocessing parameters encoded into every object-cache manifest."""

    history_frames: int = 3
    prediction_horizons_ms: tuple[int, ...] = (100, 250, 500)
    event_window_ms: int = 100
    adaptive_event_count: int | None = None
    minimum_adaptive_window_ms: int = 10
    maximum_target_slop_ms: int = 25
    maximum_history_gap_ms: int = 125
    roi_width: int = 64
    roi_height: int = 64
    roi_expansion: float = 1.25
    event_bins: int = 5
    normalize_events: bool = True
    derivative_radius: int = 2
    maximum_derivative_gap_s: float = 0.25
    shard_size: int = 512
    action_dim: int = 8
    corruption_kind: str = "none"
    corruption_severity: float = 0.0
    corruption_seed: int = 0
    include_rgb: bool = False
    rgb_width: int = 112
    rgb_height: int = 112

    def __post_init__(self) -> None:
        if self.history_frames < 2 or self.event_window_ms <= 0:
            msg = "history_frames must be >=2 and event_window_ms must be positive."
            raise ValueError(msg)
        if self.adaptive_event_count is not None and self.adaptive_event_count <= 0:
            msg = "adaptive_event_count must be positive when enabled."
            raise ValueError(msg)
        if not 0 < self.minimum_adaptive_window_ms <= self.event_window_ms:
            msg = "minimum_adaptive_window_ms must lie in (0, event_window_ms]."
            raise ValueError(msg)
        if not self.prediction_horizons_ms:
            msg = "prediction_horizons_ms must be non-empty."
            raise ValueError(msg)
        if any(horizon < self.event_window_ms for horizon in self.prediction_horizons_ms):
            msg = (
                "Every endpoint prediction horizon must be at least event_window_ms "
                "to keep future target windows disjoint from causal context."
            )
            raise ValueError(msg)
        if tuple(sorted(set(self.prediction_horizons_ms))) != self.prediction_horizons_ms:
            msg = "prediction_horizons_ms must be unique and strictly increasing."
            raise ValueError(msg)
        dimensions = (
            self.roi_width,
            self.roi_height,
            self.event_bins,
            self.shard_size,
            self.rgb_width,
            self.rgb_height,
        )
        if min(dimensions) <= 0:
            msg = "ROI dimensions, event bins and shard size must be positive."
            raise ValueError(msg)
        if self.roi_expansion <= 0 or self.action_dim <= 0:
            msg = "roi_expansion and action_dim must be positive."
            raise ValueError(msg)
        EventCorruptionSpec(
            kind=self.corruption_kind,
            severity=self.corruption_severity,
            seed=self.corruption_seed,
        )


def _normalized_box(state: EAPObjectState) -> np.ndarray:
    width, height = EAP_IMAGE_SIZE
    scale = np.asarray([width, height, width, height], dtype=np.float32)
    return np.clip(np.asarray(state.bbox_xyxy, dtype=np.float32) / scale, 0.0, 1.0)


def _full_sampling_box() -> np.ndarray:
    return np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)


def _rgb_object_crop(
    reader: EAPRGBReader,
    state: EAPObjectState,
    media_lookup: dict[str, tuple[str, str]],
    *,
    config: EAPObjectCacheConfig,
) -> np.ndarray:
    from PIL import Image

    paths = media_lookup.get(state.sample_token)
    if paths is None:
        msg = f"No RGB media row for sample token {state.sample_token!r}."
        raise ValueError(msg)
    image = reader.read(paths[0], paths[1])
    x_min, y_min, x_max, y_max = state.bbox_xyxy
    center_x = (x_min + x_max) * 0.5
    center_y = (y_min + y_max) * 0.5
    width = max(1.0, (x_max - x_min) * config.roi_expansion)
    height = max(1.0, (y_max - y_min) * config.roi_expansion)
    crop = image.crop(
        (
            max(0.0, center_x - width * 0.5),
            max(0.0, center_y - height * 0.5),
            min(float(image.width), center_x + width * 0.5),
            min(float(image.height), center_y + height * 0.5),
        )
    )
    resized = crop.resize((config.rgb_width, config.rgb_height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)


def _state_voxel(
    reader: EAPEventReader,
    state: EAPObjectState,
    *,
    config: EAPObjectCacheConfig,
    minimum_start_us: int | None = None,
    raw_event_cache: OrderedDict[tuple[int, int], dict[str, np.ndarray]] | None = None,
    maximum_raw_entries: int = 16,
) -> tuple[np.ndarray, int, int]:
    end_us = state.timestamp_us
    start_us = end_us - config.event_window_ms * 1000
    if minimum_start_us is not None:
        start_us = max(start_us, minimum_start_us)
    if start_us >= end_us:
        msg = f"Event window must have positive duration, got [{start_us}, {end_us})."
        raise ValueError(msg)
    raw_key = (start_us, end_us)
    events = raw_event_cache.pop(raw_key, None) if raw_event_cache is not None else None
    if events is None:
        events = reader.read_window(start_us, end_us)
    if raw_event_cache is not None:
        raw_event_cache[raw_key] = events
        while len(raw_event_cache) > maximum_raw_entries:
            raw_event_cache.popitem(last=False)
    cropped = crop_events_to_roi(
        events,
        state.bbox_xyxy,
        sequence_id=state.sequence_id,
        start_us=start_us,
        end_us=end_us,
        output_size=(config.roi_width, config.roi_height),
        expansion=config.roi_expansion,
    )
    if config.adaptive_event_count is not None and cropped.num_events > config.adaptive_event_count:
        first = cropped.num_events - config.adaptive_event_count
        count_start_us = int(cropped.t_us[first])
        minimum_duration_start_us = end_us - config.minimum_adaptive_window_ms * 1000
        adaptive_start_us = min(count_start_us, minimum_duration_start_us)
        selected = cropped.t_us >= adaptive_start_us
        cropped = EventBatch(
            x=cropped.x[selected],
            y=cropped.y[selected],
            t_us=cropped.t_us[selected],
            polarity=cropped.polarity[selected],
            width=cropped.width,
            height=cropped.height,
            sequence_id=cropped.sequence_id,
            t_start_us=adaptive_start_us,
            t_end_us=end_us,
        )
        start_us = adaptive_start_us
    if config.corruption_kind != "none":
        identity = f"{state.sequence_id}:{state.track_id}:{state.timestamp_us}".encode()
        cropped = corrupt_event_batch(
            cropped,
            EventCorruptionSpec(
                kind=config.corruption_kind,
                severity=config.corruption_severity,
                seed=config.corruption_seed,
            ),
            seed_offset=zlib.crc32(identity),
        )
        start_us = cropped.t_start_us
    voxel = encode_voxel_grid(
        cropped,
        bins=config.event_bins,
        separate_polarity=True,
        normalize=config.normalize_events,
    )
    return voxel.astype(np.float16), start_us, end_us


def _encode_with_lru(
    reader: EAPEventReader,
    state: EAPObjectState,
    *,
    config: EAPObjectCacheConfig,
    cache: OrderedDict[tuple[str, int, int], tuple[np.ndarray, int, int]],
    minimum_start_us: int | None = None,
    maximum_entries: int = 4096,
    raw_event_cache: OrderedDict[tuple[int, int], dict[str, np.ndarray]] | None = None,
) -> tuple[np.ndarray, int, int]:
    key = (
        state.track_id,
        state.timestamp_us,
        minimum_start_us if minimum_start_us is not None else -1,
    )
    cached = cache.pop(key, None)
    if cached is not None:
        cache[key] = cached
        return cached
    encoded = _state_voxel(
        reader,
        state,
        config=config,
        minimum_start_us=minimum_start_us,
        raw_event_cache=raw_event_cache,
    )
    cache[key] = encoded
    while len(cache) > maximum_entries:
        cache.popitem(last=False)
    return encoded


def _window_sample(
    window: EAPObjectWindow,
    reader: EAPEventReader,
    *,
    config: EAPObjectCacheConfig,
    event_cache: OrderedDict[tuple[str, int, int], tuple[np.ndarray, int, int]],
    raw_event_cache: OrderedDict[tuple[int, int], dict[str, np.ndarray]],
    split_name: str,
    rgb_reader: EAPRGBReader | None = None,
    media_lookup: dict[str, tuple[str, str]] | None = None,
) -> dict[str, np.ndarray | str]:
    horizon_to_state = dict(window.future)
    context_voxels: list[np.ndarray] = []
    context_start: list[int] = []
    context_end: list[int] = []
    for state in window.history:
        voxel, start_us, end_us = _encode_with_lru(
            reader,
            state,
            config=config,
            cache=event_cache,
            raw_event_cache=raw_event_cache,
        )
        context_voxels.append(voxel)
        context_start.append(start_us)
        context_end.append(end_us)

    channel_count = config.event_bins * 2
    future_voxels: list[np.ndarray] = []
    future_boxes: list[np.ndarray] = []
    future_depth: list[float] = []
    future_mask: list[bool] = []
    future_start: list[int] = []
    future_end: list[int] = []
    for horizon_ms in config.prediction_horizons_ms:
        state = horizon_to_state.get(horizon_ms)
        if state is None:
            future_voxels.append(
                np.zeros(
                    (channel_count, config.roi_height, config.roi_width),
                    dtype=np.float16,
                )
            )
            future_boxes.append(np.zeros(4, dtype=np.float32))
            future_depth.append(float("nan"))
            future_mask.append(False)
            future_start.append(-1)
            future_end.append(-1)
            continue
        voxel, start_us, end_us = _encode_with_lru(
            reader,
            state,
            config=config,
            cache=event_cache,
            minimum_start_us=window.target.timestamp_us,
            raw_event_cache=raw_event_cache,
        )
        if start_us < window.target.timestamp_us:
            msg = (
                f"Future target window at {horizon_ms}ms overlaps causal context: "
                f"{start_us} < {window.target.timestamp_us}."
            )
            raise ValueError(msg)
        future_voxels.append(voxel)
        future_boxes.append(_normalized_box(state))
        future_depth.append(state.nearest_depth_m)
        future_mask.append(True)
        future_start.append(start_us)
        future_end.append(end_us)

    history_count = config.history_frames
    horizon_count = len(config.prediction_horizons_ms)
    sample: dict[str, np.ndarray | str] = {
        "context_events": np.stack(context_voxels),
        "context_boxes": np.stack([_normalized_box(state) for state in window.history])[:, None, :],
        "context_sampling_boxes": np.tile(
            _full_sampling_box(),
            (history_count, 1, 1),
        ),
        "context_object_mask": np.ones((history_count, 1), dtype=np.bool_),
        "context_depth_m": np.asarray([window.target.nearest_depth_m], dtype=np.float32),
        "context_depth_history_m": np.asarray(
            [state.nearest_depth_m for state in window.history],
            dtype=np.float32,
        )[:, None],
        "context_ego_actions": np.zeros(
            (history_count, config.action_dim),
            dtype=np.float32,
        ),
        "context_ego_action_mask": np.zeros(history_count, dtype=np.bool_),
        "future_events": np.stack(future_voxels),
        "future_boxes": np.stack(future_boxes)[:, None, :],
        "future_sampling_boxes": np.tile(
            _full_sampling_box(),
            (horizon_count, 1, 1),
        ),
        "future_object_mask": np.asarray(future_mask, dtype=np.bool_)[:, None],
        "future_depth_m": np.asarray(future_depth, dtype=np.float32)[:, None],
        "future_ego_actions": np.zeros(
            (horizon_count, config.action_dim),
            dtype=np.float32,
        ),
        "future_ego_action_mask": np.zeros(horizon_count, dtype=np.bool_),
        "ttc_s": np.asarray([window.target.ttc_s], dtype=np.float32),
        "context_window_start_us": np.asarray(context_start, dtype=np.int64),
        "context_window_end_us": np.asarray(context_end, dtype=np.int64),
        "future_window_start_us": np.asarray(future_start, dtype=np.int64),
        "future_window_end_us": np.asarray(future_end, dtype=np.int64),
        "sample_token": window.target.sample_token,
        "sequence_id": window.target.sequence_id,
        "track_id": window.target.track_id,
        "category": window.target.category,
        "split": split_name,
        "ttc_source": window.target.ttc_source,
    }
    if config.include_rgb:
        if rgb_reader is None or media_lookup is None:
            msg = "RGB cache materialization requires an RGB reader and media lookup."
            raise RuntimeError(msg)
        sample["context_rgb"] = np.stack(
            [
                _rgb_object_crop(rgb_reader, state, media_lookup, config=config)
                for state in window.history
            ]
        )
    return sample


def _write_shard(
    output_dir: Path,
    *,
    sequence_id: str,
    split_name: str,
    shard_index: int,
    samples: list[dict[str, np.ndarray | str]],
    config: EAPObjectCacheConfig,
) -> dict[str, Any]:
    path = output_dir / f"{split_name}-{sequence_id}-{shard_index:05d}.npz"
    array_keys = [key for key, value in samples[0].items() if isinstance(value, np.ndarray)]
    string_keys = [key for key, value in samples[0].items() if isinstance(value, str)]
    payload: dict[str, np.ndarray] = {
        key: np.stack([np.asarray(sample[key]) for sample in samples]) for key in array_keys
    }
    payload.update(
        {key: np.asarray([str(sample[key]) for sample in samples]) for key in string_keys}
    )
    payload.update(
        {
            "prediction_horizons_s": np.asarray(
                config.prediction_horizons_ms,
                dtype=np.float32,
            )
            * 1e-3,
            "cache_format_version": np.asarray(2, dtype=np.int64),
            "future_window_semantics": np.asarray("endpoint_offset_disjoint_fixed_duration"),
        }
    )
    np.savez_compressed(path, **payload)
    return {
        "path": path.name,
        "sequence_id": sequence_id,
        "split": split_name,
        "samples": len(samples),
        "size_bytes": path.stat().st_size,
    }


def _materialize_sequence_job(
    payload: tuple[str, str, str, str, EAPObjectCacheConfig, int | None],
) -> dict[str, Any]:
    """Process-pool entry point for one independent eAP sequence."""

    eap_root, output_dir, sequence_id, split_name, config, maximum = payload
    return materialize_eap_object_cache(
        eap_root=eap_root,
        output_dir=output_dir,
        sequence_splits={sequence_id: split_name},
        config=config,
        max_windows_per_sequence=maximum,
        workers=1,
    )


def _cache_manifest(
    *,
    root: Path,
    config: EAPObjectCacheConfig,
    sequence_splits: dict[str, str],
    sequence_summaries: list[dict[str, Any]],
    shards: list[dict[str, Any]],
    workers: int,
) -> dict[str, Any]:
    return {
        "format": "eap_object_event_jepa_cache_v2",
        "pre_cropped_events": True,
        "eap_root": root.as_posix(),
        "config": asdict(config),
        "sequence_splits": sequence_splits,
        "ttc_label_status": "reconstructed_from_public_3d_tracks_not_official_garlttc",
        "ego_action_status": "unavailable_in_public_eap_media_masked_false",
        "future_teacher_uses_ego_actions": False,
        "event_window_policy": (
            "roi_density_adaptive_trailing_count"
            if config.adaptive_event_count is not None
            else "fixed_duration"
        ),
        "normalization": "occupied_voxel_noncentred_q95_magnitude",
        "materialization_workers": workers,
        "sequences": sequence_summaries,
        "shards": shards,
        "total_samples": sum(int(shard["samples"]) for shard in shards),
        "total_size_bytes": sum(int(shard["size_bytes"]) for shard in shards),
    }


def materialize_eap_object_cache(
    *,
    eap_root: str | Path,
    output_dir: str | Path,
    sequence_splits: dict[str, str],
    config: EAPObjectCacheConfig | None = None,
    max_windows_per_sequence: int | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Build sharded ROI-event tensors and auditable geometry targets."""

    if config is None:
        config = EAPObjectCacheConfig()
    if not sequence_splits:
        msg = "sequence_splits must assign at least one eAP train sequence."
        raise ValueError(msg)
    if workers <= 0:
        msg = "workers must be positive."
        raise ValueError(msg)
    allowed_splits = {"train", "validation", "calibration", "test"}
    unexpected = sorted(set(sequence_splits.values()) - allowed_splits)
    if unexpected:
        msg = f"Unsupported cache split names: {unexpected}."
        raise ValueError(msg)
    if max_windows_per_sequence is not None and max_windows_per_sequence <= 0:
        msg = "max_windows_per_sequence must be positive when provided."
        raise ValueError(msg)
    root = Path(eap_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if workers > 1 and len(sequence_splits) > 1:
        tasks = [
            (
                str(root),
                str(output / sequence_id),
                sequence_id,
                split_name,
                config,
                max_windows_per_sequence,
            )
            for sequence_id, split_name in sorted(sequence_splits.items())
        ]
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            partial_manifests = list(executor.map(_materialize_sequence_job, tasks))
        combined_shards: list[dict[str, Any]] = []
        combined_sequences: list[dict[str, Any]] = []
        for task, partial in zip(tasks, partial_manifests, strict=True):
            sequence_id = task[2]
            combined_sequences.extend(partial["sequences"])
            for shard in partial["shards"]:
                combined_shards.append(
                    {
                        **shard,
                        "path": (Path(sequence_id) / str(shard["path"])).as_posix(),
                    }
                )
        manifest = _cache_manifest(
            root=root,
            config=config,
            sequence_splits=sequence_splits,
            sequence_summaries=combined_sequences,
            shards=combined_shards,
            workers=min(workers, len(tasks)),
        )
        write_structured(output / "manifest.json", manifest)
        return manifest

    media = load_eap_media_table(root, split="train")
    available = set(media["sequence_id"].astype(str).unique().tolist())
    missing_sequences = sorted(set(sequence_splits) - available)
    if missing_sequences:
        msg = f"Sequences are absent from public eAP train metadata: {missing_sequences}."
        raise ValueError(msg)

    shards: list[dict[str, Any]] = []
    sequence_summaries: list[dict[str, Any]] = []
    for sequence_id, split_name in sorted(sequence_splits.items()):
        labels = load_eap_sequence_labels(root, sequence_id, split="train")
        sequence_media = media[media["sequence_id"].astype(str) == sequence_id]
        states = reconstruct_eap_object_states(
            sequence_media,
            labels,
            derivative_radius=config.derivative_radius,
            maximum_gap_s=config.maximum_derivative_gap_s,
        )
        windows = build_eap_object_windows(
            states,
            history_frames=config.history_frames,
            horizons_ms=config.prediction_horizons_ms,
            maximum_slop_ms=config.maximum_target_slop_ms,
            maximum_history_gap_ms=config.maximum_history_gap_ms,
        )
        if max_windows_per_sequence is not None and len(windows) > max_windows_per_sequence:
            selected = np.linspace(
                0,
                len(windows) - 1,
                max_windows_per_sequence,
                dtype=np.int64,
            )
            windows = [windows[int(index)] for index in selected]
        event_path = root / "data" / "train" / sequence_id / "events.h5"
        media_lookup = {
            str(row.sample_token): (str(row.rgb_shard_path), str(row.rgb_member_path))
            for row in sequence_media.itertuples(index=False)
        }
        samples: list[dict[str, np.ndarray | str]] = []
        shard_index = 0
        sequence_shards = 0
        event_cache: OrderedDict[
            tuple[str, int, int],
            tuple[np.ndarray, int, int],
        ] = OrderedDict()
        raw_event_cache: OrderedDict[tuple[int, int], dict[str, np.ndarray]] = OrderedDict()
        rgb_context = EAPRGBReader(root) if config.include_rgb else nullcontext(None)
        with EAPEventReader(event_path) as reader, rgb_context as rgb_reader:
            for window in windows:
                samples.append(
                    _window_sample(
                        window,
                        reader,
                        config=config,
                        event_cache=event_cache,
                        raw_event_cache=raw_event_cache,
                        split_name=split_name,
                        rgb_reader=rgb_reader,
                        media_lookup=media_lookup,
                    )
                )
                if len(samples) >= config.shard_size:
                    shards.append(
                        _write_shard(
                            output,
                            sequence_id=sequence_id,
                            split_name=split_name,
                            shard_index=shard_index,
                            samples=samples,
                            config=config,
                        )
                    )
                    samples = []
                    shard_index += 1
                    sequence_shards += 1
            if samples:
                shards.append(
                    _write_shard(
                        output,
                        sequence_id=sequence_id,
                        split_name=split_name,
                        shard_index=shard_index,
                        samples=samples,
                        config=config,
                    )
                )
                sequence_shards += 1
        sequence_summaries.append(
            {
                "sequence_id": sequence_id,
                "split": split_name,
                "projected_states": len(states),
                "windows": len(windows),
                "shards": sequence_shards,
            }
        )
    manifest = _cache_manifest(
        root=root,
        config=config,
        sequence_splits=sequence_splits,
        sequence_summaries=sequence_summaries,
        shards=shards,
        workers=1,
    )
    write_structured(output / "manifest.json", manifest)
    return manifest


class EAPObjectCacheDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Lazy PyTorch dataset over sharded eAP object-cache files."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        splits: tuple[str, ...],
    ) -> None:
        import json

        source = Path(manifest_path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        selected = [
            source.parent / shard["path"]
            for shard in payload["shards"]
            if str(shard["split"]) in splits
        ]
        if not selected:
            msg = f"No object-cache shards found for splits {splits}."
            raise ValueError(msg)
        self.shard_paths = selected
        self.counts: list[int] = []
        for path in selected:
            with np.load(path, allow_pickle=False) as shard:
                self.counts.append(int(shard["sample_token"].shape[0]))
        self.cumulative = np.cumsum(self.counts).astype(np.int64).tolist()
        self._open_index: int | None = None
        self._open_shard: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return int(self.cumulative[-1])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.cumulative, index)
        previous = 0 if shard_index == 0 else self.cumulative[shard_index - 1]
        local_index = index - previous
        shard = self._shard(shard_index)
        ignored = {
            "prediction_horizons_s",
            "cache_format_version",
            "future_window_semantics",
        }
        sample: dict[str, torch.Tensor | str] = {}
        for key in shard:
            if key in ignored:
                continue
            value = shard[key][local_index]
            if np.issubdtype(np.asarray(value).dtype, np.str_):
                sample[key] = str(value)
            else:
                sample[key] = torch.from_numpy(np.array(value, copy=True))
        sample["prediction_horizons_s"] = torch.from_numpy(
            np.array(shard["prediction_horizons_s"], copy=True)
        )
        return sample

    def _shard(self, index: int) -> dict[str, np.ndarray]:
        if self._open_index != index:
            self.close()
            # ``NpzFile`` is lazy: indexing ``shard[key]`` decompresses the whole
            # member every time.  Materialize each member once while a shard is
            # active.  Combined with ``ShardLocalSampler`` this changes training
            # I/O from roughly one decompression per sample to one per shard and
            # epoch without retaining the complete multi-GB cache in RAM.
            with np.load(self.shard_paths[index], allow_pickle=False) as archive:
                self._open_shard = {key: archive[key] for key in archive.files}
            self._open_index = index
        if self._open_shard is None:
            msg = "Object cache shard failed to open."
            raise RuntimeError(msg)
        return self._open_shard

    def close(self) -> None:
        """Close the currently cached NPZ shard."""

        if self._open_shard is not None:
            self._open_shard.clear()
        self._open_shard = None
        self._open_index = None

    def shard_index(self, index: int) -> int:
        """Return the backing shard for a validated global sample index."""

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return bisect.bisect_right(self.cumulative, index)


class ShardLocalSampler(Sampler[int]):
    """Shuffle samples while visiting each compressed shard only once per epoch.

    ``source_indices`` maps sampler positions to indices in ``dataset``.  It is
    useful for a ``Subset``: the sampler still yields subset-local positions,
    but groups them using their backing cache shards.
    """

    def __init__(
        self,
        dataset: EAPObjectCacheDataset,
        *,
        source_indices: list[int] | None = None,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.source_indices = (
            list(range(len(dataset))) if source_indices is None else list(source_indices)
        )
        if not self.source_indices:
            msg = "ShardLocalSampler requires at least one sample."
            raise ValueError(msg)
        if any(index < 0 or index >= len(dataset) for index in self.source_indices):
            msg = "source_indices contains an index outside the cache dataset."
            raise IndexError(msg)
        self.seed = int(seed)
        self.epoch = 0
        grouped: dict[int, list[int]] = {}
        for sampler_position, source_index in enumerate(self.source_indices):
            grouped.setdefault(dataset.shard_index(source_index), []).append(sampler_position)
        self._groups = grouped

    def __len__(self) -> int:
        return len(self.source_indices)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        shard_order = list(self._groups)
        rng.shuffle(shard_order)
        for shard_index in shard_order:
            positions = self._groups[shard_index].copy()
            rng.shuffle(positions)
            yield from positions


__all__ = [
    "EAPObjectCacheConfig",
    "EAPObjectCacheDataset",
    "ShardLocalSampler",
    "materialize_eap_object_cache",
]
