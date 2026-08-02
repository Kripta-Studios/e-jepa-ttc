"""Cache tensors with the exact boundary semantics of the frozen Garl release.

The canonical v4 cache uses the repository's exact half-open HDF5 reader.  The
official release, however, indexes event windows by millisecond and does not
apply a second timestamp filter.  This module deliberately keeps that behavior
separate so an accelerated official-model run cannot silently mix the two
preprocessing contracts.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tarfile
import zipfile
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import h5py
import hdf5plugin  # noqa: F401  # registers the dataset compression filters
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from e_jepa_ttc.data.garl_official_preprocessing import (
    official_resize_feature,
    official_resize_roi,
    official_square_box,
    official_timevolume_roi_np,
)
from e_jepa_ttc.data.garlttc_eap import (
    load_garlttc_train_index,
    normalize_boxes_xyxy,
    normalize_event_windows_us,
    resolve_eap_events_path,
)
from e_jepa_ttc.data.garlttc_lhr_cache import select_temporal_indices

RELEASE_CACHE_ARTIFACT = "garl_release_input_cache_v1"
RELEASE_CACHE_SCHEMA = "garl_release_input_cache_v1"
CACHE_SAMPLER_VERSION = "shard_grouped_shuffle_v2"
EVENT_QUANTIZATION = "uint16_linear_0_1"
RGB_STORAGE = "float16_normalized_release_v1"


@dataclass(frozen=True)
class GarlReleaseCacheConfig:
    """Controls for a cache that feeds the unchanged official model."""

    roi_size: int = 128
    roi_bins: int = 10
    event_pixel_diff: int = 5
    target_delta_t_s: float = 0.1
    delta_t_tolerance_s: float = 0.025
    fy: float = 1694.1323524131867
    shard_size: int = 256
    workers: int = 4
    include_rgb: bool = True
    compression_level: int = 1
    expected_rows: int = 88_744

    def __post_init__(self) -> None:
        if self.roi_size <= 0 or self.roi_bins <= 0 or self.shard_size <= 0:
            raise ValueError("ROI, bin and shard dimensions must be positive.")
        if self.roi_bins != 10:
            raise ValueError("The official Garl input requires 10 bins per endpoint.")
        if self.event_pixel_diff < 0:
            raise ValueError("event_pixel_diff must be non-negative.")
        if self.target_delta_t_s <= 0 or self.delta_t_tolerance_s <= 0:
            raise ValueError("Temporal pairing tolerances must be positive.")
        if self.workers <= 0:
            raise ValueError("workers must be positive.")
        if not 0 <= self.compression_level <= 9:
            raise ValueError("compression_level must be in [0, 9].")


def _as_list(value: object) -> list[object]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _nested_float(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_nested_float(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_nested_float(item) for item in value]
    return float(cast(Any, value)) if value is not None else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


_WORKER_EAP_ROOT: Path | None = None
_WORKER_CONFIG: GarlReleaseCacheConfig | None = None
_WORKER_H5: dict[Path, h5py.File] | None = None
_WORKER_TARS: OrderedDict[Path, tarfile.TarFile] | None = None


def _init_worker(eap_root: str, config: GarlReleaseCacheConfig) -> None:
    global _WORKER_EAP_ROOT, _WORKER_CONFIG, _WORKER_H5, _WORKER_TARS
    _WORKER_EAP_ROOT = Path(eap_root)
    _WORKER_CONFIG = config
    _WORKER_H5 = {}
    _WORKER_TARS = OrderedDict()


def _release_event_window(
    path: Path,
    start_us: int,
    end_us: int,
    pixel_diff: int,
    sensor_width: int,
    sensor_height: int,
) -> dict[str, np.ndarray]:
    """Match ``garl_ttc.utils.events.extract_from_h5_by_timewindow``."""

    if _WORKER_H5 is None:
        raise RuntimeError("Cache worker was not initialized.")
    handle = _WORKER_H5.get(path)
    if handle is None:
        handle = h5py.File(path, "r")
        _WORKER_H5[path] = handle
    ms_to_idx = np.asarray(cast(Any, handle["ms_to_idx"]), dtype="int64")
    events = cast(Any, handle["events"])
    timestamps = cast(Any, events["t"])
    if end_us > int(timestamps[-1]):
        raise ValueError(f"Event window ends after stream: {path} [{end_us}].")
    start_index = int(ms_to_idx[start_us // 1000])
    end_index = int(ms_to_idx[math.floor(end_us / 1000)])
    x = np.asarray(events["x"][start_index:end_index], dtype="int16") + pixel_diff
    y = np.asarray(events["y"][start_index:end_index], dtype="int16")
    t = np.asarray(events["t"][start_index:end_index], dtype="int64")
    if sensor_width <= 0 or sensor_height <= 0:
        raise ValueError("sensor_width and sensor_height must be positive.")
    valid = (x >= 0) & (x < sensor_width) & (y >= 0) & (y < sensor_height)
    return {"x": x[valid], "y": y[valid], "t": t[valid]}


def _read_rgb_shape(shard: object, member: object) -> tuple[int, int]:
    """Read ``(width, height)`` from one RGB member without decoding pixels."""

    if _WORKER_EAP_ROOT is None or _WORKER_TARS is None:
        raise RuntimeError("Cache worker was not initialized.")
    path = _WORKER_EAP_ROOT / str(shard)
    archive = _WORKER_TARS.get(path)
    if archive is None:
        archive = tarfile.open(path, "r")
        _WORKER_TARS[path] = archive
        while len(_WORKER_TARS) > 4:
            _, old = _WORKER_TARS.popitem(last=False)
            old.close()
    extracted = archive.extractfile(str(member))
    if extracted is None:
        raise FileNotFoundError(f"{member} inside {path}")
    with extracted, Image.open(extracted) as image:
        width, height = image.size
    return int(width), int(height)


def _read_rgb(
    shard: object,
    member: object,
    square: tuple[int, int, int, int],
    size: int,
) -> np.ndarray:
    if _WORKER_EAP_ROOT is None or _WORKER_TARS is None:
        raise RuntimeError("Cache worker was not initialized.")
    path = _WORKER_EAP_ROOT / str(shard)
    archive = _WORKER_TARS.get(path)
    if archive is None:
        archive = tarfile.open(path, "r")
        _WORKER_TARS[path] = archive
        while len(_WORKER_TARS) > 4:
            _, old = _WORKER_TARS.popitem(last=False)
            old.close()
    extracted = archive.extractfile(str(member))
    if extracted is None:
        raise FileNotFoundError(f"{member} inside {path}")
    with extracted, Image.open(extracted) as image:
        image = image.convert("RGB")
        # Keep the release operation order (normalize the full image before
        # grid_sample), but avoid two additional full-resolution float32
        # temporaries per worker.  This matters on the 32 GiB host when RGB
        # materialization uses several processes concurrently.
        array = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
        normalized = np.array(array, dtype=np.float32, copy=True)
        normalized *= 1.0 / 255.0
        mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
        std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]
        normalized -= mean
        normalized /= std
        # The official loader normalizes the full image before grid_sample.
        # Keeping that order preserves zero-padding at sensor boundaries.
        return official_resize_roi(normalized, square, (size, size)).numpy()


def _materialize_one(payload: tuple[dict[str, object], str]) -> dict[str, object]:
    row, split = payload
    if _WORKER_EAP_ROOT is None or _WORKER_CONFIG is None:
        raise RuntimeError("Cache worker was not initialized.")
    config = _WORKER_CONFIG
    boxes = normalize_boxes_xyxy(row["boxes_xyxy"])
    windows = normalize_event_windows_us(row["event_windows_us"])
    frame_timestamps = [int(cast(Any, value)) for value in _as_list(row["frame_timestamps_us"])]
    first, second, _ = select_temporal_indices(
        frame_timestamps,
        anchor_timestamp_us=int(cast(Any, row["timestamp_us"])),
        target_delta_t_s=config.target_delta_t_s,
        tolerance_s=config.delta_t_tolerance_s,
        context_delta_t_s=0.1,
        context_tolerance_s=0.05,
    )
    endpoint_indices = (first, second)
    squares = [official_square_box(boxes, index) for index in endpoint_indices]
    rgb_shards = _as_list(row["rgb_shard_paths"])
    rgb_members = _as_list(row["rgb_member_paths"])
    sensor_width, sensor_height = _read_rgb_shape(rgb_shards[-1], rgb_members[-1])
    event_planes: list[np.ndarray] = []
    for index, square in zip(endpoint_indices, squares, strict=True):
        start_us, end_us = windows[index]
        raw = _release_event_window(
            resolve_eap_events_path(_WORKER_EAP_ROOT, str(row["events_path"])),
            int(start_us),
            int(end_us),
            config.event_pixel_diff,
            sensor_width,
            sensor_height,
        )
        feature, _ = official_timevolume_roi_np(
            square,
            raw["x"].astype(np.int64),
            raw["y"].astype(np.int64),
            raw["t"],
            number_of_planes=config.roi_bins * 2,
        )
        event_planes.append(
            official_resize_feature(feature, (config.roi_size, config.roi_size)).numpy()
        )
    event_float = np.concatenate(event_planes, axis=0).astype(np.float32)
    event_q = np.rint(np.clip(event_float, 0.0, 1.0) * 65535.0).astype(np.uint16)

    frame_ttc = _as_list(row["frame_ttc"])
    ttc = float(cast(Any, frame_ttc[second]))
    if not math.isfinite(ttc):
        raise ValueError("Official frame_ttc[t2] is not finite.")
    delta_t_s = (frame_timestamps[second] - frame_timestamps[first]) * 1e-6
    max_edge = max(
        official_square_box(boxes, index)[3] - official_square_box(boxes, index)[1]
        for index in range(len(boxes))
    )
    box3d_h = float(cast(Any, row["box3d_h"]))
    corners = _as_list(row["box3d_Fcam"])
    visible = []
    for index in endpoint_indices:
        values = np.asarray(_nested_float(corners[index]), dtype=np.float64).reshape(-1, 3)
        minimum_depth = float(values[:, 2].min())
        if minimum_depth <= 0:
            raise ValueError("Official box3d_Fcam has non-positive depth.")
        visible.append(config.fy * box3d_h / minimum_depth * (config.roi_size / max_edge))

    record: dict[str, object] = {
        "event_q": event_q,
        "ttc_s": np.float32(ttc),
        "visible_height": np.asarray(visible, dtype=np.float32),
        "delta_t_s": np.float32(delta_t_s),
        "sequence_id": str(row["sequence_id"]),
        "sample_token": str(row["sample_token"]),
        "track_id": str(row["track_id"]),
        "public_track_id": str(row["public_track_id"]),
        "timestamp_us": np.int64(cast(Any, row["timestamp_us"])),
        "sensor_width": np.int32(sensor_width),
        "sensor_height": np.int32(sensor_height),
        "split": split,
    }
    if config.include_rgb:
        # The frozen release passes the last loop's ``square_box`` to the
        # concatenated two-frame RGB tensor.  Consequently both RGB endpoints
        # are sampled with the second endpoint's square; this is intentionally
        # kept separate from the per-endpoint event ROI semantics above.
        visual_square = squares[-1]
        record["rgb_f16"] = np.stack(
            [
                _read_rgb(
                    rgb_shards[index],
                    rgb_members[index],
                    visual_square,
                    config.roi_size,
                )
                for index in endpoint_indices
            ]
        ).astype(np.float16)
    return record


def _save_shard(
    path: Path,
    records: Sequence[dict[str, object]],
    compression_level: int,
) -> str:
    arrays: dict[str, np.ndarray] = {
        "event_q": np.stack([np.asarray(record["event_q"]) for record in records]),
        "ttc_s": np.asarray([record["ttc_s"] for record in records], dtype=np.float32),
        "visible_height": np.stack(
            [np.asarray(record["visible_height"]) for record in records]
        ).astype(np.float32),
        "delta_t_s": np.asarray([record["delta_t_s"] for record in records], dtype=np.float32),
        "sequence_id": np.asarray([record["sequence_id"] for record in records]),
        "sample_token": np.asarray([record["sample_token"] for record in records]),
        "track_id": np.asarray([record["track_id"] for record in records]),
        "public_track_id": np.asarray([record["public_track_id"] for record in records]),
        "timestamp_us": np.asarray([record["timestamp_us"] for record in records], dtype=np.int64),
        "sensor_width": np.asarray([record["sensor_width"] for record in records], dtype=np.int32),
        "sensor_height": np.asarray(
            [record["sensor_height"] for record in records], dtype=np.int32
        ),
    }
    if "rgb_f16" in records[0]:
        arrays["rgb_f16"] = np.stack([np.asarray(record["rgb_f16"]) for record in records])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    # ``numpy.savez_compressed`` does not expose the deflate level.  Write the
    # same standard NPZ members explicitly so the cache manifest's compression
    # setting is real and reproducible.
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=compression_level,
        allowZip64=True,
    ) as archive:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            archive.writestr(
                f"{name}.npy",
                buffer.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=compression_level,
            )
    os.replace(temporary, path)
    return _sha256_file(path)


def build_garl_release_cache(
    *,
    eap_root: str | Path,
    garlttc_root: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    config: GarlReleaseCacheConfig,
    max_samples_per_split: int | None = None,
) -> dict[str, object]:
    """Materialize a release-semantic cache without changing the release tree."""

    eap_root = Path(eap_root).resolve()
    garlttc_root = Path(garlttc_root).resolve()
    split_path = Path(split_path).resolve()
    output = Path(output_dir).resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite cache manifest: {output / 'manifest.json'}")
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    assignments = split_payload.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("Split artifact lacks assignments.")
    sequence_to_split = {
        str(sequence): role
        for role in ("train", "validation")
        for sequence in assignments.get(role, [])
    }
    sequences = sorted(sequence_to_split)
    index = load_garlttc_train_index(garlttc_root, sequences)
    if len(index.merged) != config.expected_rows:
        raise ValueError(f"Expected {config.expected_rows} public rows, got {len(index.merged)}.")
    rows_by_split: dict[str, list[dict[str, object]]] = {"train": [], "validation": []}
    columns = [
        "sequence_id",
        "sample_token",
        "track_id",
        "public_track_id",
        "timestamp_us",
        "boxes_xyxy",
        "frame_timestamps_us",
        "event_windows_us",
        "events_path",
        "rgb_shard_paths",
        "rgb_member_paths",
        "frame_ttc",
        "box3d_h",
        "box3d_Fcam",
    ]
    records = cast(Any, cast(Any, index.merged[columns]).to_dict("records"))
    for row in records:
        split = sequence_to_split.get(str(row["sequence_id"]))
        if split is not None:
            rows_by_split[split].append(row)
    for rows in rows_by_split.values():
        rows.sort(key=lambda item: (str(item["sequence_id"]), str(item["sample_token"])))
        if max_samples_per_split is not None:
            if max_samples_per_split <= 0:
                raise ValueError("max_samples_per_split must be positive.")
            del rows[max_samples_per_split:]
    output.mkdir(parents=True, exist_ok=True)
    shard_metadata: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    sensor_resolutions: set[tuple[int, int]] = set()
    try:
        with ProcessPoolExecutor(
            max_workers=config.workers,
            initializer=_init_worker,
            initargs=(str(eap_root), config),
        ) as pool:
            for split in ("train", "validation"):
                rows = rows_by_split[split]
                split_dir = output / split
                for shard_index, start in enumerate(range(0, len(rows), config.shard_size)):
                    chunk = rows[start : start + config.shard_size]
                    records = list(
                        pool.map(
                            _materialize_one,
                            ((row, split) for row in chunk),
                        )
                    )
                    sensor_resolutions.update(
                        (
                            int(cast(Any, record["sensor_width"])),
                            int(cast(Any, record["sensor_height"])),
                        )
                        for record in records
                    )
                    path = split_dir / f"shard-{shard_index:05d}.npz"
                    digest = _save_shard(path, records, config.compression_level)
                    shard_metadata.append(
                        {
                            "split": split,
                            "path": path.relative_to(output).as_posix(),
                            "count": len(records),
                            "sha256": digest,
                        }
                    )
                counts[split] = len(rows)
    except Exception as exc:
        failure = {
            "artifact_type": "garl_release_input_cache_failure_v1",
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_shards": len(shard_metadata),
            "negative_result_preserved": True,
        }
        _write_json(output / "FAILURE.json", failure)
        raise
    manifest: dict[str, object] = {
        "artifact_type": RELEASE_CACHE_ARTIFACT,
        "schema_version": RELEASE_CACHE_SCHEMA,
        "status": "pass",
        "release_semantics": "garl_ttc.utils.events.extract_from_h5_by_timewindow_v1",
        "release_window_boundary_filter": False,
        "release_event_pixel_diff": config.event_pixel_diff,
        "release_commit": "256661242b8a7f5e56aa3c1c02348b30f6e89de6",
        "bbox_protocol": "P0_oracle_bbox_roi",
        "bbox_source": "public_garl_data_parquet_boxes_xyxy",
        "bbox_used_to_construct_model_input": True,
        "split_path": split_path.as_posix(),
        "split_sha256": _sha256_file(split_path),
        "garlttc_root": garlttc_root.as_posix(),
        "eap_root": eap_root.as_posix(),
        "garlttc_data_sha256": _sha256_file(garlttc_root / "data" / "train.parquet"),
        "garlttc_annotations_sha256": _sha256_file(garlttc_root / "annotations" / "train.parquet"),
        "sensor_resolution_source": "rgb_member_header_per_sample",
        "sensor_resolutions_width_height": [list(item) for item in sorted(sensor_resolutions)],
        "config": {
            "roi_size": config.roi_size,
            "roi_bins": config.roi_bins,
            "target_delta_t_s": config.target_delta_t_s,
            "delta_t_tolerance_s": config.delta_t_tolerance_s,
            "fy": config.fy,
            "shard_size": config.shard_size,
            "workers": config.workers,
            "include_rgb": config.include_rgb,
            "event_quantization": EVENT_QUANTIZATION,
            "rgb_storage": RGB_STORAGE if config.include_rgb else None,
        },
        "model_input_fields": ["event_roi"] + (["rgb_pair"] if config.include_rgb else []),
        "supervision_fields": ["ttc_s", "visible_height", "delta_t_s"],
        "forbidden_model_input_fields": [
            "ttc_s",
            "visible_height",
            "category",
            "box3d_Fcam",
            "box3d_h",
            "mask",
            "depth",
        ],
        "split_counts": counts,
        "shards": shard_metadata,
        "total_rows": sum(counts.values()),
        "negative_result_preserved": True,
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(
        output / "build_state.json",
        {
            "artifact_type": "garl_release_input_cache_build_state_v1",
            "status": "completed",
            "manifest_sha256": _sha256_file(output / "manifest.json"),
            "completed_shards": len(shard_metadata),
        },
    )
    return manifest


class GarlReleaseCacheDataset(Dataset[dict[str, Any]]):
    """Lazy reader that decodes quantized cache arrays once per active shard."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        fields: Sequence[str] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("artifact_type") != RELEASE_CACHE_ARTIFACT:
            raise ValueError("Unexpected release cache artifact type.")
        shards = [item for item in self.manifest.get("shards", []) if item.get("split") == split]
        self.shards = [(self.root / str(item["path"])).resolve() for item in shards]
        self.entries: list[tuple[int, int]] = []
        for shard_index, item in enumerate(shards):
            self.entries.extend((shard_index, local) for local in range(int(item["count"])))
        if not self.entries:
            raise ValueError(f"Release cache has no rows for split={split!r}.")
        if fields is not None:
            normalized_fields = frozenset(str(field) for field in fields)
            required_fields = {
                "ttc_s",
                "visible_height",
                "delta_t_s",
                "sequence_id",
                "sample_token",
            }
            missing = sorted(required_fields.difference(normalized_fields))
            if missing:
                raise ValueError(f"Selective cache fields omit required fields: {missing}")
            self.fields: frozenset[str] | None = normalized_fields
        else:
            self.fields = None
        self._active_index: int | None = None
        self._active: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.entries)

    def shard_index(self, index: int) -> int:
        return self.entries[index][0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_index, local_index = self.entries[index]
        if self._active_index != shard_index:
            with np.load(self.shards[shard_index], allow_pickle=False) as archive:
                selected_fields = getattr(self, "fields", None)
                keys = (
                    archive.files
                    if selected_fields is None
                    else selected_fields.intersection(archive.files)
                )
                self._active = {key: archive[key] for key in keys}
            self._active_index = shard_index
        if self._active is None:
            raise RuntimeError("Release cache shard was not loaded.")
        values = self._active
        output: dict[str, Any] = {
            "ttc_s": torch.from_numpy(np.asarray(values["ttc_s"][local_index])),
            "visible_height": torch.from_numpy(values["visible_height"][local_index]),
            "delta_t_s": torch.from_numpy(np.asarray(values["delta_t_s"][local_index])),
            "sequence_id": str(values["sequence_id"][local_index]),
            "sample_token": str(values["sample_token"][local_index]),
        }
        if "event_q" in values:
            output["event_roi"] = (
                torch.from_numpy(values["event_q"][local_index].astype(np.float32)) / 65535.0
            )
        if "rgb_f16" in values:
            output["rgb_pair"] = torch.from_numpy(values["rgb_f16"][local_index].astype(np.float32))
        if "sensor_width" in values and "sensor_height" in values:
            output["sensor_width"] = torch.as_tensor(values["sensor_width"][local_index])
            output["sensor_height"] = torch.as_tensor(values["sensor_height"][local_index])
        return output


class GarlReleaseShardBatchSampler(Sampler[list[int]]):
    """Shuffle within shards while never forcing a compressed shard reload per row."""

    def __init__(self, dataset: GarlReleaseCacheDataset, batch_size: int, seed: int = 0) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = int(seed)
        self.epoch = 0
        grouped: dict[int, list[int]] = {}
        for index in range(len(dataset)):
            grouped.setdefault(dataset.shard_index(index), []).append(index)
        self.groups = grouped

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        batches: list[list[int]] = []
        # Shuffle shard order, but keep the batches belonging to one compressed
        # shard adjacent.  The dataset keeps one decoded shard active; a global
        # batch shuffle would make the same worker reopen/decompress a shard for
        # every batch and leaves the accelerator idle on the full cache.
        shard_indices = list(self.groups)
        rng.shuffle(shard_indices)
        for shard_index in shard_indices:
            indices = self.groups[shard_index]
            values = list(indices)
            rng.shuffle(values)
            batches.extend(
                values[start : start + self.batch_size]
                for start in range(0, len(values), self.batch_size)
            )
        yield from batches

    def __len__(self) -> int:
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.groups.values())


__all__ = [
    "EVENT_QUANTIZATION",
    "CACHE_SAMPLER_VERSION",
    "GarlReleaseCacheConfig",
    "GarlReleaseCacheDataset",
    "GarlReleaseShardBatchSampler",
    "RELEASE_CACHE_ARTIFACT",
    "RGB_STORAGE",
    "build_garl_release_cache",
]
