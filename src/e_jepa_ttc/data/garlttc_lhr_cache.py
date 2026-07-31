"""Official GarlTTC-label object cache for LHR + object-JEPA training.

This module is deliberately separate from the public-track reconstruction cache.
It requires the public GarlTTC train parquets, joins them with the existing
audited five-key loader, and never falls back to reconstructed TTC labels.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import tarfile
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from e_jepa_ttc.data.eap import EAP_IMAGE_SIZE, EAPEventReader
from e_jepa_ttc.data.eap_geometry_v2 import (
    EAP_GEOMETRY_V2_DIM,
    category_index,
    coarse_category,
)
from e_jepa_ttc.data.eap_representation import base_compatible_voxel, downsample_full_frame
from e_jepa_ttc.data.garlttc_eap import (
    GARLTTC_JOIN_KEYS,
    load_garlttc_train_index,
    normalize_boxes_xyxy,
    normalize_event_windows_us,
    resolve_eap_events_path,
    validate_garlttc_train_index,
)
from e_jepa_ttc.data.types import EventBatch
from e_jepa_ttc.representations.voxel_grid import encode_voxel_grid
from e_jepa_ttc.utils.io import read_structured, write_structured

OBSERVABLE_MOTION_NAMES: tuple[str, ...] = (
    "center_x",
    "center_y",
    "width",
    "height",
    "center_velocity_x",
    "center_velocity_y",
    "log_width_rate",
    "log_height_rate",
    "lateral_speed",
    "radiality",
    "visibility_fraction",
    "truncated_left",
    "truncated_right",
    "corridor_center_distance",
    "corridor_overlap",
    "corridor_velocity",
    "corridor_entry_direction",
    "delta_t",
)
OBSERVABLE_MOTION_DIM = len(OBSERVABLE_MOTION_NAMES)

FORBIDDEN_MODEL_INPUT_KEYS = frozenset(
    {
        "ttc_s",
        "geometry_v2_target",
        "geometry_v2_valid",
        "category_index",
        "category_valid",
        "context_depth_history_m",
        "box3d_Fcam",
        "box3d_h",
        "target_ttc",
    }
)


@dataclass(frozen=True)
class GarlTTCLHRCacheConfig:
    """Materialization controls for the official-label cache."""

    full_width: int = 160
    full_height: int = 90
    full_bins: int = 5
    roi_size: int = 128
    roi_bins: int = 10
    shard_size: int = 256
    include_rgb: bool = False
    expected_train_rows: int = 88_744
    allow_dataset_version_change: bool = False

    def __post_init__(self) -> None:
        positive = (
            self.full_width,
            self.full_height,
            self.full_bins,
            self.roi_size,
            self.roi_bins,
            self.shard_size,
            self.expected_train_rows,
        )
        if min(positive) <= 0:
            raise ValueError("Cache dimensions and counts must be positive.")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _as_list(value: object) -> list[Any]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        stripped = value.strip()
        try:
            value = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            value = json.loads(stripped)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Expected a list-like value, got {type(value)!r}.")
    return list(value)


def _nested_float(value: object) -> Any:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (list, tuple)):
        return [_nested_float(item) for item in value]
    return float(value)


def _normalized_box(box: tuple[float, float, float, float]) -> np.ndarray:
    width, height = EAP_IMAGE_SIZE
    x0, y0, x1, y1 = (float(item) for item in box)
    return np.asarray([x0 / width, y0 / height, x1 / width, y1 / height], dtype=np.float32)


def _box_features(
    first_box: tuple[float, float, float, float],
    second_box: tuple[float, float, float, float],
    delta_t_s: float,
    *,
    corridor_half_width: float = 0.18,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build strictly observable 2-D motion features.

    No TTC, depth, 3-D geometry, category, or future frame outside the supplied
    two causal endpoints is accessed here.
    """

    first = _normalized_box(first_box)
    second = _normalized_box(second_box)
    first_center = 0.5 * (first[:2] + first[2:])
    second_center = 0.5 * (second[:2] + second[2:])
    first_size = np.maximum(first[2:] - first[:2], 1e-6)
    second_size = np.maximum(second[2:] - second[:2], 1e-6)
    dt = max(float(delta_t_s), 1e-6)
    velocity = (second_center - first_center) / dt
    log_size_rate = (np.log(second_size) - np.log(first_size)) / dt
    lateral = float(np.abs(velocity).sum())
    radiality = float(np.tanh(abs(float(log_size_rate[1])) / (lateral + 1e-3)))

    image_width, _image_height = EAP_IMAGE_SIZE
    x0, _y0, x1, _y1 = (float(item) for item in second_box)
    clipped_width = max(0.0, min(x1, image_width) - max(x0, 0.0))
    raw_width = max(x1 - x0, 1e-6)
    visibility = float(np.clip(clipped_width / raw_width, 0.0, 1.0))
    truncated_left = float(x0 <= 0.0)
    truncated_right = float(x1 >= image_width - 1.0)

    corridor_left = 0.5 - corridor_half_width
    corridor_right = 0.5 + corridor_half_width
    overlap = max(0.0, min(float(second[2]), corridor_right) - max(float(second[0]), corridor_left))
    corridor_overlap = overlap / max(float(second_size[0]), 1e-6)
    center_offset = float(second_center[0] - 0.5)
    corridor_distance = min(abs(center_offset) / corridor_half_width, 2.0) / 2.0
    corridor_velocity = -math.copysign(1.0, center_offset) * float(velocity[0])
    entry_direction = float(np.sign(corridor_velocity))

    values = np.asarray(
        [
            second_center[0],
            second_center[1],
            second_size[0],
            second_size[1],
            np.clip(velocity[0] / 3.0, -1.0, 1.0),
            np.clip(velocity[1] / 3.0, -1.0, 1.0),
            np.clip(log_size_rate[0] / 5.0, -1.0, 1.0),
            np.clip(log_size_rate[1] / 5.0, -1.0, 1.0),
            np.clip(lateral / 3.0, 0.0, 1.0),
            radiality,
            visibility,
            truncated_left,
            truncated_right,
            corridor_distance,
            np.clip(corridor_overlap, 0.0, 1.0),
            np.clip(corridor_velocity / 3.0, -1.0, 1.0),
            entry_direction,
            np.clip(dt / 0.5, 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    metadata = {
        "lateral_speed_raw": lateral,
        "radiality_raw": radiality,
        "visibility_fraction": visibility,
        "corridor_overlap": float(np.clip(corridor_overlap, 0.0, 1.0)),
        "log_height_rate_raw": float(log_size_rate[1]),
        "log_area_rate_raw": float(log_size_rate.sum()),
        "center_velocity_x_raw": float(velocity[0]),
        "center_velocity_y_raw": float(velocity[1]),
    }
    return values, metadata


def observable_motion_from_boxes_torch(
    boxes: torch.Tensor,
    delta_t_s: torch.Tensor,
) -> torch.Tensor:
    """Torch adapter used by zero-shot EvTTC without privileged inputs."""

    if boxes.ndim == 4:
        boxes = boxes[:, :, 0]
    if boxes.ndim != 3 or boxes.shape[1] < 2 or boxes.shape[-1] != 4:
        raise ValueError("context_boxes must have shape [B,T,4] or [B,T,N,4].")
    rows = []
    boxes_cpu = boxes.detach().cpu().float()
    delta_cpu = delta_t_s.detach().cpu().float().reshape(-1)
    for index in range(boxes_cpu.shape[0]):
        first = tuple(float(value) for value in boxes_cpu[index, -2])
        second = tuple(float(value) for value in boxes_cpu[index, -1])
        # EvTTC caches normally store normalized boxes. Detect pixel boxes.
        if max(first + second) <= 2.0:
            width, height = EAP_IMAGE_SIZE
            first = (first[0] * width, first[1] * height, first[2] * width, first[3] * height)
            second = (
                second[0] * width,
                second[1] * height,
                second[2] * width,
                second[3] * height,
            )
        values, _ = _box_features(first, second, float(delta_cpu[index]))
        rows.append(torch.from_numpy(values))
    return torch.stack(rows).to(device=boxes.device, dtype=torch.float32)


def _square_box(boxes: list[tuple[float, float, float, float]], index: int) -> tuple[float, float, float, float]:
    max_edge = max(max(box[2] - box[0], box[3] - box[1]) for box in boxes)
    x0, y0, x1, y1 = boxes[index]
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    half = max(max_edge, 2.0) * 0.5
    return (cx - half, cy - half, cx + half, cy + half)


def _roi_voxel(
    events: dict[str, np.ndarray],
    square: tuple[float, float, float, float],
    *,
    size: int,
    bins: int,
    sequence_id: str,
    start_us: int,
    end_us: int,
) -> torch.Tensor:
    x0, y0, x1, y1 = square
    x = np.asarray(events["x"], dtype=np.float32)
    y = np.asarray(events["y"], dtype=np.float32)
    t = np.asarray(events["t"], dtype=np.int64)
    p = np.asarray(events["p"])
    inside = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    edge = max(x1 - x0, y1 - y0, 1e-6)
    rx = np.clip(((x[inside] - x0) * size / edge).astype(np.int32), 0, size - 1)
    ry = np.clip(((y[inside] - y0) * size / edge).astype(np.int32), 0, size - 1)
    event_batch = EventBatch(
        x=rx,
        y=ry,
        t_us=t[inside],
        polarity=np.where(p[inside] > 0, 1, -1).astype(np.int8),
        width=size,
        height=size,
        sequence_id=sequence_id,
        t_start_us=start_us,
        t_end_us=end_us,
    )
    return torch.from_numpy(encode_voxel_grid(event_batch, bins=bins, normalize=True))


class _RGBTarReader:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: OrderedDict[Path, tarfile.TarFile] = OrderedDict()

    def read(
        self,
        shard: object,
        member: object,
        *,
        square: tuple[float, float, float, float],
        size: int,
    ) -> torch.Tensor:
        path = self.root / str(shard)
        archive = self._cache.get(path)
        if archive is None:
            archive = tarfile.open(path, "r")
            self._cache[path] = archive
            while len(self._cache) > 4:
                _, old = self._cache.popitem(last=False)
                old.close()
        extracted = archive.extractfile(str(member))
        if extracted is None:
            raise FileNotFoundError(f"{member} inside {path}")
        image = Image.open(io.BytesIO(extracted.read())).convert("RGB")
        x0, y0, x1, y1 = square
        image = image.crop(
            (
                int(math.floor(x0)),
                int(math.floor(y0)),
                int(math.ceil(x1)),
                int(math.ceil(y1)),
            )
        ).resize((size, size))
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(array)


def _visible_heights(row: Any, endpoint_indices: tuple[int, int], boxes: list[tuple[float, float, float, float]], roi_size: int) -> np.ndarray:
    if "box3d_h" not in row or "box3d_Fcam" not in row or "K_event" not in row:
        raise ValueError("Official LHR targets require box3d_h, box3d_Fcam, and K_event.")
    height_3d = float(row["box3d_h"])
    corners_per_frame = _as_list(row["box3d_Fcam"])
    intrinsics = np.asarray(_nested_float(row["K_event"]), dtype=np.float64)
    if intrinsics.size != 9:
        raise ValueError("K_event must contain a 3x3 intrinsic matrix.")
    fy = float(intrinsics.reshape(3, 3)[1, 1])
    max_edge = max(max(box[2] - box[0], box[3] - box[1]) for box in boxes)
    scaling = roi_size / max(max_edge, 1e-6)
    output = []
    for index in endpoint_indices:
        corners = np.asarray(_nested_float(corners_per_frame[index]), dtype=np.float64).reshape(-1, 3)
        min_depth = float(np.min(corners[:, 2]))
        if min_depth <= 0:
            raise ValueError("box3d_Fcam contains non-positive depth.")
        output.append(fy * height_3d / min_depth * scaling)
    return np.asarray(output, dtype=np.float32)


def _geometry_target(
    observable: np.ndarray,
    metadata: dict[str, float],
    *,
    depths: tuple[float, float] | None,
    delta_t_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(EAP_GEOMETRY_V2_DIM, dtype=np.float32)
    valid = np.zeros(EAP_GEOMETRY_V2_DIM, dtype=np.bool_)
    # Observable targets.
    values[0:4] = observable[0:4]
    values[5] = observable[7]
    values[6:10] = observable[4:6].tolist() + [observable[8], observable[9]]
    values[10:13] = observable[10:13]
    values[13] = observable[17]
    values[14] = np.clip(metadata["log_area_rate_raw"] / 5.0, -1.0, 1.0)
    values[15] = float(abs(metadata["log_area_rate_raw"]) > 3.0)
    values[16:20] = observable[13:17]
    valid[:] = True
    valid[4] = False
    if depths is not None:
        closing = -(depths[1] - depths[0]) / max(delta_t_s, 1e-6)
        values[4] = np.clip(closing / 20.0, -1.0, 1.0)
        valid[4] = True
    return values, valid


def _sampling_group(category: str, metadata: dict[str, float]) -> str:
    motion = "transverse" if metadata["lateral_speed_raw"] > 0.25 else "longitudinal"
    visibility = "partial" if metadata["visibility_fraction"] < 0.95 else "visible"
    corridor = "intersecting" if metadata["corridor_overlap"] >= 0.25 else "off_corridor"
    return f"{coarse_category(category)}:{motion}:{visibility}:{corridor}"


def _materialize_row(
    row: Any,
    *,
    eap_root: Path,
    config: GarlTTCLHRCacheConfig,
    event_readers: dict[Path, EAPEventReader],
    rgb_reader: _RGBTarReader | None,
) -> dict[str, Any]:
    boxes = normalize_boxes_xyxy(row["boxes_xyxy"])
    windows = normalize_event_windows_us(row["event_windows_us"])
    frame_timestamps = [int(value) for value in _as_list(row["frame_timestamps_us"])]
    if min(len(boxes), len(windows), len(frame_timestamps)) < 2:
        raise ValueError("LHR sample requires at least two aligned frames.")
    first_index, second_index = 0, min(len(boxes), len(windows), len(frame_timestamps)) - 1
    delta_t_s = (frame_timestamps[second_index] - frame_timestamps[first_index]) * 1e-6
    if delta_t_s <= 0:
        raise ValueError("Frame timestamps must increase.")

    events_path = resolve_eap_events_path(eap_root, str(row["events_path"]))
    reader = event_readers.get(events_path)
    if reader is None:
        reader = EAPEventReader(events_path)
        event_readers[events_path] = reader
    endpoint_events = []
    full_frames = []
    for index in (first_index, second_index):
        start_us, end_us = windows[index]
        raw = reader.read_window(start_us, end_us)
        endpoint_events.append(
            _roi_voxel(
                raw,
                _square_box(boxes, index),
                size=config.roi_size,
                bins=config.roi_bins,
                sequence_id=str(row["sequence_id"]),
                start_us=start_us,
                end_us=end_us,
            )
        )
        full_batch = downsample_full_frame(
            raw,
            sequence_id=str(row["sequence_id"]),
            start_us=start_us,
            end_us=end_us,
            width=config.full_width,
            height=config.full_height,
        )
        full_frames.append(base_compatible_voxel(full_batch, bins=config.full_bins))

    observable, metadata = _box_features(boxes[first_index], boxes[second_index], delta_t_s)
    visible_heights = _visible_heights(row, (first_index, second_index), boxes, config.roi_size)
    depths = None
    try:
        corners = _as_list(row["box3d_Fcam"])
        first_corners = np.asarray(_nested_float(corners[first_index]), dtype=np.float64).reshape(-1, 3)
        second_corners = np.asarray(_nested_float(corners[second_index]), dtype=np.float64).reshape(-1, 3)
        depths = (float(first_corners[:, 2].min()), float(second_corners[:, 2].min()))
    except (KeyError, TypeError, ValueError, IndexError):
        depths = None
    geometry, geometry_valid = _geometry_target(
        observable,
        metadata,
        depths=depths,
        delta_t_s=delta_t_s,
    )
    category = str(row.get("category", row.get("category_name", "unknown")))
    sample = {
        "full_frame_events": torch.stack(full_frames).numpy().astype(np.float32),
        "garl_event_roi": torch.cat(endpoint_events, dim=0).numpy().astype(np.float32),
        "garl_delta_t_s": np.float32(delta_t_s),
        "observable_motion": observable,
        "geometry_v2_target": geometry,
        "geometry_v2_valid": geometry_valid,
        "garl_visible_heights_px": visible_heights,
        "ttc_s": np.float32(float(row["ttc"])),
        "category_index": np.int64(category_index(category)),
        "category_valid": np.bool_(category != "unknown"),
        "sampling_group": _sampling_group(category, metadata),
        "category": category,
        "sequence_id": str(row["sequence_id"]),
        "sample_token": str(row["sample_token"]),
        "track_id": str(row["track_id"]),
        "public_track_id": str(row["public_track_id"]),
        "timestamp_us": np.int64(row["timestamp_us"]),
        "ttc_label_source": "official_garlttc_annotations_train_parquet",
    }
    if config.include_rgb:
        if rgb_reader is None:
            raise RuntimeError("RGB reader is unavailable.")
        shards = _as_list(row["rgb_shard_paths"])
        members = _as_list(row["rgb_member_paths"])
        sample["garl_rgb_pair"] = torch.stack(
            [
                rgb_reader.read(
                    shards[first_index],
                    members[first_index],
                    square=_square_box(boxes, first_index),
                    size=config.roi_size,
                ),
                rgb_reader.read(
                    shards[second_index],
                    members[second_index],
                    square=_square_box(boxes, second_index),
                    size=config.roi_size,
                ),
            ]
        ).numpy().astype(np.float32)
    return sample


def materialize_garlttc_lhr_cache(
    *,
    eap_root: str | Path,
    garlttc_root: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    config: GarlTTCLHRCacheConfig,
    max_samples_per_split: int | None = None,
) -> dict[str, Any]:
    """Build an auditable cache using official public GarlTTC train labels only."""

    eap_root = Path(eap_root).resolve()
    garlttc_root = Path(garlttc_root).resolve()
    split_artifact = read_structured(split_path)
    assignments = split_artifact.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("Split artifact has no assignments mapping.")
    sequence_role = {
        str(sequence): role
        for role in ("train", "validation")
        for sequence in assignments.get(role, [])
    }
    index = load_garlttc_train_index(garlttc_root, sorted(sequence_role))
    validate_garlttc_train_index(
        index,
        expected_rows=config.expected_train_rows,
        allow_version_change=config.allow_dataset_version_change,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    event_readers: dict[Path, EAPEventReader] = {}
    rgb_reader = _RGBTarReader(eap_root) if config.include_rgb else None
    shard_records: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    shard_meta: list[dict[str, Any]] = []
    discard_reasons: dict[str, int] = {}
    split_counts = {"train": 0, "validation": 0}

    def flush(role: str) -> None:
        records = shard_records[role]
        if not records:
            return
        role_dir = output / role
        role_dir.mkdir(parents=True, exist_ok=True)
        shard_index = sum(1 for item in shard_meta if item["split"] == role)
        path = role_dir / f"shard-{shard_index:05d}.pt"
        torch.save(records, path)
        shard_meta.append(
            {
                "split": role,
                "path": path.relative_to(output).as_posix(),
                "count": len(records),
                "sha256": _sha256_file(path),
            }
        )
        records.clear()

    rows = index.merged.sort_values(
        ["sequence_id", "timestamp_us", "track_id", "sample_token"],
        kind="mergesort",
    )
    for _, row in rows.iterrows():
        role = sequence_role.get(str(row["sequence_id"]))
        if role not in shard_records:
            continue
        if max_samples_per_split is not None and split_counts[role] >= max_samples_per_split:
            continue
        try:
            sample = _materialize_row(
                row,
                eap_root=eap_root,
                config=config,
                event_readers=event_readers,
                rgb_reader=rgb_reader,
            )
        except Exception as exc:  # explicit accounting; never silently substitute labels
            key = f"{type(exc).__name__}:{str(exc)[:120]}"
            discard_reasons[key] = discard_reasons.get(key, 0) + 1
            continue
        shard_records[role].append(sample)
        split_counts[role] += 1
        if len(shard_records[role]) >= config.shard_size:
            flush(role)
    flush("train")
    flush("validation")
    if min(split_counts.values()) <= 0:
        raise RuntimeError(f"Official cache produced an empty split: {split_counts}")

    data_path = garlttc_root / "data" / "train.parquet"
    annotations_path = garlttc_root / "annotations" / "train.parquet"
    manifest = {
        "artifact_type": "garlttc_official_lhr_object_cache_v2",
        "format": "torch_sharded_list_v1",
        "config": asdict(config),
        "eap_root": eap_root.as_posix(),
        "garlttc_root": garlttc_root.as_posix(),
        "split_path": Path(split_path).as_posix(),
        "split_sha256": _sha256_file(Path(split_path)),
        "garlttc_data_sha256": _sha256_file(data_path),
        "garlttc_annotations_sha256": _sha256_file(annotations_path),
        "garlttc_join_keys_sha256": index.join_keys_sha256,
        "join_keys": GARLTTC_JOIN_KEYS,
        "ttc_label_source": "official_garlttc_annotations_train_parquet",
        "uses_official_garl_ttc_labels": True,
        "uses_reconstructed_public_eap_ttc": False,
        "no_label_fallback": True,
        "model_input_fields": [
            "full_frame_events",
            "garl_event_roi",
            "garl_delta_t_s",
            "observable_motion",
            *([] if not config.include_rgb else ["garl_rgb_pair"]),
        ],
        "supervision_only_fields": [
            "ttc_s",
            "garl_visible_heights_px",
            "geometry_v2_target",
            "geometry_v2_valid",
            "category_index",
            "category_valid",
        ],
        "forbidden_model_input_fields": sorted(FORBIDDEN_MODEL_INPUT_KEYS),
        "observable_motion_names": list(OBSERVABLE_MOTION_NAMES),
        "geometry_target_dim": EAP_GEOMETRY_V2_DIM,
        "split_counts": split_counts,
        "discard_reasons": dict(sorted(discard_reasons.items())),
        "shards": shard_meta,
    }
    write_structured(output / "manifest.json", manifest)
    return manifest


class GarlTTCLHRCacheDataset(Dataset[dict[str, Any]]):
    """Read materialized official-label shards with a one-shard process cache."""

    def __init__(self, manifest_path: str | Path, *, splits: tuple[str, ...]) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = read_structured(self.manifest_path)
        if self.manifest.get("uses_official_garl_ttc_labels") is not True:
            raise ValueError("LHR-v2 training requires an official-label cache.")
        self.entries: list[tuple[Path, int]] = []
        for shard in self.manifest.get("shards", []):
            if shard.get("split") not in splits:
                continue
            path = self.root / str(shard["path"])
            for index in range(int(shard["count"])):
                self.entries.append((path, index))
        if not self.entries:
            raise ValueError(f"No cache samples found for splits={splits}.")
        self._cached_path: Path | None = None
        self._cached_records: list[dict[str, Any]] | None = None

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, local_index = self.entries[index]
        if self._cached_path != path:
            records = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(records, list):
                raise TypeError(f"Shard {path} is not a list.")
            self._cached_path = path
            self._cached_records = records
        assert self._cached_records is not None
        record = self._cached_records[local_index]
        output: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, np.ndarray):
                output[key] = torch.from_numpy(value)
            elif isinstance(value, np.generic):
                output[key] = value.item()
            else:
                output[key] = value
        return output
