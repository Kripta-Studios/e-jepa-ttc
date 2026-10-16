"""Official GarlTTC-label object cache for LHR + object-JEPA training.

This module is deliberately separate from the public-track reconstruction cache.
It requires the public GarlTTC train parquets, joins them with the existing
audited five-key loader, and never falls back to reconstructed TTC labels.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import io
import json
import math
import os
import subprocess
import tarfile
from collections import Counter, OrderedDict
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

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
from e_jepa_ttc.data.eap_representation import (
    base_compatible_voxel,
    downsample_full_frame,
    event_voxel_with_scalars,
)
from e_jepa_ttc.data.event_v4_geometry import (
    EVENT_V4_STEPS,
    box_in_common_roi,
    common_square_from_boxes,
    event_v4_channel_count,
    event_v4_channel_names,
    shifted_precontext_window,
)
from e_jepa_ttc.data.garl_input_contract import (
    EVENT_CHANNEL_NAMES,
    INPUT_SCHEMA_VERSION,
    NORMALIZATION_ID,
    validate_cache_manifest_input_schema,
)
from e_jepa_ttc.data.garl_official_preprocessing import (
    official_resize_feature,
    official_square_box,
    official_timevolume_roi_np,
)
from e_jepa_ttc.data.garlttc_calibration import CalibrationResolver
from e_jepa_ttc.data.garlttc_eap import (
    GARLTTC_JOIN_KEYS,
    load_garlttc_train_index,
    normalize_boxes_xyxy,
    normalize_event_windows_us,
    resolve_eap_events_path,
    validate_garlttc_train_index,
)
from e_jepa_ttc.data.garlttc_sampling import select_balanced_cache_rows, signed_ttc_bucket
from e_jepa_ttc.data.types import EventBatch
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
    store_full_frame_events: bool = True
    store_garl_event_roi: bool = True
    store_jepa_event_roi: bool = True
    store_event_v4_common_roi: bool = False
    event_v4_margin_fraction: float = 0.25
    event_v4_require_precontext: bool = True
    event_v4_precontext_fallback: str = "shifted_event_window"
    event_v4_bins_per_polarity: int = 5
    event_v4_storage_dtype: str = "float32"
    materialize_splits: tuple[Literal["train", "validation"], ...] = (
        "train",
        "validation",
    )
    include_rgb: bool = False
    include_masks: bool = False
    mask_required: bool = False
    jepa_roi_bins: int = 5
    expected_train_rows: int = 88_744
    allow_dataset_version_change: bool = False
    target_delta_t_s: float = 0.1
    delta_t_tolerance_s: float = 0.025
    jepa_context_delta_t_s: float = 0.1
    jepa_context_tolerance_s: float = 0.05
    calibration_mode: str = "official_constant_fy"
    selection_seed: int = 7
    event_pixel_diff: int = 5
    preprocessing_device: str = "cpu"
    workers: int = 1
    compression: str = "none"
    compression_level: int = 1

    def __post_init__(self) -> None:
        positive = (
            self.full_width,
            self.full_height,
            self.full_bins,
            self.roi_size,
            self.roi_bins,
            self.jepa_roi_bins,
            self.event_v4_bins_per_polarity,
            self.shard_size,
            self.expected_train_rows,
        )
        if min(positive) <= 0:
            raise ValueError("Cache dimensions and counts must be positive.")
        if self.roi_bins != 10:
            raise ValueError("Official Garl-TTC input v4 requires exactly 10 ROI bins.")
        if not any(
            (
                self.store_full_frame_events,
                self.store_garl_event_roi,
                self.store_jepa_event_roi,
                self.store_event_v4_common_roi,
            )
        ):
            raise ValueError("At least one event representation must be materialized.")
        temporal = (
            self.target_delta_t_s,
            self.delta_t_tolerance_s,
            self.jepa_context_delta_t_s,
            self.jepa_context_tolerance_s,
        )
        if min(temporal) <= 0:
            raise ValueError("Temporal pairing controls must be positive.")
        if self.event_v4_margin_fraction < 0.0:
            raise ValueError("event_v4_margin_fraction must be non-negative.")
        if self.event_v4_storage_dtype not in {"float16", "float32"}:
            raise ValueError("event_v4_storage_dtype must be float16 or float32.")
        if not self.materialize_splits or set(self.materialize_splits) - {
            "train",
            "validation",
        }:
            raise ValueError("materialize_splits must contain train and/or validation.")
        if len(set(self.materialize_splits)) != len(self.materialize_splits):
            raise ValueError("materialize_splits must not contain duplicates.")
        if self.event_v4_precontext_fallback not in {
            "disabled",
            "shifted_event_window",
        }:
            raise ValueError(
                "event_v4_precontext_fallback must be 'disabled' or "
                "'shifted_event_window'."
            )
        if self.calibration_mode not in {
            "official_constant_fy",
            "per_sample_eap_intrinsics",
        }:
            raise ValueError(f"Unsupported calibration mode: {self.calibration_mode!r}.")
        if self.preprocessing_device not in {"cpu", "cuda"}:
            raise ValueError("preprocessing_device must be 'cpu' or 'cuda'.")
        if self.preprocessing_device == "cuda" and not torch.cuda.is_available():
            raise ValueError("preprocessing_device='cuda' requires an available CUDA device.")
        if self.workers <= 0:
            raise ValueError("workers must be positive.")
        if self.compression not in {"none", "gzip"}:
            raise ValueError("compression must be 'none' or 'gzip'.")
        if not 0 <= self.compression_level <= 9:
            raise ValueError("compression_level must lie in [0, 9].")
        if self.include_rgb and self.workers > 1:
            raise ValueError("include_rgb with workers > 1 is not supported by the tar reader.")
        if self.mask_required and not self.include_masks:
            raise ValueError("mask_required=True requires include_masks=True.")


_CACHE_WORKER_EAP_ROOT: Path | None = None
_CACHE_WORKER_CONFIG: GarlTTCLHRCacheConfig | None = None
_CACHE_WORKER_EVENT_READERS: dict[Path, EAPEventReader] | None = None
_CACHE_WORKER_RGB_READER: _RGBTarReader | None = None
_CACHE_WORKER_MASK_READER: _MaskReader | None = None
_CACHE_WORKER_CALIBRATION: CalibrationResolver | None = None


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON control artifact without exposing a half-written file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(
    value: object,
    path: Path,
    *,
    compression: str = "none",
    compression_level: int = 1,
) -> None:
    """Write a torch shard atomically, optionally with lossless gzip."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if compression == "none":
        temporary = path.with_name(f".{path.name}.tmp")
        torch.save(value, temporary)
        os.replace(temporary, path)
        return
    if compression != "gzip":
        raise ValueError(f"Unsupported shard compression: {compression!r}.")
    raw_temporary = path.with_name(f".{path.name}.raw.tmp")
    compressed_temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(value, raw_temporary)
        with (
            raw_temporary.open("rb") as source,
            gzip.open(
                compressed_temporary,
                "wb",
                compresslevel=compression_level,
            ) as target,
        ):
            while chunk := source.read(8 * 1024 * 1024):
                target.write(chunk)
        os.replace(compressed_temporary, path)
    finally:
        raw_temporary.unlink(missing_ok=True)
        compressed_temporary.unlink(missing_ok=True)


def _load_torch_records(path: Path) -> list[dict[str, Any]]:
    """Load a raw or gzip-compressed cache shard without changing its values."""

    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as handle:
                records = torch.load(
                    cast(Any, handle), map_location="cpu", weights_only=False
                )
        else:
            records = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        size = path.stat().st_size if path.is_file() else None
        raise RuntimeError(
            f"Unable to read cache shard {path} (size_bytes={size}). "
            "Rebuild the cache with the object_lhr_minimal storage profile "
            "and a small shard size; oversized multi-gigabyte torch archives "
            "are not reliable on all Windows/PyTorch readers."
        ) from exc
    if not isinstance(records, list):
        raise TypeError(f"Shard {path} is not a list.")
    return records


def _cache_row_identity(row: Mapping[str, object]) -> tuple[str, ...]:
    """Return the stable annotation identity used by cache fingerprints."""

    return tuple(
        str(row.get(key, ""))
        for key in (
            "sequence_id",
            "sample_token",
            "track_id",
            "public_track_id",
            "timestamp_us",
        )
    )


def _cache_fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _init_cache_worker(eap_root: str, config: GarlTTCLHRCacheConfig) -> None:
    """Initialize one process-local reader/calibration set.

    The official preprocessing remains unchanged. Only ownership and scheduling
    move into the worker so HDF5 handles and calibration indexes are reused.
    """

    global _CACHE_WORKER_CONFIG
    global _CACHE_WORKER_EAP_ROOT
    global _CACHE_WORKER_EVENT_READERS
    global _CACHE_WORKER_RGB_READER
    global _CACHE_WORKER_MASK_READER
    global _CACHE_WORKER_CALIBRATION

    _CACHE_WORKER_EAP_ROOT = Path(eap_root)
    _CACHE_WORKER_CONFIG = config
    _CACHE_WORKER_EVENT_READERS = {}
    _CACHE_WORKER_CALIBRATION = CalibrationResolver(config.calibration_mode, eap_root=eap_root)
    _CACHE_WORKER_RGB_READER = _RGBTarReader(_CACHE_WORKER_EAP_ROOT) if config.include_rgb else None
    _CACHE_WORKER_MASK_READER = (
        _MaskReader(_CACHE_WORKER_EAP_ROOT) if config.include_masks else None
    )
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only allows this before its first parallel operation.
        pass


def _materialize_cache_worker(
    task: tuple[int, dict[str, object], int],
) -> tuple[int, dict[str, Any] | None, str | None]:
    """Materialize one row in a process with persistent local readers."""

    order, row, first_track_timestamp_us = task
    if (
        _CACHE_WORKER_EAP_ROOT is None
        or _CACHE_WORKER_CONFIG is None
        or _CACHE_WORKER_EVENT_READERS is None
        or _CACHE_WORKER_CALIBRATION is None
    ):
        raise RuntimeError("Cache worker was not initialized.")
    try:
        sample = _materialize_row(
            row,
            eap_root=_CACHE_WORKER_EAP_ROOT,
            config=_CACHE_WORKER_CONFIG,
            event_readers=_CACHE_WORKER_EVENT_READERS,
            rgb_reader=_CACHE_WORKER_RGB_READER,
            mask_reader=_CACHE_WORKER_MASK_READER,
            calibration=_CACHE_WORKER_CALIBRATION,
            first_track_timestamp_us=first_track_timestamp_us,
        )
    except Exception as exc:  # explicit accounting; never silently substitute labels
        return order, None, f"{type(exc).__name__}:{str(exc)[:120]}"
    return order, sample, None


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _repository_provenance() -> dict[str, str | None]:
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    diff = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    return {
        "code_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty_diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
    }


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


def _nested_float(value: object) -> object:
    as_py = getattr(value, "as_py", None)
    if callable(as_py):
        value = as_py()
    if isinstance(value, np.ndarray):
        value = value.tolist()
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

    image_width, image_height = EAP_IMAGE_SIZE
    x0, y0, x1, y1 = (float(item) for item in second_box)
    clipped_width = max(0.0, min(x1, image_width) - max(x0, 0.0))
    clipped_height = max(0.0, min(y1, image_height) - max(y0, 0.0))
    raw_area = max((x1 - x0) * (y1 - y0), 1e-6)
    visibility = float(np.clip((clipped_width * clipped_height) / raw_area, 0.0, 1.0))
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


def _square_box(
    boxes: list[tuple[float, float, float, float]],
    index: int,
) -> tuple[int, int, int, int]:
    return official_square_box(boxes, index)


def _roi_voxel(
    events: dict[str, np.ndarray],
    square: tuple[float, float, float, float],
    *,
    size: int,
    bins: int,
    sequence_id: str,
    start_us: int,
    end_us: int,
    event_pixel_diff: int,
    preprocessing_device: str,
) -> torch.Tensor:
    del sequence_id, start_us, end_us
    source_width, source_height = EAP_IMAGE_SIZE
    x = np.asarray(events["x"], dtype=np.int64) + event_pixel_diff
    y = np.asarray(events["y"], dtype=np.int64)
    timestamps = np.asarray(events["t"], dtype=np.int64)
    valid = (x >= 0) & (x < source_width) & (y >= 0) & (y < source_height)
    feature, _ = official_timevolume_roi_np(
        square,
        x[valid],
        y[valid],
        timestamps[valid],
        number_of_planes=bins * 2,
    )
    return official_resize_feature(feature, (size, size), device=preprocessing_device)


def _jepa_roi_voxel(
    events: dict[str, np.ndarray],
    square: tuple[float, float, float, float],
    *,
    size: int,
    bins: int,
    sequence_id: str,
    start_us: int,
    end_us: int,
    event_pixel_diff: int,
) -> torch.Tensor:
    """Build the same 21-channel representation used by the JEPA backbone.

    Coordinates are mapped inside the official square ROI, including zero
    padding when the square extends outside the source image. This preserves the
    pretraining channel contract while making the downstream sample object
    specific.
    """

    source_width, source_height = EAP_IMAGE_SIZE
    x = np.asarray(events["x"], dtype=np.float64) + event_pixel_diff
    y = np.asarray(events["y"], dtype=np.float64)
    t = np.asarray(events["t"], dtype=np.int64)
    p = np.asarray(events["p"])
    x0, y0, x1, y1 = (float(value) for value in square)
    edge_x = max(x1 - x0, 1.0)
    edge_y = max(y1 - y0, 1.0)
    valid = (
        (x >= 0)
        & (x < source_width)
        & (y >= 0)
        & (y < source_height)
        & (x >= x0)
        & (x < x1)
        & (y >= y0)
        & (y < y1)
    )
    mapped_x = np.floor((x[valid] - x0) * size / edge_x).astype(np.int32)
    mapped_y = np.floor((y[valid] - y0) * size / edge_y).astype(np.int32)
    mapped_x = np.clip(mapped_x, 0, size - 1)
    mapped_y = np.clip(mapped_y, 0, size - 1)
    batch = EventBatch(
        x=mapped_x,
        y=mapped_y,
        t_us=t[valid],
        polarity=np.where(p[valid] > 0, 1, -1).astype(np.int8),
        width=size,
        height=size,
        sequence_id=sequence_id,
        t_start_us=start_us,
        t_end_us=end_us,
    )
    return base_compatible_voxel(batch, bins=bins)


class _MaskReader:
    """Read optional direct mask paths without inventing rectangular masks."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _resolve(self, value: object) -> Path | None:
        as_py = getattr(value, "as_py", None)
        if callable(as_py):
            value = as_py()
        if value is None or str(value).strip() in {"", "None", "nan"}:
            return None
        raw = Path(str(value))
        candidates = (raw, self.root / raw, self.root / "data" / raw)
        return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)

    def read(
        self,
        value: object,
        *,
        square: tuple[float, float, float, float],
        size: int,
    ) -> tuple[torch.Tensor, bool]:
        path = self._resolve(value)
        if path is None:
            return torch.zeros((1, size, size), dtype=torch.float32), False
        if path.suffix.lower() == ".npy":
            loaded = np.load(path, allow_pickle=True)
            if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.shape == ():
                loaded = loaded.item()
            if isinstance(loaded, Mapping):
                loaded = loaded.get("mask")
            array = np.asarray(loaded)
        elif path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                key = "mask" if "mask" in archive.files else archive.files[0]
                array = np.asarray(archive[key])
        else:
            array = np.asarray(Image.open(path).convert("L"))
        if array.ndim == 3:
            array = array[..., 0]
        if array.ndim != 2:
            raise ValueError(f"Mask {path} must be 2-D, got {array.shape}")
        image = Image.fromarray((array > 0).astype(np.uint8) * 255)
        x0, y0, x1, y1 = square
        image = image.crop(
            (int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1)))
        ).resize((size, size), resample=Image.Resampling.NEAREST)
        tensor = torch.from_numpy((np.asarray(image) > 0).astype(np.float32)).unsqueeze(0)
        return tensor, True


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


def _visible_heights(
    row: Mapping[str, object],
    endpoint_indices: tuple[int, int],
    boxes: list[tuple[float, float, float, float]],
    roi_size: int,
    *,
    calibration: CalibrationResolver | None = None,
) -> np.ndarray:
    if "box3d_h" not in row or "box3d_Fcam" not in row:
        raise ValueError("Official LHR targets require box3d_h and box3d_Fcam.")
    height_3d = float(row["box3d_h"])
    corners_per_frame = _as_list(row["box3d_Fcam"])
    resolver = calibration or CalibrationResolver()
    max_edge = max(
        official_square_box(boxes, index)[3] - official_square_box(boxes, index)[1]
        for index in range(len(boxes))
    )
    scaling = roi_size / max(max_edge, 1e-6)
    output = []
    for index, resolved in zip(
        endpoint_indices,
        resolver.resolve_pair(row, endpoint_indices),
        strict=True,
    ):
        corners = np.asarray(_nested_float(corners_per_frame[index]), dtype=np.float64).reshape(
            -1, 3
        )
        min_depth = float(np.min(corners[:, 2]))
        if min_depth <= 0:
            raise ValueError("box3d_Fcam contains non-positive depth.")
        output.append(resolved.fy * height_3d / min_depth * scaling)
    return np.asarray(output, dtype=np.float32)


def _geometry_target(
    observable: np.ndarray,
    metadata: dict[str, float],
    *,
    depths: tuple[float, float] | None,
    delta_t_s: float,
    track_age_s: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(EAP_GEOMETRY_V2_DIM, dtype=np.float32)
    valid = np.zeros(EAP_GEOMETRY_V2_DIM, dtype=np.bool_)
    # Observable targets.
    values[0:4] = observable[0:4]
    values[5] = observable[7]
    values[6:10] = observable[4:6].tolist() + [observable[8], observable[9]]
    values[10:13] = observable[10:13]
    track_age_valid = bool(
        track_age_s is not None and math.isfinite(track_age_s) and track_age_s >= 0.0
    )
    values[13] = np.clip(float(track_age_s) / 2.0, 0.0, 1.0) if track_age_valid else 0.0
    values[14] = np.clip(metadata["log_area_rate_raw"] / 5.0, -1.0, 1.0)
    values[15] = float(abs(metadata["log_area_rate_raw"]) > 3.0)
    values[16:20] = observable[13:17]
    valid[:] = True
    valid[4] = False
    valid[13] = track_age_valid
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


def _nearest_index(timestamps_us: list[int], target_us: int) -> tuple[int, int]:
    distances = [abs(int(value) - int(target_us)) for value in timestamps_us]
    index = int(np.argmin(np.asarray(distances, dtype=np.int64)))
    return index, int(distances[index])


def select_temporal_indices(
    timestamps_us: list[int],
    *,
    anchor_timestamp_us: int,
    target_delta_t_s: float,
    tolerance_s: float,
    context_delta_t_s: float,
    context_tolerance_s: float,
) -> tuple[int, int, int | None]:
    """Select t1/t2 near the declared horizon and an optional causal t0."""
    if len(timestamps_us) < 2:
        raise ValueError("At least two frame timestamps are required.")
    second_index, second_error = _nearest_index(timestamps_us, anchor_timestamp_us)
    first_target = int(anchor_timestamp_us - target_delta_t_s * 1_000_000.0)
    first_index, first_error = _nearest_index(timestamps_us, first_target)
    endpoint_tolerance_us = int(tolerance_s * 1_000_000.0)
    if (
        first_index >= second_index
        or first_error > endpoint_tolerance_us
        or second_error > endpoint_tolerance_us
    ):
        raise ValueError("No causal LHR endpoint pair lies within the requested tolerance.")
    context_target = int(timestamps_us[first_index] - context_delta_t_s * 1_000_000.0)
    context_index, context_error = _nearest_index(timestamps_us, context_target)
    if context_index >= first_index or context_error > int(context_tolerance_s * 1_000_000.0):
        context_index = None
    return first_index, second_index, context_index


def _official_ttc_at_endpoint(
    row: Mapping[str, object],
    second_index: int,
    *,
    allow_row_ttc_compatibility: bool = False,
) -> float:
    try:
        values = _as_list(row["frame_ttc"])
        value = float(values[second_index])
        if math.isfinite(value):
            return value
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        if not allow_row_ttc_compatibility:
            raise ValueError(
                "Official GarlTTC requires frame_ttc[t2]; enable the explicit "
                "compatibility mode only for a version-checked annotation."
            ) from exc
    if not allow_row_ttc_compatibility:
        raise ValueError("Official GarlTTC frame_ttc[t2] is not finite.")
    value = float(row["ttc"])
    if not math.isfinite(value):
        raise ValueError("Official GarlTTC target is not finite.")
    return value


def _materialize_row(
    row: Mapping[str, object],
    *,
    eap_root: Path,
    config: GarlTTCLHRCacheConfig,
    event_readers: dict[Path, EAPEventReader],
    rgb_reader: _RGBTarReader | None,
    mask_reader: _MaskReader | None,
    first_track_timestamp_us: int | None,
    calibration: CalibrationResolver,
) -> dict[str, Any]:
    boxes = normalize_boxes_xyxy(row["boxes_xyxy"])
    windows = normalize_event_windows_us(row["event_windows_us"])
    frame_timestamps = [int(value) for value in _as_list(row["frame_timestamps_us"])]
    if min(len(boxes), len(windows), len(frame_timestamps)) < 2:
        raise ValueError("LHR sample requires at least two aligned frames.")
    usable = min(len(boxes), len(windows), len(frame_timestamps))
    boxes, windows, frame_timestamps = boxes[:usable], windows[:usable], frame_timestamps[:usable]
    first_index, second_index, context_index = select_temporal_indices(
        frame_timestamps,
        anchor_timestamp_us=int(row["timestamp_us"]),
        target_delta_t_s=config.target_delta_t_s,
        tolerance_s=config.delta_t_tolerance_s,
        context_delta_t_s=config.jepa_context_delta_t_s,
        context_tolerance_s=config.jepa_context_tolerance_s,
    )
    delta_t_s = (frame_timestamps[second_index] - frame_timestamps[first_index]) * 1e-6
    if delta_t_s <= 0:
        raise ValueError("Frame timestamps must increase.")

    events_path = resolve_eap_events_path(eap_root, str(row["events_path"]))
    reader = event_readers.get(events_path)
    if reader is None:
        reader = EAPEventReader(events_path)
        event_readers[events_path] = reader
    event_v4_context_window: tuple[int, int] | None = None
    event_v4_context_box: tuple[float, float, float, float] | None = None
    event_v4_precontext_source: str | None = None
    if config.store_event_v4_common_roi:
        if context_index is not None:
            event_v4_context_window = windows[context_index]
            event_v4_context_box = boxes[context_index]
            event_v4_precontext_source = "annotated_frame_window"
        elif config.event_v4_precontext_fallback == "shifted_event_window":
            event_v4_context_window = shifted_precontext_window(
                windows[first_index],
                shift_s=config.jepa_context_delta_t_s,
            )
            # GarlTTC does not expose a t0 box in the same annotation row.
            # Reusing B1 only defines the shared crop/diagnostic coordinate;
            # the t0 tensor is read from the real earlier event interval.
            event_v4_context_box = boxes[first_index]
            event_v4_precontext_source = "shifted_event_window_t1_box_proxy"
        elif config.event_v4_require_precontext:
            raise ValueError(
                "event_v4_common_roi requires causal precontext events, but the "
                "row has no annotated t0 and fallback is disabled."
            )
        else:
            raise ValueError("event_v4_common_roi cannot be built without t0 events.")

    endpoint_events: list[torch.Tensor] = []
    jepa_endpoint_events: list[torch.Tensor] = []
    event_v4_frames: list[torch.Tensor] = []
    full_frames: list[torch.Tensor] = []
    raw_by_index: dict[int, dict[str, np.ndarray]] = {}
    for index in (first_index, second_index):
        start_us, end_us = windows[index]
        raw = reader.read_window(start_us, end_us)
        raw_by_index[index] = raw
        square = _square_box(boxes, index)
        if config.store_garl_event_roi:
            endpoint_events.append(
                _roi_voxel(
                    raw,
                    square,
                    size=config.roi_size,
                    bins=config.roi_bins,
                    sequence_id=str(row["sequence_id"]),
                    start_us=start_us,
                    end_us=end_us,
                    event_pixel_diff=config.event_pixel_diff,
                    preprocessing_device=config.preprocessing_device,
                )
            )
        if config.store_jepa_event_roi:
            jepa_endpoint_events.append(
                _jepa_roi_voxel(
                    raw,
                    square,
                    size=config.roi_size,
                    bins=config.jepa_roi_bins,
                    sequence_id=str(row["sequence_id"]),
                    start_us=start_us,
                    end_us=end_us,
                    event_pixel_diff=config.event_pixel_diff,
                )
            )
        if config.store_full_frame_events:
            full_batch = downsample_full_frame(
                raw,
                sequence_id=str(row["sequence_id"]),
                start_us=start_us,
                end_us=end_us,
                width=config.full_width,
                height=config.full_height,
            )
            full_frames.append(base_compatible_voxel(full_batch, bins=config.full_bins))

    event_v4_common_square: tuple[float, float, float, float] | None = None
    event_v4_boxes: np.ndarray | None = None
    if config.store_event_v4_common_roi:
        if event_v4_context_window is None or event_v4_context_box is None:
            raise RuntimeError("V4 precontext resolution was not completed.")
        # The crop is defined by the labelled t1/t2 boxes only.  This keeps the
        # supervised endpoint scale exact and applies that same coordinate frame
        # to the real earlier t0 event window.
        event_v4_common_square = common_square_from_boxes(
            boxes,
            (first_index, second_index),
            margin_fraction=config.event_v4_margin_fraction,
        )
        event_v4_specs = (
            (event_v4_context_window, event_v4_context_box, None),
            (windows[first_index], boxes[first_index], first_index),
            (windows[second_index], boxes[second_index], second_index),
        )
        for (start_us, end_us), _, source_index in event_v4_specs:
            raw = raw_by_index.get(source_index) if source_index is not None else None
            if raw is None:
                raw = reader.read_window(start_us, end_us)
                if source_index is not None:
                    raw_by_index[source_index] = raw
            source_width, source_height = EAP_IMAGE_SIZE
            event_x = np.asarray(raw["x"], dtype=np.float64) + config.event_pixel_diff
            event_y = np.asarray(raw["y"], dtype=np.float64)
            event_t = np.asarray(raw["t"], dtype=np.int64)
            event_p = np.asarray(raw["p"])
            square_x0, square_y0, square_x1, square_y1 = event_v4_common_square
            valid = (
                (event_x >= 0)
                & (event_x < source_width)
                & (event_y >= 0)
                & (event_y < source_height)
                & (event_x >= square_x0)
                & (event_x < square_x1)
                & (event_y >= square_y0)
                & (event_y < square_y1)
            )
            mapped_x = np.floor(
                (event_x[valid] - square_x0) * config.roi_size / (square_x1 - square_x0)
            ).astype(np.int32)
            mapped_y = np.floor(
                (event_y[valid] - square_y0) * config.roi_size / (square_y1 - square_y0)
            ).astype(np.int32)
            endpoint = EventBatch(
                x=np.clip(mapped_x, 0, config.roi_size - 1),
                y=np.clip(mapped_y, 0, config.roi_size - 1),
                t_us=event_t[valid],
                polarity=np.where(event_p[valid] > 0, 1, -1).astype(np.int8),
                width=config.roi_size,
                height=config.roi_size,
                sequence_id=str(row["sequence_id"]),
                t_start_us=start_us,
                t_end_us=end_us,
            )
            event_v4_frames.append(
                event_voxel_with_scalars(
                    endpoint,
                    bins_per_polarity=config.event_v4_bins_per_polarity,
                )
            )
        event_v4_boxes = np.stack(
            [
                box_in_common_roi(
                    box,
                    event_v4_common_square,
                    roi_size=config.roi_size,
                )
                for _, box, _ in event_v4_specs
            ],
            axis=0,
        ).astype(np.float32)

    observable, metadata = _box_features(boxes[first_index], boxes[second_index], delta_t_s)
    visible_heights = _visible_heights(
        row,
        (first_index, second_index),
        boxes,
        config.roi_size,
        calibration=calibration,
    )
    depths = None
    try:
        corners = _as_list(row["box3d_Fcam"])
        first_corners = np.asarray(_nested_float(corners[first_index]), dtype=np.float64).reshape(
            -1, 3
        )
        second_corners = np.asarray(_nested_float(corners[second_index]), dtype=np.float64).reshape(
            -1, 3
        )
        depths = (float(first_corners[:, 2].min()), float(second_corners[:, 2].min()))
    except (KeyError, TypeError, ValueError, IndexError):
        depths = None
    geometry, geometry_valid = _geometry_target(
        observable,
        metadata,
        depths=depths,
        delta_t_s=delta_t_s,
        track_age_s=(
            (int(row["timestamp_us"]) - int(first_track_timestamp_us)) * 1e-6
            if first_track_timestamp_us is not None
            else None
        ),
    )
    category = str(row.get("category", row.get("category_name", "unknown")))
    if context_index is not None:
        context_dt_s = (frame_timestamps[first_index] - frame_timestamps[context_index]) * 1e-6
        jepa_context_motion, _ = _box_features(
            boxes[context_index], boxes[first_index], context_dt_s
        )
        precontext_motion_valid = True
    else:
        jepa_context_motion = np.zeros(OBSERVABLE_MOTION_DIM, dtype=np.float32)
        precontext_motion_valid = False
    jepa_pair_valid = bool(first_index < second_index and math.isfinite(delta_t_s))
    sample = {
        "garl_delta_t_s": np.float32(delta_t_s),
        "observable_motion": observable,
        "jepa_context_motion": jepa_context_motion,
        "jepa_pair_valid": np.bool_(jepa_pair_valid),
        "precontext_motion_valid": np.bool_(precontext_motion_valid),
        "geometry_v2_target": geometry,
        "geometry_v2_valid": geometry_valid,
        "garl_visible_heights_px": visible_heights,
        "ttc_s": np.float32(_official_ttc_at_endpoint(row, second_index)),
        "category_index": np.int64(category_index(category)),
        "category_valid": np.bool_(category != "unknown"),
        "sampling_group": _sampling_group(category, metadata),
        "category": category,
        "sequence_id": str(row["sequence_id"]),
        "sample_token": str(row["sample_token"]),
        "track_id": str(row["track_id"]),
        "public_track_id": str(row["public_track_id"]),
        "timestamp_us": np.int64(row["timestamp_us"]),
        "endpoint_first_timestamp_us": np.int64(frame_timestamps[first_index]),
        "endpoint_second_timestamp_us": np.int64(frame_timestamps[second_index]),
        "endpoint_delta_error_s": np.float32(abs(delta_t_s - config.target_delta_t_s)),
        "ttc_label_index": np.int64(second_index),
        "ttc_label_timestamp_us": np.int64(frame_timestamps[second_index]),
        "ttc_label_source": "official_garlttc_annotations_train_parquet.frame_ttc[t2]",
    }
    if config.store_full_frame_events:
        sample["full_frame_events"] = (
            torch.stack(full_frames).numpy().astype(np.float32)
        )
    if config.store_garl_event_roi:
        sample["garl_event_roi"] = (
            torch.cat(endpoint_events, dim=0).numpy().astype(np.float32)
        )
    if config.store_jepa_event_roi:
        sample["jepa_event_roi"] = (
            torch.stack(jepa_endpoint_events).numpy().astype(np.float32)
        )
    if config.store_event_v4_common_roi:
        if event_v4_common_square is None or event_v4_boxes is None:
            raise RuntimeError("V4 common-ROI materialization was not completed.")
        storage_dtype = (
            np.float16 if config.event_v4_storage_dtype == "float16" else np.float32
        )
        sample["event_v4_common_roi"] = torch.stack(event_v4_frames).numpy().astype(
            storage_dtype
        )
        sample["event_v4_boxes_xyxy"] = event_v4_boxes
        sample["event_v4_common_square_xyxy"] = np.asarray(
            event_v4_common_square, dtype=np.float32
        )
        sample["event_v4_precontext_valid"] = np.bool_(True)
        sample["event_v4_precontext_source"] = str(event_v4_precontext_source)
        sample["event_v4_t0_box_is_proxy"] = np.bool_(
            event_v4_precontext_source == "shifted_event_window_t1_box_proxy"
        )

    if config.include_masks:
        if mask_reader is None:
            raise RuntimeError("Mask reader is unavailable.")
        mask_paths = _as_list(row.get("mask_paths", []))
        mask_tensors: list[torch.Tensor] = []
        mask_valid: list[bool] = []
        for index in (first_index, second_index):
            value = mask_paths[index] if index < len(mask_paths) else None
            mask, valid = mask_reader.read(
                value,
                square=_square_box(boxes, index),
                size=config.roi_size,
            )
            mask_tensors.append(mask)
            mask_valid.append(valid)
        if config.mask_required and not all(mask_valid):
            raise ValueError("Required object masks are unavailable for one or both endpoints.")
        sample["garl_mask_pair"] = torch.stack(mask_tensors).numpy().astype(np.float32)
        sample["garl_mask_valid"] = np.asarray(mask_valid, dtype=np.bool_)

    if config.include_rgb:
        if rgb_reader is None:
            raise RuntimeError("RGB reader is unavailable.")
        shards = _as_list(row["rgb_shard_paths"])
        members = _as_list(row["rgb_member_paths"])
        sample["garl_rgb_pair"] = (
            torch.stack(
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
            )
            .numpy()
            .astype(np.float32)
        )
    return sample


def materialize_garlttc_lhr_cache(
    *,
    eap_root: str | Path,
    garlttc_root: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    config: GarlTTCLHRCacheConfig,
    max_samples_per_split: int | None = None,
    resume: bool = False,
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
        for role in config.materialize_splits
        for sequence in assignments.get(role, [])
    }
    index = load_garlttc_train_index(garlttc_root, sorted(sequence_role))
    validate_garlttc_train_index(
        index,
        expected_rows=config.expected_train_rows,
        allow_version_change=config.allow_dataset_version_change,
    )
    track_start_lookup = (
        index.merged.groupby(["sequence_id", "track_id"], dropna=False)["timestamp_us"]
        .min()
        .to_dict()
    )
    rows = index.merged.sort_values(
        ["sequence_id", "timestamp_us", "track_id", "sample_token"],
        kind="mergesort",
    )
    selected_rows, selection_report = select_balanced_cache_rows(
        rows,
        sequence_role,
        seed=config.selection_seed,
        max_samples_per_split=max_samples_per_split,
    )
    role_rows: dict[str, list[dict[str, object]]] = {"train": [], "validation": []}
    for _, row in selected_rows.iterrows():
        role = sequence_role.get(str(row["sequence_id"]))
        if role in role_rows:
            role_rows[role].append(dict(row))

    selection_fingerprint = _cache_fingerprint(
        {
            role: [_cache_row_identity(row) for row in role_rows[role]]
            for role in ("train", "validation")
        }
    )
    config_fingerprint = _cache_fingerprint(asdict(config))
    split_path = Path(split_path).resolve()
    split_sha256 = _sha256_file(split_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "build_state.json"
    manifest_path = output / "manifest.json"
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume {output}: build_state.json is missing; "
                "refusing to adopt old shards."
            )
        state = read_structured(state_path)
        expected_state = {
            "selection_fingerprint": selection_fingerprint,
            "config_fingerprint": config_fingerprint,
            "split_sha256": split_sha256,
        }
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected_state.items()
            if state.get(key) != value
        }
        recoverable_train_only_closure = (
            state.get("status") == "failed"
            and state.get("error") == "empty_split_after_materialization"
            and all(
                int(state.get("split_counts", {}).get(role, 0)) > 0
                for role in config.materialize_splits
            )
        )
        if mismatches and not recoverable_train_only_closure:
            raise RuntimeError(
                f"Resume state does not match this selection/configuration: {mismatches}"
            )
        if state.get("status") == "completed" and manifest_path.is_file():
            return read_structured(manifest_path)
    else:
        existing = [path.name for path in output.iterdir()]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite non-empty cache output {output}; "
                "use a new path or --resume."
            )
        state = {}

    if config.preprocessing_device == "cuda" and config.workers != 1:
        raise ValueError(
            "CUDA preprocessing must use exactly one worker to avoid competing for the 12 GB GPU."
        )
    _atomic_json(
        state_path,
        {
            "artifact_type": "garlttc_lhr_cache_build_state_v1",
            "status": "running",
            "selection_fingerprint": selection_fingerprint,
            "config_fingerprint": config_fingerprint,
            "split_sha256": split_sha256,
            "workers": config.workers,
            "preprocessing_device": config.preprocessing_device,
            "selected_count": int(selection_report["selected_count"]),
            "selection_report": selection_report,
        },
    )

    shard_meta: list[dict[str, Any]] = []
    discard_reasons: Counter[str] = Counter()
    split_counts = {"train": 0, "validation": 0}
    jepa_pair_valid_count = 0
    precontext_motion_valid_count = 0
    event_v4_precontext_valid_count = 0
    event_v4_precontext_sources: Counter[str] = Counter()
    mask_valid_endpoint_count = 0
    mask_endpoint_count = 0
    ttc_bucket_counts: Counter[str] = Counter()
    ttc_positive_negative_counts: Counter[str] = Counter()
    calibration = CalibrationResolver(config.calibration_mode, eap_root=eap_root)
    rgb_reader = _RGBTarReader(eap_root) if config.include_rgb else None
    mask_reader = _MaskReader(eap_root) if config.include_masks else None
    event_readers: dict[Path, EAPEventReader] = {}
    executor: ProcessPoolExecutor | None = None

    def consume(
        role: str,
        records: list[dict[str, Any]],
        errors: Mapping[str, int],
    ) -> None:
        nonlocal jepa_pair_valid_count, precontext_motion_valid_count
        nonlocal event_v4_precontext_valid_count
        nonlocal mask_valid_endpoint_count, mask_endpoint_count
        split_counts[role] += len(records)
        for error, count in errors.items():
            discard_reasons[error] += int(count)
        for sample in records:
            jepa_pair_valid_count += int(sample["jepa_pair_valid"])
            precontext_motion_valid_count += int(sample["precontext_motion_valid"])
            if "event_v4_precontext_valid" in sample:
                event_v4_precontext_valid_count += int(
                    sample["event_v4_precontext_valid"]
                )
                event_v4_precontext_sources[
                    str(sample.get("event_v4_precontext_source", "unknown"))
                ] += 1
            if "garl_mask_valid" in sample:
                mask_valid = np.asarray(sample["garl_mask_valid"], dtype=np.bool_)
                mask_valid_endpoint_count += int(mask_valid.sum())
                mask_endpoint_count += int(mask_valid.size)
            ttc_value = float(sample["ttc_s"])
            ttc_bucket_counts[signed_ttc_bucket(ttc_value)] += 1
            ttc_positive_negative_counts["positive" if ttc_value > 0.0 else "negative"] += 1

    def direct_task(
        task: tuple[int, dict[str, object], int],
    ) -> tuple[int, dict[str, Any] | None, str | None]:
        order, row, first_timestamp = task
        try:
            sample = _materialize_row(
                row,
                eap_root=eap_root,
                config=config,
                event_readers=event_readers,
                rgb_reader=rgb_reader,
                mask_reader=mask_reader,
                calibration=calibration,
                first_track_timestamp_us=first_timestamp,
            )
        except Exception as exc:  # explicit accounting; never silently substitute labels
            return order, None, f"{type(exc).__name__}:{str(exc)[:120]}"
        return order, sample, None

    try:
        if config.workers > 1:
            executor = ProcessPoolExecutor(
                max_workers=config.workers,
                initializer=_init_cache_worker,
                initargs=(eap_root.as_posix(), config),
            )
        for role in config.materialize_splits:
            rows_for_role = role_rows[role]
            for shard_index, start in enumerate(range(0, len(rows_for_role), config.shard_size)):
                chunk_rows = rows_for_role[start : start + config.shard_size]
                chunk_hash = _cache_fingerprint(
                    {
                        "role": role,
                        "shard_index": shard_index,
                        "rows": [_cache_row_identity(row) for row in chunk_rows],
                    }
                )
                role_dir = output / role
                role_dir.mkdir(parents=True, exist_ok=True)
                shard_suffix = ".pt.gz" if config.compression == "gzip" else ".pt"
                path = role_dir / f"shard-{shard_index:05d}{shard_suffix}"
                sidecar_path = role_dir / f"shard-{shard_index:05d}.meta.json"

                if resume:
                    if not path.is_file() or not sidecar_path.is_file():
                        if path.exists() or sidecar_path.exists():
                            raise RuntimeError(
                                f"Incomplete shard pair for {role}/{shard_index}; "
                                "refusing to overwrite it."
                            )
                    else:
                        metadata = read_structured(sidecar_path)
                        valid_metadata = (
                            metadata.get("selection_fingerprint") == chunk_hash
                            and metadata.get("config_fingerprint") == config_fingerprint
                            and metadata.get("split_sha256") == split_sha256
                            and metadata.get("sha256") == _sha256_file(path)
                        )
                        if not valid_metadata:
                            raise RuntimeError(
                                f"Shard integrity metadata does not match {path}; "
                                "refusing to overwrite it."
                            )
                        records = _load_torch_records(path)
                        consume(role, records, metadata.get("discard_reasons", {}))
                        shard_meta.append(metadata)
                        continue
                elif path.exists() or sidecar_path.exists():
                    raise FileExistsError(f"Refusing to overwrite existing shard {path}.")

                tasks = [
                    (
                        order,
                        row,
                        int(
                            track_start_lookup.get(
                                (row["sequence_id"], row["track_id"]), row["timestamp_us"]
                            )
                        ),
                    )
                    for order, row in enumerate(chunk_rows)
                ]
                if executor is None:
                    results = [direct_task(task) for task in tasks]
                else:
                    results = list(executor.map(_materialize_cache_worker, tasks, chunksize=1))
                results.sort(key=lambda item: item[0])
                records = [sample for _, sample, error in results if sample is not None]
                errors = Counter(error for _, sample, error in results if sample is None and error)
                _atomic_torch_save(
                    records,
                    path,
                    compression=config.compression,
                    compression_level=config.compression_level,
                )
                verified_records = _load_torch_records(path)
                if len(verified_records) != len(records):
                    raise RuntimeError(
                        f"Shard verification count mismatch for {path}: "
                        f"expected {len(records)}, loaded {len(verified_records)}."
                    )
                metadata = {
                    "split": role,
                    "path": path.relative_to(output).as_posix(),
                    "count": len(records),
                    "requested_count": len(chunk_rows),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                    "torch_load_verified": True,
                    "selection_fingerprint": chunk_hash,
                    "config_fingerprint": config_fingerprint,
                    "split_sha256": split_sha256,
                    "discard_count": int(sum(errors.values())),
                    "discard_reasons": dict(sorted(errors.items())),
                }
                _atomic_json(sidecar_path, metadata)
                consume(role, records, errors)
                shard_meta.append(metadata)
    except BaseException as exc:
        failed_state = {
            "artifact_type": "garlttc_lhr_cache_build_state_v1",
            "status": "failed",
            "selection_fingerprint": selection_fingerprint,
            "config_fingerprint": config_fingerprint,
            "split_sha256": split_sha256,
            "workers": config.workers,
            "preprocessing_device": config.preprocessing_device,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _atomic_json(state_path, failed_state)
        _atomic_json(
            output / "FAILURE.json",
            {
                **failed_state,
                "output_dir": output.as_posix(),
                "completed_shards": len(shard_meta),
                "shards": shard_meta,
            },
        )
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    if any(split_counts[role] <= 0 for role in config.materialize_splits):
        failure = {
            "artifact_type": "garlttc_lhr_cache_build_failure_v2",
            "status": "failed",
            "error": "empty_split_after_materialization",
            "split_counts": split_counts,
            "selected_count": int(selection_report["selected_count"]),
            "discard_count": int(sum(discard_reasons.values())),
            "discard_reasons": dict(sorted(discard_reasons.items())),
            "shards": shard_meta,
        }
        _atomic_json(output / "FAILURE.json", failure)
        _atomic_json(state_path, failure)
        raise RuntimeError(
            "Official cache produced an empty split: "
            f"{split_counts}; discard_reasons={dict(sorted(discard_reasons.items()))}"
        )

    data_path = garlttc_root / "data" / "train.parquet"
    annotations_path = garlttc_root / "annotations" / "train.parquet"
    provenance = _repository_provenance()
    protocol_path = (
        Path(__file__).resolve().parents[3] / "configs/protocol/garlttc_official_v1.yaml"
    )
    manifest = {
        "artifact_type": "garlttc_official_lhr_object_cache_v4",
        "schema_version": "garlttc_cache_v4",
        "created_at": datetime.now(UTC).isoformat(),
        **provenance,
        "format": "torch_sharded_list_v1",
        "shard_compression": config.compression,
        "shard_compression_level": config.compression_level,
        "config": asdict(config),
        "eap_root": eap_root.as_posix(),
        "garlttc_root": garlttc_root.as_posix(),
        "split_path": split_path.as_posix(),
        "split_sha256": split_sha256,
        "garlttc_data_sha256": _sha256_file(data_path),
        "garlttc_annotations_sha256": _sha256_file(annotations_path),
        "eap_data_parquet_sha256": _sha256_file(eap_root / "data" / "train.parquet"),
        "protocol_sha256": _sha256_file(protocol_path) if protocol_path.is_file() else None,
        "garlttc_join_keys_sha256": index.join_keys_sha256,
        "join_keys": GARLTTC_JOIN_KEYS,
        "ttc_label_source": "official_garlttc_annotations_train_parquet",
        "uses_official_garl_ttc_labels": True,
        "uses_reconstructed_public_eap_ttc": False,
        "no_label_fallback": True,
        "model_input_fields": [
            *(["full_frame_events"] if config.store_full_frame_events else []),
            *(["garl_event_roi"] if config.store_garl_event_roi else []),
            *(["jepa_event_roi"] if config.store_jepa_event_roi else []),
            *(["event_v4_common_roi"] if config.store_event_v4_common_roi else []),
            *(["event_v4_boxes_xyxy"] if config.store_event_v4_common_roi else []),
            *(["event_v4_common_square_xyxy"] if config.store_event_v4_common_roi else []),
            *(["event_v4_precontext_valid"] if config.store_event_v4_common_roi else []),
            *(["event_v4_precontext_source"] if config.store_event_v4_common_roi else []),
            *(["event_v4_t0_box_is_proxy"] if config.store_event_v4_common_roi else []),
            "garl_delta_t_s",
            "observable_motion",
            "jepa_context_motion",
            "jepa_pair_valid",
            "precontext_motion_valid",
            *([] if not config.include_rgb else ["garl_rgb_pair"]),
        ],
        "supervision_only_fields": [
            "ttc_s",
            "garl_visible_heights_px",
            "geometry_v2_target",
            "geometry_v2_valid",
            "category_index",
            "category_valid",
            *([] if not config.include_masks else ["garl_mask_pair", "garl_mask_valid"]),
        ],
        "forbidden_model_input_fields": sorted(FORBIDDEN_MODEL_INPUT_KEYS),
        "observable_motion_names": list(OBSERVABLE_MOTION_NAMES),
        "geometry_target_dim": EAP_GEOMETRY_V2_DIM,
        "input_schema": {
            "version": INPUT_SCHEMA_VERSION,
            "event_roi_shape": [2, config.roi_bins * 2, config.roi_size, config.roi_size],
            "channel_names": list(EVENT_CHANNEL_NAMES),
            "normalization": NORMALIZATION_ID,
        },
        "object_lhr_extension": {
            "version": 3,
            "storage_profile": (
                "object_lhr_minimal"
                if (
                    not config.store_full_frame_events
                    and not config.store_garl_event_roi
                    and config.store_jepa_event_roi
                    and not config.store_event_v4_common_roi
                )
                else "event_v4_common_roi"
                if (
                    not config.store_full_frame_events
                    and not config.store_garl_event_roi
                    and not config.store_jepa_event_roi
                    and config.store_event_v4_common_roi
                )
                else "custom"
            ),
            "jepa_event_roi_shape": [2, 21, config.roi_size, config.roi_size],
            "jepa_event_representation": "base_compatible_voxel_roi_v1",
            "jepa_roi_bins": config.jepa_roi_bins,
            "event_v4_common_roi_shape": (
                [
                    EVENT_V4_STEPS,
                    event_v4_channel_count(config.event_v4_bins_per_polarity),
                    config.roi_size,
                    config.roi_size,
                ]
                if config.store_event_v4_common_roi
                else None
            ),
            "event_v4_channel_names": (
                list(event_v4_channel_names(config.event_v4_bins_per_polarity))
                if config.store_event_v4_common_roi
                else None
            ),
            "event_v4_coordinate_frame": (
                "single_union_square_t1_t2_applied_to_real_t0_t1_t2_events"
                if config.store_event_v4_common_roi
                else None
            ),
            "event_v4_margin_fraction": (
                config.event_v4_margin_fraction
                if config.store_event_v4_common_roi
                else None
            ),
            "event_v4_bins_per_polarity": config.event_v4_bins_per_polarity,
            "event_v4_storage_dtype": config.event_v4_storage_dtype,
            "event_v4_requires_precontext": config.event_v4_require_precontext,
            "event_v4_precontext_fallback": config.event_v4_precontext_fallback,
            "event_v4_precontext_is_real_events": True,
            "event_v4_t0_box_policy": (
                "annotated_when_available_else_t1_proxy_for_crop_diagnostics"
                if config.store_event_v4_common_roi
                else None
            ),
            "event_v4_independent_endpoint_resize": False,
            "event_v4_preserves_absolute_scale_inside_common_roi": True,
            "object_identity_source": "official_boxes_xyxy_per_track",
            "height_target_source": "official_box3d_projection_visible_height",
            "mask_policy": (
                "required"
                if config.mask_required
                else "optional"
                if config.include_masks
                else "disabled"
            ),
            "uses_boxes_for_cache_preprocessing": True,
            "uses_boxes_for_model_input": False,
        },
        "calibration_mode": config.calibration_mode,
        "bbox_protocol": "official_gt_square_roi_p0",
        "temporal_pairing": {
            "target_delta_t_s": config.target_delta_t_s,
            "delta_t_tolerance_s": config.delta_t_tolerance_s,
            "jepa_context_delta_t_s": config.jepa_context_delta_t_s,
            "jepa_context_tolerance_s": config.jepa_context_tolerance_s,
            "jepa_predictor_is_strictly_causal": True,
        },
        "visibility_definition": "clipped_bbox_area_over_raw_bbox_area",
        "split_counts": split_counts,
        "selection": selection_report,
        "jepa_pair_valid_fraction": float(
            jepa_pair_valid_count / max(sum(split_counts.values()), 1)
        ),
        "precontext_motion_valid_fraction": float(
            precontext_motion_valid_count / max(sum(split_counts.values()), 1)
        ),
        "event_v4_precontext_valid_fraction": float(
            event_v4_precontext_valid_count / max(sum(split_counts.values()), 1)
        ),
        "event_v4_precontext_source_counts": dict(
            sorted(event_v4_precontext_sources.items())
        ),
        "mask_valid_endpoint_fraction": (
            float(mask_valid_endpoint_count / max(mask_endpoint_count, 1))
            if config.include_masks
            else None
        ),
        "ttc_positive_negative_counts": dict(sorted(ttc_positive_negative_counts.items())),
        "ttc_bucket_counts": dict(sorted(ttc_bucket_counts.items())),
        "discard_count": int(sum(discard_reasons.values())),
        "discard_fraction": float(
            sum(discard_reasons.values()) / max(selection_report["selected_count"], 1)
        ),
        "discard_reasons": dict(sorted(discard_reasons.items())),
        "shards": shard_meta,
        "materialization_backend": "process_pool" if config.workers > 1 else "single_process",
        "workers": config.workers,
        "resumable": True,
    }
    write_structured(output / "manifest.json", manifest)
    _atomic_json(
        state_path,
        {
            "artifact_type": "garlttc_lhr_cache_build_state_v1",
            "status": "completed",
            "selection_fingerprint": selection_fingerprint,
            "config_fingerprint": config_fingerprint,
            "split_sha256": split_sha256,
            "workers": config.workers,
            "preprocessing_device": config.preprocessing_device,
            "manifest_sha256": _sha256_file(output / "manifest.json"),
            "shard_count": len(shard_meta),
        },
    )
    (output / "FAILURE.json").unlink(missing_ok=True)
    return manifest


class GarlTTCLHRCacheDataset(Dataset[dict[str, Any]]):
    """Read materialized official-label shards with a one-shard process cache."""

    def __init__(self, manifest_path: str | Path, *, splits: tuple[str, ...]) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = read_structured(self.manifest_path)
        if self.manifest.get("uses_official_garl_ttc_labels") is not True:
            raise ValueError("LHR-v2 training requires an official-label cache.")
        validate_cache_manifest_input_schema(self.manifest)
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

    def shard_index_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return dataset indices grouped by backing shard in manifest order."""

        groups: dict[Path, list[int]] = {}
        for dataset_index, (path, _) in enumerate(self.entries):
            groups.setdefault(path, []).append(dataset_index)
        return tuple(tuple(indices) for indices in groups.values())

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, local_index = self.entries[index]
        if self._cached_path != path:
            records = _load_torch_records(path)
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
