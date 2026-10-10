"""Real RGB archive decoding, shared-current-ROI crops and resumable role batches."""

from __future__ import annotations

import hashlib
import io
import json
import math
import tarfile
import time
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .contracts import RGB_SIZE
from .history import deltas_seconds

if TYPE_CHECKING:
    from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch


def as_list(value: object) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class TarFrameReader:
    """Bounded archive reader returning original uint8 RGB frames."""

    def __init__(
        self,
        root: Path,
        *,
        max_open: int = 4,
        max_frame_cache_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self.root = root.resolve(strict=True)
        self.max_open = int(max_open)
        self.max_frame_cache_bytes = int(max_frame_cache_bytes)
        if self.max_frame_cache_bytes < 0:
            raise ValueError("frame cache capacity cannot be negative")
        self._archives: OrderedDict[Path, tarfile.TarFile] = OrderedDict()
        self._frames: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        self._frame_cache_bytes = 0
        self.frame_cache_hits = 0
        self.frame_cache_misses = 0

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()
        self._frames.clear()
        self._frame_cache_bytes = 0

    def _archive(self, relative: str) -> tarfile.TarFile:
        path = (self.root / relative).resolve(strict=True)
        if not path.is_relative_to(self.root):
            raise ValueError("RGB archive escapes the authorized eAP root")
        archive = self._archives.get(path)
        if archive is None:
            archive = tarfile.open(path, mode="r")
            self._archives[path] = archive
            while len(self._archives) > self.max_open:
                _, stale = self._archives.popitem(last=False)
                stale.close()
        else:
            self._archives.move_to_end(path)
        return archive

    def read_uint8(self, shard: str, member: str) -> np.ndarray:
        key = (str(shard), str(member))
        cached = self._frames.get(key)
        if cached is not None:
            self.frame_cache_hits += 1
            self._frames.move_to_end(key)
            return cached
        self.frame_cache_misses += 1
        payload: bytes | None = None
        for attempt in range(3):
            try:
                extracted = self._archive(shard).extractfile(member)
                if extracted is None:
                    raise FileNotFoundError(f"{member} is absent from {shard}")
                with extracted:
                    payload = extracted.read()
                break
            except OSError:
                if attempt == 2:
                    raise
                self.close()
                time.sleep(0.05 * (attempt + 1))
        if payload is None:
            raise OSError(f"unable to read {member} from {shard}")
        with Image.open(io.BytesIO(payload)) as image:
            result = np.asarray(image.convert("RGB"), dtype=np.uint8)
        result.setflags(write=False)
        if result.nbytes <= self.max_frame_cache_bytes:
            self._frames[key] = result
            self._frame_cache_bytes += result.nbytes
            while self._frame_cache_bytes > self.max_frame_cache_bytes and self._frames:
                _, stale = self._frames.popitem(last=False)
                self._frame_cache_bytes -= stale.nbytes
        return result


def square_from_current_box(box: Sequence[float]) -> tuple[int, int, int, int]:
    """Build one integer square used unchanged for every frame in an observation."""
    if len(box) != 4:
        raise ValueError("ROI box must contain four coordinates")
    x0, y0, x1, y1 = map(float, box)
    edge = max(int(x1) - int(x0), int(y1) - int(y0))
    if edge <= 0:
        raise ValueError("ROI box has no positive extent")
    cx, cy = int((int(x0) + int(x1)) / 2.0), int((int(y0) + int(y1)) / 2.0)
    return (
        int(math.ceil(cx - edge / 2.0)),
        int(math.ceil(cy - edge / 2.0)),
        int(math.ceil(cx + edge / 2.0)),
        int(math.ceil(cy + edge / 2.0)),
    )


def _effective_square(frame: np.ndarray, square: Sequence[float]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = map(float, square)
    clipped = (
        max(0, int(x0)),
        max(0, int(y0)),
        min(frame.shape[1], int(x1)),
        min(frame.shape[0], int(y1)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("shared RGB ROI is empty after image-bound clipping")
    return clipped


def crop_uint8(frame: np.ndarray, square: Sequence[float], *, size: int = RGB_SIZE) -> np.ndarray:
    """Crop one RGB frame using the shared ROI and bilinear resize."""
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("raw teacher frame must be uint8 HWC RGB")
    x0, y0, x1, y1 = _effective_square(frame, square)
    image = Image.fromarray(frame, mode="RGB").crop((x0, y0, x1, y1))
    resized = image.resize((size, size), resample=Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1))


def _foreground(boxes: list[list[float]], square: Sequence[int], size: int) -> np.ndarray:
    x0, y0, x1, y1 = map(float, square)
    scale_x, scale_y = size / (x1 - x0), size / (y1 - y0)
    masks = np.zeros((len(boxes), 1, size, size), dtype=np.bool_)
    for index, box in enumerate(boxes):
        bx0, by0, bx1, by1 = map(float, box)
        left = max(0, min(size, math.floor((bx0 - x0) * scale_x)))
        top = max(0, min(size, math.floor((by0 - y0) * scale_y)))
        right = max(0, min(size, math.ceil((bx1 - x0) * scale_x)))
        bottom = max(0, min(size, math.ceil((by1 - y0) * scale_y)))
        if right > left and bottom > top:
            masks[index, 0, top:bottom, left:right] = True
    return masks


def _boxes_in_crop(boxes: list[list[float]], square: Sequence[int], size: int) -> np.ndarray:
    """Transform public RGB boxes into the shared resized crop coordinate frame."""
    x0, y0, x1, y1 = map(float, square)
    scale_x, scale_y = size / (x1 - x0), size / (y1 - y0)
    result = np.asarray(
        [
            [
                (box[0] - x0) * scale_x,
                (box[1] - y0) * scale_y,
                (box[2] - x0) * scale_x,
                (box[3] - y0) * scale_y,
            ]
            for box in boxes
        ],
        dtype=np.float32,
    )
    result[:, (0, 2)] = np.clip(result[:, (0, 2)], 0, size)
    result[:, (1, 3)] = np.clip(result[:, (1, 3)], 0, size)
    return result


def decode_query(
    row: Mapping[str, Any], *, eap_root: Path, reader: TarFrameReader
) -> dict[str, Any]:
    """Decode a T2/T3 producer observation with one current ROI for all frames."""
    shards = list(map(str, as_list(row["producer_shards"])))
    members = list(map(str, as_list(row["producer_members"])))
    annotation_times = np.asarray(as_list(row["producer_times_us"]), dtype=np.int64)
    sensor_times = np.asarray(as_list(row["producer_sensor_times_us"]), dtype=np.int64)
    boxes = [list(map(float, as_list(value))) for value in as_list(row["producer_boxes_xyxy"])]
    if len(shards) not in {2, 3} or not (
        len(shards) == len(members) == len(annotation_times) == len(sensor_times) == len(boxes)
    ):
        raise ValueError("producer observation must contain two or three distinct real frames")
    if (
        len(set(members)) != len(members)
        or np.any(np.diff(annotation_times) <= 0)
        or np.any(np.diff(sensor_times) <= 0)
    ):
        raise ValueError("producer frames must be distinct and chronological")
    requested_square = tuple(map(float, as_list(row["roi_xyxy"])))
    raw_frames = [
        reader.read_uint8(shard, member) for shard, member in zip(shards, members, strict=True)
    ]
    effective_squares = [_effective_square(frame, requested_square) for frame in raw_frames]
    if len(set(effective_squares)) != 1:
        raise ValueError("RGB frames disagree on the shared clipped ROI")
    square = effective_squares[0]
    rgb_uint8 = np.stack([crop_uint8(frame, requested_square) for frame in raw_frames])
    rgb = rgb_uint8.astype(np.float32) / np.float32(255.0)
    heights = np.asarray([box[3] - box[1] for box in boxes], dtype=np.float32)
    if np.any(heights <= 0):
        raise ValueError("visible RGB boxes require positive heights")
    return {
        "rgb": np.ascontiguousarray(rgb),
        "rgb_uint8": rgb_uint8,
        "frame_times_us": sensor_times,
        "sensor_frame_times_us": sensor_times,
        "annotation_frame_times_us": annotation_times,
        "delta_t_s": deltas_seconds(sensor_times.tolist()),
        "frame_valid": np.ones(len(sensor_times), dtype=np.bool_),
        "foreground_mask": _foreground(boxes, square, rgb.shape[-1]),
        "boxes_in_crop_xyxy": _boxes_in_crop(boxes, square, rgb.shape[-1]),
        "visible_heights_px": heights,
        "log_visible_heights": np.log(heights).astype(np.float32),
        "roi_xyxy": np.asarray(square, dtype=np.int32),
        "requested_roi_xyxy": np.asarray(requested_square, dtype=np.float32),
    }


class PreparedRGBCache:
    """Validated lazy reader for homogeneous lossless T2/T3 input shards."""

    def __init__(self, binding_path: Path, role_manifest_path: Path) -> None:
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if binding.get("status") != "COMPLETE":
            raise ValueError("RGB cache binding is incomplete")
        if binding.get("role_manifest_sha256") != _file_sha256(role_manifest_path):
            raise ValueError("RGB cache binding role manifest changed")
        self.manifest_path = Path(binding["cache_manifest_path"]).resolve(strict=True)
        self.binding_path = binding_path.resolve(strict=True)
        self.binding_sha256 = _file_sha256(self.binding_path)
        self.equivalence = str(binding.get("equivalence"))
        if _file_sha256(self.manifest_path) != binding.get("cache_manifest_sha256"):
            raise ValueError("RGB cache manifest hash mismatch")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETE" or not manifest.get("decode_parity_verified"):
            raise ValueError("RGB prepared cache is incomplete or lacks parity admission")
        if manifest.get("role_manifest_sha256") != _file_sha256(role_manifest_path):
            raise ValueError("RGB prepared cache belongs to another role population")
        self.manifest_sha256 = _file_sha256(self.manifest_path)
        self._lookup: dict[int, tuple[dict[str, Any], int]] = {}
        for shard in manifest["shards"]:
            indices = list(map(int, shard["indices"]))
            for local, index in enumerate(indices):
                if index in self._lookup:
                    raise ValueError("RGB prepared cache contains duplicate dataset indices")
                self._lookup[index] = (shard, local)
        if len(self._lookup) != int(manifest["population_size"]):
            raise ValueError("RGB prepared cache population is incomplete")
        self._loaded: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._loaded_bytes = 0
        self._max_loaded_bytes = 256 * 1024 * 1024

    def get(self, index: int, token: str) -> dict[str, Any]:
        """Return one decoded query after first-use shard hash and identity checks."""
        if index not in self._lookup:
            raise IndexError(f"dataset index {index} is absent from the RGB cache")
        shard, local = self._lookup[index]
        key = str(shard["path"])
        stored = self._loaded.get(key)
        if stored is None:
            path = Path(key).resolve(strict=True)
            if _file_sha256(path) != shard["sha256"]:
                raise ValueError(f"RGB prepared shard hash mismatch: {path.name}")
            with np.load(path, allow_pickle=False) as payload:
                stored = {name: payload[name] for name in payload.files}
            if stored["tokens"].astype(str).tolist() != list(map(str, shard["tokens"])):
                raise ValueError("RGB prepared shard token identity mismatch")
            self._loaded[key] = stored
            self._loaded_bytes += sum(array.nbytes for array in stored.values())
            while self._loaded_bytes > self._max_loaded_bytes and len(self._loaded) > 1:
                _, stale = self._loaded.popitem(last=False)
                self._loaded_bytes -= sum(array.nbytes for array in stale.values())
        else:
            self._loaded.move_to_end(key)
        if str(stored["tokens"][local]) != token:
            raise ValueError("RGB prepared cache row token mismatch")
        result = {
            name: np.ascontiguousarray(values[local])
            for name, values in stored.items()
            if name not in {"indices", "tokens"}
        }
        result["rgb"] = result["rgb_uint8"].astype(np.float32) / np.float32(255.0)
        return result


class RGBPortDataset(Dataset[dict[str, Any]]):
    """On-demand real-frame dataset for one immutable P/H/V role manifest."""

    def __init__(self, manifest_path: str | Path, *, role: str) -> None:
        import json

        import pandas as pd

        self.manifest_path = Path(manifest_path).resolve(strict=True)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETE" or manifest.get("role") != role:
            raise ValueError("role manifest is incomplete or mismatched")
        self.role = role
        self.eap_root = Path(manifest["eap_root"]).resolve(strict=True)
        self.rows_path = Path(manifest["rows_path"]).resolve(strict=True)
        self.rows = pd.read_parquet(self.rows_path).to_dict(orient="records")
        self.reader: TarFrameReader | None = None
        self.frame_counts = np.asarray([len(as_list(row["producer_members"])) for row in self.rows])
        binding_path = self.manifest_path.with_name(f"{role}_RGB_CACHE_BINDING.json")
        self.prepared_cache = (
            PreparedRGBCache(binding_path, self.manifest_path) if binding_path.is_file() else None
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if self.prepared_cache is not None:
            decoded = self.prepared_cache.get(index, str(row["sample_token"]))
        else:
            if self.reader is None:
                self.reader = TarFrameReader(self.eap_root)
            decoded = decode_query(row, eap_root=self.eap_root, reader=self.reader)
        return {
            **decoded,
            "teacher_rgb_uint8": decoded["rgb_uint8"],
            "dino_teacher_available": np.asarray(False, dtype=np.bool_),
            "target_ttc": np.asarray(row["target_ttc"], dtype=np.float32),
            "target_phase": np.asarray(row["target_phase"], dtype=np.float32),
            "mass": np.asarray(row["mass"], dtype=np.float32),
            "query_time_us": np.asarray(row["query_time_us"], dtype=np.int64),
            "rgb_anchor_us": np.asarray(row["rgb_anchor_us"], dtype=np.int64),
            "query_delay_us": np.asarray(row["query_delay_us"], dtype=np.int64),
            "ordinal": int(row["ordinal"]),
            "sample_token": str(row["sample_token"]),
            "group_id": str(row["group_id"]),
            "role": self.role,
            "track_id": str(row["track_id"]),
            "sequence_id": str(row["sequence_id"]),
        }


def collate_rgb_port(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate a same-T bucket without repeating T2 cold-start frames."""
    if not samples:
        raise ValueError("cannot collate an empty RGB-PORT batch")
    counts = {len(sample["rgb"]) for sample in samples}
    if len(counts) != 1:
        raise ValueError("T2 and T3 observations require separate sampler buckets")
    tensor_fields = (
        "rgb",
        "rgb_uint8",
        "teacher_rgb_uint8",
        "frame_times_us",
        "sensor_frame_times_us",
        "annotation_frame_times_us",
        "delta_t_s",
        "frame_valid",
        "foreground_mask",
        "boxes_in_crop_xyxy",
        "visible_heights_px",
        "log_visible_heights",
        "roi_xyxy",
        "requested_roi_xyxy",
        "dino_teacher_available",
        "target_ttc",
        "target_phase",
        "mass",
        "query_time_us",
        "rgb_anchor_us",
        "query_delay_us",
    )
    result: dict[str, Any] = {
        field: torch.stack([torch.as_tensor(sample[field]) for sample in samples])
        for field in tensor_fields
    }
    for field in ("ordinal", "sample_token", "group_id", "role", "track_id", "sequence_id"):
        result[field] = [sample[field] for sample in samples]
    return result


def history_observations(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the frozen, label-free, distinct triplets for one query row."""
    names = list(map(str, as_list(row["observation_ids"])))
    if not names and int(row["producer_frame_count"]) == 2:
        anchor = int(row["rgb_anchor_us"])
        return [
            {
                "observation_id": hashlib.sha256(
                    f"{row['track_id']}\0cold_t2\0{anchor}".encode()
                ).hexdigest(),
                "kind": "genuine_T2_cold_start",
                "frame_ids": as_list(row["producer_frame_ids"]),
                "producer_times_us": as_list(row["producer_times_us"]),
                "producer_available_us": as_list(row["producer_available_us"]),
                "producer_sensor_times_us": as_list(row["producer_sensor_times_us"]),
                "producer_shards": as_list(row["producer_shards"]),
                "producer_members": as_list(row["producer_members"]),
                "producer_boxes_xyxy": as_list(row["producer_boxes_xyxy"]),
                "roi_xyxy": as_list(row["roi_xyxy"]),
                "anchor_us": anchor,
                "query_age_us": int(row["query_delay_us"]),
                "query_time_us": int(row["query_time_us"]),
                "sample_token": str(row["sample_token"]),
            }
        ]
    fields = {
        "frame_ids": as_list(row["observation_frame_ids"]),
        "producer_times_us": as_list(row["observation_times_us"]),
        "producer_available_us": as_list(row["observation_available_us"]),
        "producer_sensor_times_us": as_list(row["observation_sensor_times_us"]),
        "producer_shards": as_list(row["observation_shards"]),
        "producer_members": as_list(row["observation_members"]),
        "producer_boxes_xyxy": as_list(row["observation_boxes_xyxy"]),
        "roi_xyxy": as_list(row["observation_rois_xyxy"]),
        "anchor_us": as_list(row["observation_anchor_us"]),
        "query_age_us": as_list(row["observation_query_age_us"]),
    }
    if any(len(values) != len(names) for values in fields.values()):
        raise ValueError("RGB history arrays are not aligned")
    return [
        {
            "observation_id": name,
            "kind": "distinct_T3_triplet",
            **{field: values[index] for field, values in fields.items()},
            "query_time_us": int(row["query_time_us"]),
            "sample_token": str(row["sample_token"]),
        }
        for index, name in enumerate(names)
    ]


class RGBInferenceSource:
    """P/H/V random-access RGB source for frozen endpoint inference."""

    def __init__(self, manifest_path: str | Path) -> None:
        path = Path(manifest_path).resolve(strict=True)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        role = str(manifest.get("role"))
        if role not in {"P", "H", "V"}:
            raise ValueError("inference source manifest has no valid role")
        self.dataset = RGBPortDataset(path, role=role)
        self.population_size = len(self.dataset)
        self.frame_counts = self.dataset.frame_counts
        self.identity = {
            "schema": "rgb_port_inference_source_v1",
            "role": role,
            "role_manifest_path": str(path),
            "role_manifest_sha256": _file_sha256(path),
            "rows_sha256": manifest["rows_sha256"],
            "split_assignment_sha256": manifest["split_assignment_sha256"],
            "population_size": self.population_size,
            "target_anchor": "query_time_us",
            "rgb_anchor_and_query_delay_explicit": True,
            **_prepared_cache_identity(self.dataset),
        }

    def batch(self, indices: Sequence[int], modality: str) -> ObjectEventV4Batch:
        """Decode a homogeneous-T inference batch with targets outside model inputs."""
        from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch

        if modality != "rgb" or not indices:
            raise ValueError("RGBInferenceSource only serves nonempty RGB batches")
        collated = collate_rgb_port([self.dataset[int(index)] for index in indices])
        batch_size = len(indices)
        return ObjectEventV4Batch(
            events=collated["rgb"].float(),
            delta_t_s=collated["delta_t_s"].float(),
            observable_motion=torch.empty((batch_size, 0), dtype=torch.float32),
            visible_heights_px=collated["visible_heights_px"].float(),
            target_ttc_s=collated["target_ttc"].float(),
            boxes_xyxy=collated["boxes_in_crop_xyxy"].float(),
            common_square_xyxy=collated["roi_xyxy"].float(),
            sequence_ids=list(collated["sequence_id"]),
            sample_tokens=list(collated["sample_token"]),
            track_ids=list(collated["track_id"]),
        )

    def metadata(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        """Return clocks, IDs and frozen observation histories without decoding pixels."""
        return [
            {
                "ordinal": int(self.dataset.rows[int(index)]["ordinal"]),
                "sample_token": str(self.dataset.rows[int(index)]["sample_token"]),
                "query_time_us": int(self.dataset.rows[int(index)]["query_time_us"]),
                "rgb_anchor_us": int(self.dataset.rows[int(index)]["rgb_anchor_us"]),
                "query_delay_us": int(self.dataset.rows[int(index)]["query_delay_us"]),
                "roi_xyxy": as_list(self.dataset.rows[int(index)]["roi_xyxy"]),
                "observations": history_observations(self.dataset.rows[int(index)]),
            }
            for index in indices
        ]


class RGBProducerSource:
    """Random-access P source bound to a frozen generic-DINO sidecar."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.dataset = RGBPortDataset(manifest_path, role="P")
        self.population_size = len(self.dataset)
        self.frame_counts = self.dataset.frame_counts
        manifest_path = Path(manifest_path).resolve(strict=True)
        role_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sidecar_manifest_path = manifest_path.with_name("P_DINO_MANIFEST.json")
        if not sidecar_manifest_path.is_file():
            raise FileNotFoundError(
                "Frozen generic-DINO sidecar is required before producer fit: "
                f"{sidecar_manifest_path}"
            )
        sidecar_manifest = json.loads(sidecar_manifest_path.read_text(encoding="utf-8"))
        if sidecar_manifest.get("status") != "COMPLETE":
            raise ValueError("generic-DINO sidecar is not COMPLETE")
        if sidecar_manifest.get("role_manifest_sha256") != _file_sha256(manifest_path):
            raise ValueError("generic-DINO sidecar was built for another P manifest")
        if sidecar_manifest.get("input_storage") != "raw_uint8_rgb_no_double_divide":
            raise ValueError("DINO sidecar input route is not the frozen raw-uint8 route")
        self._dino_values: np.ndarray | None = None
        self._dino_valid: np.ndarray | None = None
        self._dino_shards: list[dict[str, Any]] = []
        self._dino_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        if "targets_path" in sidecar_manifest:
            sidecar_path = Path(sidecar_manifest["targets_path"]).resolve(strict=True)
            if _file_sha256(sidecar_path) != sidecar_manifest.get("targets_sha256"):
                raise ValueError("generic-DINO sidecar hash mismatch")
            payload = np.load(sidecar_path, allow_pickle=False)
            tokens = payload["sample_token"].astype(str).tolist()
            expected = [str(row["sample_token"]) for row in self.dataset.rows]
            if tokens != expected:
                raise ValueError("generic-DINO sidecar token order differs from P")
            self._dino_values = payload["relation_targets"]
            self._dino_valid = payload["relation_valid"]
            values = self._dino_values
            valid = self._dino_valid
            if (
                values is None
                or valid is None
                or len(values) != self.population_size
                or len(valid) != self.population_size
            ):
                raise ValueError("generic-DINO sidecar population differs from P")
        else:
            if not sidecar_manifest.get("all_shards_sha256_verified"):
                raise ValueError("sharded generic-DINO sidecar was not fully verified")
            self._dino_shards = list(sidecar_manifest.get("shards", []))
            if not self._dino_shards:
                raise ValueError("sharded generic-DINO sidecar is empty")
        self.identity = {
            "schema": "rgb_port_producer_source_v1",
            "role": "P",
            "role_manifest_path": str(manifest_path),
            "role_manifest_sha256": _file_sha256(manifest_path),
            "rows_sha256": role_manifest["rows_sha256"],
            "split_assignment_sha256": role_manifest["split_assignment_sha256"],
            "population_size": self.population_size,
            "dino_manifest_path": str(sidecar_manifest_path.resolve()),
            "dino_manifest_sha256": _file_sha256(sidecar_manifest_path),
            "dino_targets_sha256": sidecar_manifest.get("targets_sha256"),
            "dino_input_storage": "raw_uint8_rgb_no_double_divide",
            "dino_teacher_weights_sha256": sidecar_manifest.get("teacher_weights_sha256"),
            "dino_frame_roi_alignment_sha256": sidecar_manifest.get("frame_roi_alignment_sha256"),
            "target_anchor": "query_time_us",
            "rgb_anchor_and_query_delay_explicit": True,
            **_prepared_cache_identity(self.dataset),
        }

    def batch(self, indices: Sequence[int], modality: str) -> ObjectEventV4Batch:
        """Return a homogeneous-T ObjectEventV4Batch for the RGB producer."""
        from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch

        if modality != "rgb" or not indices:
            raise ValueError("RGBProducerSource only serves nonempty RGB batches")
        samples = [self.dataset[int(index)] for index in indices]
        collated = collate_rgb_port(samples)
        values, valid = self._dino_for_indices(indices)
        dino_values = torch.from_numpy(values.astype(np.float32, copy=False))
        dino_valid = torch.from_numpy(valid.astype(np.bool_, copy=False))
        batch_size = len(indices)
        return ObjectEventV4Batch(
            events=collated["rgb"].float(),
            delta_t_s=collated["delta_t_s"].float(),
            observable_motion=torch.empty((batch_size, 0), dtype=torch.float32),
            visible_heights_px=collated["visible_heights_px"].float(),
            target_ttc_s=collated["target_ttc"].float(),
            boxes_xyxy=collated["boxes_in_crop_xyxy"].float(),
            common_square_xyxy=collated["roi_xyxy"].float(),
            sequence_ids=list(collated["sequence_id"]),
            sample_tokens=list(collated["sample_token"]),
            track_ids=list(collated["track_id"]),
            dinov3_relation_targets=dino_values,
            dinov3_relation_valid=dino_valid,
        )

    def _dino_for_indices(self, indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        if self._dino_values is not None and self._dino_valid is not None:
            return self._dino_values[list(indices)], self._dino_valid[list(indices)]
        values: list[np.ndarray] = []
        valid: list[np.ndarray] = []
        for index in indices:
            row = self.dataset.rows[int(index)]
            ordinal = int(row["source_ordinal"])
            item = next(
                (
                    shard
                    for shard in self._dino_shards
                    if int(shard["start"]) <= ordinal < int(shard["stop"])
                ),
                None,
            )
            if item is None:
                raise ValueError(f"no generic-DINO shard covers source ordinal {ordinal}")
            key = str(item["path"])
            stored = self._dino_cache.get(key)
            if stored is None:
                with np.load(Path(key), allow_pickle=False) as payload:
                    stored = {name: payload[name] for name in payload.files}
                self._dino_cache[key] = stored
                while len(self._dino_cache) > 4:
                    self._dino_cache.popitem(last=False)
            else:
                self._dino_cache.move_to_end(key)
            local = ordinal - int(item["start"])
            if int(stored["ordinals"][local]) != ordinal or str(stored["tokens"][local]) != str(
                row["sample_token"]
            ):
                raise ValueError("generic-DINO shard row identity mismatch")
            values.append(stored["relation_targets"][local])
            valid.append(stored["relation_valid"][local])
        return np.stack(values), np.stack(valid)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _prepared_cache_identity(dataset: RGBPortDataset) -> dict[str, Any]:
    cache = dataset.prepared_cache
    if cache is None:
        return {
            "prepared_cache_mode": "on_demand_tar",
            "prepared_cache_binding_path": None,
            "prepared_cache_binding_sha256": None,
            "prepared_cache_manifest_path": None,
            "prepared_cache_manifest_sha256": None,
            "prepared_cache_equivalence": None,
        }
    return {
        "prepared_cache_mode": "validated_lossless_shards",
        "prepared_cache_binding_path": str(cache.binding_path),
        "prepared_cache_binding_sha256": cache.binding_sha256,
        "prepared_cache_manifest_path": str(cache.manifest_path),
        "prepared_cache_manifest_sha256": cache.manifest_sha256,
        "prepared_cache_equivalence": cache.equivalence,
    }


def make_rgb_producer_source(manifest_path: str | Path) -> RGBProducerSource:
    """CLI factory used as `e_jepa_ttc.rgb_port.data:make_rgb_producer_source`."""
    return RGBProducerSource(manifest_path)


def make_rgb_inference_source(manifest_path: str | Path) -> RGBInferenceSource:
    """Factory for P/H/V producer inference and history preparation."""
    return RGBInferenceSource(manifest_path)


class StatefulRoleBatchSampler(Sampler[list[int]]):
    """Deterministic resumable T2/T3-bucketed batches for one frozen role."""

    def __init__(
        self, dataset: RGBPortDataset, *, batch_size: int, seed: int, shuffle: bool = True
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.cursor = 0
        identity = "\n".join(str(row["sample_token"]) for row in dataset.rows)
        self.population_sha256 = hashlib.sha256(identity.encode()).hexdigest()

    def _batches(self) -> list[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        batches: list[list[int]] = []
        for count in (2, 3):
            ids = np.flatnonzero(self.dataset.frame_counts == count)
            if self.shuffle:
                rng.shuffle(ids)
            batches.extend(
                ids[start : start + self.batch_size].astype(int).tolist()
                for start in range(0, len(ids), self.batch_size)
            )
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._batches()
        while self.cursor < len(batches):
            batch = batches[self.cursor]
            self.cursor += 1
            yield batch

    def __len__(self) -> int:
        return len(self._batches()) - self.cursor

    def state_dict(self) -> dict[str, Any]:
        return {
            "role": self.dataset.role,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "population_sha256": self.population_sha256,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        frozen = ("role", "seed", "shuffle", "batch_size", "population_sha256")
        current = self.state_dict()
        if any(state[name] != current[name] for name in frozen):
            raise ValueError("RGB-PORT sampler identity changed")
        epoch, cursor = int(state["epoch"]), int(state["cursor"])
        if epoch < 0:
            raise ValueError("invalid sampler epoch")
        self.epoch = epoch
        if not 0 <= cursor <= len(self._batches()):
            raise ValueError("invalid sampler cursor")
        self.cursor = cursor

    def next_epoch(self) -> None:
        self.epoch += 1
        self.cursor = 0


__all__ = [
    "RGBPortDataset",
    "RGBProducerSource",
    "RGBInferenceSource",
    "StatefulRoleBatchSampler",
    "TarFrameReader",
    "as_list",
    "collate_rgb_port",
    "crop_uint8",
    "decode_query",
    "history_observations",
    "make_rgb_inference_source",
    "make_rgb_producer_source",
    "square_from_current_box",
]
