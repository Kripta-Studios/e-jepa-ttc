"""GarlTTC ↔ eAP linkage dataset for JEPA + signed-TTC pretraining.

This module provides the data pipeline for the ``ttc`` objective of the
eAP pretraining system.  It joins the official GarlTTC parquets on five
exact keys, filters to the sequences present in the eAP split manifest,
and produces samples compatible with the existing 21-channel encoder.

IMPORTANT
---------
* It groups annotations by unique context (sequence_id, timestamp_us).
* The JEPA temporal structure exactly matches SSL/Geo protocols.
* Zero-event windows yield zero-tensors and are NOT discarded.
* Bbox spatial masks are provided to pool tokens at the downstream head.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from e_jepa_ttc.models.motion_head import TubeletTokenGeometry

from e_jepa_ttc.data.eap import EAP_IMAGE_SIZE, EAPEventReader
from e_jepa_ttc.data.eap_representation import (
    base_compatible_voxel,
    downsample_full_frame,
)
from e_jepa_ttc.training.eap_jepa import EAPJEPATrainerConfig


class EventWindowOutOfBoundsError(ValueError):
    pass


GARLTTC_JOIN_KEYS: list[str] = [
    "sequence_id",
    "sample_token",
    "track_id",
    "public_track_id",
    "timestamp_us",
]


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_string(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_eap_events_path(eap_root: Path, events_path: str) -> Path:
    """Canonical resolution of eAP events_path relative to eAP root directory."""
    if not events_path or not isinstance(events_path, str):
        raise ValueError(f"Invalid events_path: {events_path}")

    p = Path(events_path)
    if p.is_absolute():
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Absolute events_path does not exist: {p}")

    candidates = [
        eap_root / p,
        eap_root / "data" / "train" / p,
        eap_root / "data" / p,
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(
        f"Could not resolve events_path '{events_path}' under eAP root '{eap_root}'"
    )


def canonical_context_id(
    *,
    resolved_events_path: Path,
    sequence_id: str,
    timestamp_us: int,
) -> str:
    """Canonical ID for a context window."""
    return f"{resolved_events_path.as_posix()}|{sequence_id}|{timestamp_us}"


def _parse_single_box(item: object) -> tuple[float, float, float, float]:
    if isinstance(item, np.ndarray):
        item = item.tolist()
    if not isinstance(item, (list, tuple)):
        raise ValueError(f"Box element must be list or tuple, got {type(item)}")
    if len(item) != 4:
        raise ValueError(f"Box must contain exactly 4 coordinates, got {len(item)}")
    coords = []
    for val in item:
        fval = float(val)
        if np.isnan(fval) or np.isinf(fval):
            raise ValueError(f"Box coordinate {val} is not finite")
        coords.append(fval)
    return (coords[0], coords[1], coords[2], coords[3])


def normalize_boxes_xyxy(value: object) -> list[tuple[float, float, float, float]]:
    """Normalize a heterogeneous boxes_xyxy field into a list of (x0, y0, x1, y1) tuples."""
    if value is None:
        raise ValueError("boxes_xyxy cannot be None")

    if hasattr(value, "as_py"):
        value = value.as_py()

    if isinstance(value, str):
        s = value.strip()
        try:
            value = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            try:
                value = json.loads(s)
            except Exception as err:
                raise ValueError(f"Could not parse string as box structure: {value}") from err

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Expected list or tuple for boxes_xyxy, got {type(value)}")

    if len(value) == 0:
        raise ValueError("boxes_xyxy is empty")

    # Check if value itself is a flat box [x0, y0, ...]
    is_flat_elements = all(
        isinstance(v, (int, float, np.number)) or (isinstance(v, str) and not v.startswith("["))
        for v in value
    )
    if is_flat_elements:
        if len(value) != 4:
            raise ValueError(f"Flat box must contain exactly 4 coordinates, got {len(value)}")
        return [_parse_single_box(value)]

    boxes = []
    for item in value:
        boxes.append(_parse_single_box(item))
    return boxes


def _extract_last_bbox(boxes_raw: object) -> tuple[float, float, float, float] | None:
    """Extract the last-frame bounding box from boxes_xyxy."""
    try:
        boxes = normalize_boxes_xyxy(boxes_raw)
    except ValueError:
        return None

    if not boxes:
        return None

    x0, y0, x1, y1 = boxes[-1]

    if not all(np.isfinite([x0, y0, x1, y1])):
        return None

    img_w, img_h = EAP_IMAGE_SIZE
    x0 = max(0.0, min(x0, float(img_w)))
    y0 = max(0.0, min(y0, float(img_h)))
    x1 = max(0.0, min(x1, float(img_w)))
    y1 = max(0.0, min(y1, float(img_h)))

    if (x1 - x0) <= 0 or (y1 - y0) <= 0:
        return None
    return (x0, y0, x1, y1)


def normalize_event_windows_us(
    value: object,
) -> list[tuple[int, int]]:
    import ast

    if hasattr(value, "as_py"):
        value = value.as_py()

    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (
            ValueError,
            SyntaxError,
        ) as exc:
            raise ValueError("Invalid event_windows_us string") from exc

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if not isinstance(
        value,
        (list, tuple),
    ):
        raise ValueError("event_windows_us must be a list of windows")

    if not value:
        raise ValueError("event_windows_us is empty")

    normalized: list[tuple[int, int]] = []

    previous_start: int | None = None

    for item in value:
        if hasattr(item, "as_py"):
            item = item.as_py()

        if isinstance(item, np.ndarray):
            item = item.tolist()

        if (
            not isinstance(
                item,
                (list, tuple),
            )
            or len(item) != 2
        ):
            raise ValueError("Each event window must contain [start_us, end_us]")

        start_us = int(item[0])
        end_us = int(item[1])

        if end_us <= start_us:
            raise ValueError("Event window must have positive duration")

        if previous_start is not None and start_us < previous_start:
            raise ValueError("Event windows are not ordered")

        normalized.append((start_us, end_us))

        previous_start = start_us

    return normalized


def bbox_to_patch_mask(
    bbox_xyxy_original: tuple[float, float, float, float],
    *,
    original_width: int,
    original_height: int,
    input_width: int,
    input_height: int,
    geometry: TubeletTokenGeometry,
) -> tuple[
    torch.Tensor,
    tuple[float, float, float, float],
]:
    x0, y0, x1, y1 = (float(v) for v in bbox_xyxy_original)

    x0 = min(max(x0, 0.0), float(original_width))
    x1 = min(max(x1, 0.0), float(original_width))
    y0 = min(max(y0, 0.0), float(original_height))
    y1 = min(max(y1, 0.0), float(original_height))

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Bounding box has non-positive area: {(x0, y0, x1, y1)}")

    scale_x = input_width / float(original_width)
    scale_y = input_height / float(original_height)

    scaled = (
        x0 * scale_x,
        y0 * scale_y,
        x1 * scale_x,
        y1 * scale_y,
    )

    sx0, sy0, sx1, sy1 = scaled

    x_centers = (
        torch.arange(geometry.grid_w, dtype=torch.float32) * geometry.stride_w
        + geometry.kernel_w / 2.0
    )

    y_centers = (
        torch.arange(geometry.grid_h, dtype=torch.float32) * geometry.stride_h
        + geometry.kernel_h / 2.0
    )

    mask = (
        (y_centers[:, None] >= sy0)
        & (y_centers[:, None] <= sy1)
        & (x_centers[None, :] >= sx0)
        & (x_centers[None, :] <= sx1)
    )

    if not bool(mask.any()):
        bbox_center_x = (sx0 + sx1) / 2.0

        bbox_center_y = (sy0 + sy1) / 2.0

        distances_sq = (x_centers[None, :] - bbox_center_x) ** 2 + (
            y_centers[:, None] - bbox_center_y
        ) ** 2

        nearest_flat_index = int(torch.argmin(distances_sq).item())

        nearest_y = nearest_flat_index // geometry.grid_w

        nearest_x = nearest_flat_index % geometry.grid_w

        mask[nearest_y, nearest_x] = True

    if mask.shape != (
        geometry.grid_h,
        geometry.grid_w,
    ):
        raise RuntimeError(f"Unexpected bbox mask shape: {tuple(mask.shape)}")

    if not bool(mask.any()):
        raise RuntimeError("BBox mask fallback produced an empty mask")

    return mask.to(torch.bool), scaled


@dataclass(frozen=True)
class GarlTTCEAPIndex:
    """Merged GarlTTC-eAP index with provenance hashes."""

    merged: pd.DataFrame
    data_sha256: str
    annotations_sha256: str
    join_keys_sha256: str
    sequence_ids: list[str]
    train_sequences: list[str]
    validation_sequences: list[str]
    source_data_row_count: int
    source_annotation_row_count: int
    source_merged_row_count: int
    selected_row_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "merged",
            self.merged.copy(deep=True),
        )


def load_garlttc_train_index(
    garlttc_root: Path,
    eap_split_sequences: list[str],
) -> GarlTTCEAPIndex:
    """Load, merge, and filter GarlTTC train parquets for eAP-TTC."""
    data_path = garlttc_root / "data" / "train.parquet"
    ann_path = garlttc_root / "annotations" / "train.parquet"

    if "test_inputs.parquet" in str(data_path) or "test_inputs.parquet" in str(ann_path):
        raise ValueError("Loading test_inputs.parquet is forbidden in pretraining.")

    if not data_path.exists():
        raise FileNotFoundError(f"Data parquet not found: {data_path}")
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations parquet not found: {ann_path}")

    data_df = pd.read_parquet(data_path)
    ann_df = pd.read_parquet(ann_path)

    # Check required columns
    required_cols = GARLTTC_JOIN_KEYS + ["events_path", "event_windows_us"]
    for col in required_cols:
        if col not in data_df.columns:
            raise ValueError(f"Missing column '{col}' in data_df")
    for col in GARLTTC_JOIN_KEYS:
        if col not in ann_df.columns:
            raise ValueError(f"Missing column '{col}' in ann_df")

    if "boxes_xyxy" not in ann_df.columns and "boxes_xyxy" not in data_df.columns:
        raise ValueError("Missing column 'boxes_xyxy' in both data_df and ann_df")

    if "ttc" not in ann_df.columns and "ttc_time" not in ann_df.columns:
        raise ValueError("Missing 'ttc' or 'ttc_time' column in ann_df")
    if "ttc" in ann_df.columns and "ttc_time" in ann_df.columns:
        diff = (
            (ann_df["ttc"] != ann_df["ttc_time"])
            & ann_df["ttc"].notna()
            & ann_df["ttc_time"].notna()
        )
        if diff.any():
            raise ValueError("ttc and ttc_time columns disagree")
        ann_df = ann_df.drop(columns=["ttc_time"])

    # Check nulls in join keys
    for k in GARLTTC_JOIN_KEYS:
        if data_df[k].isnull().any():
            raise ValueError(f"Null values found in data_df join key '{k}'")
        if ann_df[k].isnull().any():
            raise ValueError(f"Null values found in ann_df join key '{k}'")

    # Check duplicates before merge
    if data_df.duplicated(GARLTTC_JOIN_KEYS, keep=False).any():
        raise ValueError("Duplicate join keys found in data_df")
    if ann_df.duplicated(GARLTTC_JOIN_KEYS, keep=False).any():
        raise ValueError("Duplicate join keys found in ann_df")

    source_data_row_count = len(data_df)
    source_annotation_row_count = len(ann_df)

    diagnostic = pd.merge(
        data_df, ann_df, on=GARLTTC_JOIN_KEYS, how="outer", indicator=True, validate="one_to_one"
    )
    left_only = int((diagnostic["_merge"] == "left_only").sum())
    right_only = int((diagnostic["_merge"] == "right_only").sum())

    if left_only > 0 or right_only > 0:
        raise ValueError(
            f"Outer merge revealed unlinked rows: left_only={left_only}, right_only={right_only}"
        )

    full_merged = pd.merge(
        data_df, ann_df, on=GARLTTC_JOIN_KEYS, how="inner", validate="one_to_one"
    )
    source_merged_row_count = len(full_merged)

    if "boxes_xyxy" not in full_merged.columns:
        if "boxes_xyxy_x" in full_merged.columns and "boxes_xyxy_y" in full_merged.columns:
            full_merged["boxes_xyxy"] = full_merged["boxes_xyxy_y"]
        elif "boxes_xyxy_x" in full_merged.columns:
            full_merged["boxes_xyxy"] = full_merged["boxes_xyxy_x"]
        elif "boxes_xyxy_y" in full_merged.columns:
            full_merged["boxes_xyxy"] = full_merged["boxes_xyxy_y"]

    if "ttc" not in full_merged.columns and "ttc_time" in full_merged.columns:
        full_merged["ttc"] = full_merged["ttc_time"]

    raw_ttc = full_merged["ttc"]
    numeric_ttc = pd.to_numeric(
        raw_ttc,
        errors="coerce",
    )
    non_numeric_mask = numeric_ttc.isna() & raw_ttc.notna()
    if bool(non_numeric_mask.any()):
        examples = raw_ttc[non_numeric_mask].head(10).tolist()
        raise ValueError(f"Non-numeric TTC values found: {examples}")
    if bool(numeric_ttc.isna().any()):
        raise ValueError("Null TTC values found")

    full_merged["ttc"] = numeric_ttc.to_numpy(dtype=np.float64)

    seq_set = set(eap_split_sequences)
    merged = full_merged[full_merged["sequence_id"].astype(str).isin(seq_set)].copy()
    selected_row_count = len(merged)

    merged = merged.sort_values(
        ["sequence_id", "timestamp_us", "track_id", "sample_token"],
        ignore_index=True,
    )

    data_sha = _sha256_file(data_path)
    ann_sha = _sha256_file(ann_path)

    import json

    full_merged_for_hash = full_merged.sort_values(
        GARLTTC_JOIN_KEYS,
        kind="mergesort",
        ignore_index=True,
    )
    full_join_key_lines = [
        json.dumps(
            [
                (
                    int(row[key])
                    if isinstance(
                        row[key],
                        np.integer,
                    )
                    else str(row[key])
                )
                for key in GARLTTC_JOIN_KEYS
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for _, row in full_merged_for_hash.iterrows()
    ]

    keys_sha = _sha256_string("\n".join(full_join_key_lines))

    return GarlTTCEAPIndex(
        merged=merged,
        data_sha256=data_sha,
        annotations_sha256=ann_sha,
        join_keys_sha256=keys_sha,
        sequence_ids=sorted(merged["sequence_id"].astype(str).unique().tolist()),
        train_sequences=[],
        validation_sequences=[],
        source_data_row_count=source_data_row_count,
        source_annotation_row_count=source_annotation_row_count,
        source_merged_row_count=source_merged_row_count,
        selected_row_count=selected_row_count,
    )


def validate_garlttc_train_index(
    index: GarlTTCEAPIndex,
    *,
    expected_rows: int,
    allow_version_change: bool,
) -> None:
    """Validate the merged index and raise ValueError if invalid."""
    errors: list[str] = []
    merged = index.merged

    if len(merged) == 0:
        errors.append("Merged index is empty")

    if merged.duplicated(subset=GARLTTC_JOIN_KEYS).any():
        errors.append("Duplicate rows on join keys after merge")

    ttc = merged["ttc"].to_numpy(dtype=np.float64)
    if np.any(np.isnan(ttc)):
        errors.append(f"{int(np.isnan(ttc).sum())} null TTC values")
    if np.any(np.isinf(ttc)):
        errors.append(f"{int(np.isinf(ttc).sum())} infinite TTC values")

    finite = ttc[np.isfinite(ttc)]
    out = int(np.sum((finite < -10.0 - 1e-6) | (finite > 10.0 + 1e-6)))
    if out > 0:
        errors.append(f"{out} TTC values outside [-10, 10]")

    try:
        merged["event_windows_us"].apply(normalize_event_windows_us)
    except Exception:
        errors.append("event_windows_us fails strict format validation")
    if index.source_merged_row_count != expected_rows and not allow_version_change:
        errors.append(
            "Unexpected GarlTTC source row count: "
            f"expected {expected_rows}, "
            f"got {index.source_merged_row_count}"
        )

    if errors:
        raise ValueError("GarlTTC train index validation failed:\n" + "\n".join(errors))


def _uniform_downsample(items: list[Any], maximum: int) -> list[Any]:
    if len(items) <= maximum:
        return items
    indices = np.linspace(0, len(items) - 1, maximum, dtype=np.int64)
    return [items[int(i)] for i in np.unique(indices)]


@dataclass(frozen=True)
class _GarlTTCContextSample:
    """A unique temporal context containing multiple tracks."""

    sequence_id: str
    timestamp_us: int
    event_reference_end_us: int
    events_path: str
    bboxes_xyxy: list[tuple[float, float, float, float]]
    ttc_seconds: list[float]
    track_ids: list[str]
    future_valid: tuple[bool, ...]


@dataclass
class GarlTTCBatch:
    """Collated batch for E-JEPA-TTC with full metadata."""

    context: torch.Tensor  # [B, 21, H, W]
    futures: torch.Tensor  # [B, num_horizons, 21, H, W]
    future_valid: torch.Tensor  # [B, num_horizons]
    bbox_masks: list[torch.Tensor]  # list of length B, each [N, grid_h, grid_w]
    target_ttc: list[torch.Tensor]  # list of length B, each [N]
    sequence_ids: list[str]  # length B
    track_ids: list[list[str]]  # list of length B, each list of length N
    timestamp_us: torch.Tensor  # [B]
    events_paths: list[str]  # length B
    original_bboxes: list[list[tuple[float, float, float, float]]]  # length B
    transformed_bboxes: list[list[tuple[float, float, float, float]]]  # length B
    context_event_counts: torch.Tensor  # [B]
    future_event_counts: torch.Tensor  # [B, num_horizons]


def collate_garlttc(batch: list[tuple[Any, ...]]) -> GarlTTCBatch:
    """Collate function for the GarlTTC-eAP dataset."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    contexts = []
    futures = []
    valid = []
    masks = []
    ttcs = []
    seqs = []
    tracks = []
    timestamps = []
    epaths = []
    orig_boxes = []
    trans_boxes = []
    ctx_counts = []
    fut_counts = []

    for item in batch:
        ctx, fut, val, msk, ttc, seq, trk, ts, ep, orig_b, trans_b, c_cnt, f_cnt = item
        contexts.append(ctx)
        futures.append(fut)
        valid.append(val)
        masks.append(msk)
        ttcs.append(ttc)
        seqs.append(seq)
        tracks.append(trk)
        timestamps.append(ts)
        epaths.append(ep)
        orig_boxes.append(orig_b)
        trans_boxes.append(trans_b)
        ctx_counts.append(c_cnt)
        fut_counts.append(f_cnt)

    return GarlTTCBatch(
        context=torch.stack(contexts),
        futures=torch.stack(futures),
        future_valid=torch.stack(valid),
        bbox_masks=masks,
        target_ttc=ttcs,
        sequence_ids=seqs,
        track_ids=tracks,
        timestamp_us=torch.tensor(timestamps, dtype=torch.int64),
        events_paths=epaths,
        original_bboxes=orig_boxes,
        transformed_bboxes=trans_boxes,
        context_event_counts=torch.tensor(ctx_counts, dtype=torch.int64),
        future_event_counts=torch.stack(fut_counts),
    )


class GarlTTCEAPDataset(
    Dataset[
        tuple[
            torch.Tensor,  # context
            torch.Tensor,  # futures
            torch.Tensor,  # future_valid
            torch.Tensor,  # bbox_masks [N, grid_h, grid_w]
            torch.Tensor,  # target_ttc_normalized [N]
            str,  # sequence_id
            list[str],  # track_ids
            int,  # timestamp_us
            str,  # events_path
            list[tuple[float, float, float, float]],  # original_bboxes
            list[tuple[float, float, float, float]],  # transformed_bboxes
            int,  # context_event_count
            torch.Tensor,  # future_event_counts [num_horizons]
        ]
    ]
):
    """Dataset providing fixed JEPA contexts grouped by timestamp."""

    def __init__(
        self,
        eap_root: str | Path,
        index: GarlTTCEAPIndex,
        sequence_ids: list[str],
        config: EAPJEPATrainerConfig,
        geometry: TubeletTokenGeometry,
    ) -> None:
        self.root = Path(eap_root).resolve()
        self.config = config
        self.geometry = geometry
        self._image_width, self._image_height = EAP_IMAGE_SIZE

        self._readers: dict[Path, EAPEventReader] = {}

        self._voxel_cache: OrderedDict[
            tuple[Path, int, int],
            tuple[torch.Tensor, int],
        ] = OrderedDict()

        self._discard_reasons: dict[str, int] = {}
        self.samples: list[_GarlTTCContextSample] = []
        self.selected_context_ids: list[str] = []

        merged = index.merged
        seq_set = set(sequence_ids)
        filtered = merged[merged["sequence_id"].astype(str).isin(seq_set)].copy()

        for seq_id in sorted(seq_set):
            seq_rows = filtered[filtered["sequence_id"].astype(str) == seq_id]
            if len(seq_rows) == 0:
                continue

            grouped = seq_rows.groupby("timestamp_us")
            unique_timestamps = sorted(grouped.groups.keys())

            seq_samples = []
            for ts in unique_timestamps:
                ctx_rows = grouped.get_group(ts)
                bboxes = []
                ttcs = []
                tracks = []
                events_paths = []

                for _, row in ctx_rows.iterrows():
                    bbox = _extract_last_bbox(row.get("boxes_xyxy"))
                    if bbox is None:
                        self._discard_reasons["invalid_bbox"] = (
                            self._discard_reasons.get("invalid_bbox", 0) + 1
                        )
                        continue

                    raw_ttc = row.get("ttc", row.get("ttc_time", np.nan))
                    ttc_val = float(raw_ttc)
                    if not np.isfinite(ttc_val):
                        self._discard_reasons["non_finite_ttc"] = (
                            self._discard_reasons.get("non_finite_ttc", 0) + 1
                        )
                        continue

                    bboxes.append(bbox)
                    ttcs.append(ttc_val)
                    tracks.append(str(row["track_id"]))
                    if "events_path" in row and pd.notna(row["events_path"]):
                        events_paths.append(str(row["events_path"]))

                if not bboxes:
                    self._discard_reasons["empty_context_after_filtering"] = (
                        self._discard_reasons.get("empty_context_after_filtering", 0) + 1
                    )
                    continue

                if events_paths:
                    unique_paths = set(events_paths)
                    if len(unique_paths) > 1:
                        raise ValueError(
                            f"Conflicting events_path in context {seq_id} {ts}: {unique_paths}"
                        )
                    context_events_path = events_paths[0]
                else:
                    raise ValueError(f"Missing required events_path in context {seq_id} {ts}")

                from e_jepa_ttc.data.eap import build_eap_temporal_windows

                resolved_path = resolve_eap_events_path(
                    self.root,
                    context_events_path,
                )

                reader = self._reader(context_events_path)

                first_row = ctx_rows.iloc[0]
                parsed_windows = normalize_event_windows_us(first_row["event_windows_us"])
                reference_end_us = parsed_windows[-1][1]

                windows = build_eap_temporal_windows(
                    reference_end_us=reference_end_us,
                    event_window_ms=config.event_window_ms,
                    horizons_ms=config.horizons_ms,
                )

                context_valid = (
                    windows.context_start_us >= reader.t_start_us
                    and windows.context_end_us <= reader.t_end_us
                )

                future_valid = tuple(
                    start_us >= reader.t_start_us and end_us <= reader.t_end_us
                    for start_us, end_us in windows.future_windows_us
                )

                if not context_valid:
                    self._discard_reasons["context_out_of_bounds"] = (
                        self._discard_reasons.get("context_out_of_bounds", 0) + 1
                    )
                    continue

                if not any(future_valid):
                    self._discard_reasons["no_valid_future"] = (
                        self._discard_reasons.get("no_valid_future", 0) + 1
                    )
                    continue

                sample = _GarlTTCContextSample(
                    sequence_id=seq_id,
                    timestamp_us=int(ts),
                    event_reference_end_us=reference_end_us,
                    events_path=context_events_path,
                    bboxes_xyxy=bboxes,
                    ttc_seconds=ttcs,
                    track_ids=tracks,
                    future_valid=future_valid,
                )

                context_identifier = canonical_context_id(
                    resolved_events_path=resolved_path,
                    sequence_id=seq_id,
                    timestamp_us=int(ts),
                )
                seq_samples.append((sample, context_identifier))

            if (
                config.max_windows_per_sequence is not None
                and len(seq_samples) > config.max_windows_per_sequence
            ):
                indices = np.linspace(
                    0, len(seq_samples) - 1, config.max_windows_per_sequence, dtype=int
                )
                seq_samples = [seq_samples[i] for i in indices]

            for sample, context_identifier in seq_samples:
                self.samples.append(sample)
                self.selected_context_ids.append(context_identifier)

        if not self.samples:
            reason = self._discard_reasons
            raise ValueError(
                f"No GarlTTC-eAP samples satisfy the protocol. Discard reasons: {reason}"
            )

        self.selected_context_ids_hash = hashlib.sha256(
            "\n".join(self.selected_context_ids).encode("utf-8")
        ).hexdigest()

    @property
    def discard_reasons(self) -> dict[str, int]:
        return dict(self._discard_reasons)

    def __len__(self) -> int:
        return len(self.samples)

    def _reader(self, events_path: str) -> EAPEventReader:
        resolved = resolve_eap_events_path(self.root, events_path)
        reader = self._readers.get(resolved)
        if reader is None:
            reader = EAPEventReader(resolved)
            reader.open()
            self._readers[resolved] = reader
        return reader

    def _voxel(
        self, events_path: str, start_us: int, end_us: int, sequence_id: str
    ) -> tuple[torch.Tensor, int]:
        resolved = resolve_eap_events_path(self.root, events_path)
        key = (resolved, start_us, end_us)
        cached = self._voxel_cache.pop(key, None)
        if cached is not None:
            self._voxel_cache[key] = cached
            return cached[0].clone(), cached[1]

        reader = self._reader(events_path)

        if end_us > reader.t_end_us or start_us < reader.t_start_us:
            msg = f"Window [{start_us}, {end_us}] out of bounds"
            raise EventWindowOutOfBoundsError(msg)

        events = reader.read_window(start_us, end_us)
        event_count = int(events["x"].shape[0])

        if event_count == 0:
            voxel = torch.zeros((21, self.config.height, self.config.width), dtype=torch.float32)
        else:
            voxel = base_compatible_voxel(
                downsample_full_frame(
                    events,
                    sequence_id=sequence_id,
                    start_us=start_us,
                    end_us=end_us,
                    width=self.config.width,
                    height=self.config.height,
                ),
                bins=self.config.bins,
            )

        self._voxel_cache[key] = (voxel, event_count)
        while len(self._voxel_cache) > 32:
            self._voxel_cache.popitem(last=False)
        return voxel.clone(), event_count

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        from e_jepa_ttc.data.eap import build_eap_temporal_windows

        sample = self.samples[index]

        ref_end_us = sample.event_reference_end_us

        windows = build_eap_temporal_windows(
            reference_end_us=ref_end_us,
            event_window_ms=self.config.event_window_ms,
            horizons_ms=self.config.horizons_ms,
        )

        context, ctx_event_cnt = self._voxel(
            sample.events_path, windows.context_start_us, windows.context_end_us, sample.sequence_id
        )

        futures: list[torch.Tensor] = []
        valid_futures: list[bool] = []
        future_cnts: list[int] = []
        for f_start_us, f_end_us in windows.future_windows_us:
            try:
                future, fut_cnt = self._voxel(
                    sample.events_path, f_start_us, f_end_us, sample.sequence_id
                )
                futures.append(future)
                valid_futures.append(True)
                future_cnts.append(fut_cnt)
            except EventWindowOutOfBoundsError:
                futures.append(torch.zeros_like(context))
                valid_futures.append(False)
                future_cnts.append(0)

        masks = []
        ttcs = []
        transformed_bboxes = []
        for bbox, ttc_s in zip(sample.bboxes_xyxy, sample.ttc_seconds, strict=True):
            mask, transformed_bbox = bbox_to_patch_mask(
                bbox,
                original_width=self._image_width,
                original_height=self._image_height,
                input_width=self.config.width,
                input_height=self.config.height,
                geometry=self.geometry,
            )
            masks.append(mask)
            transformed_bboxes.append(transformed_bbox)
            ttc_seconds = float(ttc_s)

            if not np.isfinite(ttc_seconds):
                raise ValueError(f"Non-finite TTC value: {ttc_seconds}")

            if ttc_seconds < -10.0 or ttc_seconds > 10.0:
                raise ValueError(f"TTC outside audited [-10, 10] range: {ttc_seconds}")

            ttcs.append(ttc_seconds / 10.0)

        return (
            context,
            torch.stack(futures),
            torch.tensor(valid_futures, dtype=torch.bool),
            torch.stack(masks),
            torch.tensor(ttcs, dtype=torch.float32),
            sample.sequence_id,
            sample.track_ids,
            sample.timestamp_us,
            sample.events_path,
            sample.bboxes_xyxy,
            transformed_bboxes,
            ctx_event_cnt,
            torch.tensor(future_cnts, dtype=torch.int64),
        )

    def close(self) -> None:
        readers = getattr(self, "_readers", {})
        for reader in readers.values():
            reader.close()
        readers.clear()
        cache = getattr(self, "_voxel_cache", None)
        if cache is not None:
            cache.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_readers"] = {}
        state["_voxel_cache"] = OrderedDict()
        return state

    def __del__(self) -> None:
        self.close()


__all__ = [
    "GARLTTC_JOIN_KEYS",
    "GarlTTCEAPDataset",
    "GarlTTCEAPIndex",
    "GarlTTCBatch",
    "collate_garlttc",
    "load_garlttc_train_index",
    "validate_garlttc_train_index",
    "normalize_boxes_xyxy",
    "resolve_eap_events_path",
]
