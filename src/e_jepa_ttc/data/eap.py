"""Public eAP media adapter and auditable object-TTC reconstruction.

The public ``NAIL-HNU/eAP-dataset`` release contains synchronized media and
3-D tracks, but the GarlTTC-specific TTC parquet files are a separate asset.
This module therefore keeps reconstructed targets explicitly marked as such:
they are useful for development and pretraining, but are not byte-equivalent
to the private, smoothed labels used by the official benchmark.
"""

from __future__ import annotations

import io
import math
import tarfile
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from e_jepa_ttc.data.types import EventBatch

if TYPE_CHECKING:
    import pandas as pd
    from PIL.Image import Image


EAP_IMAGE_SIZE = (1280, 720)
EAP_REQUIRED_MEDIA_COLUMNS = frozenset(
    {
        "sample_token",
        "split",
        "sequence_id",
        "rgb_shard_path",
        "rgb_member_path",
        "events_path",
        "labels_path",
        "rgb_exposure_start_timestamp_us",
        "rgb_exposure_end_timestamp_us",
        "K_event",
        "T_event_ego",
    }
)
EAP_REQUIRED_LABEL_COLUMNS = frozenset(
    {
        "sample_token",
        "sequence_id",
        "track_id",
        "category",
        "bbox_3d_ego",
    }
)


@dataclass(frozen=True)
class EAPObjectState:
    """One projected object state with reconstructed depth dynamics."""

    sample_token: str
    sequence_id: str
    track_id: str
    category: str
    timestamp_us: int
    bbox_xyxy: tuple[float, float, float, float]
    bbox_3d_ego: tuple[float, ...]
    nearest_depth_m: float
    visible_height_px: float
    depth_velocity_mps: float
    ttc_s: float
    ttc_source: str = "reconstructed_public_3d_track_local_linear"

    @property
    def center_xy(self) -> tuple[float, float]:
        """Return the projected box center in event-camera pixels."""

        x_min, y_min, x_max, y_max = self.bbox_xyxy
        return ((x_min + x_max) * 0.5, (y_min + y_max) * 0.5)


@dataclass(frozen=True)
class EAPObjectWindow:
    """Causal object history paired with disjoint future object states."""

    history: tuple[EAPObjectState, ...]
    future: tuple[tuple[int, EAPObjectState], ...]
    ego_action: tuple[float, ...] = ()
    ego_action_valid: bool = False

    @property
    def target(self) -> EAPObjectState:
        """Return the last causal state used as the supervised target time."""

        return self.history[-1]


def _require_pandas() -> ModuleType:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised by clean-install checks
        msg = "Reading eAP parquet metadata requires pandas and pyarrow."
        raise RuntimeError(msg) from exc
    return pd


def _validate_columns(
    columns: Iterable[object],
    required: frozenset[str],
    *,
    source: Path,
) -> None:
    missing = sorted(required - set(str(column) for column in columns))
    if missing:
        msg = f"{source} is missing required eAP columns: {missing}."
        raise ValueError(msg)


def load_eap_media_table(root: str | Path, *, split: str = "train") -> pd.DataFrame:
    """Load and validate the public eAP synchronized-media table."""

    pd = _require_pandas()
    source = Path(root) / "data" / f"{split}.parquet"
    if not source.is_file():
        raise FileNotFoundError(source)
    table = pd.read_parquet(source)
    _validate_columns(table.columns, EAP_REQUIRED_MEDIA_COLUMNS, source=source)
    if not table["sample_token"].is_unique:
        msg = f"{source} contains duplicate sample_token values."
        raise ValueError(msg)
    return table


def load_eap_sequence_labels(
    root: str | Path,
    sequence_id: str,
    *,
    split: str = "train",
) -> pd.DataFrame:
    """Load public 3-D object labels for one eAP sequence."""

    pd = _require_pandas()
    source = Path(root) / "data" / split / sequence_id / "labels.parquet"
    if not source.is_file():
        raise FileNotFoundError(source)
    table = pd.read_parquet(source)
    _validate_columns(table.columns, EAP_REQUIRED_LABEL_COLUMNS, source=source)
    observed_sequences = set(table["sequence_id"].astype(str).unique().tolist())
    if observed_sequences != {sequence_id}:
        msg = f"{source} contains unexpected sequence IDs: {sorted(observed_sequences)}."
        raise ValueError(msg)
    return table


def box_corners_ego(bbox_3d_ego: object) -> np.ndarray:
    """Return eight ego-frame corners from ``[x,y,z,l,w,h,yaw]``."""

    box = _numeric_array(bbox_3d_ego).reshape(-1)
    if box.shape != (7,) or not np.all(np.isfinite(box)):
        msg = "bbox_3d_ego must contain seven finite values [x,y,z,l,w,h,yaw]."
        raise ValueError(msg)
    center = box[:3]
    length, width, height, yaw = box[3:]
    if min(length, width, height) <= 0:
        msg = "3-D box dimensions must be positive."
        raise ValueError(msg)
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    local = signs * np.asarray([length, width, height], dtype=np.float64)[None, :] * 0.5
    cosine = math.cos(float(yaw))
    sine = math.sin(float(yaw))
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return local @ rotation.T + center[None, :]


def project_box_3d_to_event(
    bbox_3d_ego: object,
    intrinsic: object,
    event_from_ego: object,
    *,
    image_size: tuple[int, int] = EAP_IMAGE_SIZE,
    minimum_depth_m: float = 0.1,
) -> tuple[np.ndarray, tuple[float, float, float, float], float, float]:
    """Project an ego-frame 3-D box into the event camera.

    Returns camera-frame corners, a clipped 2-D box, nearest camera depth and
    the projected vertical extent. Boxes crossing or behind the camera fail
    closed instead of silently producing infinities.
    """

    if minimum_depth_m <= 0:
        msg = "minimum_depth_m must be positive."
        raise ValueError(msg)
    k_event = _numeric_array(intrinsic)
    transform = _numeric_array(event_from_ego)
    if k_event.shape != (3, 3) or transform.shape != (4, 4):
        msg = "K_event must be 3x3 and T_event_ego must be 4x4."
        raise ValueError(msg)
    corners_ego = box_corners_ego(bbox_3d_ego)
    homogeneous = np.concatenate(
        [corners_ego, np.ones((corners_ego.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    corners_camera = (transform @ homogeneous.T).T[:, :3]
    depth = corners_camera[:, 2]
    nearest_depth = float(np.min(depth))
    if nearest_depth <= minimum_depth_m:
        msg = f"3-D box is not fully in front of the event camera (depth={nearest_depth:.3f}m)."
        raise ValueError(msg)
    projected_homogeneous = (k_event @ corners_camera.T).T
    pixels = projected_homogeneous[:, :2] / projected_homogeneous[:, 2:3]
    width, height = image_size
    x_min = float(np.clip(np.min(pixels[:, 0]), 0.0, float(width - 1)))
    y_min = float(np.clip(np.min(pixels[:, 1]), 0.0, float(height - 1)))
    x_max = float(np.clip(np.max(pixels[:, 0]), 0.0, float(width - 1)))
    y_max = float(np.clip(np.max(pixels[:, 1]), 0.0, float(height - 1)))
    if x_max <= x_min or y_max <= y_min:
        msg = "Projected 3-D box does not intersect the event image."
        raise ValueError(msg)
    return corners_camera, (x_min, y_min, x_max, y_max), nearest_depth, y_max - y_min


def _numeric_array(value: object) -> np.ndarray:
    """Decode Arrow nested arrays, including one-dimensional object wrappers."""

    array = np.asarray(value)
    payload = array.tolist() if isinstance(array, np.ndarray) else value
    return np.asarray(payload, dtype=np.float64)


def _local_linear_slopes(
    timestamps_us: np.ndarray,
    values: np.ndarray,
    *,
    radius: int,
    maximum_gap_s: float,
) -> np.ndarray:
    """Estimate a smoothed derivative without assuming constant full-track motion."""

    if radius < 1 or maximum_gap_s <= 0:
        msg = "radius and maximum_gap_s must be positive."
        raise ValueError(msg)
    times_s = (timestamps_us.astype(np.float64) - float(timestamps_us[0])) * 1e-6
    slopes = np.full(values.shape, np.nan, dtype=np.float64)
    for index in range(values.shape[0]):
        start = max(0, index - radius)
        stop = min(values.shape[0], index + radius + 1)
        local_times = times_s[start:stop]
        local_values = values[start:stop]
        distance = np.abs(local_times - times_s[index])
        valid = np.isfinite(local_values) & (distance <= maximum_gap_s)
        if np.count_nonzero(valid) < 2:
            continue
        selected_times = local_times[valid]
        selected_values = local_values[valid]
        centered_times = selected_times - float(np.mean(selected_times))
        denominator = float(np.dot(centered_times, centered_times))
        if denominator <= np.finfo(np.float64).eps:
            continue
        centered_values = selected_values - float(np.mean(selected_values))
        slopes[index] = float(np.dot(centered_times, centered_values) / denominator)
    return slopes


def reconstruct_eap_object_states(
    media_table: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    derivative_radius: int = 2,
    maximum_gap_s: float = 0.25,
    minimum_speed_mps: float = 0.05,
) -> list[EAPObjectState]:
    """Reconstruct object TTC from public 3-D tracks.

    The signed convention is ``TTC = -depth / depth_velocity``. Approaching
    objects have positive TTC; receding objects have negative TTC. Near-zero
    depth speed is represented by ``NaN`` and filtered by benchmark builders.
    """

    if minimum_speed_mps <= 0:
        msg = "minimum_speed_mps must be positive."
        raise ValueError(msg)
    _validate_columns(
        media_table.columns,
        EAP_REQUIRED_MEDIA_COLUMNS,
        source=Path("<media_table>"),
    )
    _validate_columns(labels.columns, EAP_REQUIRED_LABEL_COLUMNS, source=Path("<labels>"))
    media_lookup = media_table.set_index("sample_token", drop=False)
    projected: list[dict[str, Any]] = []
    for row in labels.itertuples(index=False):
        token = str(row.sample_token)
        if token not in media_lookup.index:
            continue
        media = media_lookup.loc[token]
        if getattr(media, "ndim", 1) != 1:
            msg = f"Duplicate media rows found for sample_token {token}."
            raise ValueError(msg)
        timestamp_us = int(
            round(
                (
                    int(media["rgb_exposure_start_timestamp_us"])
                    + int(media["rgb_exposure_end_timestamp_us"])
                )
                * 0.5
            )
        )
        try:
            _corners, bbox_xyxy, nearest_depth, visible_height = project_box_3d_to_event(
                row.bbox_3d_ego,
                media["K_event"],
                media["T_event_ego"],
            )
        except ValueError:
            continue
        projected.append(
            {
                "sample_token": token,
                "sequence_id": str(row.sequence_id),
                "track_id": str(row.track_id),
                "category": str(row.category),
                "timestamp_us": timestamp_us,
                "bbox_xyxy": bbox_xyxy,
                "bbox_3d_ego": tuple(float(value) for value in row.bbox_3d_ego),
                "nearest_depth_m": nearest_depth,
                "visible_height_px": visible_height,
            }
        )

    states: list[EAPObjectState] = []
    track_keys = sorted({(row["sequence_id"], row["track_id"]) for row in projected})
    for sequence_id, track_id in track_keys:
        track = sorted(
            (
                row
                for row in projected
                if row["sequence_id"] == sequence_id and row["track_id"] == track_id
            ),
            key=lambda row: int(row["timestamp_us"]),
        )
        timestamps = np.asarray([row["timestamp_us"] for row in track], dtype=np.int64)
        if timestamps.shape[0] < 2 or np.any(np.diff(timestamps) <= 0):
            continue
        depths = np.asarray([row["nearest_depth_m"] for row in track], dtype=np.float64)
        slopes = _local_linear_slopes(
            timestamps,
            depths,
            radius=derivative_radius,
            maximum_gap_s=maximum_gap_s,
        )
        for row, slope in zip(track, slopes, strict=True):
            ttc = (
                -float(row["nearest_depth_m"]) / float(slope)
                if np.isfinite(slope) and abs(slope) >= minimum_speed_mps
                else float("nan")
            )
            states.append(
                EAPObjectState(
                    sample_token=str(row["sample_token"]),
                    sequence_id=sequence_id,
                    track_id=track_id,
                    category=str(row["category"]),
                    timestamp_us=int(row["timestamp_us"]),
                    bbox_xyxy=tuple(row["bbox_xyxy"]),
                    bbox_3d_ego=tuple(row["bbox_3d_ego"]),
                    nearest_depth_m=float(row["nearest_depth_m"]),
                    visible_height_px=float(row["visible_height_px"]),
                    depth_velocity_mps=float(slope),
                    ttc_s=ttc,
                )
            )
    return sorted(states, key=lambda state: (state.sequence_id, state.track_id, state.timestamp_us))


def build_eap_object_windows(
    states: list[EAPObjectState],
    *,
    history_frames: int = 3,
    horizons_ms: tuple[int, ...] = (100, 250, 500),
    maximum_slop_ms: int = 25,
    maximum_history_gap_ms: int = 125,
    maximum_interpolation_gap_ms: int = 150,
    ttc_range_s: tuple[float, float] | None = (-10.0, 10.0),
) -> list[EAPObjectWindow]:
    """Build causal histories and future geometry targets by object track."""

    if history_frames < 2:
        msg = "history_frames must be at least two for object dynamics."
        raise ValueError(msg)
    if not horizons_ms or any(horizon <= 0 for horizon in horizons_ms):
        msg = "horizons_ms must contain positive values."
        raise ValueError(msg)
    if maximum_slop_ms < 0 or maximum_history_gap_ms <= 0 or maximum_interpolation_gap_ms <= 0:
        msg = "Invalid window slop or history-gap threshold."
        raise ValueError(msg)
    windows: list[EAPObjectWindow] = []
    track_keys = sorted({(state.sequence_id, state.track_id) for state in states})
    for sequence_id, track_id in track_keys:
        track = [
            state
            for state in states
            if state.sequence_id == sequence_id and state.track_id == track_id
        ]
        timestamps = np.asarray([state.timestamp_us for state in track], dtype=np.int64)
        for end_index in range(history_frames - 1, len(track)):
            history = tuple(track[end_index - history_frames + 1 : end_index + 1])
            gaps_ms = np.diff([state.timestamp_us for state in history]) * 1e-3
            if np.any(gaps_ms > maximum_history_gap_ms):
                continue
            target = history[-1]
            if not np.isfinite(target.ttc_s):
                continue
            if ttc_range_s is not None and not ttc_range_s[0] <= target.ttc_s <= ttc_range_s[1]:
                continue
            future: list[tuple[int, EAPObjectState]] = []
            for horizon_ms in horizons_ms:
                desired = target.timestamp_us + horizon_ms * 1000
                insertion = int(np.searchsorted(timestamps, desired, side="left"))
                candidates = [
                    index for index in (insertion - 1, insertion) if end_index < index < len(track)
                ]
                if not candidates:
                    continue
                best = min(candidates, key=lambda index: abs(int(timestamps[index]) - desired))
                if abs(int(timestamps[best]) - desired) <= maximum_slop_ms * 1000:
                    future.append((horizon_ms, track[best]))
                    continue
                if 0 < insertion < len(track):
                    left = track[insertion - 1]
                    right = track[insertion]
                    gap_us = right.timestamp_us - left.timestamp_us
                    if (
                        left.timestamp_us < desired < right.timestamp_us
                        and gap_us <= maximum_interpolation_gap_ms * 1000
                    ):
                        future.append(
                            (
                                horizon_ms,
                                _interpolate_object_state(left, right, desired),
                            )
                        )
            if future:
                windows.append(EAPObjectWindow(history=history, future=tuple(future)))
    return sorted(
        windows,
        key=lambda window: (
            window.target.sequence_id,
            window.target.timestamp_us,
            window.target.track_id,
        ),
    )


def _interpolate_object_state(
    left: EAPObjectState,
    right: EAPObjectState,
    timestamp_us: int,
) -> EAPObjectState:
    if (left.sequence_id, left.track_id) != (right.sequence_id, right.track_id):
        msg = "Object-state interpolation requires the same sequence and track."
        raise ValueError(msg)
    duration = right.timestamp_us - left.timestamp_us
    if duration <= 0 or not left.timestamp_us < timestamp_us < right.timestamp_us:
        msg = "Interpolated timestamp must lie strictly between ordered object states."
        raise ValueError(msg)
    weight = (timestamp_us - left.timestamp_us) / duration

    def interpolate_tuple(
        left_values: tuple[float, ...],
        right_values: tuple[float, ...],
    ) -> tuple[float, ...]:
        return tuple(
            float(left_value + weight * (right_value - left_value))
            for left_value, right_value in zip(left_values, right_values, strict=True)
        )

    depth = float(left.nearest_depth_m + weight * (right.nearest_depth_m - left.nearest_depth_m))
    velocity = float(
        left.depth_velocity_mps + weight * (right.depth_velocity_mps - left.depth_velocity_mps)
    )
    ttc = -depth / velocity if np.isfinite(velocity) and abs(velocity) >= 0.05 else float("nan")
    return EAPObjectState(
        sample_token=f"{left.sequence_id}:{left.track_id}:interpolated:{timestamp_us}",
        sequence_id=left.sequence_id,
        track_id=left.track_id,
        category=left.category,
        timestamp_us=timestamp_us,
        bbox_xyxy=interpolate_tuple(left.bbox_xyxy, right.bbox_xyxy),
        bbox_3d_ego=interpolate_tuple(left.bbox_3d_ego, right.bbox_3d_ego),
        nearest_depth_m=depth,
        visible_height_px=float(
            left.visible_height_px + weight * (right.visible_height_px - left.visible_height_px)
        ),
        depth_velocity_mps=velocity,
        ttc_s=ttc,
        ttc_source="interpolated_public_3d_track_local_linear",
    )


class EAPEventReader:
    """Indexed, exact-boundary event reader for eAP HDF5 streams."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._handle: h5py.File | None = None

    def __enter__(self) -> EAPEventReader:
        """Open the HDF5 stream once for repeated indexed reads."""

        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def open(self) -> None:
        """Open the reader if it is not already open."""

        _require_hdf5plugin()
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")

    def close(self) -> None:
        """Close an explicitly opened HDF5 stream."""

        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def read_window(self, start_us: int, end_us: int) -> dict[str, np.ndarray]:
        """Read events in the half-open interval ``[start_us, end_us)``."""

        if start_us < 0 or end_us <= start_us:
            msg = "Event window requires 0 <= start_us < end_us."
            raise ValueError(msg)
        if self._handle is not None:
            return self._read_from_handle(self._handle, start_us=start_us, end_us=end_us)
        _require_hdf5plugin()
        with h5py.File(self.path, "r") as handle:
            return self._read_from_handle(handle, start_us=start_us, end_us=end_us)

    def _read_from_handle(
        self,
        handle: h5py.File,
        *,
        start_us: int,
        end_us: int,
    ) -> dict[str, np.ndarray]:
        required = {"events/x", "events/y", "events/t", "events/p", "ms_to_idx"}
        missing = sorted(name for name in required if name not in handle)
        if missing:
            msg = f"{self.path} is missing eAP event datasets: {missing}."
            raise ValueError(msg)
        millisecond_index = handle["ms_to_idx"]
        maximum_ms = int(millisecond_index.shape[0] - 1)
        start_ms = min(maximum_ms, max(0, start_us // 1000))
        end_ms = min(maximum_ms, max(0, math.ceil(end_us / 1000)))
        first = int(millisecond_index[start_ms])
        last = int(millisecond_index[end_ms])
        timestamps = np.asarray(handle["events/t"][first:last], dtype=np.int64)
        exact = (timestamps >= start_us) & (timestamps < end_us)
        return {
            "x": np.asarray(handle["events/x"][first:last], dtype=np.int32)[exact],
            "y": np.asarray(handle["events/y"][first:last], dtype=np.int32)[exact],
            "t": timestamps[exact],
            "p": np.asarray(handle["events/p"][first:last], dtype=np.int8)[exact],
        }


def _require_hdf5plugin() -> None:
    try:
        import hdf5plugin  # noqa: F401
    except ImportError as exc:
        msg = "eAP event files require the hdf5plugin compression filters."
        raise RuntimeError(msg) from exc


def crop_events_to_roi(
    events: dict[str, np.ndarray],
    bbox_xyxy: tuple[float, float, float, float],
    *,
    sequence_id: str,
    start_us: int,
    end_us: int,
    output_size: tuple[int, int] = (64, 64),
    expansion: float = 1.25,
    image_size: tuple[int, int] = EAP_IMAGE_SIZE,
) -> EventBatch:
    """Crop an event dictionary to an expanded object ROI and resize coordinates."""

    if end_us <= start_us or expansion <= 0:
        msg = "ROI event windows require positive duration and expansion."
        raise ValueError(msg)
    output_width, output_height = output_size
    if output_width <= 0 or output_height <= 0:
        msg = "ROI output dimensions must be positive."
        raise ValueError(msg)
    x_min, y_min, x_max, y_max = bbox_xyxy
    center_x = (x_min + x_max) * 0.5
    center_y = (y_min + y_max) * 0.5
    width = max(1.0, (x_max - x_min) * expansion)
    height = max(1.0, (y_max - y_min) * expansion)
    square_edge = max(width, height)
    image_width, image_height = image_size
    crop_x_min = max(0.0, center_x - square_edge * 0.5)
    crop_y_min = max(0.0, center_y - square_edge * 0.5)
    crop_x_max = min(float(image_width), center_x + square_edge * 0.5)
    crop_y_max = min(float(image_height), center_y + square_edge * 0.5)
    source_width = max(crop_x_max - crop_x_min, 1.0)
    source_height = max(crop_y_max - crop_y_min, 1.0)
    x = np.asarray(events["x"], dtype=np.float64)
    y = np.asarray(events["y"], dtype=np.float64)
    timestamps = np.asarray(events["t"], dtype=np.int64)
    polarity = np.asarray(events["p"], dtype=np.int8)
    if not (x.shape == y.shape == timestamps.shape == polarity.shape):
        msg = "Event dictionary arrays must be aligned one-dimensional arrays."
        raise ValueError(msg)
    inside = (
        (x >= crop_x_min)
        & (x < crop_x_max)
        & (y >= crop_y_min)
        & (y < crop_y_max)
        & (timestamps >= start_us)
        & (timestamps < end_us)
    )
    resized_x = np.floor((x[inside] - crop_x_min) / source_width * output_width).astype(np.int32)
    resized_y = np.floor((y[inside] - crop_y_min) / source_height * output_height).astype(np.int32)
    resized_x = np.clip(resized_x, 0, output_width - 1)
    resized_y = np.clip(resized_y, 0, output_height - 1)
    normalized_polarity = np.where(polarity[inside] > 0, 1, -1).astype(np.int8)
    return EventBatch(
        x=resized_x,
        y=resized_y,
        t_us=timestamps[inside],
        polarity=normalized_polarity,
        width=output_width,
        height=output_height,
        sequence_id=sequence_id,
        t_start_us=start_us,
        t_end_us=end_us,
    )


def _h5_dataset_names(handle: h5py.File) -> list[str]:
    names: list[str] = []

    def collect(name: str, value: h5py.Group | h5py.Dataset) -> None:
        if isinstance(value, h5py.Dataset):
            names.append(name)

    handle.visititems(collect)
    return names


def read_rgb_tar_member(shard_path: str | Path, member_path: str) -> Image:
    """Decode one RGB PNG directly from an eAP tar shard."""

    from PIL import Image

    source = Path(shard_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with tarfile.open(source, mode="r") as archive:
        member = archive.getmember(member_path)
        extracted = archive.extractfile(member)
        if extracted is None:
            msg = f"Could not extract {member_path!r} from {source}."
            raise ValueError(msg)
        payload = extracted.read()
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB").copy()


class EAPRGBReader:
    """Reuse open tar indices while decoding synchronized eAP RGB members."""

    def __init__(self, root: str | Path, *, maximum_open_shards: int = 2) -> None:
        if maximum_open_shards <= 0:
            msg = "maximum_open_shards must be positive."
            raise ValueError(msg)
        self.root = Path(root)
        self.maximum_open_shards = maximum_open_shards
        self._archives: OrderedDict[Path, tarfile.TarFile] = OrderedDict()

    def read(self, relative_shard_path: str | Path, member_path: str) -> Image:
        """Decode one RGB image and retain a bounded number of tar handles."""

        from PIL import Image

        path = self.root / Path(relative_shard_path)
        archive = self._archives.pop(path, None)
        if archive is None:
            if not path.is_file():
                raise FileNotFoundError(path)
            archive = tarfile.open(path, mode="r")
        self._archives[path] = archive
        while len(self._archives) > self.maximum_open_shards:
            _, stale = self._archives.popitem(last=False)
            stale.close()
        member = archive.getmember(member_path)
        extracted = archive.extractfile(member)
        if extracted is None:
            msg = f"Could not extract {member_path!r} from {path}."
            raise ValueError(msg)
        with Image.open(extracted) as image:
            return image.convert("RGB").copy()

    def close(self) -> None:
        """Close all retained tar handles."""

        while self._archives:
            _, archive = self._archives.popitem(last=False)
            archive.close()

    def __enter__(self) -> EAPRGBReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "EAP_IMAGE_SIZE",
    "EAPEventReader",
    "EAPObjectState",
    "EAPObjectWindow",
    "EAPRGBReader",
    "box_corners_ego",
    "build_eap_object_windows",
    "crop_events_to_roi",
    "load_eap_media_table",
    "load_eap_sequence_labels",
    "project_box_3d_to_event",
    "read_rgb_tar_member",
    "reconstruct_eap_object_states",
]
