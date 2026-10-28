"""Read-only sensor readers for the EvTTC competition inputs.

The public API uses sequence-relative integer microseconds.  Road HDF5 inputs
already use a relative clock.  Slider inputs use the first RGB header stamp as
their origin; consequently, early event timestamps may be negative.
"""

from __future__ import annotations

import csv
import hashlib
import struct
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, BinaryIO

import h5py
import numpy as np

_BAG_MAGIC = b"#ROSBAG V2.0\n"
_EVENT_TYPE = "dv_ros_msgs/EventArray"
_IMAGE_TYPE = "sensor_msgs/Image"
_EVENT_DTYPE = np.dtype(
    [("x", "<u2"), ("y", "<u2"), ("sec", "<u4"), ("nsec", "<u4"), ("p", "u1")],
    align=False,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"Unexpected end of file: needed {size} bytes, got {len(data)}")
    return data


def _u32(data: bytes) -> int:
    if len(data) != 4:
        raise ValueError("Expected a four-byte ROS uint32 field")
    return struct.unpack("<I", data)[0]


def _parse_fields(blob: bytes) -> dict[str, bytes]:
    fields: dict[str, bytes] = {}
    cursor = 0
    while cursor < len(blob):
        if cursor + 4 > len(blob):
            raise ValueError("Malformed ROS field length")
        size = struct.unpack_from("<I", blob, cursor)[0]
        cursor += 4
        if cursor + size > len(blob):
            raise ValueError("ROS field exceeds record header")
        item = blob[cursor : cursor + size]
        cursor += size
        if b"=" not in item:
            raise ValueError("Malformed ROS field without '='")
        key, value = item.split(b"=", 1)
        fields[key.decode("ascii")] = value
    return fields


@dataclass(frozen=True)
class _Record:
    header: dict[str, bytes]
    data_offset: int
    data_length: int


def _read_record(stream: BinaryIO, *, limit: int | None = None) -> _Record | None:
    start = stream.tell()
    if limit is not None and start >= limit:
        return None
    raw_length = stream.read(4)
    if not raw_length:
        return None
    if len(raw_length) != 4:
        raise ValueError("Truncated ROS record header length")
    header_length = struct.unpack("<I", raw_length)[0]
    header = _parse_fields(_read_exact(stream, header_length))
    data_length = struct.unpack("<I", _read_exact(stream, 4))[0]
    data_offset = stream.tell()
    end = data_offset + data_length
    if limit is not None and end > limit:
        raise ValueError("ROS record exceeds enclosing chunk")
    stream.seek(end)
    return _Record(header, data_offset, data_length)


def _read_ros_string(stream: BinaryIO) -> str:
    size = struct.unpack("<I", _read_exact(stream, 4))[0]
    return _read_exact(stream, size).decode("utf-8", errors="strict")


def _read_ros_header(stream: BinaryIO) -> tuple[int, int, str]:
    _seq, seconds, nanoseconds = struct.unpack("<III", _read_exact(stream, 12))
    if nanoseconds >= 1_000_000_000:
        raise ValueError("Invalid ROS header nanoseconds")
    return seconds, nanoseconds, _read_ros_string(stream)


def _ns(seconds: int, nanoseconds: int) -> int:
    if nanoseconds >= 1_000_000_000:
        raise ValueError("Invalid ROS timestamp nanoseconds")
    return seconds * 1_000_000_000 + nanoseconds


def _h5_lower_bound(dataset: h5py.Dataset, target: int) -> int:
    low, high = 0, len(dataset)
    while low < high:
        middle = (low + high) // 2
        if int(dataset[middle]) < target:
            low = middle + 1
        else:
            high = middle
    return low


def _dataset(handle: h5py.File | h5py.Group, path: str) -> h5py.Dataset:
    value = handle[path]
    if not isinstance(value, h5py.Dataset):
        raise ValueError(f"Expected HDF5 dataset at {path}")
    return value


def _validate_ms_index(timestamps: h5py.Dataset, index: np.ndarray) -> None:
    if index.ndim != 1 or len(index) == 0:
        raise ValueError("Road millisecond index must be a nonempty vector")
    if int(index[0]) != 0 or np.any(np.diff(index.astype(np.int64)) < 0):
        raise ValueError("Road millisecond index must begin at zero and be monotone")
    event_count = len(timestamps)
    if np.any(index > event_count):
        raise ValueError("Road millisecond index exceeds event array length")
    for millisecond, raw_position in enumerate(index):
        position = int(raw_position)
        boundary_us = millisecond * 1000
        if position < event_count and int(timestamps[position]) < boundary_us:
            raise ValueError("Road millisecond index starts before its timestamp boundary")
        if position > 0 and int(timestamps[position - 1]) >= boundary_us:
            raise ValueError("Road millisecond index skips events at its timestamp boundary")


@dataclass(frozen=True)
class _EventPacket:
    events_offset: int
    count: int
    header_ns: int
    first_ns: int | None
    last_ns: int | None


@dataclass(frozen=True)
class _ImageRecord:
    data_offset: int
    data_length: int
    stamp_ns: int


class SensorSequence:
    """Read one road-HDF5 or Slider-ROS1 sequence without mutating inputs."""

    def __init__(self, root: Path, sequence_id: str) -> None:
        self.root = Path(root)
        self.sequence_id = sequence_id
        self.sequence_dir = self.root / sequence_id
        if not self.sequence_dir.is_dir():
            raise FileNotFoundError(f"Sequence directory does not exist: {self.sequence_dir}")
        self._closed = False
        self._h5: h5py.File | None = None
        self._bag: BinaryIO | None = None
        self._source_files: list[Path] = []
        self._event_packets: list[_EventPacket] = []
        self._image_records: list[_ImageRecord] = []
        self._event_origin_ns = 0
        self._rgb_timestamps = np.empty(0, dtype=np.int64)

        hdf5 = self.sequence_dir / "data.hdf5"
        bags = sorted(self.sequence_dir.glob("*.bag"))
        try:
            if hdf5.is_file() and bags:
                raise ValueError(
                    f"Ambiguous sequence contains both HDF5 and ROS bag: {sequence_id}"
                )
            if hdf5.is_file():
                self._kind = "road_hdf5"
                self._open_road(hdf5)
            elif len(bags) == 1:
                self._kind = "slider_ros1"
                self._open_slider(bags[0])
            elif len(bags) > 1:
                raise ValueError(f"Expected one Slider bag, found {len(bags)}")
            else:
                raise FileNotFoundError(f"No data.hdf5 or ROS bag found in {self.sequence_dir}")
        except Exception:
            self.close()
            raise

    def __enter__(self) -> SensorSequence:
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
        if self._bag is not None:
            self._bag.close()
            self._bag = None
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("SensorSequence is closed")

    def _open_road(self, path: Path) -> None:
        handle = h5py.File(path, "r")
        self._h5 = handle
        self._source_files = [path]
        required = (
            "blackflys/right/data",
            "blackflys/right/ts",
            "blackflys/right/calib/resolution",
            "prophesee/event_cam_right/x",
            "prophesee/event_cam_right/y",
            "prophesee/event_cam_right/t",
            "prophesee/event_cam_right/p",
            "prophesee/event_cam_right/ms_map_idx",
            "prophesee/event_cam_right/calib/resolution",
        )
        missing = [name for name in required if name not in handle]
        if missing:
            self.close()
            raise ValueError(f"Road HDF5 is missing right-camera datasets: {missing}")
        event_lengths = {
            len(_dataset(handle, f"prophesee/event_cam_right/{field}"))
            for field in ("x", "y", "t", "p")
        }
        if len(event_lengths) != 1 or next(iter(event_lengths)) == 0:
            self.close()
            raise ValueError("Road right-event arrays must be nonempty and equal length")
        rgb_data = _dataset(handle, "blackflys/right/data")
        rgb_ts = np.asarray(_dataset(handle, "blackflys/right/ts"), dtype=np.int64)
        if rgb_data.ndim != 4 or rgb_data.shape[-1] != 3 or rgb_data.dtype != np.uint8:
            self.close()
            raise ValueError("Road right RGB dataset must be uint8 [N,H,W,3]")
        if len(rgb_ts) != len(rgb_data) or len(rgb_ts) == 0 or np.any(np.diff(rgb_ts) <= 0):
            self.close()
            raise ValueError("Road right RGB timestamps must be nonempty, aligned, and increasing")
        event_t = _dataset(handle, "prophesee/event_cam_right/t")
        first, last = int(event_t[0]), int(event_t[-1])
        if first < 0 or last < first:
            self.close()
            raise ValueError("Road right event timestamp endpoints are invalid")
        self._rgb_timestamps = rgb_ts
        ms_index = np.asarray(
            _dataset(handle, "prophesee/event_cam_right/ms_map_idx"), dtype=np.uint64
        )
        _validate_ms_index(event_t, ms_index)
        indexed_last_boundary_us = (len(ms_index) - 1) * 1000
        self._event_start = 0
        self._event_end = max(indexed_last_boundary_us, last + 1)
        self._coverage_evidence = {
            "kind": "road_ms_map_idx",
            "authoritative_start_us": 0,
            "last_indexed_boundary_us": indexed_last_boundary_us,
            "last_observed_event_us": last,
            "coverage_end_rule": "max(last_indexed_boundary_us,last_observed_event_us+1)",
            "silent_edge_uncertainty": False,
        }
        event_resolution = np.asarray(
            _dataset(handle, "prophesee/event_cam_right/calib/resolution"), dtype=np.int64
        )
        rgb_resolution = np.asarray(
            _dataset(handle, "blackflys/right/calib/resolution"), dtype=np.int64
        )
        if event_resolution.shape != (2,) or rgb_resolution.shape != (2,):
            self.close()
            raise ValueError("Road calibration resolutions must contain [width,height]")
        self._event_width, self._event_height = map(int, event_resolution)
        self._rgb_width, self._rgb_height = map(int, rgb_resolution)
        if tuple(rgb_data.shape[1:3]) != (self._rgb_height, self._rgb_width):
            self.close()
            raise ValueError("Road RGB data shape disagrees with right-camera calibration")

    def _open_slider(self, path: Path) -> None:
        stream = path.open("rb")
        self._bag = stream
        self._source_files = [path]
        if stream.read(len(_BAG_MAGIC)) != _BAG_MAGIC:
            raise ValueError("Slider input is not a ROS bag v2.0 file")
        connections: dict[int, tuple[str, str]] = {}
        message_records: list[tuple[int, int, int]] = []
        while True:
            record = _read_record(stream)
            if record is None:
                break
            op = record.header.get("op", b"\x00")
            if len(op) != 1:
                raise ValueError("Malformed ROS operation code")
            if op[0] == 7:  # CONNECTION
                connection_id = _u32(record.header["conn"])
                topic = record.header["topic"].decode("utf-8")
                stream.seek(record.data_offset)
                connection_header = _parse_fields(_read_exact(stream, record.data_length))
                message_type = connection_header.get("type", b"").decode("utf-8")
                connections[connection_id] = (topic, message_type)
            elif op[0] == 5:  # CHUNK
                compression = record.header.get("compression", b"none").decode("ascii")
                if compression != "none":
                    raise ValueError(f"Unsupported Slider bag compression: {compression}")
                chunk_end = record.data_offset + record.data_length
                stream.seek(record.data_offset)
                while stream.tell() < chunk_end:
                    inner = _read_record(stream, limit=chunk_end)
                    if inner is None:
                        break
                    inner_op = inner.header.get("op", b"\x00")
                    if len(inner_op) == 1 and inner_op[0] == 2:
                        message_records.append(
                            (_u32(inner.header["conn"]), inner.data_offset, inner.data_length)
                        )
                stream.seek(chunk_end)
        if not connections:
            self.close()
            raise ValueError("Slider bag contains no connection records")
        event_ids = [
            cid
            for cid, value in connections.items()
            if value == ("/capture_node/events", _EVENT_TYPE)
        ]
        image_ids = [
            cid for cid, value in connections.items() if value == ("/image_raw", _IMAGE_TYPE)
        ]
        if len(event_ids) != 1 or len(image_ids) != 1:
            self.close()
            raise ValueError("Slider bag must contain exactly one audited event and RGB connection")
        event_id, image_id = event_ids[0], image_ids[0]
        for connection_id, offset, length in message_records:
            if connection_id == event_id:
                self._event_packets.append(self._index_event_packet(offset, length))
            elif connection_id == image_id:
                self._image_records.append(self._index_image_record(offset, length))
        if not self._image_records:
            self.close()
            raise ValueError("Slider bag contains no RGB frames")
        if not self._event_packets or not any(packet.count for packet in self._event_packets):
            self.close()
            raise ValueError("Slider bag contains no events")
        nonempty_packets = [packet for packet in self._event_packets if packet.count]
        packet_starts = np.asarray([packet.first_ns for packet in nonempty_packets], dtype=np.int64)
        packet_ends = np.asarray([packet.last_ns for packet in nonempty_packets], dtype=np.int64)
        if np.any(np.diff(packet_starts) < 0) or np.any(np.diff(packet_ends) < 0):
            self.close()
            raise ValueError("Slider event packets are not chronological")
        image_stamps = np.asarray(
            [record.stamp_ns for record in self._image_records], dtype=np.int64
        )
        if np.any(np.diff(image_stamps) <= 0):
            self.close()
            raise ValueError("Slider RGB header timestamps are not strictly increasing")
        self._event_origin_ns = int(image_stamps[0])
        relative_rgb_ns = image_stamps - self._event_origin_ns
        self._rgb_timestamps = relative_rgb_ns // 1000
        first_ns = min(
            int(packet.first_ns) for packet in nonempty_packets if packet.first_ns is not None
        )
        last_ns = max(
            int(packet.last_ns) for packet in nonempty_packets if packet.last_ns is not None
        )
        nonzero_header_ns = [packet.header_ns for packet in self._event_packets if packet.header_ns]
        coverage_points_ns = [first_ns, last_ns, *nonzero_header_ns]
        self._event_start = min(
            (timestamp_ns - self._event_origin_ns) // 1000 for timestamp_ns in coverage_points_ns
        )
        self._event_end = (
            max(
                (timestamp_ns - self._event_origin_ns) // 1000
                for timestamp_ns in coverage_points_ns
            )
            + 1
        )
        relative_headers_us = [
            (timestamp_ns - self._event_origin_ns) // 1000 for timestamp_ns in nonzero_header_ns
        ]
        self._coverage_evidence = {
            "kind": "slider_packet_headers_and_events",
            "nonzero_packet_header_count": len(nonzero_header_ns),
            "zero_packet_header_count": sum(
                packet.header_ns == 0 for packet in self._event_packets
            ),
            "first_nonzero_packet_header_us": min(relative_headers_us, default=None),
            "last_nonzero_packet_header_us": max(relative_headers_us, default=None),
            "first_observed_event_us": (first_ns - self._event_origin_ns) // 1000,
            "last_observed_event_us": (last_ns - self._event_origin_ns) // 1000,
            "coverage_end_rule": "max(nonzero_packet_headers,observed_events)+1us",
            "silent_edge_uncertainty": True,
            "authoritative_recording_interval": False,
        }

    def _index_event_packet(self, offset: int, length: int) -> _EventPacket:
        stream = self._bag
        assert stream is not None
        stream.seek(offset)
        header_seconds, header_nanoseconds, _frame_id = _read_ros_header(stream)
        header_ns = _ns(header_seconds, header_nanoseconds)
        height, width, count = struct.unpack("<III", _read_exact(stream, 12))
        if height <= 0 or width <= 0:
            raise ValueError("Slider event packet has invalid resolution")
        if hasattr(self, "_event_width") and (width, height) != (
            self._event_width,
            self._event_height,
        ):
            raise ValueError("Slider event resolution changes within the bag")
        self._event_width, self._event_height = width, height
        events_offset = stream.tell()
        expected_end = events_offset + count * _EVENT_DTYPE.itemsize
        if expected_end != offset + length:
            raise ValueError("Slider event packet length does not match event count")
        if count == 0:
            return _EventPacket(events_offset, 0, header_ns, None, None)
        stream.seek(events_offset + 4)
        first_sec, first_nsec = struct.unpack("<II", _read_exact(stream, 8))
        stream.seek(events_offset + (count - 1) * _EVENT_DTYPE.itemsize + 4)
        last_sec, last_nsec = struct.unpack("<II", _read_exact(stream, 8))
        first_ns, last_ns = _ns(first_sec, first_nsec), _ns(last_sec, last_nsec)
        if last_ns < first_ns:
            raise ValueError("Slider event packet timestamps are not chronological")
        return _EventPacket(events_offset, count, header_ns, first_ns, last_ns)

    def _index_image_record(self, offset: int, length: int) -> _ImageRecord:
        stream = self._bag
        assert stream is not None
        stream.seek(offset)
        seconds, nanoseconds, _frame_id = _read_ros_header(stream)
        return _ImageRecord(offset, length, _ns(seconds, nanoseconds))

    @property
    def event_start_us(self) -> int:
        self._require_open()
        return self._event_start

    @property
    def event_end_us(self) -> int:
        self._require_open()
        return self._event_end

    @property
    def rgb_timestamps_us(self) -> np.ndarray:
        self._require_open()
        result = self._rgb_timestamps.copy()
        result.setflags(write=False)
        return result

    @property
    def event_width(self) -> int:
        self._require_open()
        return self._event_width

    @property
    def event_height(self) -> int:
        self._require_open()
        return self._event_height

    @property
    def rgb_width(self) -> int:
        self._require_open()
        if self._kind == "slider_ros1" and not hasattr(self, "_rgb_width"):
            self._read_slider_image_metadata(0)
        return self._rgb_width

    @property
    def rgb_height(self) -> int:
        self._require_open()
        if self._kind == "slider_ros1" and not hasattr(self, "_rgb_height"):
            self._read_slider_image_metadata(0)
        return self._rgb_height

    def read_events(self, start_us: int, end_us: int) -> np.ndarray:
        """Read chronological events in the half-open relative interval ``[start,end)``."""

        self._require_open()
        start, end = int(start_us), int(end_us)
        if end < start:
            raise ValueError("end_us must be greater than or equal to start_us")
        if end == start or end <= self._event_start or start >= self._event_end:
            return np.empty((0, 4), dtype=np.int64)
        if self._kind == "road_hdf5":
            return self._read_road_events(start, end)
        return self._read_slider_events(start, end)

    def _read_road_events(self, start: int, end: int) -> np.ndarray:
        handle = self._h5
        assert handle is not None
        base = "prophesee/event_cam_right"
        timestamps = _dataset(handle, f"{base}/t")
        left = _h5_lower_bound(timestamps, start)
        right = _h5_lower_bound(timestamps, end)
        if left == right:
            return np.empty((0, 4), dtype=np.int64)
        t = np.asarray(timestamps[left:right], dtype=np.int64)
        if np.any(np.diff(t) < 0):
            raise ValueError("Road right-event timestamps are not chronological")
        result = np.column_stack(
            (
                np.asarray(_dataset(handle, f"{base}/x")[left:right], dtype=np.int64),
                np.asarray(_dataset(handle, f"{base}/y")[left:right], dtype=np.int64),
                t,
                np.asarray(_dataset(handle, f"{base}/p")[left:right], dtype=np.int64),
            )
        )
        self._validate_events(result)
        return result

    def _read_slider_events(self, start: int, end: int) -> np.ndarray:
        stream = self._bag
        assert stream is not None
        pieces: list[np.ndarray] = []
        for packet in self._event_packets:
            if packet.count == 0 or packet.first_ns is None or packet.last_ns is None:
                continue
            first_us = (packet.first_ns - self._event_origin_ns) // 1000
            last_us = (packet.last_ns - self._event_origin_ns) // 1000
            if last_us < start or first_us >= end:
                continue
            stream.seek(packet.events_offset)
            raw = _read_exact(stream, packet.count * _EVENT_DTYPE.itemsize)
            events = np.frombuffer(raw, dtype=_EVENT_DTYPE, count=packet.count)
            timestamp_ns = events["sec"].astype(np.int64) * 1_000_000_000 + events["nsec"].astype(
                np.int64
            )
            timestamp_us = (timestamp_ns - self._event_origin_ns) // 1000
            keep = (timestamp_us >= start) & (timestamp_us < end)
            if not np.any(keep):
                continue
            pieces.append(
                np.column_stack(
                    (
                        events["x"][keep].astype(np.int64),
                        events["y"][keep].astype(np.int64),
                        timestamp_us[keep],
                        events["p"][keep].astype(np.int64),
                    )
                )
            )
        if not pieces:
            return np.empty((0, 4), dtype=np.int64)
        result = np.concatenate(pieces)
        if np.any(np.diff(result[:, 2]) < 0):
            result = result[np.argsort(result[:, 2], kind="stable")]
        self._validate_events(result)
        return result

    def _validate_events(self, events: np.ndarray) -> None:
        if events.ndim != 2 or events.shape[1] != 4 or events.dtype != np.int64:
            raise ValueError("Decoded events must be int64 [N,4]")
        if len(events) == 0:
            return
        if np.any(np.diff(events[:, 2]) < 0):
            raise ValueError("Decoded events are not chronological")
        if np.any(events[:, 0] < 0) or np.any(events[:, 0] >= self._event_width):
            raise ValueError("Decoded event x coordinate is outside the sensor raster")
        if np.any(events[:, 1] < 0) or np.any(events[:, 1] >= self._event_height):
            raise ValueError("Decoded event y coordinate is outside the sensor raster")
        if not np.all(np.isin(events[:, 3], (0, 1))):
            raise ValueError("Decoded event polarity must be 0 or 1")

    def _read_slider_image_metadata(self, index: int) -> tuple[str, int, int, int, int, int]:
        stream = self._bag
        assert stream is not None
        record = self._image_records[index]
        stream.seek(record.data_offset)
        _read_ros_header(stream)
        height, width = struct.unpack("<II", _read_exact(stream, 8))
        encoding = _read_ros_string(stream)
        big_endian = _read_exact(stream, 1)[0]
        step, data_size = struct.unpack("<II", _read_exact(stream, 8))
        if height <= 0 or width <= 0 or step <= 0 or data_size != step * height:
            raise ValueError("Malformed Slider RGB image dimensions")
        if hasattr(self, "_rgb_width") and (width, height) != (
            self._rgb_width,
            self._rgb_height,
        ):
            raise ValueError("Slider RGB resolution changes within the bag")
        self._rgb_width, self._rgb_height = width, height
        return (
            encoding,
            big_endian,
            step,
            data_size,
            stream.tell(),
            record.data_offset + record.data_length,
        )

    def read_rgb(self, index: int) -> np.ndarray:
        """Read one RGB frame as a detached ``uint8 [H,W,3]`` array."""

        self._require_open()
        if not isinstance(index, (int, np.integer)):
            raise TypeError("RGB frame index must be an integer")
        frame_index = int(index)
        if frame_index < 0 or frame_index >= len(self._rgb_timestamps):
            raise IndexError(f"RGB frame index out of range: {frame_index}")
        if self._kind == "road_hdf5":
            handle = self._h5
            assert handle is not None
            frame = np.asarray(
                _dataset(handle, "blackflys/right/data")[frame_index], dtype=np.uint8
            )
            return frame.copy()
        stream = self._bag
        assert stream is not None
        encoding, big_endian, step, data_size, data_offset, record_end = (
            self._read_slider_image_metadata(frame_index)
        )
        if big_endian != 0:
            raise ValueError("Big-endian Slider RGB images are unsupported")
        if data_offset + data_size != record_end:
            raise ValueError("Slider RGB payload length mismatch")
        stream.seek(data_offset)
        raw = _read_exact(stream, data_size)
        height, width = self._rgb_height, self._rgb_width
        if encoding in {"rgb8", "bgr8"}:
            if step < width * 3:
                raise ValueError("Slider RGB step is smaller than packed row width")
            rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, step)
            frame = rows[:, : width * 3].reshape(height, width, 3).copy()
            if encoding == "bgr8":
                frame = frame[:, :, ::-1].copy()
            return frame
        if encoding == "mono8":
            if step < width:
                raise ValueError("Slider mono8 step is smaller than image width")
            mono = np.frombuffer(raw, dtype=np.uint8).reshape(height, step)[:, :width]
            return np.repeat(mono[:, :, None], 3, axis=2)
        raise ValueError(f"Unsupported Slider image encoding: {encoding}")

    def bbox_rows(self) -> list[dict[str, Any]]:
        """Return raw CSV boxes and exact timestamp text anchored to RGB frames."""

        self._require_open()
        path = self.sequence_dir / "bbox.csv"
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row_index, row in enumerate(csv.reader(stream)):
                if len(row) != 6:
                    raise ValueError(f"Malformed bbox.csv row {row_index + 1}: expected 6 columns")
                try:
                    frame_index = int(row[0])
                    timestamp = Decimal(row[1])
                    coordinates = tuple(Decimal(value) for value in row[2:])
                except (ValueError, InvalidOperation) as exc:
                    raise ValueError(f"Malformed bbox.csv row {row_index + 1}") from exc
                if frame_index < 0 or frame_index >= len(self._rgb_timestamps):
                    raise ValueError(f"bbox.csv frame index out of range at row {row_index + 1}")
                if not timestamp.is_finite() or any(not value.is_finite() for value in coordinates):
                    raise ValueError(f"bbox.csv contains non-finite values at row {row_index + 1}")
                if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
                    raise ValueError(f"bbox.csv has a non-positive box at row {row_index + 1}")
                anchor_us = int(self._rgb_timestamps[frame_index])
                timestamp_us = timestamp * Decimal(1_000_000)
                if abs(timestamp_us - Decimal(anchor_us)) > Decimal(2):
                    raise ValueError(
                        f"bbox.csv timestamp is not aligned to RGB frame at row {row_index + 1}"
                    )
                rows.append(
                    {
                        "index": frame_index,
                        "row_index": row_index,
                        "frame_index": frame_index,
                        "timestamp_s": timestamp,
                        "timestamp_text": row[1],
                        "anchor_us": anchor_us,
                        "box_xyxy": tuple(
                            int(value) if value == value.to_integral_value() else float(value)
                            for value in coordinates
                        ),
                        "box_text": tuple(row[2:]),
                    }
                )
        return rows

    def source_metadata(self) -> dict[str, Any]:
        """Return immutable source identity, selected camera paths, and clock origins."""

        self._require_open()
        sidecars = [self.sequence_dir / "bbox.csv", self.sequence_dir / "calibParams.mat"]
        files = [path for path in [*self._source_files, *sidecars] if path.is_file()]
        metadata: dict[str, Any] = {
            "sequence_id": self.sequence_id,
            "kind": self._kind,
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)} for path in files
            },
            "event_clock": "relative_microseconds",
            "rgb_clock": "relative_microseconds",
            "polarity_encoding": "0_or_1",
            "bbox_timestamp_preservation": "Decimal timestamp_s plus original timestamp_text",
            "coverage_evidence": dict(self._coverage_evidence),
        }
        if self._kind == "road_hdf5":
            metadata.update(
                {
                    "clock_origin_us": 0,
                    "camera_selection": {
                        "rgb": "blackflys/right",
                        "events": "prophesee/event_cam_right",
                    },
                }
            )
        else:
            metadata.update(
                {
                    "clock_origin": "first /image_raw Header.stamp",
                    "clock_origin_ns": self._event_origin_ns,
                    "event_topic": "/capture_node/events",
                    "rgb_topic": "/image_raw",
                    "negative_event_times_allowed": True,
                }
            )
        return metadata


__all__ = ["SensorSequence"]
