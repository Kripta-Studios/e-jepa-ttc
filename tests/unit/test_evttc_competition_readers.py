from __future__ import annotations

import struct
from decimal import Decimal
from pathlib import Path

import h5py
import numpy as np
import pytest

from operational.evttc_competition.readers import SensorSequence


def _field(key: str, value: bytes) -> bytes:
    item = key.encode("ascii") + b"=" + value
    return struct.pack("<I", len(item)) + item


def _record(header: dict[str, bytes], data: bytes) -> bytes:
    encoded_header = b"".join(_field(key, value) for key, value in header.items())
    return (
        struct.pack("<I", len(encoded_header))
        + encoded_header
        + struct.pack("<I", len(data))
        + data
    )


def _ros_header(seq: int, seconds: int, nanoseconds: int, frame_id: str = "") -> bytes:
    frame = frame_id.encode("utf-8")
    return struct.pack("<IIII", seq, seconds, nanoseconds, len(frame)) + frame


def _event_packet(
    seq: int,
    header_stamp: tuple[int, int],
    events: list[tuple[int, int, int, int, bool]],
) -> bytes:
    payload = _ros_header(seq, *header_stamp)
    payload += struct.pack("<III", 3, 4, len(events))
    payload += b"".join(
        struct.pack("<HHIIB", x, y, seconds, nanoseconds, polarity)
        for x, y, seconds, nanoseconds, polarity in events
    )
    return payload


def _image(seq: int, seconds: int, nanoseconds: int, pixels: np.ndarray) -> bytes:
    height, width, channels = pixels.shape
    assert channels == 3
    encoding = b"rgb8"
    raw = pixels.tobytes()
    return (
        _ros_header(seq, seconds, nanoseconds, "camera")
        + struct.pack("<II", height, width)
        + struct.pack("<I", len(encoding))
        + encoding
        + b"\x00"
        + struct.pack("<II", width * 3, len(raw))
        + raw
    )


def _message(connection: int, data: bytes, bag_seconds: int = 99) -> bytes:
    return _record(
        {
            "op": b"\x02",
            "conn": struct.pack("<I", connection),
            "time": struct.pack("<II", bag_seconds, 0),
        },
        data,
    )


def _write_slider(
    root: Path,
    name: str = "Slider-test",
    *,
    event_topic: bytes = b"/capture_node/events",
    rgb_topic: bytes = b"/image_raw",
) -> Path:
    directory = root / name
    directory.mkdir()
    event_connection = _record(
        {"op": b"\x07", "conn": struct.pack("<I", 1), "topic": event_topic},
        _field("type", b"dv_ros_msgs/EventArray") + _field("md5sum", b"test"),
    )
    image_connection = _record(
        {"op": b"\x07", "conn": struct.pack("<I", 2), "topic": rgb_topic},
        _field("type", b"sensor_msgs/Image") + _field("md5sum", b"test"),
    )
    zero_header_empty = _message(1, _event_packet(0, (0, 0), []))
    leading_empty = _message(1, _event_packet(1, (9, 999_000_000), []))
    events = _message(
        1,
        _event_packet(
            2,
            (10, 5_000),
            [
                (1, 2, 9, 999_995_000, False),
                (2, 1, 10, 5_000, True),
                (3, 0, 10, 505_000, False),
            ],
        ),
    )
    frame0 = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    frame1 = np.full((2, 3, 3), 77, dtype=np.uint8)
    images = _message(2, _image(0, 10, 5_000, frame0)) + _message(
        2, _image(1, 10, 1_005_000, frame1)
    )
    trailing_empty = _message(1, _event_packet(3, (10, 2_005_000), []))
    chunk_payload = zero_header_empty + leading_empty + events + images + trailing_empty
    chunk = _record(
        {
            "op": b"\x05",
            "compression": b"none",
            "size": struct.pack("<I", len(chunk_payload)),
        },
        chunk_payload,
    )
    (directory / "slider.bag").write_bytes(
        b"#ROSBAG V2.0\n" + event_connection + image_connection + chunk
    )
    (directory / "bbox.csv").write_text("1,0.0010000000000000,1,2,3,4\n", encoding="utf-8")
    return directory


def test_external_slider_profile_is_explicit_and_preserves_header_clock(tmp_path):
    _write_slider(tmp_path, event_topic=b"/dvs/events", rgb_topic=b"/aps/image_raw")
    with pytest.raises(ValueError, match="audited"):
        SensorSequence(tmp_path, "Slider-test")
    with SensorSequence(tmp_path, "Slider-test", slider_profile="event_aided_ttc") as source:
        assert source.rgb_timestamps_us.tolist() == [0, 1000]
        assert len(source.read_events(-20, 1000)) == 3
        metadata = source.source_metadata()
        assert metadata["event_topic"] == "/dvs/events"
        assert metadata["rgb_topic"] == "/aps/image_raw"


def _write_road(root: Path, name: str = "road") -> Path:
    directory = root / name
    directory.mkdir()
    with h5py.File(directory / "data.hdf5", "w") as handle:
        rgb = handle.create_group("blackflys/right")
        rgb.create_dataset("data", data=np.arange(36, dtype=np.uint8).reshape(2, 2, 3, 3))
        rgb.create_dataset("ts", data=np.asarray([100, 1_100], dtype=np.uint64))
        rgb.create_dataset("calib/resolution", data=np.asarray([3, 2], dtype=np.int64))
        events = handle.create_group("prophesee/event_cam_right")
        events.create_dataset("x", data=np.asarray([1, 2, 3], dtype=np.uint16))
        events.create_dataset("y", data=np.asarray([0, 1, 1], dtype=np.uint16))
        events.create_dataset("t", data=np.asarray([500, 1_500, 2_500], dtype=np.int64))
        events.create_dataset("p", data=np.asarray([0, 1, 0], dtype=np.int8))
        events.create_dataset("ms_map_idx", data=np.asarray([0, 1, 2], dtype=np.uint64))
        events.create_dataset("calib/resolution", data=np.asarray([4, 3], dtype=np.int64))
        # A left camera is deliberately present; the reader must never select it.
        handle.create_dataset("blackflys/left/data", data=np.zeros((1, 1, 1, 3), dtype=np.uint8))
    (directory / "bbox.csv").write_text("0,0.0001000000000000,0,0,2,2\n", encoding="utf-8")
    return directory


def test_road_reader_selects_right_camera_and_half_open_windows(tmp_path: Path) -> None:
    _write_road(tmp_path)
    with SensorSequence(tmp_path, "road") as sequence:
        assert (sequence.event_width, sequence.event_height) == (4, 3)
        assert (sequence.rgb_width, sequence.rgb_height) == (3, 2)
        assert sequence.event_start_us == 0
        assert sequence.event_end_us == 2_501
        np.testing.assert_array_equal(sequence.rgb_timestamps_us, [100, 1_100])
        np.testing.assert_array_equal(sequence.read_events(1_000, 2_000), [[2, 1, 1_500, 1]])
        assert sequence.read_events(2_501, 9_000).shape == (0, 4)
        np.testing.assert_array_equal(
            sequence.read_rgb(1), np.arange(36, dtype=np.uint8).reshape(2, 2, 3, 3)[1]
        )
        bbox = sequence.bbox_rows()[0]
        assert bbox["timestamp_s"] == Decimal("0.0001000000000000")
        assert bbox["timestamp_text"] == "0.0001000000000000"
        assert bbox["anchor_us"] == 100
        assert bbox["box_xyxy"] == (0, 0, 2, 2)
        metadata = sequence.source_metadata()
        assert metadata["camera_selection"]["events"].endswith("right")
        assert metadata["coverage_evidence"]["authoritative_start_us"] == 0
        assert metadata["coverage_evidence"]["last_indexed_boundary_us"] == 2_000


def test_slider_payload_clocks_empty_packets_and_frame_mapping(tmp_path: Path) -> None:
    _write_slider(tmp_path)
    with SensorSequence(tmp_path, "Slider-test") as sequence:
        assert (sequence.event_width, sequence.event_height) == (4, 3)
        assert (sequence.rgb_width, sequence.rgb_height) == (3, 2)
        assert sequence.event_start_us == -1_005
        assert sequence.event_end_us == 2_001
        np.testing.assert_array_equal(sequence.rgb_timestamps_us, [0, 1_000])
        np.testing.assert_array_equal(
            sequence.read_events(-10, 1),
            [[1, 2, -10, 0], [2, 1, 0, 1]],
        )
        np.testing.assert_array_equal(sequence.read_events(0, 1), [[2, 1, 0, 1]])
        np.testing.assert_array_equal(
            sequence.read_rgb(0), np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        )
        bbox = sequence.bbox_rows()[0]
        assert bbox["frame_index"] == 1
        assert bbox["anchor_us"] == 1_000
        assert bbox["timestamp_s"] == Decimal("0.0010000000000000")
        metadata = sequence.source_metadata()
        assert metadata["negative_event_times_allowed"] is True
        assert metadata["coverage_evidence"]["zero_packet_header_count"] == 1
        assert metadata["coverage_evidence"]["first_nonzero_packet_header_us"] == -1_005
        assert metadata["coverage_evidence"]["last_nonzero_packet_header_us"] == 2_000


def test_future_slider_read_returns_without_touching_bag(tmp_path: Path) -> None:
    _write_slider(tmp_path)
    with SensorSequence(tmp_path, "Slider-test") as sequence:
        assert sequence._bag is not None
        position = sequence._bag.tell()
        assert sequence.read_events(sequence.event_end_us, sequence.event_end_us + 1_000).size == 0
        assert sequence._bag.tell() == position


def test_closed_reader_rejects_access(tmp_path: Path) -> None:
    _write_road(tmp_path)
    sequence = SensorSequence(tmp_path, "road")
    sequence.close()
    with pytest.raises(RuntimeError, match="closed"):
        _ = sequence.rgb_timestamps_us


def test_malformed_inputs_are_rejected(tmp_path: Path) -> None:
    broken_road = tmp_path / "broken-road"
    broken_road.mkdir()
    with h5py.File(broken_road / "data.hdf5", "w") as handle:
        handle.create_dataset("blackflys/left/data", data=np.zeros((1, 1, 1, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="right-camera"):
        SensorSequence(tmp_path, "broken-road")

    broken_slider = tmp_path / "broken-slider"
    broken_slider.mkdir()
    (broken_slider / "bad.bag").write_bytes(b"not a bag")
    with pytest.raises(ValueError, match="ROS bag"):
        SensorSequence(tmp_path, "broken-slider")


def test_bbox_frame_out_of_range_is_rejected(tmp_path: Path) -> None:
    directory = _write_slider(tmp_path)
    (directory / "bbox.csv").write_text("9,0.1,1,2,3,4\n", encoding="utf-8")
    with SensorSequence(tmp_path, "Slider-test") as sequence:
        with pytest.raises(ValueError, match="frame index out of range"):
            sequence.bbox_rows()
