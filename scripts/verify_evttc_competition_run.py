"""Independently verify a completed EvTTC run without importing its inference code.

Only sensor metadata, annotations, prediction artifacts and integrity receipts are
read. No model is loaded, no ground truth is accessed, and no accuracy is computed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import struct
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, BinaryIO

ROOT = Path(__file__).resolve().parents[1]
SEQUENCES = (
    "CCRs1-low",
    "CCRs1-medium",
    "CCRs1-high",
    "CCRs2-low",
    "CCRs2-medium",
    "CCRs2-high",
    "CCRm-low",
    "CCRm-medium",
    "Slider-750",
    "Slider-1000",
)
EXPECTED_ROWS = 434
COLUMNS = ["Index", "timestamp (s)", "ttc (s)", "cost time(s)"]
MAX_SPAN_US = 650_000
MODEL_NAMES = {
    "E_A5",
    "E_C2F",
    "R_A5",
    "R_C2F",
    "PAIR_E",
    "PAIR_R",
    "E_CTX_MATCHED",
    "R_CTX",
    "PHASE_ONLY_EVENT_S7_TRAIN40",
    "event_normalizer",
    "rgb_normalizer",
    "MODEL_SEAL.json",
    "OOF_RECIPE.json",
}


class AuditError(ValueError):
    """A completed artifact violated an independently checked invariant."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def reject_constant(value: str) -> None:
    raise AuditError(f"Non-finite JSON constant: {value}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def canonical_sha(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root.resolve()), f"Path escapes declared root: {relative}")
    return path


def finite(value: Any, name: str) -> float:
    require(isinstance(value, (float, int)) and not isinstance(value, bool), f"{name}: not numeric")
    result = float(value)
    require(math.isfinite(result), f"{name}: non-finite")
    return result


def integer(value: Any, name: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{name}: not integer")
    return value


def _fields(blob: bytes) -> dict[str, bytes]:
    result = {}
    position = 0
    while position < len(blob):
        size = struct.unpack_from("<I", blob, position)[0]
        position += 4
        item = blob[position : position + size]
        require(len(item) == size and b"=" in item, "Malformed bag field")
        key, value = item.split(b"=", 1)
        result[key.decode("ascii")] = value
        position += size
    return result


def _records(stream: BinaryIO, stop: int) -> Iterator[tuple[dict[str, bytes], int, int]]:
    while stream.tell() < stop:
        header_size = struct.unpack("<I", stream.read(4))[0]
        header = _fields(stream.read(header_size))
        size = struct.unpack("<I", stream.read(4))[0]
        offset = stream.tell()
        require(offset + size <= stop, "Bag record crosses enclosing boundary")
        yield header, offset, size
        stream.seek(offset + size)


def _stamp(stream: BinaryIO) -> int:
    _sequence, seconds, nanoseconds, name_size = struct.unpack("<IIII", stream.read(16))
    require(nanoseconds < 1_000_000_000, "Invalid sensor nanoseconds")
    stream.seek(name_size, 1)
    return seconds * 1_000_000_000 + nanoseconds


def sensor_clocks(folder: Path) -> dict[str, Any]:
    """Read raw clock/coverage metadata independently of the executed reader."""
    hdf5 = folder / "data.hdf5"
    if hdf5.is_file():
        import h5py

        with h5py.File(hdf5, "r") as source:

            def dataset(name: str) -> h5py.Dataset:
                value = source[name]
                if not isinstance(value, h5py.Dataset):
                    raise AuditError(f"HDF5 clock/index is not a dataset: {name}")
                return value

            clock = [int(value) for value in dataset("blackflys/right/ts")[:]]
            events = dataset("prophesee/event_cam_right/t")
            index_size = len(dataset("prophesee/event_cam_right/ms_map_idx"))
            return {
                "rgb_us": clock,
                "event_start_us": 0,
                "event_end_us": max((index_size - 1) * 1000, int(events[-1]) + 1),
                "coverage_source": "raw_HDF5_index_and_event_endpoints",
            }
    bags = list(folder.glob("*.bag"))
    require(len(bags) == 1, f"Expected one bag: {folder.name}")
    messages = []
    connections = {}
    with bags[0].open("rb") as stream:
        require(stream.read(13) == b"#ROSBAG V2.0\n", "Invalid ROS1 bag magic")
        for header, offset, size in _records(stream, bags[0].stat().st_size):
            operation = header["op"][0]
            if operation == 7:
                stream.seek(offset)
                fields = _fields(stream.read(size))
                connections[struct.unpack("<I", header["conn"])[0]] = (
                    header["topic"].decode(),
                    fields["type"].decode(),
                )
            elif operation == 5:
                require(header["compression"] == b"none", "Unsupported bag compression")
                stream.seek(offset)
                for inner, payload, length in _records(stream, offset + size):
                    if inner["op"] == b"\x02":
                        messages.append((struct.unpack("<I", inner["conn"])[0], payload, length))
        image_ns = []
        coverage_ns = []
        for connection, offset, length in messages:
            topic, kind = connections[connection]
            if topic not in {"/image_raw", "/capture_node/events"}:
                continue
            stream.seek(offset)
            stamp = _stamp(stream)
            if topic == "/image_raw":
                require(kind == "sensor_msgs/Image", "Unexpected image message type")
                image_ns.append(stamp)
                continue
            require(kind == "dv_ros_msgs/EventArray", "Unexpected event message type")
            if stamp:
                coverage_ns.append(stamp)
            _height, _width, count = struct.unpack("<III", stream.read(12))
            events_offset = stream.tell()
            require(events_offset + count * 13 == offset + length, "Event packet size mismatch")
            for index in {0, count - 1} if count else set():
                stream.seek(events_offset + 13 * index + 4)
                seconds, nanoseconds = struct.unpack("<II", stream.read(8))
                require(nanoseconds < 1_000_000_000, "Invalid event nanoseconds")
                coverage_ns.append(seconds * 1_000_000_000 + nanoseconds)
    require(bool(image_ns) and bool(coverage_ns), "Missing bag clock evidence")
    origin = image_ns[0]
    return {
        "rgb_us": [(value - origin) // 1000 for value in image_ns],
        "event_start_us": (min(coverage_ns) - origin) // 1000,
        "event_end_us": (max(coverage_ns) - origin) // 1000 + 1,
        "clock_origin_ns": origin,
        "coverage_source": "raw_packet_headers_and_event_endpoints_silent_edges_uncertain",
    }


def original_queries(folder: Path, clock: Sequence[int]) -> list[dict[str, Any]]:
    require(
        bool(clock) and all(a < b for a, b in zip(clock, clock[1:], strict=False)),
        "Nonmonotonic RGB clock",
    )
    identities = []
    with (folder / "bbox.csv").open(encoding="utf-8", newline="") as stream:
        for ordinal, row in enumerate(csv.reader(stream)):
            require(len(row) == 6, f"Malformed raw CSV: {folder.name}:{ordinal}")
            index = int(row[0])
            require(0 <= index < len(clock), "Raw CSV image index outside clock")
            timestamp = Decimal(row[1])
            box = [Decimal(value) for value in row[2:]]
            require(timestamp.is_finite() and all(x.is_finite() for x in box), "Nonfinite CSV")
            require(box[2] > box[0] and box[3] > box[1], "Degenerate CSV bbox")
            anchor = clock[index]
            require(abs(timestamp * 1_000_000 - anchor) <= 2, "CSV/sensor clock mismatch")
            identities.append(
                {
                    "sequence_id": folder.name,
                    "row_index": ordinal,
                    "index": index,
                    "timestamp_text": row[1],
                    "anchor_us": anchor,
                    "box_text": row[2:],
                }
            )
    require(bool(identities), "Empty raw query CSV")
    require(len({row["index"] for row in identities}) == len(identities), "Duplicate CSV index")
    require(
        all(
            a["anchor_us"] < b["anchor_us"]
            for a, b in zip(identities, identities[1:], strict=False)
        ),
        "Raw CSV order is not chronological",
    )
    return identities


def validate_temporal(
    row: Mapping[str, Any], clock: Mapping[str, Any], identities: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    """Check recorded evidence, not the runner's reported causal-validity booleans."""
    detail = row["input_details"]
    query = row["identity"]["anchor_us"]
    require(detail["query_us"] == query, "Input/query clock mismatch")
    start, end = clock["event_start_us"], clock["event_end_us"]
    require(detail["event_recording_start_us"] == start, "Event coverage start mismatch")
    require(detail["event_recording_end_us"] == end, "Event coverage end mismatch")
    events, rgb = detail["event_observations"], detail["rgb_observations"]
    require(len(events) <= 4 and len(rgb) <= 8, "Invalid observation history count")
    maximum_span = 0
    zero_windows = 0
    zero_observations = 0
    for modality, observations in (("event", events), ("rgb", rgb)):
        previous = None
        for observation in observations:
            anchor = integer(observation["anchor_us"], "observation anchor")
            available = integer(observation["available_us"], "observation availability")
            first = integer(observation["first_sensor_ts_us"], "first sensor timestamp")
            require(first <= anchor <= available <= query, "Future or reversed observation clocks")
            require(available == query, "Retrospective query ROI has incorrect availability")
            require(previous is None or previous < anchor, "Unordered/duplicate history anchors")
            previous = anchor
            require(query - first <= MAX_SPAN_US, "Complete sensor history exceeds 650ms")
            maximum_span = max(maximum_span, query - first)
            if modality == "event":
                windows = observation["windows_us"]
                require(len(windows) in {2, 3}, "Event observation is not T2/T3")
                require(all(len(window) == 2 for window in windows), "Malformed event windows")
                for window in windows:
                    for endpoint in window:
                        integer(endpoint, "event window endpoint")
                require(
                    all(start <= a < b <= min(end, query) for a, b in windows),
                    "Event window crosses recorded coverage or query",
                )
                require(
                    all(a[1] == b[0] for a, b in zip(windows, windows[1:], strict=False)),
                    "Event windows overlap or have gaps",
                )
                require(
                    windows[0][0] == first and windows[-1][1] == anchor,
                    "Event support endpoint mismatch",
                )
                times = [window[1] for window in windows]
                raw, roi = observation["raw_counts"], observation["roi_counts"]
                require(len(raw) == len(roi) == len(windows), "Event count length mismatch")
                require(
                    all(
                        isinstance(a, int) and isinstance(b, int) and 0 <= b <= a
                        for a, b in zip(raw, roi, strict=True)
                    ),
                    "Invalid event counts",
                )
                require(observation["admitted"] is True, "Unadmitted observation reached output")
                require(observation["zero_support"] == (sum(roi) == 0), "Zero-support mismatch")
                if "preparation_mode" in observation:
                    mode = (
                        "adaptive_observed_prefix"
                        if anchor - first < 300_000
                        else "native_100ms_windows"
                    )
                    require(
                        observation["preparation_mode"] == mode,
                        "Observation preparation mode differs from actual support",
                    )
                zero_windows += sum(value == 0 for value in roi)
                zero_observations += int(sum(roi) == 0)
            else:
                indices, times = observation["frame_indices"], observation["timestamps_us"]
                require(len(indices) == len(times) and len(times) in {2, 3}, "RGB is not T2/T3")
                require(
                    all(isinstance(i, int) and 0 <= i < len(clock["rgb_us"]) for i in indices),
                    "Invalid RGB frame index",
                )
                require(times == [clock["rgb_us"][i] for i in indices], "RGB raw clock mismatch")
                require(times[0] == first and times[-1] == anchor, "RGB endpoint mismatch")
            require(
                all(a < b for a, b in zip(times, times[1:], strict=False)),
                "Nonpositive sensor delta",
            )
            deltas = observation["delta_t_s"]
            expected = [(b - a) / 1e6 for a, b in zip(times, times[1:], strict=False)]
            require(
                len(deltas) == len(expected)
                and all(
                    math.isclose(finite(a, "sensor delta"), b, rel_tol=0, abs_tol=1e-12)
                    for a, b in zip(deltas, expected, strict=True)
                ),
                "Reported delta differs from actual sensor interval",
            )
    sources = detail["bbox_sources"]
    causal = [identity for identity in identities if identity["anchor_us"] <= query][-2:]
    require(len(sources) == len(causal), "Incorrect causal bbox source count")
    for source, identity in zip(sources, causal, strict=True):
        require(
            source["index"] == identity["index"] and source["anchor_us"] == identity["anchor_us"],
            "Future or wrong bbox annotation source",
        )
        require(
            [float(x) for x in source["raw_box"]] == [float(x) for x in identity["box_text"]],
            "BBox coordinates differ from source CSV",
        )
    validate_admission(row, events, rgb)
    has_prefix = any(item["anchor_us"] - item["first_sensor_ts_us"] < 300_000 for item in events)
    expected_mode = "adaptive_observed_prefix" if has_prefix else "native_100ms_windows"
    require(detail["startup_mode"] == expected_mode, "Startup mode label differs from support")
    return {
        "max_span_us": maximum_span,
        "zero_roi_windows": zero_windows,
        "zero_roi_observations": zero_observations,
    }


def validate_v2_plan(row: Mapping[str, Any], clock: Mapping[str, Any]) -> None:
    """Independently rebuild the frozen four-anchor, observed-prefix event plan."""
    query = row["identity"]["anchor_us"]
    start = clock["event_start_us"]
    expected = []
    if query <= clock["event_end_us"]:
        for lag in (300_000, 200_000, 100_000, 0):
            anchor = query - lag
            available = anchor - start
            if available < 3:
                continue
            first = max(start, anchor - 300_000)
            duration = anchor - first
            edges = [first + duration * index // 3 for index in range(4)]
            expected.append([[edges[index], edges[index + 1]] for index in range(3)])
    actual = [
        observation["windows_us"] for observation in row["input_details"]["event_observations"]
    ]
    require(actual == expected, "Prepared history differs from frozen v2 real-prefix anchor plan")


def validate_admission(
    row: Mapping[str, Any], events: Sequence[Mapping[str, Any]], rgb: Sequence[Mapping[str, Any]]
) -> None:
    """Recompute numerical rejection predicates and retain the complement in order."""
    prediction = row["prediction"]
    require(prediction["model_phase_delta_seconds"] == 0.1, "Frozen phase anchor changed")
    diagnostics = prediction["producer_diagnostics"]
    if not events:
        require(
            row["status"] == "unavailable" and not diagnostics,
            "Missing event preparation must remain unavailable without producer execution",
        )
        require(
            prediction["event_history_count"] == 0 and prediction["rgb_history_count"] == len(rgb),
            "Unattempted history count mismatch",
        )
        require(
            row["reason"] == "no_causally_available_t2_event_observation",
            "Wrong unavailable-history reason",
        )
        return
    version_two = "observations_requested" in diagnostics.get("event", {})
    require(
        set(diagnostics) == ({"event", "rgb"} if version_two or rgb else {"event"}),
        "Producer diagnostic modality coverage mismatch",
    )
    for modality, observations in (("event", events), ("rgb", rgb)):
        if modality not in diagnostics:
            require(not observations, "Missing producer diagnostics for prepared observations")
            continue
        values = diagnostics[modality]
        producer_ids = (
            ["E_A5_MATCHED", "E_C2F_MATCHED"] if modality == "event" else ["R_A5", "R_C2F"]
        )
        rejected = values.get("rejected", [])
        rejected_indices: list[int] = []
        rejected_known = 0
        rejected_total = 0
        rejected_support_count = 0
        rejected_times = []
        for entry in rejected:
            index = integer(entry["observation_index"], "rejected observation index")
            require(0 <= index < len(observations), "Rejection index outside prepared history")
            require(
                not rejected_indices or index > rejected_indices[-1],
                "Duplicate or unordered rejected observations",
            )
            rejected_indices.append(index)
            observation = observations[index]
            require(entry["anchor_us"] == observation["anchor_us"], "Rejected anchor mismatch")
            require(entry["reason"] == "unrepresentable_expert_ttc", "Wrong rejection reason")
            elapsed = finite(entry["compute_seconds"], "rejected compute time")
            require(elapsed >= 0, "Negative rejected compute time")
            rejected_times.append(elapsed)
            outputs = entry["all_producer_outputs"]
            require(1 <= len(outputs) <= 2, "Invalid rejected producer output count")
            require(
                [item["producer_id"] for item in outputs] == producer_ids[: len(outputs)],
                "Rejected producer execution order differs",
            )
            invalid = []
            steps = len(
                observation["windows_us"] if modality == "event" else observation["timestamps_us"]
            )
            for output in outputs:
                values_ttc = output["ttc_seconds"]
                require(len(values_ttc) == 1, "Expected one producer output per observation")
                classifications = []
                for value in values_ttc:
                    if value == "nan":
                        classifications.append("unrepresentable_nan")
                    else:
                        number = finite(value, "rejected producer TTC")
                        classifications.append(
                            "unrepresentable_zero"
                            if number == 0
                            else "unrepresentable_positive_le_model_delta"
                            if 0 < number <= 0.1
                            else "representable_finite"
                        )
                if "domain_classification" in output:
                    require(
                        output["domain_classification"] == classifications,
                        "Producer domain classification differs from recorded TTC",
                    )
                unrepresentable = any(
                    value.startswith("unrepresentable") for value in classifications
                )
                require(
                    output["unrepresentable"] is unrepresentable,
                    "Rejection predicate does not match mathematical phase domain",
                )
                if unrepresentable:
                    invalid.append(output)
                known = output["known_mask"]
                require(
                    len(known) == 1 and all(isinstance(value, bool) for value in known),
                    "Rejected known-mask shape/type mismatch",
                )
                support = output["sensor_support"]
                require(
                    len(support) == 1 and len(support[0]) == steps,
                    "Rejected support shape differs from actual T2/T3",
                )
                require(
                    all(0 <= finite(value, "rejected support") <= 1 for value in support[0]),
                    "Rejected support is outside [0,1]",
                )
                rejected_known += sum(known)
                rejected_total += 1
                rejected_support_count += steps
            require(
                invalid == [outputs[-1]] and entry["invalid_producers"] == invalid,
                "Rejection must stop at the first mathematically invalid producer",
            )
        accepted_indices = [
            index for index in range(len(observations)) if index not in rejected_indices
        ]
        if version_two:
            require(
                values["observations_requested"] == len(observations),
                "Prepared/requested observation count mismatch",
            )
            require(
                values["observations_accepted"] == len(accepted_indices)
                and values["observations_rejected_unrepresentable"] == len(rejected),
                "Accepted/rejected observation accounting mismatch",
            )
            require(
                math.isclose(
                    finite(values["rejected_compute_seconds"], "rejection total"),
                    sum(rejected_times),
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                ),
                "Rejected compute total mismatch",
            )
            if "accepted_observation_indices" in values:
                require(
                    values["accepted_observation_indices"] == accepted_indices,
                    "Accepted observation indices are not the ordered rejection complement",
                )
            if "accepted_anchor_us" in values:
                require(
                    values["accepted_anchor_us"]
                    == [observations[index]["anchor_us"] for index in accepted_indices],
                    "Accepted anchors are not aligned to accepted observations",
                )
        else:
            require(not rejected, "Legacy diagnostics cannot contain unaccounted rejections")
        accepted_steps = sum(
            len(
                observations[index]["windows_us"]
                if modality == "event"
                else observations[index]["timestamps_us"]
            )
            for index in accepted_indices
        )
        require(
            values["known_total"] == 2 * len(accepted_indices) + rejected_total
            and values["support_count"] == 2 * accepted_steps + rejected_support_count,
            "Producer diagnostic counts differ from attempted observations",
        )
        require(
            rejected_known <= values["known_true"] <= rejected_known + 2 * len(accepted_indices),
            "Known-mask aggregate cannot match executed producers",
        )
        if values["support_count"]:
            require(
                0
                <= finite(values["support_min"], "support minimum")
                <= finite(values["support_max"], "support maximum")
                <= 1,
                "Producer support outside [0,1]",
            )
        require(
            prediction[f"{modality}_history_count"] == len(accepted_indices),
            f"{modality} admitted history count mismatch",
        )
    if prediction["event_history_count"] == 0:
        require(
            row["status"] == "unavailable"
            and row["reason"] == "event_observations_unrepresentable",
            "No representable event observation must remain explicitly unavailable",
        )
    else:
        expected_reason = (
            "fused_event_rgb" if prediction["rgb_history_count"] else "event_only_fallback"
        )
        require(
            row["status"] == "ok" and row["reason"] == expected_reason,
            "Admitted histories do not match final output status/mode",
        )


def describe(values: Sequence[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p50": None,
            "p95": None,
        }

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])

    return {
        "count": len(values),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
    }


def descriptive_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    diagnostics: dict[str, dict[str, Any]] = {}
    for row in rows:
        for modality, values in row["prediction"].get("producer_diagnostics", {}).items():
            aggregate = diagnostics.setdefault(
                modality,
                {
                    "known_true": 0,
                    "known_total": 0,
                    "unknown_count": 0,
                    "support_count": 0,
                    "support_min": None,
                    "support_max": None,
                    "rows_all_unknown": 0,
                    "observations_requested": 0,
                    "observations_accepted": 0,
                    "observations_rejected_unrepresentable": 0,
                    "rejected_compute_seconds": 0.0,
                },
            )
            known, total = values["known_true"], values["known_total"]
            require(
                isinstance(known, int) and isinstance(total, int) and 0 <= known <= total,
                "Malformed producer known-mask diagnostics",
            )
            aggregate["known_true"] += known
            aggregate["known_total"] += total
            aggregate["unknown_count"] += total - known
            aggregate["support_count"] += values["support_count"]
            aggregate["rows_all_unknown"] += int(total > 0 and known == 0)
            for key in (
                "observations_requested",
                "observations_accepted",
                "observations_rejected_unrepresentable",
                "rejected_compute_seconds",
            ):
                aggregate[key] += values.get(key, 0)
            if not values["support_count"]:
                continue
            for key, reducer in (("support_min", min), ("support_max", max)):
                value = finite(values[key], key)
                aggregate[key] = value if aggregate[key] is None else reducer(aggregate[key], value)
    predictions = [float(row["prediction_ttc_seconds"]) for row in rows if row["status"] == "ok"]
    return {
        "rows": len(rows),
        "available_prediction_rows": len(predictions),
        "unavailable_prediction_rows": len(rows) - len(predictions),
        "signed_ttc_seconds": describe(predictions),
        "negative_ttc_rows": sum(value < 0 for value in predictions),
        "zero_ttc_rows": sum(value == 0 for value in predictions),
        "startup_modes": dict(Counter(row["input_details"]["startup_mode"] for row in rows)),
        "output_modes": dict(Counter(row["reason"] for row in rows)),
        "zero_roi_rows": sum(
            any(item["zero_support"] for item in row["input_details"]["event_observations"])
            for row in rows
        ),
        "producer_diagnostics": diagnostics,
        "producer_diagnostics_scope": "all_attempted_producers_including_rejected_observations",
        "runtime_seconds": {
            "per_estimate_total": describe([float(row["cost_time_seconds"]) for row in rows]),
            "preprocess": describe([float(row["preprocess_seconds"]) for row in rows]),
            "model_wall": describe([float(row["model_wall_seconds"]) for row in rows]),
        },
    }


def verify_integrity(
    freeze: Mapping[str, Any], data_root: Path, project_root: Path
) -> dict[str, Any]:
    checked = {}

    def check(path: Path, expected: str) -> None:
        actual = checked.setdefault(str(path), file_sha(path))
        require(actual == expected, f"SHA-256 changed: {path}")

    raw_count = 0
    raw_bytes = 0
    for sequence, metadata in freeze["raw_sources"].items():
        for name, expected in metadata["files"].items():
            require(Path(name).name == name, "Nonlocal raw source filename")
            path = inside(data_root, f"{sequence}/{name}")
            require(path.stat().st_size == expected["bytes"], f"Raw size changed: {path}")
            check(path, expected["sha256"])
            raw_count += 1
            raw_bytes += expected["bytes"]
    require(raw_count == 22, "Expected 22 frozen sensor/annotation/calibration files")
    sources = dict(freeze["executed_source_sha256"])
    for relative, expected in freeze["frozen_runtime"]["source_sha256"].items():
        require(
            relative not in sources or sources[relative] == expected, "Contradictory source hash"
        )
        sources[relative] = expected
    for relative, expected in sources.items():
        check(inside(project_root, relative), expected)
    candidate = Path(freeze["candidate_root_resolved"]).resolve(strict=True)
    portable_path = candidate / "PORTABLE_PATHS.json"
    check(portable_path, freeze["model"]["portable_paths_sha256"])
    portable = read_json(portable_path)
    require(set(freeze["model"]["components"]) == MODEL_NAMES, "Wrong final model component set")
    for name, expected in freeze["model"]["components"].items():
        item = portable["components"][name]
        path = Path(item["local_path"]).resolve(strict=True)
        require(path.is_relative_to(candidate), "Model component outside candidate directory")
        require(item["sha256"] == expected, "Portable/frozen component hash differs")
        check(path, expected)
    parent = freeze.get("parent_attempt")
    if parent is not None:
        require(parent.get("read_only") is True, "Parent attempt is not bound read-only")
        parent_root = Path(parent["path_resolved"]).resolve(strict=True)
        for name, key in (
            ("RUN_FREEZE.json", "run_freeze_sha256"),
            ("all_rows.jsonl", "all_rows_ledger_sha256"),
            ("errors.jsonl", "errors_ledger_sha256"),
        ):
            check(parent_root / name, parent[key])
    return {
        "raw_files_verified": raw_count,
        "raw_bytes_verified": raw_bytes,
        "source_files_verified": len(sources),
        "model_component_files_verified": len(MODEL_NAMES),
        "sha256": checked,
        "archive_rehash": "not_repeated; frozen runtime source bytes independently rehashed",
        "parent_attempt_files_verified": 3 if parent is not None else 0,
    }


def verify_package(
    run: Path,
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    by_sequence: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    partial: bool,
) -> dict[str, Any]:
    """Check either complete submission bytes or explicitly non-submission partial artifacts."""
    rows = [row for sequence in SEQUENCES for row in by_sequence[sequence]]
    expected_available = sum(row["status"] == "ok" for row in rows)
    if partial:
        require(
            not (run / "results.zip").exists() and not (run / "results").exists(),
            "PARTIAL run must not contain complete-submission artifacts",
        )
        require(
            summary.get("partial_results") == "partial_results"
            and summary.get("partial_results_zip") == "partial_results.zip",
            "Missing distinctly named terminal partial package",
        )
        root_name, archive_name, basename = (
            "partial_results",
            "partial_results.zip",
            "available_predictions.tsv",
        )
        hash_key = "partial_results_zip_sha256"
        partial_manifest = read_json(run / root_name / "partial_manifest.json")
        require(
            partial_manifest["status"] == "PARTIAL_NOT_SUBMISSION",
            "Partial manifest is mislabeled as a submission",
        )
        require(
            partial_manifest["expected_rows"] == EXPECTED_ROWS
            and partial_manifest["available_rows"] == expected_available,
            "Partial population count mismatch",
        )
        expected_unavailable = [
            {
                "sequence_id": row["identity"]["sequence_id"],
                "row_index": row["identity"]["row_index"],
                "index": row["identity"]["index"],
                "timestamp_text": row["identity"]["timestamp_text"],
                "status": row["status"],
                "reason": row["reason"],
            }
            for row in rows
            if row["status"] != "ok"
        ]
        require(
            partial_manifest["unavailable_rows"] == expected_unavailable,
            "Partial manifest drops or changes unavailable identities/reasons",
        )
        require(
            summary["partial_available_rows"] == expected_available
            and summary["partial_unavailable_rows"] == len(expected_unavailable),
            "Summary partial coverage mismatch",
        )
        entries = partial_manifest["sequence_files"]
        extras = {"partial_manifest.json", "PARTIAL_NOT_SUBMISSION.txt"}
        row_count_key = "available_rows"
        require(
            "PARTIAL" in (run / root_name / "PARTIAL_NOT_SUBMISSION.txt").read_text(),
            "Missing partial-only notice",
        )
    else:
        require(summary["results_zip"] == "results.zip", "Unexpected submission ZIP path")
        root_name, archive_name, basename = "results", "results.zip", "results.txt"
        hash_key = "results_zip_sha256"
        entries = summary["results_validation"]["files"]
        require(
            summary["results_validation"]["valid"] is True
            and not summary["results_validation"]["errors"],
            "Runner result validation was not successful",
        )
        require(
            not (run / "partial_results.zip").exists(),
            "Complete run retains a contradictory partial archive",
        )
        extras, row_count_key = set(), "rows"
    archive_sha = file_sha(run / archive_name)
    require(archive_sha == summary[hash_key] == manifest[hash_key], "Submission ZIP SHA mismatch")
    result_members = {f"{sequence}/{basename}" for sequence in SEQUENCES}
    expected_members = result_members | extras
    actual_disk = {
        path.relative_to(run / root_name).as_posix()
        for path in (run / root_name).rglob("*")
        if path.is_file()
    }
    require(actual_disk == expected_members, "Unexpected/missing disk results files")
    require(
        len(entries) == 10 and {entry["path"] for entry in entries} == result_members,
        "Results hash manifest has wrong files",
    )
    entry_by_path = {entry["path"]: entry for entry in entries}
    with zipfile.ZipFile(run / archive_name) as archive:
        require(
            len(archive.infolist()) == len(expected_members)
            and set(archive.namelist()) == expected_members,
            "ZIP has extra, missing, or duplicate members",
        )
        require(archive.testzip() is None, "ZIP CRC check failed")
        for relative in expected_members:
            require(
                archive.read(relative) == (run / root_name / relative).read_bytes(),
                "ZIP/disk bytes differ",
            )
        for sequence in SEQUENCES:
            relative = f"{sequence}/{basename}"
            payload = archive.read(relative)
            entry = entry_by_path[relative]
            require(
                hashlib.sha256(payload).hexdigest() == entry["sha256"], "Results file SHA mismatch"
            )
            content = list(csv.reader(io.StringIO(payload.decode("utf-8")), delimiter="\t"))
            require(bool(content) and content[0] == COLUMNS, "Wrong results columns")
            expected_rows = [row for row in by_sequence[sequence] if row["status"] == "ok"]
            require(
                len(content) - 1 == entry[row_count_key] == len(expected_rows),
                "Results row count mismatch",
            )
            for cells, row in zip(content[1:], expected_rows, strict=True):
                require(len(cells) == 4, "Results row is not four columns")
                identity = row["identity"]
                require(
                    cells[:2] == [str(identity["index"]), identity["timestamp_text"]],
                    "Results original index/timestamp text mismatch",
                )
                prediction, cost = float(cells[2]), float(cells[3])
                require(
                    math.isfinite(prediction) and math.isfinite(cost) and cost >= 0,
                    "Invalid numerical result",
                )
                require(
                    prediction == row["prediction_ttc_seconds"]
                    and cost == row["cost_time_seconds"],
                    "Results prediction/runtime differs from ledger",
                )
    return {
        "archive": archive_name,
        "root": root_name,
        "sha256": archive_sha,
        "members_verified": len(expected_members),
        "submission_complete": not partial,
        "available_rows": expected_available,
        "unavailable_rows": len(rows) - expected_available,
    }


def audit_run(
    run: Path,
    data_root: Path,
    *,
    project_root: Path = ROOT,
    clock_reader: Callable[[Path], dict[str, Any]] = sensor_clocks,
) -> dict[str, Any]:
    """Audit only terminal, fully accounted COMPLETE or PARTIAL populations."""
    started = time.perf_counter()
    run, data_root = run.resolve(strict=True), data_root.resolve(strict=True)
    summary = read_json(run / "summary.json")
    terminal_status = summary.get("status")
    require(terminal_status in {"COMPLETE", "PARTIAL"}, "Run is not terminal; audit must wait")
    partial = terminal_status == "PARTIAL"
    if partial:
        require(
            summary.get("terminal_population_accounted") is True
            and summary.get("prepare_only") is False
            and summary.get("limit") is None,
            "Run is not terminal COMPLETE/PARTIAL; audit must wait",
        )
    manifest = read_json(run / "manifest.json")
    require(manifest.get("status") == terminal_status, "Manifest terminal status differs")
    freeze = read_json(run / "RUN_FREEZE.json")
    freeze_sha = file_sha(run / "RUN_FREEZE.json")
    require(
        summary["freeze_sha256"] == manifest["freeze_sha256"] == freeze_sha, "Freeze SHA mismatch"
    )
    require(freeze["sequences"] == list(SEQUENCES), "Wrong frozen sequence set/order")
    require(
        freeze["queries"] == EXPECTED_ROWS and freeze["run_mode"] == "inference",
        "Wrong run mode/population",
    )
    require(
        freeze["targets_read"] is False
        and freeze["training_updates"] == 0
        and freeze["checkpoint_selection"] is False,
        "Run declares target access or fitting",
    )
    for field in ("policy", "query_manifest", "calibrations", "raw_sources"):
        require(
            canonical_sha(freeze[field]) == freeze[f"{field}_sha256"],
            f"Frozen {field} hash mismatch",
        )
    require(set(freeze["raw_sources"]) == set(SEQUENCES), "Wrong raw source sequence set")
    require(
        summary["recorded_rows"] == summary["expected_rows"] == EXPECTED_ROWS
        and summary["ok_rows"] + summary["null_rows"] == EXPECTED_ROWS
        and ((summary["null_rows"] > 0) if partial else (summary["null_rows"] == 0)),
        "Summary population mismatch",
    )
    require(
        not (run / "errors.jsonl").exists() or (run / "errors.jsonl").stat().st_size == 0,
        "Run has a nonempty error ledger",
    )
    clocks = {sequence: clock_reader(data_root / sequence) for sequence in SEQUENCES}
    queries = {
        sequence: original_queries(data_root / sequence, clocks[sequence]["rgb_us"])
        for sequence in SEQUENCES
    }
    require(sum(map(len, queries.values())) == EXPECTED_ROWS, "Raw CSV population differs from 434")
    require(
        queries == freeze["query_manifest"],
        "Frozen identities differ from original CSV/sensor clocks",
    )
    with (run / "all_rows.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line, parse_constant=reject_constant) for line in stream if line.strip()]
    require(len(rows) == EXPECTED_ROWS, "Dropped or additional ledger rows")
    expected = [identity for sequence in SEQUENCES for identity in queries[sequence]]
    require(
        [row["identity"] for row in rows] == expected, "Ledger identity/order differs from raw CSV"
    )
    temporal = []
    for row in rows:
        require(
            row["schema"] == "evttc_competition_row_v1" and row["freeze_sha256"] == freeze_sha,
            "Ledger row/freeze binding mismatch",
        )
        require(row["status"] in {"ok", "unavailable"}, "Invalid terminal row status")
        value = row["prediction_ttc_seconds"]
        if row["status"] == "ok":
            value = finite(value, "ledger TTC")
        else:
            require(partial and value is None, "Unavailable row has a fabricated finite TTC")
            require(
                row["prediction"]["event_ttc"] is None and row["prediction"]["rgb_ttc"] is None,
                "Unavailable row must not claim evaluated context predictions",
            )
        cost = finite(row["cost_time_seconds"], "ledger runtime")
        preprocessing = finite(row["preprocess_seconds"], "preprocess runtime")
        model_time = finite(row["model_wall_seconds"], "model runtime")
        require(min(cost, preprocessing, model_time) >= 0, "Negative runtime")
        require(
            math.isclose(cost, preprocessing + model_time, rel_tol=1e-12, abs_tol=1e-12),
            "Per-estimate runtime does not equal preprocessing plus model wall time",
        )
        require(row["prediction"] == row["runtime_internal"], "Duplicate runtime ledgers disagree")
        require(
            row["prediction"]["model_ttc"] == value
            and row["prediction"]["status"] == row["status"],
            "Prediction fields disagree",
        )
        require(row["reason"] == row["prediction"]["model_reason"], "Output-mode mismatch")
        sequence = row["identity"]["sequence_id"]
        temporal.append(validate_temporal(row, clocks[sequence], queries[sequence]))
        if freeze["policy"].get("version") == "evttc_competition_causal_adapter_v2":
            validate_v2_plan(row, clocks[sequence])
    available_rows = sum(row["status"] == "ok" for row in rows)
    require(
        available_rows == summary["ok_rows"] and len(rows) - available_rows == summary["null_rows"],
        "Ledger availability counts differ",
    )
    by_sequence = {
        sequence: [row for row in rows if row["identity"]["sequence_id"] == sequence]
        for sequence in SEQUENCES
    }
    package = verify_package(run, summary, manifest, by_sequence, partial=partial)
    integrity = verify_integrity(freeze, data_root, project_root)
    result = {
        "schema": "evttc_competition_independent_audit_v2",
        "status": "PARTIAL" if partial else "PASS",
        "audit_status": "PASS",
        "run_status": terminal_status,
        "audited_utc": datetime.now(UTC).isoformat(),
        "run": str(run),
        "rows_verified": len(rows),
        "sequences_verified": len(SEQUENCES),
        "zip_members_verified": package["members_verified"],
        "package": package,
        "max_complete_sensor_span_us": max(item["max_span_us"] for item in temporal),
        "zero_roi_observation_uses": sum(item["zero_roi_observations"] for item in temporal),
        "zero_roi_window_uses": sum(item["zero_roi_windows"] for item in temporal),
        "integrity": integrity,
        "aggregate": descriptive_rows(rows),
        "per_sequence": {
            sequence: descriptive_rows(by_sequence[sequence]) for sequence in SEQUENCES
        },
        "artifact_sha256": {
            name: file_sha(run / name)
            for name in (
                "RUN_FREEZE.json",
                "all_rows.jsonl",
                "summary.json",
                "manifest.json",
                package["archive"],
            )
        },
        "verifier_sha256": file_sha(Path(__file__)),
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Descriptive prediction/runtime statistics only; no accuracy or SOTA conclusion.",
            "No model execution, ground-truth reads, PDF reads, or prediction-dependent selection.",
            "Original ledger has per-row freeze bindings but no separate prior ledger SHA seal; "
            "this audit records its SHA.",
            "Sensor metadata/coverage independently checked; "
            "event tensors and event counts are not recomputed.",
            "Slider coverage from packet/event bounds retains declared silent-edge uncertainty.",
            "Official acceptance of signed values and the CSV query contract remain unverified.",
            "Producer diagnostics include attempted rejected observations; "
            "null predictions remain unavailable and cannot form a complete submission.",
        ],
    }
    require(
        result["aggregate"]["zero_roi_rows"] == summary["zero_support_rows"],
        "Summary zero-support count mismatch",
    )
    for modality, values in summary["producer_known_mask_diagnostics"].items():
        for key, value in values.items():
            require(
                result["aggregate"]["producer_diagnostics"][modality][key] == value,
                "Summary known-mask diagnostic count mismatch",
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run / "independent_audit.json"
    try:
        result = audit_run(args.run, args.data_root)
    except (AuditError, OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        result = {
            "schema": "evttc_competition_independent_audit_v1",
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {key: result[key] for key in ("status", "rows_verified", "error") if key in result}
        )
    )
    return 0 if result.get("audit_status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
