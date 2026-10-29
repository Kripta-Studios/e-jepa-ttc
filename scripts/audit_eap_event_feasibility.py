"""Read-only, bounded audit of local eAP event timing and storage contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import h5py
import hdf5plugin  # noqa: F401 -- registers the pinned Blosc2 HDF5 filter
import numpy as np
import pyarrow.parquet as pq


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_sha256(value: object) -> str:
    matrix = np.stack(value).astype("<f8")  # type: ignore[arg-type]
    return hashlib.sha256(matrix.tobytes()).hexdigest()


def _exact_index(timestamps: h5py.Dataset, index: h5py.Dataset, timestamp_us: int) -> int:
    millisecond = timestamp_us // 1000
    if millisecond < 0 or millisecond + 1 >= len(index):
        raise IndexError(f"Timestamp outside ms_to_idx: {timestamp_us}")
    first, last = int(index[millisecond]), int(index[millisecond + 1])
    values = np.asarray(timestamps[first:last], dtype=np.int64)
    return first + int(np.searchsorted(values, timestamp_us, side="left"))


def _sample_payload(
    handle: h5py.File, *, chunks: int = 5, chunk_events: int = 40_000
) -> dict[str, Any]:
    timestamps_dataset = cast(h5py.Dataset, handle["events/t"])
    x_dataset = cast(h5py.Dataset, handle["events/x"])
    y_dataset = cast(h5py.Dataset, handle["events/y"])
    polarity_dataset = cast(h5py.Dataset, handle["events/p"])
    length = len(timestamps_dataset)
    starts = sorted(
        {
            min(max(0, int(position) - chunk_events // 2), max(0, length - chunk_events))
            for position in np.linspace(0, max(0, length - 1), chunks)
        }
    )
    observed_x: list[np.ndarray] = []
    observed_y: list[np.ndarray] = []
    observed_p: list[np.ndarray] = []
    timestamp_monotonic = True
    for start in starts:
        stop = min(length, start + chunk_events)
        timestamps = np.asarray(timestamps_dataset[start:stop], dtype=np.int64)
        timestamp_monotonic &= bool(np.all(timestamps[1:] >= timestamps[:-1]))
        observed_x.append(np.asarray(x_dataset[start:stop], dtype=np.uint16))
        observed_y.append(np.asarray(y_dataset[start:stop], dtype=np.uint16))
        observed_p.append(np.asarray(polarity_dataset[start:stop], dtype=np.int8))
    x, y, polarity = map(np.concatenate, (observed_x, observed_y, observed_p))
    return {
        "sampled_events": int(len(x)),
        "sampled_chunk_starts": starts,
        "x_observed_min_max": [int(x.min()), int(x.max())],
        "y_observed_min_max": [int(y.min()), int(y.max())],
        "polarity_observed_values": sorted(int(value) for value in np.unique(polarity)),
        "timestamps_monotonic_within_sampled_chunks": timestamp_monotonic,
    }


def audit(root: Path, sequences: list[str]) -> dict[str, Any]:
    media_path = root / "data/train.parquet"
    columns = [
        "sample_token",
        "split",
        "sequence_id",
        "events_path",
        "rgb_exposure_start_timestamp_us",
        "rgb_exposure_end_timestamp_us",
        "K_event",
        "T_event_ego",
    ]
    media = pq.read_table(media_path, columns=columns).to_pandas()
    if not media["split"].eq("train").all():
        raise ValueError("Audit input contains a non-train row")
    exposure_durations = (
        media["rgb_exposure_end_timestamp_us"] - media["rgb_exposure_start_timestamp_us"]
    ).to_numpy(dtype=np.int64)
    frame_intervals = np.concatenate(
        [
            np.diff(rows["rgb_exposure_end_timestamp_us"].to_numpy(dtype=np.int64))
            for _, rows in media.groupby("sequence_id", sort=True)
        ]
    )
    calibration = {
        "unique_K_event_sha256_float64_le": sorted(
            {_matrix_sha256(value) for value in media["K_event"]}
        ),
        "unique_T_event_ego_sha256_float64_le": sorted(
            {_matrix_sha256(value) for value in media["T_event_ego"]}
        ),
    }
    inventory = []
    detailed = []
    for sequence_id, rows in media.groupby("sequence_id", sort=True):
        relative = str(rows.iloc[0]["events_path"])
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Unsafe or absent event path: {relative}")
        with h5py.File(path, "r") as handle:
            required = {"events/x", "events/y", "events/t", "events/p", "ms_to_idx"}
            missing = sorted(name for name in required if name not in handle)
            if missing:
                raise ValueError(f"{relative} missing {missing}")
            datasets = {
                name: cast(h5py.Dataset, handle[f"events/{name}"]) for name in ("x", "y", "t", "p")
            }
            lengths = {name: len(dataset) for name, dataset in datasets.items()}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"Event arrays differ in length: {relative}")
            timestamps = datasets["t"]
            millisecond_index = cast(h5py.Dataset, handle["ms_to_idx"])
            creation = timestamps.id.get_create_plist()
            compression_filter = (
                creation.get_filter(0)[3].decode("ascii") if creation.get_nfilters() else "none"
            )
            first_us, last_us = int(timestamps[0]), int(timestamps[-1])
            first_frame_end = int(rows["rgb_exposure_end_timestamp_us"].min())
            last_frame_end = int(rows["rgb_exposure_end_timestamp_us"].max())
            inventory.append(
                {
                    "sequence_id": str(sequence_id),
                    "file_size_bytes": path.stat().st_size,
                    "event_count": lengths["t"],
                    "event_first_us": first_us,
                    "event_last_us_inclusive": last_us,
                    "frame_count": len(rows),
                    "first_frame_end_us": first_frame_end,
                    "last_frame_end_us": last_frame_end,
                    "has_100ms_before_first_frame_end": first_us <= first_frame_end - 100_000,
                    # A half-open window ending at frame_end needs coverage only
                    # through frame_end-1 microsecond.
                    "has_events_through_last_frame_end": last_us >= last_frame_end - 1,
                    "ms_to_idx_entries": len(millisecond_index),
                    "compression_filter": compression_filter,
                }
            )
            if sequence_id not in sequences:
                continue
            row = rows.iloc[len(rows) // 2]
            cutoff_us = int(row["rgb_exposure_end_timestamp_us"])
            window_counts = {}
            for duration_ms in (50, 100):
                first = _exact_index(timestamps, millisecond_index, cutoff_us - duration_ms * 1000)
                last = _exact_index(timestamps, millisecond_index, cutoff_us)
                window_counts[f"{duration_ms}ms"] = last - first
            detailed.append(
                {
                    "sequence_id": str(sequence_id),
                    "sample_token": str(row["sample_token"]),
                    "rgb_exposure_start_us": int(row["rgb_exposure_start_timestamp_us"]),
                    "rgb_exposure_end_us": cutoff_us,
                    "event_window_cutoff": (
                        "rgb_exposure_end_timestamp_us; half-open [cutoff-duration, cutoff)"
                    ),
                    "window_event_counts": window_counts,
                    "K_event": np.stack(row["K_event"]).astype(float).tolist(),
                    "T_event_ego": np.stack(row["T_event_ego"]).astype(float).tolist(),
                    "payload_probe": _sample_payload(handle),
                }
            )
    return {
        "schema": "eap_event_data_feasibility_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "audit_script_sha256": _sha256(Path(__file__)),
        "scope": "local public train metadata and bounded HDF5 reads; no labels/test/GPU",
        "data_root": str(root.resolve()),
        "train_metadata_sha256": _sha256(media_path),
        "metadata_row_count": len(media),
        "metadata_columns": columns,
        "rgb_exposure_duration_us_min_median_max": [
            int(exposure_durations.min()),
            float(np.median(exposure_durations)),
            int(exposure_durations.max()),
        ],
        "rgb_frame_end_interval_us_min_median_max": [
            int(frame_intervals.min()),
            float(np.median(frame_intervals)),
            int(frame_intervals.max()),
        ],
        "calibration": calibration,
        "sequence_count": len(inventory),
        "total_hdf5_bytes": sum(row["file_size_bytes"] for row in inventory),
        "total_event_count": sum(row["event_count"] for row in inventory),
        "all_sequences_have_100ms_history": all(
            row["has_100ms_before_first_frame_end"] for row in inventory
        ),
        "all_sequences_cover_last_frame_end": all(
            row["has_events_through_last_frame_end"] for row in inventory
        ),
        "inventory": inventory,
        "detailed_probes": detailed,
        "units_interpretation": (
            "events/t and RGB exposure columns share the integer microsecond clock used by "
            "ms_to_idx[t//1000]; HDF5 has no explicit unit attribute"
        ),
        "arrival_time_available": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=["qGsgzl4Q8B", "2Y46OkK3ff", "Po6ePjWYAz"],
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new audit output path")
    result = audit(args.root, args.sequences)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("sequence_count", "total_event_count")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
