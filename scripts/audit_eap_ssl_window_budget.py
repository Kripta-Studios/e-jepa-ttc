"""Audit the event allocation bound used by the eAP SSL-Pure reader."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401  # register public eAP compression filters

from e_jepa_ttc.training.eap_jepa import EAPJEPATrainerConfig, EAPOnDemandJEPADataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_window_budget(
    root: Path,
    split_path: Path,
    output_path: Path,
    *,
    chunk_events: int,
) -> dict[str, Any]:
    config = EAPJEPATrainerConfig(max_windows_per_sequence=512, event_chunk_size=chunk_events)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    dataset = EAPOnDemandJEPADataset(root, split["assignments"]["train"], config)
    handles: dict[str, h5py.File] = {}
    maximum = (0, "", 0, 0)
    window_count = 0
    try:
        for sample in dataset.samples:
            handle = handles.get(sample.sequence_id)
            if handle is None:
                handle = h5py.File(root / "data" / "train" / sample.sequence_id / "events.h5", "r")
                handles[sample.sequence_id] = handle
            index = handle["ms_to_idx"]
            max_ms = int(index.shape[0] - 1)
            for end_us in (
                sample.timestamp_us,
                *(sample.timestamp_us + horizon * 1000 for horizon in config.horizons_ms),
            ):
                start_us = end_us - config.event_window_ms * 1000
                first = int(index[min(max_ms, max(0, start_us // 1000))])
                last = int(index[min(max_ms, max(0, math.ceil(end_us / 1000)))])
                count = last - first
                window_count += 1
                if count > maximum[0]:
                    maximum = (count, sample.sequence_id, sample.timestamp_us, end_us)
    finally:
        for handle in handles.values():
            handle.close()
        dataset.close()
    result: dict[str, Any] = {
        "artifact_type": "eap_ssl_window_budget_audit_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "root": root.as_posix(),
        "split": split_path.as_posix(),
        "split_sha256": _sha256(split_path),
        "status": "pass" if chunk_events > 0 else "fail",
        "train_sample_count": len(dataset),
        "window_count": window_count,
        "event_chunk_size": chunk_events,
        "temporary_array_bytes_upper_bound": chunk_events * (4 + 4 + 8 + 1),
        "maximum_indexed_window_events": maximum[0],
        "maximum_window_sequence_id": maximum[1],
        "maximum_window_reference_us": maximum[2],
        "maximum_window_end_us": maximum[3],
        "chunking_required": maximum[0] > chunk_events,
        "uses_ttc_labels": False,
        "uses_object_bboxes": False,
        "uses_ttc_for_sampling": False,
        "uses_boxes_for_sampling": False,
        "uses_category_for_sampling": False,
        "uses_depth_for_sampling": False,
        "uses_masks_for_sampling": False,
        "uses_3d_for_sampling": False,
        "uses_future_labels_for_sampling": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"E:\eAP_dataset"))
    parser.add_argument("--split", type=Path, default=Path("data/splits/eap_train40_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-events", type=int, default=250_000)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_window_budget(args.root, args.split, args.output, chunk_events=args.chunk_events),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
