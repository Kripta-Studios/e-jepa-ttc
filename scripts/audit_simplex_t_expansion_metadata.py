"""Audit pinned expansion inputs without labels, scores, raw media, or role reassignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes

PINS = {
    "data_roles/DATA_ROLES.json": (
        "110198392f41dfc7d7d4e6158a91ab233501fcf417d156790dff6585ce77b4e9"
    ),
    "expansion_raw_bindings/RAW_BINDING_MANIFEST.json": (
        "4b7bfab2dc9ff31a6f3c936f487c1456bec1f8fd739d28165acb12e92a9047ba"
    ),
    "expansion_raw_bindings/RAW_BINDINGS.csv": (
        "eedda241de4cd2ce44b73838b66a8efdf04e83d99c63786285dbdbf8adf439bf"
    ),
    "expansion_usability/USABLE_METADATA.parquet": (
        "e8514492952a3abd86e45e3b00c07e398d2e6def5b73062eb5b5f8f13a932064"
    ),
    "expansion_inventory/SELECTED_METADATA.csv": (
        "ab273a087b4b92e48f0f6657407a520dc033ed7b84c6f80efd9dad1a3cc3aedf"
    ),
}
EXPECTED_QUERIES = 27307
INPUT_COLUMNS = [
    "sample_token",
    "sequence_id",
    "track_id",
    "timestamp_us",
    "frame_timestamps_us",
    "event_windows_us",
    "boxes_xyxy",
    "events_path",
]


def audit(root: Path) -> dict[str, Any]:
    """Check all input rows and binding correspondence; do not certify expert availability."""
    for relative, expected in PINS.items():
        with (root / relative).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"pinned expansion interface changed: {relative}")
    roles = json.loads((root / "data_roles/DATA_ROLES.json").read_text())["roles"]
    allowed = set(roles["expansion"])
    forbidden = set(roles["original"] + roles["protected"] + roles["confirmation"])
    if len(allowed) != 22 or allowed & forbidden:
        raise ValueError("expansion role overlap or inventory mismatch")
    metadata = pd.read_parquet(
        root / "expansion_usability/USABLE_METADATA.parquet", columns=INPUT_COLUMNS
    ).sort_values("sample_token")
    bindings = pd.read_csv(
        root / "expansion_raw_bindings/RAW_BINDINGS.csv", dtype={"track_id": str}
    ).sort_values(["sample_token", "window_id"])
    if len(metadata) != EXPECTED_QUERIES or metadata.sample_token.duplicated().any():
        raise ValueError("expansion query inventory mismatch")
    selected = pd.read_csv(
        root / "expansion_inventory/SELECTED_METADATA.csv",
        usecols=pd.Index(
            ["sample_token", "sequence_id", "track_id", "timestamp_us", "events_path"]
        ),
        dtype={"track_id": str},
    ).sort_values("sample_token")
    for column in selected.columns:
        if not np.array_equal(selected[column].to_numpy(), metadata[column].to_numpy()):
            raise ValueError("usable rows differ from prospective input-only selection")
    if set(metadata.sequence_id) != allowed or set(bindings.sequence_id) != allowed:
        raise ValueError("expansion payload differs from authorized roles")
    if (
        len(bindings) != 2 * len(metadata)
        or bindings.duplicated(["sample_token", "window_id"]).any()
    ):
        raise ValueError("incomplete or duplicated window bindings")
    manifest = json.loads((root / "expansion_raw_bindings/RAW_BINDING_MANIFEST.json").read_text())
    if set(manifest["identity"]["sources"]) != allowed:
        raise ValueError("source manifest role mismatch")
    bound = list(bindings.itertuples(index=False))
    for index, row in enumerate(metadata.itertuples(index=False)):
        row = cast(Any, row)
        windows = np.stack(row.event_windows_us)
        frames = np.asarray(row.frame_timestamps_us)
        if windows.shape != (2, 2) or frames.shape != (2,):
            raise ValueError("unsupported expansion clock schema")
        if windows.dtype.kind not in "iu" or frames.dtype.kind not in "iu":
            raise ValueError("noninteger expansion timestamps")
        offsets = frames - windows[:, 1]
        if (
            row.timestamp_us != frames[1]
            or frames[1] <= frames[0]
            or np.any(windows[:, 1] <= windows[:, 0])
            or windows[0, 1] >= windows[1, 1]
            or abs(int(offsets[1] - offsets[0])) > 5
        ):
            raise ValueError("inconsistent expansion clock")
        square = common_square_from_boxes(row.boxes_xyxy, (0, 1))
        for window_id in range(2):
            item = cast(Any, bound[2 * index + window_id])
            if (
                item.sample_token != row.sample_token
                or item.window_id != window_id
                or item.sequence_id != row.sequence_id
                or item.track_id != str(row.track_id)
                or (item.window_start_us, item.window_end_us) != tuple(windows[window_id])
                or item.target_anchor_event_clock_us != windows[1, 1]
                or item.frame_to_event_clock_offset_us != offsets[window_id]
                or item.events_path_relative != f"{row.sequence_id}/events.h5"
                or item.h5_file_sha256 != manifest["identity"]["sources"][row.sequence_id]["sha256"]
                or not np.array_equal(square, [item.roi_x0, item.roi_y0, item.roi_x1, item.roi_y1])
                or item.stop_event_index - item.start_event_index != item.full_window_event_count
                or item.full_window_event_count < 0
                or item.time_unit != "us"
            ):
                raise ValueError(f"binding mismatch: {row.sample_token}:{window_id}")
    source_stats = {}
    for sequence, reference in manifest["identity"]["sources"].items():
        # Stat only: the owner's full-file hash is attributed, not claimed as repeated here.
        stamp = Path(reference["path"]).stat()
        source_stats[sequence] = {
            "bytes": stamp.st_size,
            "matches_owner_stat": (stamp.st_size, stamp.st_mtime_ns)
            == (reference["bytes"], reference["mtime_ns"]),
            "owner_sha256": reference["sha256"],
        }
    return {
        "status": "EXPANSION_INPUT_BINDINGS_VERIFIED_NOT_CACHE_OR_SCIENTIFIC_FREEZE",
        "pins": PINS,
        "queries": len(metadata),
        "windows": len(bindings),
        "queries_per_sequence": metadata.groupby("sequence_id").size().to_dict(),
        "source_stats": source_stats,
        "metadata_columns_read": INPUT_COLUMNS,
        "labels_read": False,
        "scores_read": False,
        "raw_media_read": False,
        "optimizer_updates": 0,
        "prospective_selected_rows_equal_usable_rows": True,
        "upstream_usability_code_checks_TTC": True,
        "claim_limit": "No selected rows were removed; upstream release TTC filtering persists.",
        "remaining_checks": [
            "owner acknowledgment of expansion clock/ROI provenance and eligibility filters",
            "ROI exposure availability and frozen historical third-window preprocessing",
            "whole-family A5/C2F/PAIR ancestor exclusions for each D1 fold",
            "raw source support of retrospective windows and additional exclusive replay",
        ],
        "context_semantics": "RETROSPECTIVE_CURRENT_QUERY_ROI_NOT_OBJECT_TRACKING",
    }


def main() -> int:
    """Write a new audit receipt only after every metadata check passes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage70-artifacts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.stage70_artifacts)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "queries": result["queries"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
