"""Restricted original-release decoding and target-independent observation fields."""

from __future__ import annotations

import io
import pickle
import pickletools
from pathlib import Path
from typing import Any, NoReturn


class DataOnlyUnpickler(pickle.Unpickler):
    """Refuse Python global loading, object construction and persistent references."""

    def find_class(self, module: str, name: str) -> NoReturn:
        raise ValueError("pickle globals are not allowed")

    def persistent_load(self, pid: object) -> NoReturn:
        raise ValueError("persistent pickle references are not allowed")


def read_records(path: Path) -> list[dict[str, Any]]:
    """Decode bounded primitive-only pickle; callers must project allowed fields."""
    if path.stat().st_size > 32 * 1024**2:
        raise ValueError("original annotation exceeds bounded metadata allocation")
    data = path.read_bytes()
    forbidden = {
        "GLOBAL",
        "STACK_GLOBAL",
        "REDUCE",
        "BUILD",
        "OBJ",
        "INST",
        "NEWOBJ",
        "NEWOBJ_EX",
        "EXT1",
        "EXT2",
        "EXT4",
        "PERSID",
        "BINPERSID",
    }
    for opcode, _, _ in pickletools.genops(data):
        if opcode.name in forbidden:
            raise ValueError(f"non-data pickle opcode: {opcode.name}")
    records = DataOnlyUnpickler(io.BytesIO(data)).load()
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("expected original release list of record dictionaries")
    return records


def input_observation(row: dict[str, Any]) -> dict[str, Any]:
    """Whitelist original xywh/identity/exposure fields; never inspect target fields."""
    x, y, width, height = row["bbox"]
    return {
        "sequence_id": str(row["sequence_id"]),
        "instance_id": str(row["instance_id"]),
        "file_name": str(row["file_name"]),
        "exposure_start_us": int(row["rgb_exposure_start_timestamp_us"]),
        "exposure_end_us": int(row["rgb_exposure_end_timestamp_us"]),
        "boxes_xyxy": [float(x), float(y), float(x + width), float(y + height)],
    }
