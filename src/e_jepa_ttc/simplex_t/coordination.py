"""Validate the acknowledged interface bytes without opening referenced datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def verified_ack(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Bind the explicit user-approved ACK to request, roles, time and ancestry."""

    def read_bound(record: dict[str, Any]) -> bytes:
        target = Path(record["path"])
        if target.stat().st_size > 1024 * 1024:
            raise ValueError("interface exceeds metadata bound")
        data = target.read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"acknowledged interface changed: {target}")
        return data

    ack = json.loads(read_bound({"path": str(path), "sha256": expected_sha256}))
    if ack["request_id"] != "SIMPLEX_T_2026-09-06_T0":
        raise ValueError("unexpected coordination request")
    read_bound(ack["request"])
    for key in ("role_manifest", "time_charter"):
        record = ack["interfaces"][key]
        if record["authoritative"] is not True or record["read_only_import_authorized"] is not True:
            raise ValueError("interface import not authorized")
        read_bound(record)
    read_bound(ack["producers"]["authoritative_historical_manifest"])
    read_bound(ack["resources"]["resource_amendment"])
    return ack


def shared_write_admission(free_bytes: int, reservations_bytes: int | None) -> bool:
    """Unknown outstanding reservations cannot be treated as zero."""
    if reservations_bytes is None:
        return False
    if min(free_bytes, reservations_bytes) < 0:
        raise ValueError("negative resource accounting")
    return free_bytes - reservations_bytes >= 20_000_000_000
