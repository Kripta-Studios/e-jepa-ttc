"""Atomic Windows publication with bounded retries for transient reader file locks."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path


def replace(source: str | Path, destination: str | Path) -> None:
    """Retry only sharing/access locks; retain pending bytes on permanent failure."""
    started = time.monotonic()
    delay = 0.01
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if (
                getattr(exc, "winerror", None) not in (5, 32, 33)
                or time.monotonic() - started >= 10
            ):
                raise
            time.sleep(delay)
            delay = min(0.25, delay * 2)


def atomic_json(path: Path, value: dict) -> None:
    """Flush complete JSON bytes before replacing the previous durable record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".pending", dir=path.parent)
    with os.fdopen(descriptor, "wb") as file:
        file.write(raw)
        file.flush()
        os.fsync(file.fileno())
    replace(name, path)
