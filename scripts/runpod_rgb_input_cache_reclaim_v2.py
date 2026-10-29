"""Run the frozen RGB cache controller with a one-second response interval."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

V1_SHA256 = "78648b254eab5b075751fb81c2d9b7d2474b87a3f9348ba78e8cbaa9d9bc6b20"
POLL_SECONDS = 1.0


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    source, scripts = Path(__file__).resolve(), Path(__file__).resolve().parent
    v1_path = scripts / "runpod_rgb_input_cache_reclaim_v1.py"
    if digest(v1_path) != V1_SHA256:
        raise ValueError("Frozen RGB cache controller V1 bytes differ")
    sys.path.insert(0, str(scripts))
    import runpod_rgb_input_cache_reclaim_v1 as v1

    if v1.TARGET_HEADROOM_BYTES != 8 * 1024**3:
        raise ValueError("Frozen RGB cache headroom target differs")
    original_poll, original_writer = v1.POLL_SECONDS, v1.atomic_write_json

    def write(path: Path, value: dict[str, Any]) -> None:
        value.update(
            cache_controller_overlay_schema="ejepa_rgb_input_cache_reclaim_v2",
            cache_controller_overlay_source_sha256=digest(source),
            poll_seconds=POLL_SECONDS,
        )
        original_writer(path, value)

    v1.POLL_SECONDS, v1.atomic_write_json = POLL_SECONDS, write
    try:
        return v1.main()
    finally:
        v1.POLL_SECONDS, v1.atomic_write_json = original_poll, original_writer


if __name__ == "__main__":
    raise SystemExit(main())
