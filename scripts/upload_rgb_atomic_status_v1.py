"""Retry bounded Windows sharing conflicts while retaining the frozen upload protocol."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

V2_SHA256 = "f0d0ad61f2da1cb8a9ef049f4e53f419a529d5d410388437323b612475db4677"
DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 1.0)


def retry_writer(writer: Any, sleeper: Any = time.sleep) -> Any:
    """Retry only Windows access/share violations, never silently skip a receipt."""
    def write(path: Path, value: dict[str, Any]) -> None:
        for attempt in range(len(DELAYS) + 1):
            try:
                writer(path, value)
                return
            except PermissionError as error:
                if getattr(error, "winerror", None) not in {5, 32, 33} or attempt == len(DELAYS):
                    raise
                sleeper(DELAYS[attempt])
    return write


def main() -> int:
    source = Path(__file__).resolve().with_name("upload_rgb_sparse_frames_v2.py")
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != V2_SHA256:
            raise ValueError("Frozen uploader V2 bytes differ")
    spec = importlib.util.spec_from_file_location("ejepa_frozen_rgb_upload_v2", source)
    if spec is None or spec.loader is None:
        raise ImportError(str(source))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.atomic_json = retry_writer(module.atomic_json)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
