from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(
    path: str | Path,
) -> str:
    resolved = Path(path)

    hasher = hashlib.sha256()

    with resolved.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
            hasher.update(chunk)

    return hasher.hexdigest()
