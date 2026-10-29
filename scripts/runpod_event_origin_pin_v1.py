"""Keep the immutable R2/E_A5 migration origin addressable across retention."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: dict[str, object]) -> None:
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protected", type=Path, required=True)
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    args = parser.parse_args()
    protected = args.protected.resolve(strict=True)
    if digest(protected) != args.sha256:
        raise ValueError("Protected R2 origin checkpoint digest differs")
    args.origin.parent.resolve(strict=True)
    repairs = 0
    verified_inode: int | None = None
    while True:
        if args.origin.exists():
            observed = args.origin.stat()
            if not args.origin.is_file() or (
                observed.st_ino != verified_inode and digest(args.origin) != args.sha256
            ):
                raise ValueError("R2 origin path exists with another payload")
            verified_inode = observed.st_ino
        else:
            try:
                os.link(protected, args.origin)
            except FileExistsError:
                continue
            repairs += 1
            verified_inode = protected.stat().st_ino
        value: dict[str, object] = {
            "schema": "ejepa_event_origin_checkpoint_pin_v1",
            "status": "PINNED",
            "protected": str(protected),
            "origin": str(args.origin),
            "sha256": args.sha256,
            "hardlink_same_inode": protected.stat().st_ino == args.origin.stat().st_ino,
            "repairs": repairs,
            "pid": os.getpid(),
            "updated_utc": datetime.now(UTC).isoformat(),
        }
        atomic_json(args.receipt, value)
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
