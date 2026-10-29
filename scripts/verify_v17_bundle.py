"""Independent standard-library ZIP/CRC/SHA verifier for the V17 results bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


def verify(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members")
        if archive.testzip() is not None:
            raise ValueError("CRC failure")
        manifest = json.loads(archive.read("MANIFEST.json"))
        expected = {row["path"] for row in manifest["files"]}
        if len(expected) != len(manifest["files"]) or set(names) != expected | {"MANIFEST.json"}:
            raise ValueError("Manifest coverage mismatch")
        for row in manifest["files"]:
            name = row["path"]
            parts = PurePosixPath(name)
            if parts.is_absolute() or ".." in parts.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe member path")
            digest = hashlib.sha256()
            size = 0
            with archive.open(name) as stream:
                while block := stream.read(1024**2):
                    digest.update(block)
                    size += len(block)
            if size != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise ValueError("SHA/size mismatch: " + name)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.exists() and sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise ValueError("Outer SHA-256 mismatch")
    result = {
        "status": "PASS",
        "path": str(path.resolve()),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "manifest_files": len(expected),
        "members": len(names),
        "crc": "PASS",
        "all_file_hashes": "PASS",
    }
    path.with_suffix(path.suffix + ".verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    print(json.dumps(verify(parser.parse_args().zip), ensure_ascii=False))
