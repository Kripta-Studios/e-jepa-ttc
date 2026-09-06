"""Safe essential archive creation with extracted-byte SHA256 and ZIP CRC checks."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, digest


def safe_member(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or not path.parts:
        raise ValueError("unsafe archive member")
    return path.as_posix()


def create_bundle(files: dict[str, Path], output: Path) -> dict:
    """Reject unsafe names and verify every file after extraction, including CRC."""
    members = {safe_member(name): p for name, p in files.items()}
    if len(members) != len(files) or "ESSENTIAL_MANIFEST.json" in members:
        raise ValueError("duplicate archive member")
    if any(p.resolve() == output.resolve() for p in files.values()):
        raise ValueError("archive cannot contain itself")
    manifest = {
        name: {k: v for k, v in binding(p).items() if k != "path"} for name, p in members.items()
    }
    temporary = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in members.items():
            archive.write(path, name)
        archive.writestr("ESSENTIAL_MANIFEST.json", json.dumps(manifest, indent=2))
    with tempfile.TemporaryDirectory(prefix="stage66_bundle_", dir=output.parent) as directory:
        with zipfile.ZipFile(temporary) as archive:
            names = archive.namelist()
            if len(set(names)) != len(names) or archive.testzip() is not None:
                raise ValueError("ZIP CRC/duplicate check failed")
            for name in names:
                safe_member(name)
            archive.extractall(directory)
        for name, record in manifest.items():
            path = Path(directory) / name
            if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                raise ValueError("extracted file hash failed")
    temporary.replace(output)
    sha = digest(output)
    output.with_suffix(".zip.sha256").write_text(sha + "  " + output.name + "\n", encoding="utf-8")
    result = dict(
        zip=binding(output),
        crc_verified=True,
        extracted_hashes_verified=len(manifest),
        members=len(manifest) + 1,
    )
    atomic_json(output.with_suffix(".verification.json"), result)
    return result
