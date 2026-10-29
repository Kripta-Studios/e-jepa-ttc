"""Archive explicit small Pod1 operational evidence without following symlinks."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tarfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "ejepa_pod1_operational_supplement_v1"
MAX_FILE_BYTES = 64 * 1024 * 1024
TOP_LEVEL_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".py", ".sh"}
SOURCE_SUFFIXES = {
    ".cff",
    ".cfg",
    ".ini",
    ".json",
    ".lock",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SOURCE_NAMES = {"LICENSE", "Makefile", "Dockerfile", ".gitignore"}
EXCLUDED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "checkpoints",
    "data",
    "drives",
    "rgb_shards",
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binding(value: str) -> tuple[PurePosixPath, Path]:
    archive, separator, source = value.partition("=")
    relative = PurePosixPath(archive)
    if (
        not separator
        or not archive
        or relative.is_absolute()
        or ".." in relative.parts
        or not source
    ):
        raise ValueError("Bindings use SAFE/ARCHIVE/PATH=/absolute/source")
    return relative, Path(source)


def plain_file(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f"Supplement refuses symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"Supplement requires a regular file: {path}")
    size = resolved.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(f"Supplement file exceeds {MAX_FILE_BYTES} bytes: {path}")
    return resolved


def collect(files: list[str], top_levels: list[str], trees: list[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}

    def add(relative: PurePosixPath, path: Path) -> None:
        name = relative.as_posix()
        source = plain_file(path)
        if name == "MANIFEST.json" or name in records:
            raise ValueError(f"Duplicate/reserved supplement path: {name}")
        records[name] = {
            "source_path": str(source),
            "bytes": source.stat().st_size,
            "sha256": digest(source),
        }

    for raw in files:
        relative, source = binding(raw)
        add(relative, source)
    for raw in top_levels:
        prefix, root = binding(raw)
        if root.is_symlink():
            raise ValueError(f"Supplement refuses root symlink: {root}")
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"Supplement top-level root is not a directory: {root}")
        for path in sorted(root.iterdir()):
            if (
                path.is_file()
                and not path.is_symlink()
                and path.suffix.lower() in TOP_LEVEL_SUFFIXES
                and not path.name.startswith(".")
                and not path.name.endswith(".pending")
            ):
                add(prefix / path.name, path)
    for raw in trees:
        prefix, root = binding(raw)
        root = root.resolve(strict=True)
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"Supplement source tree is unsafe: {root}")
        for directory, directories, names in os.walk(root, followlinks=False):
            directories[:] = sorted(
                name
                for name in directories
                if name not in EXCLUDED_DIRECTORIES and not (Path(directory) / name).is_symlink()
            )
            base = Path(directory)
            for name in sorted(names):
                path = base / name
                if (
                    not path.is_symlink()
                    and not name.startswith(".")
                    and not name.endswith(".pending")
                    and (path.suffix.lower() in SOURCE_SUFFIXES or name in SOURCE_NAMES)
                ):
                    add(prefix / PurePosixPath(path.relative_to(root).as_posix()), path)
    if not records:
        raise ValueError("Supplement selection is empty")
    return records


def publish(
    records: dict[str, dict[str, Any]], output: Path, primary_archive_sha256: str
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", primary_archive_sha256):
        raise ValueError("Primary archive SHA-256 is invalid")
    output = output.resolve()
    receipt_path = output.with_name(output.name + ".json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError("Supplement archive/receipt already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": SCHEMA,
        "status": "READY",
        "created_utc": datetime.now(UTC).isoformat(),
        "primary_archive_sha256": primary_archive_sha256,
        "selection_is_explicit_files_top_levels_and_source_trees": True,
        "recursive_directory_capture": True,
        "recursive_capture_limited_to_source_documents": True,
        "excluded_directories": sorted(EXCLUDED_DIRECTORIES),
        "symlinks_followed": False,
        "large_payloads_included": False,
        "global_atomic_snapshot_claimed": False,
        "per_file_stability_verified_before_and_after_tar": True,
        "remote_deletion_performed": False,
        "files": records,
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    temporary = output.with_name(f".{output.name}.{os.getpid()}.pending")
    try:
        with tarfile.open(temporary, "w") as archive:
            info = tarfile.TarInfo("MANIFEST.json")
            info.size = len(encoded)
            info.mode = 0o444
            archive.addfile(info, io.BytesIO(encoded))
            for relative, identity in sorted(records.items()):
                source = Path(identity["source_path"])
                archive.add(source, arcname=relative, recursive=False)
        for relative, identity in records.items():
            source = Path(identity["source_path"])
            if source.stat().st_size != identity["bytes"] or digest(source) != identity["sha256"]:
                raise ValueError(f"Supplement source changed while archiving: {relative}")
        os.replace(temporary, output)
        receipt = {
            "schema": SCHEMA,
            "status": "READY",
            "archive": str(output),
            "archive_bytes": output.stat().st_size,
            "archive_sha256": digest(output),
            "primary_archive_sha256": primary_archive_sha256,
            "files": len(records),
            "remote_deletion_performed": False,
        }
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return receipt
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", action="append", default=[], metavar="ARCHIVE=/ABS")
    parser.add_argument("--top-level", action="append", default=[], metavar="PREFIX=/ABS")
    parser.add_argument("--tree", action="append", default=[], metavar="PREFIX=/ABS")
    parser.add_argument("--primary-archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = publish(
        collect(args.file, args.top_level, args.tree),
        args.output,
        args.primary_archive_sha256,
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
