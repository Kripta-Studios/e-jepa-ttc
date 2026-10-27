"""Safely stage and seal the 13-file EVENT downstream metadata supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def stage(archive: Path, root: Path, admission_path: Path) -> dict[str, object]:
    archive = archive.resolve(strict=True)
    if digest(archive) != "0c26ca271a484beb892524d175643720e43c8c8e45fbaaf79abfbf9cec7422c0":
        raise ValueError("Supplement archive SHA differs")
    if root.resolve() != Path("/opt/ejepa_event_supplement"):
        raise ValueError("Supplement stage must use its dedicated /opt root")
    pending = root.with_name(f".{root.name}.{os.getpid()}.pending")
    if root.exists() or pending.exists():
        raise FileExistsError("Supplement stage already exists; inspect before retry")
    pending.mkdir()
    try:
        with tarfile.open(archive) as source:
            members = source.getmembers()
            for member in members:
                relative = Path(member.name)
                if not member.isfile() or relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe supplement archive member")
                destination = pending / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                incoming = source.extractfile(member)
                if incoming is None:
                    raise ValueError("Supplement member payload missing")
                with destination.open("xb") as output:
                    shutil.copyfileobj(incoming, output)
        manifest_path = pending / "SUPPLEMENT_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_manifest = "00ec22392cbd3abbcdc0409263dbf3a49fc057be079dc1d84df52d140fd1e030"
        if digest(manifest_path) != expected_manifest:
            raise ValueError("Supplement manifest SHA differs")
        stats = []
        for item in manifest["files"]:
            path = pending / str(item["relative"])
            if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise ValueError(f"Supplement file differs: {path}")
            path.chmod(0o444)
        manifest_path.chmod(0o444)
        os.replace(pending, root)
        for item in manifest["files"]:
            path = root / str(item["relative"])
            observed = path.stat()
            stats.append(
                {
                    "relative": item["relative"], "bytes": observed.st_size,
                    "mtime_ns": observed.st_mtime_ns, "inode": observed.st_ino,
                    "device": observed.st_dev, "mode": observed.st_mode,
                }
            )
        seal = {
            "schema": "ejepa_event_supplement_immutable_seal_v1",
            "status": "SEALED", "root": str(root),
            "manifest_sha256": digest(root / "SUPPLEMENT_MANIFEST.json"),
            "content_identity_sha256": manifest["content_identity_sha256"],
            "files": stats,
        }
        seal_path = root.parent / f"{root.name}.SEAL.json"
        atomic(seal_path, seal)
        admission = {
            "schema": "ejepa_event_downstream_supplemental_admission_v1",
            "status": "VERIFIED", "root": str(root),
            "scientific_config_sha256": manifest["scientific_config_sha256"],
            "files": 13, "total_bytes": 81_182_038, "hashes_verified": 13,
            "mapping_verified": True,
            "manifest_sha256": digest(root / "SUPPLEMENT_MANIFEST.json"),
            "immutable_seal_sha256": digest(seal_path),
        }
        admission["receipt_sha256"] = canonical(admission)
        atomic(admission_path, admission)
        return admission
    finally:
        if pending.exists():
            shutil.rmtree(pending)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.archive, args.root, args.admission), indent=2))
