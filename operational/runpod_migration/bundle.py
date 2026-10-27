"""Pack and verify explicit files, with no network or cloud API side effects."""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from pathlib import Path, PurePosixPath

from .inventory import digest, read


def safe_member(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise ValueError(f"Unsafe bundle member: {name}")
    return str(path)


def pack(inventory: Path, profile: str, output: Path) -> dict:
    """Stream exact files into a tar; verify bytes actually written, not a prior stat."""
    source = read(inventory)
    if profile not in source["profiles"]:
        raise ValueError("Unknown profile")
    if output.exists():
        raise FileExistsError(output)
    selected = [dict(x) for x in source["files"] if profile in x["profiles"]]
    target = output.with_suffix(output.suffix + ".partial")
    if target.exists():
        raise FileExistsError(target)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(target, "w") as archive:
            for item in selected:
                path = Path(item["source"])
                before = path.stat()
                actual = digest(path)
                if before.st_size != item["bytes"] or (
                    item["sha256"] is not None and actual != item["sha256"]
                ):
                    raise ValueError(f"Source changed: {path}")
                info = archive.gettarinfo(str(path), arcname=safe_member(item["relative"]))
                if not info.isfile():
                    raise ValueError("Bundle accepts regular files only")
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f"Source changed during copy: {path}")
                item["sha256"] = actual
                item["hash_status"] = "VERIFIED_AT_PACK"
            manifest = {
                **source,
                "profile": profile,
                "files": selected,
                "inventory_sha256": digest(inventory),
                "execution_ready": False,
            }
            payload = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
            info = tarfile.TarInfo("BUNDLE_MANIFEST.json")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        # Read back the tar stream: guarantees the copied bytes satisfy the manifest.
        import hashlib

        with tarfile.open(target, "r") as archive:
            for item in selected:
                stream = archive.extractfile(item["relative"])
                checksum = hashlib.sha256()
                if stream is not None:
                    with stream:
                        while block := stream.read(8 * 1024**2):
                            checksum.update(block)
                if stream is None or checksum.hexdigest() != item["sha256"]:
                    raise ValueError(f"Packed file digest differs: {item['relative']}")
        os.replace(target, output)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    receipt = {
        "archive": str(output.resolve()),
        "sha256": digest(output),
        "bytes": output.stat().st_size,
        "profile": profile,
        "files": len(selected),
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def verify(root: Path) -> dict:
    """Reject missing, escaped, symlinked, truncated or altered remote bundle members."""
    root = root.resolve(strict=True)
    manifest = read(root / "BUNDLE_MANIFEST.json")
    seen = set()
    for item in manifest["files"]:
        relative = safe_member(item["relative"])
        if relative in seen:
            raise ValueError("Duplicate bundle member")
        seen.add(relative)
        path = root / relative
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("Bundle member escapes root")
        if (
            path.is_symlink()
            or path.stat().st_size != item["bytes"]
            or digest(path) != item["sha256"]
        ):
            raise ValueError(f"Bundle member changed: {relative}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pack_parser = sub.add_parser("pack")
    pack_parser.add_argument("--inventory", type=Path, required=True)
    pack_parser.add_argument(
        "--profile", choices=("rgb_pilot", "all_producers", "rotations"), required=True
    )
    pack_parser.add_argument("--output", type=Path, required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = (
        pack(args.inventory, args.profile, args.output)
        if args.command == "pack"
        else {"status": "VERIFIED", "profile": verify(args.root)["profile"]}
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
