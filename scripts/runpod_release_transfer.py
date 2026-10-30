"""Unpack, download and verify the sealed transfer kit on the future Pod."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_path(root: Path, name: str) -> Path:
    """Reject traversal, absolute names and symlinks escaping the destination."""
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe target: {name}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Target escapes root: {name}")
    return path


def selected(row: dict[str, Any], profile: str) -> bool:
    """Training never includes public test media or test labels."""
    if profile == "all":
        return True
    if profile.startswith("events-"):
        return row.get("modality") == "events" and row.get("role") == profile[7:]
    if profile == "phase-inputs":
        return "phase-inputs" in row.get("profiles", [])
    return "rgb" in row.get("profiles", ["rgb"] if row.get("modality") != "events" else [])


def verify_file(path: Path, row: dict[str, Any]) -> None:
    if not path.is_file() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
        raise ValueError(f"Missing or altered payload: {path}")


def fetch_one(root: Path, row: dict[str, Any]) -> None:
    """Resume a public file only when the server acknowledges the exact range."""
    target = safe_path(root, row["target"])
    if target.exists():
        verify_file(target, row)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if row.get("requires_hf_token"):
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError("Set HF_TOKEN in the Pod environment for the gated teacher")
        executable = Path(sys.executable).with_name("hf.exe" if os.name == "nt" else "hf")
        if not executable.is_file():
            raise RuntimeError("Use the prepared PHASE environment, which includes the hf CLI")
        stage = target.parent / ".hf_download_stage"
        subprocess.run(
            [
                str(executable),
                "download",
                row["repository"],
                "--revision",
                row["revision"],
                "--include",
                row["repository_path"],
                "--local-dir",
                str(stage),
            ],
            check=True,
        )
        downloaded = safe_path(stage, row["repository_path"])
        verify_file(downloaded, row)
        downloaded.replace(target)
        return
    partial = target.with_name(target.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > row["bytes"]:
        raise ValueError(f"Oversized partial: {partial}")
    if offset < row["bytes"]:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(row["url"], headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status == 206:
                match = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                )
                if not match or int(match[1]) != offset or int(match[3]) != row["bytes"]:
                    raise ValueError("Download range does not match the pinned object")
                mode = "ab"
            elif response.status == 200:
                mode = "wb"
            else:
                raise ValueError(f"Unexpected HTTP status: {response.status}")
            with partial.open(mode) as stream:
                while chunk := response.read(8 * 2**20):
                    stream.write(chunk)
    verify_file(partial, row)
    partial.replace(target)
    print(json.dumps({"downloaded": row["target"], "bytes": row["bytes"]}), flush=True)


def unpack(
    root: Path,
    incoming: Path,
    manifest: dict[str, Any],
    profile: str = "all",
    remove_verified: bool = False,
) -> None:
    """Verify each archive and allow only regular manifest-listed files."""
    expected = {
        row["target"]: row
        for row in manifest["files"]
        if row["kind"] == "private" and selected(row, profile)
    }
    seen = set()
    for part in json.loads((incoming / "PARTS.json").read_text()):
        if profile not in part.get("profiles", ["rgb", "all"]):
            continue
        archive_path = safe_path(incoming, part["name"])
        verify_file(archive_path, part)
        with tarfile.open(archive_path, "r:") as archive:
            for member in archive:
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError(f"Unexpected archive member: {member.name}")
                row = expected[member.name]
                if member.size != row["bytes"]:
                    raise ValueError(f"Archive member size differs: {member.name}")
                target = safe_path(root, member.name)
                if target.exists():
                    verify_file(target, row)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    payload = archive.extractfile(member)
                    if payload is None:
                        raise ValueError("Missing regular-file payload")
                    temporary = target.with_name(target.name + ".unpack-part")
                    with payload, temporary.open("wb") as stream:
                        while chunk := payload.read(8 * 2**20):
                            stream.write(chunk)
                    verify_file(temporary, row)
                    temporary.replace(target)
                seen.add(member.name)
        print(json.dumps({"unpacked": part["name"]}), flush=True)
        if remove_verified:
            archive_path.unlink()  # Only the verified uploaded duplicate; payloads remain intact.
    if seen != set(expected):
        raise ValueError("Private payload population is incomplete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("unpack", "fetch", "verify"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--incoming", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument(
        "--profile",
        choices=(
            "rgb",
            "phase-inputs",
            "events-train",
            "events-selection",
            "events-calibration",
            "all",
        ),
        default="rgb",
    )
    parser.add_argument("--remove-verified-archives", action="store_true")
    parser.add_argument("--reserve-gib", type=int, default=50)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args()
    path = args.incoming / "MANIFEST.json"
    if digest(path) != args.manifest_sha256:
        raise ValueError("Manifest SHA-256 differs from the local sealed receipt")
    manifest = json.loads(path.read_text())
    rows = [row for row in manifest["files"] if selected(row, args.profile)]
    if args.mode == "unpack":
        unpack(args.root, args.incoming, manifest, args.profile, args.remove_verified_archives)
    elif args.mode == "fetch":
        public = [row for row in rows if row["kind"] == "public"]
        if args.reserve_gib < 20:
            raise ValueError("Keep at least 20 GiB reserved for runtime and outputs")
        args.root.mkdir(parents=True, exist_ok=True)
        needed = sum(
            row["bytes"] for row in public if not safe_path(args.root, row["target"]).is_file()
        )
        if needed + args.reserve_gib * 2**30 > shutil.disk_usage(args.root).free:
            raise RuntimeError("Insufficient disk with reserve: use a smaller event-role profile")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda row: fetch_one(args.root, row), public))
    else:
        for row in rows:
            verify_file(safe_path(args.root, row["target"]), row)
        receipt = {
            "status": "ALL_SELECTED_PAYLOAD_HASHES_VERIFIED",
            "profile": args.profile,
            "files": len(rows),
            "manifest_sha256": args.manifest_sha256,
        }
        (args.root / f"VERIFIED_{args.profile}.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt))


if __name__ == "__main__":
    main()
