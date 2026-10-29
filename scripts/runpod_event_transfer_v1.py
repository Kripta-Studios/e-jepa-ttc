"""Resume manifest-bound event files over one authenticated SSH connection."""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

REMOTE_RECEIVER = r"""
import hashlib, json, os, pathlib, sys, tempfile

root = pathlib.Path(sys.argv[1]).resolve(strict=True)

def reply(value):
    sys.stdout.buffer.write((json.dumps(value, separators=(",", ":")) + "\n").encode())
    sys.stdout.buffer.flush()

def fsync_dir(path):
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

while True:
    line = sys.stdin.buffer.readline()
    if not line:
        raise EOFError("transfer sender disconnected before END")
    header = json.loads(line)
    if header.get("command") == "END":
        reply({"status": "COMPLETE"})
        break
    relative = pathlib.PurePosixPath(header["relative"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("unsafe relative path")
    size, expected = int(header["bytes"]), str(header["sha256"])
    if size < 0 or len(expected) != 64:
        raise ValueError("invalid transfer identity")
    target = root.joinpath(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.parent.resolve().is_relative_to(root):
        raise ValueError("target escapes root")
    seal = target.with_name(target.name + ".transfer.json")
    if target.is_file() and seal.is_file():
        value = json.loads(seal.read_text(encoding="utf-8"))
        if (
            value.get("bytes") == size
            and value.get("sha256") == expected
            and target.stat().st_size == size
            and target.stat().st_mode & 0o222 == 0
        ):
            reply({"status": "SKIP", "relative": str(relative), "bytes": size})
            continue
    partial = target.with_name("." + target.name + ".partial")
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset > size:
        partial.unlink()
        offset = 0
    reply({"status": "OFFSET", "relative": str(relative), "offset": offset})
    checksum = hashlib.sha256()
    if offset:
        with partial.open("rb") as existing:
            while block := existing.read(8 * 1024**2):
                checksum.update(block)
    remaining = size - offset
    with partial.open("ab") as output:
        while remaining:
            block = sys.stdin.buffer.read(min(8 * 1024**2, remaining))
            if not block:
                raise EOFError("transfer sender disconnected inside a file")
            output.write(block)
            checksum.update(block)
            remaining -= len(block)
        output.flush()
        os.fsync(output.fileno())
    if partial.stat().st_size != size or checksum.hexdigest() != expected:
        raise ValueError("received file differs from manifest")
    partial.chmod(0o444)
    os.replace(partial, target)
    payload = {"schema": "ejepa_ssh_transfer_seal_v1", "bytes": size, "sha256": expected}
    descriptor, temporary_name = tempfile.mkstemp(prefix="." + seal.name, dir=seal.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_name, seal)
    seal.chmod(0o444)
    fsync_dir(target.parent)
    reply({"status": "OK", "relative": str(relative), "bytes": size})
"""


def _safe_relative(value: str) -> str:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe manifest path: {value}")
    return str(relative)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Manifest must be a JSON object")
    return value


def transfer(
    manifest_path: Path,
    host: str,
    user: str,
    port: int,
    identity_file: Path,
    remote_root: str,
) -> dict:
    manifest = _read(manifest_path.resolve(strict=True))
    files = list(manifest.get("files", []))
    if not files or manifest.get("all_files_content_bound") is not True:
        raise ValueError("Transfer requires a nonempty, fully content-bound manifest")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", host) or host.startswith("-"):
        raise ValueError("Invalid SSH host")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", user) or user.startswith("-"):
        raise ValueError("Invalid SSH user")
    if not 1 <= port <= 65535:
        raise ValueError("Invalid SSH port")
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", remote_root) or ".." in remote_root.split("/"):
        raise ValueError("Invalid remote root")
    encoded = base64.b64encode(REMOTE_RECEIVER.encode()).decode()
    remote = f"python3 -u -c \"import base64;exec(base64.b64decode('{encoded}'))\" {remote_root}"
    command = [
        "ssh",
        "-T",
        "-p",
        str(port),
        "-i",
        str(identity_file.resolve(strict=True)),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        f"{user}@{host}",
        remote,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("SSH transfer pipes are unavailable")
    sent = skipped = 0
    try:
        for number, item in enumerate(files, start=1):
            source = Path(str(item["source"])).resolve(strict=True)
            relative = _safe_relative(str(item["relative"]))
            size, sha256 = int(item["bytes"]), str(item["sha256"])
            before = source.stat()
            if not source.is_file() or source.is_symlink() or before.st_size != size:
                raise ValueError(f"Local source differs: {source}")
            header = {"relative": relative, "bytes": size, "sha256": sha256}
            process.stdin.write((json.dumps(header, separators=(",", ":")) + "\n").encode())
            process.stdin.flush()
            response = json.loads(process.stdout.readline())
            if response.get("status") == "SKIP":
                skipped += size
                continue
            if response.get("status") != "OFFSET":
                raise RuntimeError(f"Unexpected receiver response: {response}")
            offset = int(response["offset"])
            if not 0 <= offset <= size:
                raise ValueError("Receiver returned an invalid offset")
            with source.open("rb") as stream:
                stream.seek(offset)
                while block := stream.read(8 * 1024**2):
                    process.stdin.write(block)
            process.stdin.flush()
            response = json.loads(process.stdout.readline())
            if response.get("status") != "OK" or response.get("relative") != relative:
                raise RuntimeError(f"Receiver rejected {relative}: {response}")
            sent += size - offset
            if number % 100 == 0:
                print(json.dumps({"files_done": number, "files_total": len(files), "sent": sent}))
        process.stdin.write(b'{"command":"END"}\n')
        process.stdin.flush()
        terminal = json.loads(process.stdout.readline())
        process.stdin.close()
        code = process.wait()
        if code != 0 or terminal.get("status") != "COMPLETE":
            raise RuntimeError(f"SSH receiver failed: exit={code}, terminal={terminal}")
    except BaseException:
        process.kill()
        process.wait()
        raise
    return {
        "status": "COMPLETE",
        "manifest": str(manifest_path.resolve()),
        "files": len(files),
        "bytes_sent_this_call": sent,
        "bytes_reused": skipped,
        "remote_root": remote_root,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--remote-root", default="/workspace/ejepa_event_bundle")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        manifest = _read(args.manifest.resolve(strict=True))
        print(
            json.dumps(
                {
                    "status": "DRY_RUN",
                    "files": manifest.get("file_count"),
                    "bytes": manifest.get("total_bytes"),
                    "remote_root": args.remote_root,
                },
                indent=2,
            )
        )
        return 0
    print(
        json.dumps(
            transfer(
                args.manifest,
                args.host,
                args.user,
                args.port,
                args.identity_file,
                args.remote_root,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
