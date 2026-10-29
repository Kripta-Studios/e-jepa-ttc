# ruff: noqa: E501
"""Incrementally publish verified sparse RGB TARs over four resumable SSH streams."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

GIT_SSH = Path("C:/Program Files/Git/usr/bin/ssh.exe")

RECEIVER = r"""
import hashlib,json,os,pathlib,re,stat,sys,tempfile
root=pathlib.Path(sys.argv[1]).resolve(strict=True);reserve=int(sys.argv[2])
def reply(v):
 sys.stdout.buffer.write((json.dumps(v,separators=(",",":"))+"\n").encode());sys.stdout.buffer.flush()
def syncdir(p):
 fd=os.open(p,os.O_RDONLY)
 try: os.fsync(fd)
 finally: os.close(fd)
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def seal(target,size,expected):
 p=target.with_name(target.name+".transfer.json");fd,n=tempfile.mkstemp(prefix="."+p.name,dir=p.parent)
 with os.fdopen(fd,"w",encoding="utf-8") as f:
  json.dump({"schema":"ejepa_ssh_transfer_seal_v1","bytes":size,"sha256":expected},f,sort_keys=True,separators=(",",":"));f.write("\n");f.flush();os.fsync(f.fileno())
 os.chmod(n,0o444);os.replace(n,p);syncdir(p.parent)
while True:
 line=sys.stdin.buffer.readline()
 if not line:raise EOFError("sender disconnected before END")
 h=json.loads(line)
 if h.get("command")=="END":reply({"status":"COMPLETE"});break
 rel=pathlib.PurePosixPath(h["relative"]);size=int(h["bytes"]);expected=str(h["sha256"])
 if rel.is_absolute() or ".." in rel.parts or not rel.parts or size<0 or not re.fullmatch(r"[0-9a-f]{64}",expected):raise ValueError("unsafe transfer identity")
 target=root.joinpath(*rel.parts);target.parent.mkdir(parents=True,exist_ok=True)
 if not target.parent.resolve().is_relative_to(root):raise ValueError("target escapes root")
 if target.exists():
  s=target.lstat()
  if target.is_symlink() or not stat.S_ISREG(s.st_mode) or s.st_size!=size or sha(target)!=expected:raise ValueError("existing target differs")
  target.chmod(0o444);seal(target,size,expected);reply({"status":"SKIP","relative":str(rel),"bytes":size});continue
 partial=target.with_name("."+target.name+".partial")
 if partial.is_symlink() or (partial.exists() and not partial.is_file()):raise ValueError("unsafe partial")
 offset=partial.stat().st_size if partial.exists() else 0
 if offset>size:partial.unlink();offset=0
 v=os.statvfs(root);free=v.f_bavail*v.f_frsize
 if free<size-offset+reserve:raise OSError("remote reserve would be violated")
 reply({"status":"OFFSET","relative":str(rel),"offset":offset});checksum=hashlib.sha256()
 if offset:
  with partial.open("rb") as f:
   while True:
    b=f.read(8*1024**2)
    if not b:break
    checksum.update(b)
 remaining=size-offset
 with partial.open("ab") as f:
  while remaining:
   b=sys.stdin.buffer.read(min(8*1024**2,remaining))
   if not b:raise EOFError("sender disconnected inside file")
   f.write(b);checksum.update(b);remaining-=len(b)
  f.flush();os.fsync(f.fileno())
 if partial.stat().st_size!=size or checksum.hexdigest()!=expected:raise ValueError("received file differs")
 partial.chmod(0o444);os.replace(partial,target);syncdir(target.parent);seal(target,size,expected);reply({"status":"OK","relative":str(rel),"bytes":size})
"""


@dataclass(frozen=True)
class Ready:
    tar: Path
    marker: Path
    relative: str
    size: int
    sha256: str
    marker_size: int
    marker_sha256: str
    refiner: bool
    plan_sha256: str

    @property
    def total(self) -> int:
        return self.size + self.marker_size


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value: dict[str, Any], omit: str) -> str:
    payload = {key: item for key, item in value.items() if key != omit}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def safe_relative(value: object) -> str:
    relative = PurePosixPath(str(value))
    pattern = r"data/train/[A-Za-z0-9_-]+/rgb_shards/[A-Za-z0-9_.-]+\.tar"
    if relative.is_absolute() or ".." in relative.parts or not re.fullmatch(pattern, str(relative)):
        raise ValueError(f"Unsafe sparse RGB path: {value}")
    return str(relative)


def load_ready(marker: Path, root: Path) -> Ready:
    root = root.resolve(strict=True)
    marker = marker.resolve(strict=True)
    value = json.loads(marker.read_text(encoding="utf-8"))
    relative = safe_relative(value.get("relative"))
    tar = root.joinpath(*PurePosixPath(relative).parts)
    members = value.get("members")
    if marker != tar.with_name(tar.name + ".json") or tar.is_symlink() or not tar.is_file():
        raise ValueError("Sparse TAR/marker path binding differs")
    if not (
        value.get("schema") == "ejepa_rgb_sparse_tar_receipt_v1"
        and value.get("status") == "COMPLETE_BYTE_AND_PIXEL_VERIFIED"
        and value.get("payload_recompressed") is False
        and value.get("decoded_pixel_parity") is True
        and isinstance(members, list)
        and members
        and value.get("receipt_sha256") == canonical(value, "receipt_sha256")
    ):
        raise ValueError("Sparse TAR marker is not complete and verified")
    before = tar.stat()
    observed = digest(tar)
    after = tar.stat()

    def identity(stat_result: os.stat_result) -> tuple[int, int, int, int]:
        return (
            stat_result.st_size,
            stat_result.st_mtime_ns,
            stat_result.st_ino,
            stat_result.st_dev,
        )

    size, expected = int(value["sparse_tar_bytes"]), str(value["sparse_tar_sha256"])
    if identity(before) != identity(after) or size != after.st_size or observed != expected:
        raise ValueError("Sparse TAR changed or differs from verified marker")
    roles = {str(role) for member in members for role in member.get("roles", [])}
    plan_sha256 = str(value.get("plan_sha256"))
    if not re.fullmatch(r"[0-9a-f]{64}", plan_sha256):
        raise ValueError("Sparse TAR marker lacks a valid plan identity")
    return Ready(
        tar,
        marker,
        relative,
        size,
        expected,
        marker.stat().st_size,
        digest(marker),
        "refiner" in roles,
        plan_sha256,
    )


def can_schedule(free: int, reserve: int, inflight: int, candidate: int) -> bool:
    return free - inflight - candidate >= reserve


def ssh_base(host: str, user: str, port: int, identity: Path) -> list[str]:
    executable = Path(os.environ.get("EJEPA_TRANSFER_SSH", str(GIT_SSH))).resolve(strict=True)
    return [
        str(executable),
        "-T",
        "-p",
        str(port),
        "-i",
        str(identity.resolve(strict=True)),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        f"{user}@{host}",
    ]


def remote_free(command: list[str], root: str) -> int:
    code = "import os;v=os.statvfs(" + repr(root) + ");print(v.f_bavail*v.f_frsize)"
    encoded = base64.b64encode(code.encode()).decode()
    result = subprocess.run(
        command + [f"python3 -c \"import base64;exec(base64.b64decode('{encoded}'))\""],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return int(result.stdout.strip())


def transfer(item: Ready, command: list[str], root: str, reserve: int) -> dict[str, int]:
    encoded = base64.b64encode(RECEIVER.encode()).decode()
    remote = f"python3 -u -c \"import base64;exec(base64.b64decode('{encoded}'))\" {root} {reserve}"
    process = subprocess.Popen(command + [remote], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("SSH pipes unavailable")
    sent = reused = 0
    files = (
        (item.tar, item.relative, item.size, item.sha256),
        (item.marker, item.relative + ".json", item.marker_size, item.marker_sha256),
    )
    try:
        for source, relative, size, expected in files:
            header = {"relative": relative, "bytes": size, "sha256": expected}
            process.stdin.write((json.dumps(header, separators=(",", ":")) + "\n").encode())
            process.stdin.flush()
            response = json.loads(process.stdout.readline())
            if response.get("status") == "SKIP":
                reused += size
                continue
            if response.get("status") != "OFFSET":
                raise RuntimeError(f"Receiver rejected header: {response}")
            offset = int(response["offset"])
            if not 0 <= offset <= size:
                raise ValueError("Invalid resume offset")
            with source.open("rb") as stream:
                stream.seek(offset)
                while block := stream.read(8 * 1024**2):
                    process.stdin.write(block)
            process.stdin.flush()
            response = json.loads(process.stdout.readline())
            if response.get("status") != "OK" or response.get("relative") != relative:
                raise RuntimeError(f"Receiver rejected payload: {response}")
            sent += size - offset
        process.stdin.write(b'{"command":"END"}\n')
        process.stdin.flush()
        terminal = json.loads(process.stdout.readline())
        process.stdin.close()
        code = process.wait()
        if code or terminal.get("status") != "COMPLETE":
            raise RuntimeError(f"Receiver failed: exit={code}, terminal={terminal}")
    except BaseException:
        process.kill()
        process.wait()
        raise
    if digest(item.tar) != item.sha256 or digest(item.marker) != item.marker_sha256:
        raise ValueError("Local published inputs changed during transfer")
    return {"bytes_sent": sent, "bytes_reused": reused}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--reserve-bytes", type=int, default=10_000_000_000)
    parser.add_argument("--external-inflight-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--streams", type=int, choices=range(4, 9), default=4)
    parser.add_argument("--require-role", choices=("refiner", "generation"), default="refiner")
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "DRY_RUN", "streams": args.streams}, indent=2))
        return 0
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", args.host) or not 1 <= args.port <= 65535:
        raise ValueError("Invalid SSH endpoint")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.user):
        raise ValueError("Invalid SSH user")
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", args.remote_root):
        raise ValueError("Invalid remote root")
    if not re.fullmatch(r"[0-9a-f]{64}", args.plan_sha256):
        raise ValueError("Plan identity must be a SHA-256")
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        raise ValueError("Deadline must include timezone")
    source = args.source.resolve(strict=True)
    command = ssh_base(args.host, args.user, args.port, args.identity_file)
    old = json.loads(args.status.read_text(encoding="utf-8")) if args.status.exists() else {}
    if old and (
        old.get("source") != str(source)
        or old.get("remote_root") != args.remote_root
        or old.get("plan_sha256") != args.plan_sha256
    ):
        raise ValueError("Prior V2 ledger belongs to another source/destination/plan")
    completed = dict(old.get("completed", {}))
    base = {
        "schema": "ejepa_rgb_sparse_parallel_uploader_v2",
        "source": str(source),
        "remote_root": args.remote_root,
        "streams": args.streams,
        "reserve_bytes": args.reserve_bytes,
        "external_inflight_bytes": args.external_inflight_bytes,
        "required_role": args.require_role,
        "plan_sha256": args.plan_sha256,
        "scientific_execution_authorized_by_upload": False,
    }

    def publish(stage: str, active: dict[Future[dict[str, int]], Ready]) -> None:
        atomic_json(
            args.status,
            {
                **base,
                "stage": stage,
                "utc": datetime.now(UTC).isoformat(),
                "completed": completed,
                "active": [item.relative for item in active.values()],
                "bytes_inflight_upper": args.external_inflight_bytes
                + sum(item.total for item in active.values()),
                "bytes_completed": sum(int(value["bytes"]) for value in completed.values()),
            },
        )

    active: dict[Future[dict[str, int]], Ready] = {}
    known: dict[str, Ready] = {}
    with ThreadPoolExecutor(max_workers=args.streams) as pool:
        while datetime.now(UTC) < deadline:
            for future in list(active):
                if not future.done():
                    continue
                item = active.pop(future)
                result = future.result()
                completed[item.relative] = {
                    "bytes": item.size,
                    "sha256": item.sha256,
                    "marker_sha256": item.marker_sha256,
                    "completed_utc": datetime.now(UTC).isoformat(),
                    **result,
                }
            for marker in source.glob("data/train/*/rgb_shards/*.tar.json"):
                hint = marker.relative_to(source).as_posix()[:-5]
                if hint not in known and hint not in completed:
                    item = load_ready(marker, source)
                    if item.plan_sha256 != args.plan_sha256:
                        raise ValueError(f"READY marker belongs to another plan: {hint}")
                    if (args.require_role == "refiner") != item.refiner:
                        raise ValueError(
                            f"READY marker has the wrong role for this uploader: {hint}"
                        )
                    known[hint] = item
            pending = sorted(
                (
                    item
                    for relative, item in known.items()
                    if relative not in completed and item not in active.values()
                ),
                key=lambda item: item.relative,
            )
            if pending and len(active) < args.streams:
                free = remote_free(command, args.remote_root)
                inflight = args.external_inflight_bytes + sum(
                    item.total for item in active.values()
                )
                while pending and len(active) < args.streams:
                    item = pending.pop(0)
                    if not can_schedule(free, args.reserve_bytes, inflight, item.total):
                        break
                    active[
                        pool.submit(transfer, item, command, args.remote_root, args.reserve_bytes)
                    ] = item
                    inflight += item.total
            publish("TRANSFERRING" if active else "WAITING_READY", active)
            if args.status.with_name("UPLOAD_V2_STOP_REQUEST").exists():
                break
            time.sleep(2)
    publish("STOPPED" if datetime.now(UTC) < deadline else "DEADLINE_EXPIRED", active)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
