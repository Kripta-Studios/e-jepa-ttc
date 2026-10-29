"""Download and fully verify bounded Pod2 artifact archives on local disk."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "artifacts/runpod_event_migration_20261026/resume_20261027"))

RECEIPTS = ROOT / "artifacts/runpod_event_migration_20261026/final_20261027/POD2_RETURN"
DEADLINE = 1793145600.0


def _load_transport_call() -> Callable[..., Any]:
    """Load the session-only transport when a remote operation is requested."""
    try:
        from rpc import call as transport_call
    except ModuleNotFoundError as error:
        if error.name != "rpc":
            raise
        raise RuntimeError(
            "RunPod preservation transport is unavailable: the session module 'rpc' "
            "must be present to run remote preservation operations. Offline archive "
            "verification does not require it."
        ) from error
    return transport_call


def call(code: str, timeout: int = 45) -> Any:
    """Retry pre-execution SSH/DNS failures until the authorized deadline."""
    transport_call = _load_transport_call()
    attempt = 0
    while time.time() < DEADLINE and not (RECEIPTS / "STOP_BACKUP_REQUEST").exists():
        try:
            return transport_call(code, timeout=timeout)
        except RuntimeError as error:
            if not any(
                marker in str(error) for marker in ("banner exchange", "Could not resolve hostname")
            ):
                raise
            attempt += 1
            time.sleep(min(30, 3 * attempt))
    raise RuntimeError("Preservation transport stopped at deadline or explicit stop request")


def publish(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def verify_archive(path: Path, expected: dict) -> dict:
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected["sha256"]:
            raise ValueError("Archive SHA mismatch")
    with tarfile.open(path, "r:*") as archive:
        manifest = json.load(archive.extractfile("PRESERVATION_MANIFEST.json"))
        actual = {
            entry.name: entry
            for entry in archive.getmembers()
            if entry.name != "PRESERVATION_MANIFEST.json"
        }
        if set(actual) != set(manifest["files"]):
            raise ValueError("Archive member coverage mismatch")
        for name, entry in actual.items():
            if not entry.isfile() or name.startswith("/") or ".." in Path(name).parts:
                raise ValueError("Unsafe/non-file archive member")
            bound = manifest["files"][name]
            stream = archive.extractfile(entry)
            if (
                entry.size != bound["bytes"]
                or hashlib.file_digest(stream, "sha256").hexdigest() != bound["sha256"]
            ):
                raise ValueError(f"Member SHA mismatch: {name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--destination", type=Path, default=Path("D:/E_JEPA_POD2_RETURN_20261027"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    payload = base64.b64encode(
        (ROOT / "scripts/runpod_final_preservation_remote.py").read_bytes()
    ).decode()
    call(
        "import pathlib,base64,json,uuid,os;"
        "p=pathlib.Path('/workspace/ejepa_preservation/runpod_final_preservation_remote.py');"
        "data=base64.b64decode(" + repr(payload) + ");compile(data,str(p),'exec');"
        "temp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.pending');"
        "temp.write_bytes(data);os.replace(temp,p);"
        "print('PODSTATUS:'+json.dumps({'uploaded':True}))"
    )
    server = call(
        "import pathlib,json;"
        "paths=pathlib.Path('/root/.local/share/jupyter/runtime').glob('jpserver-*.json');"
        "print('PODSTATUS:'+json.dumps([json.loads(p.read_text()) for p in paths]))"
    )[0]
    session = requests.Session()
    session.headers["Authorization"] = "token " + server["token"]
    for group in args.groups:
        receipt_path = RECEIPTS / f"{group}.json"
        if receipt_path.exists() and not args.refresh:
            prior = json.loads(receipt_path.read_text())
            if prior.get("status") == "LOCAL_VERIFIED" and Path(prior["local_path"]).exists():
                continue
        if time.time() >= DEADLINE:
            raise TimeoutError("Authorized deadline reached")
        start = call(
            """import pathlib,json,subprocess,shutil,psutil
ops=pathlib.Path('/workspace/ejepa_preservation');group=GROUP
receipt=ops/(group+'.json')
command=['/opt/ejepa_env/bin/python',str(ops/'runpod_final_preservation_remote.py'),'archive','--group',group]
existing=json.loads(receipt.read_text()) if receipt.exists() else None
active=[]
for p in psutil.process_iter(['cmdline','status']):
 if p.info['cmdline']==command and p.info['status']!=psutil.STATUS_ZOMBIE:active.append(p.pid)
if len(active)>1:raise RuntimeError('Duplicate preservation workers')
if existing and not REFRESH and pathlib.Path(existing['path']).is_file():
 result={'pid':None,'receipt':existing}
elif active:result={'pid':active[0]}
else:
 if receipt.exists():receipt.unlink()
 with (ops/(group+'.log')).open('a') as log:
  p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 result={'pid':p.pid}
print('PODSTATUS:'+json.dumps(result))""".replace("GROUP", repr(group)).replace(
                "REFRESH", repr(args.refresh)
            )
        )
        while True:
            if start.get("receipt"):
                remote = start["receipt"]
                break
            if time.time() >= DEADLINE:
                raise TimeoutError("Authorized deadline reached")
            state = call(
                """import pathlib,json,psutil
p=pathlib.Path('/workspace/ejepa_preservation')/(GROUP+'.json')
alive=psutil.pid_exists(PID) and psutil.Process(PID).status()!=psutil.STATUS_ZOMBIE
receipt=json.loads(p.read_text()) if p.exists() else None
print('PODSTATUS:'+json.dumps({'receipt':receipt,'alive':alive}))""".replace(
                    "GROUP", repr(group)
                ).replace("PID", str(start["pid"]))
            )
            if state["receipt"]:
                remote = state["receipt"]
                break
            if not state["alive"]:
                raise RuntimeError(f"Archive worker failed for {group}")
            publish(
                RECEIPTS / "STATUS.json",
                {
                    "status": "ARCHIVING",
                    "group": group,
                    "pid": start["pid"],
                    "timestamp": time.time(),
                },
            )
            time.sleep(5)
        if shutil.disk_usage(args.destination).free < remote["bytes"] + 10 * 1024**3:
            raise RuntimeError("Insufficient local backup disk reserve")
        suffix = ".tar.gz" if remote["path"].endswith(".tar.gz") else ".tar"
        destination = args.destination / f"{group}_{remote['sha256'][:16]}{suffix}"
        temporary = destination.with_suffix(".partial")
        url = "https://lreiivpf18vxu7-8888.proxy.runpod.net/files/" + remote["path"].lstrip("/")
        response = session.get(url, stream=True, timeout=(15, 120))
        response.raise_for_status()
        received = 0
        last = 0.0
        with temporary.open("wb") as stream:
            for block in response.iter_content(8 * 1024**2):
                stream.write(block)
                received += len(block)
                if time.time() - last >= 5:
                    publish(
                        RECEIPTS / "STATUS.json",
                        {
                            "status": "DOWNLOADING",
                            "group": group,
                            "received": received,
                            "total": remote["bytes"],
                            "timestamp": time.time(),
                        },
                    )
                    last = time.time()
            stream.flush()
            os.fsync(stream.fileno())
        if received != remote["bytes"]:
            raise ValueError("Downloaded length differs")
        manifest = verify_archive(temporary, remote)
        os.replace(temporary, destination)
        result = dict(
            remote,
            status="LOCAL_VERIFIED",
            local_path=str(destination.resolve()),
            member_sha_verified=True,
            manifest_files=len(manifest["files"]),
            verified_timestamp=time.time(),
        )
        publish(receipt_path, result)
        publish(RECEIPTS / "versions" / f"{group}_{remote['sha256']}.json", result)
        publish(RECEIPTS / f"{group}.manifest.json", manifest)
        # Parent explicitly authorized removal of only this newly created temporary
        # archive after durable local receipt and all member hashes pass.
        cleanup = call(
            """import pathlib,json,hashlib,os
p=pathlib.Path(PATH);assert p.parent==pathlib.Path('/tmp/ejepa_preservation')
assert p.stat().st_size==SIZE
h=hashlib.sha256();offset=0
with p.open('rb') as stream:
 for data in iter(lambda:stream.read(8*1024**2),b''):
  h.update(data)
  os.posix_fadvise(stream.fileno(),offset,len(data),os.POSIX_FADV_DONTNEED)
  offset+=len(data)
assert h.hexdigest()==SHA
p.unlink();print('PODSTATUS:'+json.dumps({'removed_verified_temporary_archive':str(p)}))""".replace(
                "PATH", repr(remote["path"])
            )
            .replace("SIZE", str(remote["bytes"]))
            .replace("SHA", repr(remote["sha256"]))
        )
        publish(RECEIPTS / f"{group}.cleanup.json", cleanup)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "status",
                        "group",
                        "bytes",
                        "sha256",
                        "files",
                        "local_path",
                        "member_sha_verified",
                    )
                }
            ),
            flush=True,
        )
    publish(
        RECEIPTS / "STATUS.json",
        {
            "status": "REQUESTED_GROUPS_VERIFIED",
            "groups": args.groups,
            "timestamp": time.time(),
            "all_project_outputs_complete": False,
        },
    )


if __name__ == "__main__":
    main()
