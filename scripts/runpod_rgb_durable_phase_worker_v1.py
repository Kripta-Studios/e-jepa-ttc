# ruff: noqa: E501
"""Recoverably sequence the remaining RGB media, R4 return, and R2 upload phases."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Self

CONFIG_SCHEMA = "ejepa_rgb_durable_phase_worker_config_v1"
STATE_SCHEMA = "ejepa_rgb_durable_phase_worker_state_v1"
ATOMIC_HELPER_SHA256 = "c630e17f040852a29935ea4896cc753ec36cd03ac1fb4fc2c64024b9ba9b6bce"
V4_SHA256 = "5756d64ffc502568053eb9f5f2a18e141ed85d8e3ccb405623b7916c900c274d"
V5_SHA256 = "27d330086abff89aff4d4a33b8f6536c737f4cafb498894fc9d26814dd8157b4"
STAGES = (
    "WAIT_R3_UPLOAD",
    "STOP_R3_UPLOAD",
    "WAIT_R4_UPLOAD",
    "STOP_R4_UPLOAD",
    "WAIT_R4_RETURN",
    "EVICT_R4_RAW",
    "WAIT_R2_UPLOAD",
    "STOP_R2_UPLOAD",
    "COMPLETE",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def manifest_records(path: Path, expected_sha256: str) -> dict[str, dict[str, Any]]:
    if sha256(path) != expected_sha256:
        raise ValueError(f"Manifest bytes differ: {path}")
    value = read_json(path)
    tars = value.get("tars")
    if not (
        value.get("schema") == "ejepa_rgb_frame_supplement_v1"
        and value.get("status") == "COMPLETE_BYTE_AND_PIXEL_VERIFIED"
        and value.get("decoded_pixel_parity") is True
        and value.get("payload_recompressed") is False
        and value.get("optimizer_updates") == 0
        and isinstance(tars, list)
        and len(tars) == value.get("tar_count")
    ):
        raise ValueError(f"Manifest is not a verified media endpoint: {path}")
    records: dict[str, dict[str, Any]] = {}
    for raw in tars:
        if not isinstance(raw, dict):
            raise ValueError("Manifest TAR record is not an object")
        relative = PurePosixPath(str(raw.get("relative")))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix != ".tar"
            or not str(relative).startswith("data/train/")
            or str(relative) in records
        ):
            raise ValueError(f"Unsafe/duplicate manifest TAR: {relative}")
        if not isinstance(raw.get("sparse_tar_bytes"), int) or not isinstance(
            raw.get("sparse_tar_sha256"), str
        ):
            raise ValueError("Manifest TAR identity is incomplete")
        records[str(relative)] = raw
    if sum(int(item["sparse_tar_bytes"]) for item in records.values()) != value.get(
        "sparse_tar_bytes"
    ):
        raise ValueError("Manifest byte total differs")
    return records


def upload_complete(
    status_path: Path, records: dict[str, dict[str, Any]], plan_sha256: str
) -> bool:
    if not status_path.is_file():
        return False
    status = read_json(status_path)
    completed = status.get("completed")
    if (
        status.get("schema") != "ejepa_rgb_sparse_parallel_uploader_v2"
        or status.get("plan_sha256") != plan_sha256
        or not isinstance(completed, dict)
    ):
        raise ValueError("Upload ledger identity differs")
    if set(completed) - set(records):
        raise ValueError("Upload ledger contains TARs outside the manifest")
    for relative, identity in completed.items():
        expected = records[relative]
        if (
            identity.get("bytes") != expected["sparse_tar_bytes"]
            or identity.get("sha256") != expected["sparse_tar_sha256"]
            or not isinstance(identity.get("marker_sha256"), str)
        ):
            raise ValueError(f"Upload ledger identity differs: {relative}")
    return set(completed) == set(records)


def process_matches(pid: int, argv: list[str]) -> bool:
    if pid <= 0:
        return False
    if os.name == "posix":
        try:
            observed = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
            return [item.decode() for item in observed if item] == argv
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
            return False
    query = (
        f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}';"
        "if($null -eq $p){exit 3};"
        "$p.CommandLine|ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", query],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        return False
    try:
        observed_line = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False
    return observed_line == subprocess.list2cmdline(argv)


class SingleInstance(AbstractContextManager["SingleInstance"]):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.stream: Any = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return self

    def __exit__(self, *_args: object) -> None:
        assert self.stream is not None
        self.stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


def validate_eviction_authority(path: Path) -> dict[str, Any]:
    value = read_json(path)
    root = path.parent.resolve(strict=True)
    manifest = root / "MANIFEST.json"
    local = root / "LOCAL_VERIFICATION.json"
    scientific = root / "scientific_proofs/RGB_COMPLETE_REUSE.json"
    if (
        value.get("schema") != "ejepa_rgb_raw_rotation_eviction_eligibility_v1"
        or value.get("status") != "PASS"
        or value.get("rotation") != 4
        or value.get("scope") != "RAW_R4_REPRODUCIBLE"
        or value.get("scientific_genealogy_verified") is not True
        or value.get("external_uploader_may_use_as_eviction_authority") is not True
        or value.get("remote_deletion_performed") is not False
        or value.get("local_deletion_performed") is not False
        or value.get("archive_manifest_sha256") != sha256(manifest)
        or value.get("local_verification_sha256") != sha256(local)
        or value.get("scientific_reusability_verification_sha256") != sha256(scientific)
    ):
        raise ValueError("Individual R4 eviction authority is absent or inconsistent")
    proof = read_json(scientific)
    if (
        proof.get("schema") != "ejepa_local_rgb_scientific_reusability_v1"
        or proof.get("status") != "PASS"
        or proof.get("archive_sha256") != value.get("archive_sha256")
        or proof.get("selected_root_path_maps_verified") is not True
        or set(proof.get("rotations", {})) != {"4"}
    ):
        raise ValueError("R4 scientific proof does not authorize eviction")
    return value


def validate_eviction_config(value: object) -> None:
    expected_prefix = [
        "C:/Program Files/Git/usr/bin/ssh.exe",
        "-T",
        "-p",
        "13608",
        "-i",
        str(Path.home() / ".ssh/id_ed25519"),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "root@213.173.109.37",
    ]
    if not isinstance(value, dict) or (
        value.get("remote_root") != "/workspace/ejepa_rgb_frames"
        or value.get("ssh_argv_prefix") != expected_prefix
    ):
        raise ValueError("R4 eviction remote root or SSH identity differs")


REMOTE_EVICTION = r"""
import hashlib,json,os,pathlib,sys,tempfile
root=pathlib.Path(sys.argv[1]).resolve(strict=True); payload=json.loads(sys.argv[2]); intent=root/'R4_RAW_EVICTION_INTENT.json'; done=root/'R4_RAW_EVICTION_COMPLETE.json'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canon(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v):
 fd,n=tempfile.mkstemp(prefix='.'+p.name,dir=p.parent)
 with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,separators=(',',':'));f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(n,p)
 if os.name=='posix':
  d=os.open(p.parent,os.O_RDONLY)
  try:os.fsync(d)
  finally:os.close(d)
records=payload['records']; expected={'schema':'ejepa_rgb_raw_eviction_intent_v1','authority_sha256':payload['authority_sha256'],'manifest_sha256':payload['manifest_sha256'],'records':records}
def bound(r):
 p=root.joinpath(*pathlib.PurePosixPath(r['relative']).parts)
 if not p.parent.resolve(strict=True).is_relative_to(root):raise ValueError('TAR parent escapes root '+r['relative'])
 return p
prior=intent.exists()
if prior:
 if json.loads(intent.read_text())!=expected:raise ValueError('eviction intent differs')
else:
 for r in records:
  p=bound(r);m=p.with_name(p.name+'.json')
  if not p.is_file() or p.is_symlink() or not p.resolve(strict=True).is_relative_to(root) or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('pre-delete TAR differs '+r['relative'])
  if m.is_symlink() or not m.resolve(strict=True).is_relative_to(root):raise ValueError('retained marker escapes root '+r['relative'])
  v=json.loads(m.read_text());c={k:x for k,x in v.items() if k!='receipt_sha256'}
  if v.get('receipt_sha256')!=r['receipt_sha256'] or v.get('receipt_sha256')!=canon(c) or v.get('sparse_tar_sha256')!=r['sha256']:raise ValueError('retained marker differs '+r['relative'])
 write(intent,expected)
for r in records:
 p=bound(r);m=p.with_name(p.name+'.json')
 if not m.is_file() or m.is_symlink() or not m.resolve(strict=True).is_relative_to(root):raise ValueError('retained marker vanished/differs '+r['relative'])
 if p.exists():
  if p.is_symlink() or not p.resolve(strict=True).is_relative_to(root) or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('TAR changed after intent '+r['relative'])
  p.unlink()
  if os.name=='posix':
   d=os.open(p.parent,os.O_RDONLY)
   try:os.fsync(d)
   finally:os.close(d)
write(done,{'schema':'ejepa_rgb_raw_eviction_complete_v1','intent_sha256':sha(intent),'deleted_tars':len(records),'json_sidecars_retained':True})
print(json.dumps({'status':'COMPLETE','deleted_tars':len(records)}))
"""


class Worker:
    def __init__(
        self, config: dict[str, Any], *, popen: Any = subprocess.Popen, run: Any = subprocess.run
    ) -> None:
        self.config = config
        self.popen = popen
        self.run = run
        self.state_path = Path(config["state_path"])
        self.helper = self._atomic_helper()
        self.state = (
            read_json(self.state_path)
            if self.state_path.is_file()
            else {
                "schema": STATE_SCHEMA,
                "stage": "WAIT_R3_UPLOAD",
                "processes": {},
                "attempts": {},
            }
        )
        if not self.state_path.is_file():
            adoption = read_json(Path(config["r3_adoption_receipt"]))
            r3_argv = config["uploads"]["r3_generation"]["argv"]
            if (
                adoption.get("schema") != "ejepa_upload_atomic_status_recovery_v1"
                or adoption.get("argv") != r3_argv
                or adoption.get("source_sha256") != ATOMIC_HELPER_SHA256
                or adoption.get("frozen_v2_sha256")
                != "f0d0ad61f2da1cb8a9ef049f4e53f419a529d5d410388437323b612475db4677"
            ):
                raise ValueError("Existing R3 uploader adoption receipt differs")
            pid = int(adoption["pid"])
            if process_matches(pid, r3_argv):
                self.state["processes"]["r3_generation"] = {
                    "pid": pid,
                    "argv": r3_argv,
                    "adopted": True,
                }

    def _atomic_helper(self) -> Any:
        path = Path(self.config["atomic_helper_path"])
        if sha256(path) != ATOMIC_HELPER_SHA256:
            raise ValueError("Atomic helper bytes differ")
        sys.path.insert(0, str(path.parent))
        import upload_rgb_atomic_status_v1 as helper

        return helper.retry_writer(self._write)

    @staticmethod
    def _write(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
        with pending.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)

    def publish(self, detail: str) -> None:
        self.state.update(utc=datetime.now(UTC).isoformat(), pid=os.getpid(), detail=detail)
        self.helper(self.state_path, self.state)

    def advance(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(stage)
        self.state["stage"] = stage
        self.publish("ADVANCED")

    def alive(self, name: str, argv: list[str]) -> bool:
        record = self.state["processes"].get(name)
        if not isinstance(record, dict):
            return False
        if record.get("argv") != argv:
            raise ValueError(f"Persisted {name} argv differs")
        pid = int(record["pid"])
        if process_matches(pid, argv):
            return True
        record["exited"] = True
        return False

    def launch(self, name: str, argv: list[str]) -> None:
        if self.alive(name, argv):
            return
        attempts = int(self.state["attempts"].get(name, 0)) + 1
        if attempts > int(self.config["max_launch_attempts"]):
            raise RuntimeError(f"{name} exhausted bounded launch attempts")
        log_root = Path(self.config["log_root"])
        log_root.mkdir(parents=True, exist_ok=True)
        out = (log_root / f"{name}.stdout.log").open("ab")
        err = (log_root / f"{name}.stderr.log").open("ab")
        process = self.popen(argv, stdout=out, stderr=err, cwd=self.config["repository"])
        self.state["attempts"][name] = attempts
        self.state["processes"][name] = {"pid": process.pid, "argv": argv}
        self.publish(f"LAUNCHED_{name}")

    def stop(self, name: str, argv: list[str], status_path: Path) -> bool:
        record = self.state["processes"].get(name)
        if not isinstance(record, dict):
            return True
        if not self.alive(name, argv):
            return True
        stop = status_path.with_name("UPLOAD_V2_STOP_REQUEST")
        stop.parent.mkdir(parents=True, exist_ok=True)
        stop.write_text("phase worker graceful stop\n", encoding="utf-8")
        self.publish(f"GRACEFUL_STOP_REQUESTED_{name}")
        return False

    def _upload(self, key: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        item = self.config["uploads"][key]
        records = manifest_records(Path(item["manifest_path"]), item["manifest_sha256"])
        if sum(int(v["sparse_tar_bytes"]) for v in records.values()) != item["tar_bytes"]:
            raise ValueError(f"{key} configured byte total differs")
        return item, records

    def evict(self) -> None:
        eviction = self.config["eviction"]
        authority_path = Path(eviction["authority_path"])
        validate_eviction_authority(authority_path)
        r4, records = self._upload("r4_refiner")
        _, r2_records = self._upload("r2_generation")
        if set(records) & set(r2_records):
            raise ValueError("R4 raw eviction overlaps R2 generation inputs")
        payload = {
            "authority_sha256": sha256(authority_path),
            "manifest_sha256": r4["manifest_sha256"],
            "records": [
                {
                    "relative": relative,
                    "bytes": item["sparse_tar_bytes"],
                    "sha256": item["sparse_tar_sha256"],
                    "receipt_sha256": item["receipt_sha256"],
                }
                for relative, item in sorted(records.items())
            ],
        }
        encoded = base64.b64encode(REMOTE_EVICTION.encode()).decode()
        argv = list(eviction["ssh_argv_prefix"]) + [
            f"python3 -c \"import base64;exec(base64.b64decode('{encoded}'))\" "
            + eviction["remote_root"]
            + " '"
            + json.dumps(payload, sort_keys=True, separators=(",", ":"))
            + "'"
        ]
        self.run(argv, check=True, timeout=1800)

    def step(self) -> None:
        if datetime.now(UTC) >= datetime.fromisoformat(self.config["deadline"]):
            raise TimeoutError("RGB phase worker deadline expired")
        stage = self.state["stage"]
        r3, r3_records = self._upload("r3_generation")
        r4, r4_records = self._upload("r4_refiner")
        r2, r2_records = self._upload("r2_generation")
        if stage == "WAIT_R3_UPLOAD":
            if upload_complete(Path(r3["status_path"]), r3_records, r3["plan_sha256"]):
                self.advance("STOP_R3_UPLOAD")
            else:
                self.launch("r3_generation", r3["argv"])
        elif stage == "STOP_R3_UPLOAD":
            if self.stop("r3_generation", r3["argv"], Path(r3["status_path"])):
                self.launch("r4_refiner", r4["argv"])
                self.advance("WAIT_R4_UPLOAD")
        elif stage == "WAIT_R4_UPLOAD":
            if upload_complete(Path(r4["status_path"]), r4_records, r4["plan_sha256"]):
                self.advance("STOP_R4_UPLOAD")
            else:
                self.launch("r4_refiner", r4["argv"])
        elif stage == "STOP_R4_UPLOAD":
            if self.stop("r4_refiner", r4["argv"], Path(r4["status_path"])):
                self.advance("WAIT_R4_RETURN")
        elif stage == "WAIT_R4_RETURN":
            if not self.config["return_coordinator_external"]:
                self.run(self.config["return_coordinator_argv"], check=True, timeout=1800)
            return_status = Path(self.config["return_status_path"])
            if not return_status.is_file():
                self.publish("WAITING_EXTERNAL_R4_RETURN_COORDINATOR")
                return
            status = read_json(return_status)
            if (
                status.get("stage") == "COMPLETE"
                and status.get("raw_r4_eviction_eligibility") == "PASS_EXTERNAL_ACTION_ONLY"
            ):
                self.advance("EVICT_R4_RAW")
        elif stage == "EVICT_R4_RAW":
            self.evict()
            self.launch("r2_generation", r2["argv"])
            self.advance("WAIT_R2_UPLOAD")
        elif stage == "WAIT_R2_UPLOAD":
            if upload_complete(Path(r2["status_path"]), r2_records, r2["plan_sha256"]):
                self.advance("STOP_R2_UPLOAD")
            else:
                self.launch("r2_generation", r2["argv"])
        elif stage == "STOP_R2_UPLOAD":
            if self.stop("r2_generation", r2["argv"], Path(r2["status_path"])):
                self.advance("COMPLETE")
        elif stage == "COMPLETE":
            self.publish("COMPLETE")
        else:
            raise ValueError(f"Unknown persisted stage: {stage}")


def load_config(path: Path) -> dict[str, Any]:
    value = read_json(path.resolve(strict=True))
    if value.get("schema") != CONFIG_SCHEMA or value.get("poll_seconds") != 15:
        raise ValueError("Phase worker config schema/poll differs")
    if value.get("max_launch_attempts") != 12:
        raise ValueError("Phase worker retry bound differs")
    deadline = datetime.fromisoformat(str(value.get("deadline")))
    if deadline.tzinfo is None:
        raise ValueError("Deadline lacks timezone")
    repository = Path(value["repository"]).resolve(strict=True)
    expected_v4 = repository / "scripts/upload_rgb_sparse_frames_v4.py"
    expected_v5 = repository / "scripts/upload_rgb_sparse_frames_v5.py"
    if sha256(expected_v4) != V4_SHA256:
        raise ValueError("Uploader V4 bytes differ")
    if sha256(expected_v5) != V5_SHA256:
        raise ValueError("Uploader V5 bytes differ")
    helper = Path(value["atomic_helper_path"]).resolve(strict=True)
    if sha256(helper) != ATOMIC_HELPER_SHA256:
        raise ValueError("Atomic helper bytes differ")
    uploads = value.get("uploads")
    if not isinstance(uploads, dict) or set(uploads) != {
        "r3_generation",
        "r4_refiner",
        "r2_generation",
    }:
        raise ValueError("Upload phase set differs")
    for key, item in uploads.items():
        if not isinstance(item, dict) or not isinstance(item.get("argv"), list):
            raise ValueError(f"Invalid upload contract: {key}")
        manifest_records(Path(item["manifest_path"]), item["manifest_sha256"])
        if item["argv"][0] != value["python_executable"] or item["argv"][-1] != "--execute":
            raise ValueError(f"Upload argv is not exact/executable: {key}")
        if key != "r3_generation" and str(expected_v5) not in item["argv"]:
            raise ValueError(f"New upload phase must use V5: {key}")
        if key != "r3_generation" and (
            item["argv"][item["argv"].index("--complete-manifest") + 1] != item["manifest_path"]
            or item["argv"][item["argv"].index("--complete-manifest-sha256") + 1]
            != item["manifest_sha256"]
        ):
            raise ValueError(f"V5 manifest closure differs: {key}")
    coordinator_argv = value.get("return_coordinator_argv")
    if not isinstance(coordinator_argv, list) or "--config" not in coordinator_argv:
        raise ValueError("Return coordinator argv differs")
    coordinator_config = Path(coordinator_argv[coordinator_argv.index("--config") + 1]).resolve(
        strict=True
    )
    if sha256(coordinator_config) != value.get("return_coordinator_config_sha256"):
        raise ValueError("Individual R4 return coordinator config bytes differ")
    if value.get("return_coordinator_external") is not True:
        raise ValueError("R4 return coordinator must remain the admitted external singleton")
    validate_eviction_config(value.get("eviction"))
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.validate_config:
        print(json.dumps({"status": "PASS", "schema": config["schema"]}, indent=2))
        return 0
    with SingleInstance(Path(config["lock_path"])):
        worker = Worker(config)
        while True:
            worker.step()
            if args.once or worker.state["stage"] == "COMPLETE":
                return 0
            time.sleep(config["poll_seconds"])


if __name__ == "__main__":
    raise SystemExit(main())
