"""Apply workspace quota accounting around the frozen resumable RGB uploader V2."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

V2_SHA256 = "f0d0ad61f2da1cb8a9ef049f4e53f419a529d5d410388437323b612475db4677"
ATOMIC_STATUS_SHA256 = "c630e17f040852a29935ea4896cc753ec36cd03ac1fb4fc2c64024b9ba9b6bce"
RETRY_DELAYS_SECONDS = (0.1, 0.25, 0.5)
_DU_ENOENT = re.compile(
    r"^du: cannot access ['\"](?P<path>/workspace(?:/[^'\"]*)?)['\"]: "
    r"No such file or directory$"
)


def digest(path: Path) -> str:
    import hashlib

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def transient_du_enoent(error: subprocess.CalledProcessError) -> bool:
    if error.returncode != 1 or not isinstance(error.stderr, str):
        return False
    lines = [line.strip() for line in error.stderr.splitlines() if line.strip()]
    return bool(lines) and all(_DU_ENOENT.fullmatch(line) is not None for line in lines)


def _remote_python(command: list[str], code: str, runner: Callable[..., Any]) -> Any:
    encoded = base64.b64encode(code.encode()).decode()
    return runner(
        command + [f'python3 -c "import base64;exec(base64.b64decode(\'{encoded}\'))"'],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def remote_inventory(
    command: list[str],
    remote_root: str,
    quota_root: str,
    quota_bytes: int,
    *,
    runner: Callable[..., Any] = subprocess.run,
    delays: tuple[float, ...] = RETRY_DELAYS_SECONDS,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[int, dict[str, Any]]:
    """Return min(statvfs, quota-du), retrying only bounded GNU-du ENOENT races."""
    pattern = r"/[A-Za-z0-9_./-]+"
    if (
        not re.fullmatch(pattern, remote_root)
        or not re.fullmatch(pattern, quota_root)
        or quota_bytes <= 0
    ):
        raise ValueError("Unsafe remote quota contract")
    code = (
        "import json,os,pathlib;"
        f"r=pathlib.Path({remote_root!r}).resolve(strict=True);"
        f"q=pathlib.Path({quota_root!r}).resolve(strict=True);"
        "v=os.statvfs(r);"
        "print(json.dumps({'root':str(r),'quota_root':str(q),'root_dev':r.stat().st_dev,"
        "'quota_dev':q.stat().st_dev,'statvfs_free':v.f_bavail*v.f_frsize}))"
    )
    stat = json.loads(_remote_python(command, code, runner).stdout)
    root_path, quota_path = Path(str(stat["root"])), Path(str(stat["quota_root"]))
    if (
        root_path != Path(remote_root)
        or quota_path != Path(quota_root)
        or not root_path.is_relative_to(quota_path)
        or stat.get("root_dev") != stat.get("quota_dev")
    ):
        raise ValueError("Remote destination is outside quota root or on another filesystem")
    retries = 0
    for attempt in range(len(delays) + 1):
        try:
            result = runner(
                command + [f"du -sb -- {quota_root}"],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            fields = result.stdout.strip().split(maxsplit=1)
            if len(fields) != 2 or fields[1] != quota_root:
                raise ValueError("GNU du quota output differs")
            used = int(fields[0])
            if not 0 <= used <= quota_bytes:
                raise ValueError("Workspace quota usage is outside authorization")
            quota_free = quota_bytes - used
            available = min(int(stat["statvfs_free"]), quota_free)
            return available, {
                "schema": "ejepa_rgb_workspace_quota_inventory_v4",
                "remote_root": remote_root,
                "quota_root": quota_root,
                "quota_bytes": quota_bytes,
                "statvfs_free_bytes": int(stat["statvfs_free"]),
                "quota_used_bytes": used,
                "quota_free_bytes": quota_free,
                "available_bytes": available,
                "du_enoent_retries": retries,
                "retry_bounded": True,
                "stale_fallback_used": False,
            }
        except subprocess.CalledProcessError as error:
            if not transient_du_enoent(error):
                raise
            retries += 1
            if attempt == len(delays):
                raise RuntimeError(
                    "GNU du remained unstable after bounded ENOENT retries"
                ) from error
            sleeper(delays[attempt])
    raise AssertionError("unreachable")


def _consume(argv: list[str], name: str) -> tuple[str, list[str]]:
    if argv.count(name) != 1:
        raise ValueError(f"Exactly one {name} is required")
    index = argv.index(name)
    if index + 1 >= len(argv):
        raise ValueError(f"Missing {name} value")
    return argv[index + 1], [*argv[:index], *argv[index + 2 :]]


def _option(argv: list[str], name: str) -> str:
    if argv.count(name) != 1 or argv.index(name) + 1 >= len(argv):
        raise ValueError(f"Exactly one valued {name} is required")
    return argv[argv.index(name) + 1]


def _option_default(argv: list[str], name: str, default: str) -> str:
    return _option(argv, name) if name in argv else default


def main() -> int:
    source = Path(__file__).resolve()
    v2_path = source.parent / "upload_rgb_sparse_frames_v2.py"
    atomic_status_path = source.parent / "upload_rgb_atomic_status_v1.py"
    if digest(v2_path) != V2_SHA256:
        raise ValueError("Frozen sparse RGB uploader V2 bytes differ")
    if digest(atomic_status_path) != ATOMIC_STATUS_SHA256:
        raise ValueError("Frozen atomic status helper bytes differ")
    quota_raw, argv = _consume(sys.argv, "--quota-bytes")
    quota_root, argv = _consume(argv, "--quota-root")
    reserve_raw, argv = _consume(argv, "--reserve-bytes")
    quota_bytes, reserve_bytes = int(quota_raw), int(reserve_raw)
    if quota_bytes != 50_000_000_000 or reserve_bytes != 4_294_967_296:
        raise ValueError("RGB upload requires exact 50GB quota and 4GiB reserve")
    argv.extend(["--reserve-bytes", str(reserve_bytes)])
    sys.path.insert(0, str(source.parent))
    import upload_rgb_atomic_status_v1 as atomic_status
    import upload_rgb_sparse_frames_v2 as v2

    retry_atomic_json = atomic_status.retry_writer(atomic_json)

    if "--execute" not in argv:
        original, sys.argv = sys.argv, argv
        try:
            return v2.main()
        finally:
            sys.argv = original
    remote_root = _option(argv, "--remote-root")
    if remote_root != "/workspace/ejepa_rgb_frames" or quota_root != "/workspace":
        raise ValueError("RGB upload requires the authorized workspace and media roots")
    source_root = Path(_option(argv, "--source")).resolve(strict=True)
    status_path = Path(_option(argv, "--status"))
    command = v2.ssh_base(
        _option(argv, "--host"),
        _option_default(argv, "--user", "root"),
        int(_option(argv, "--port")),
        Path(_option(argv, "--identity-file")),
    )
    available, inventory = remote_inventory(command, remote_root, quota_root, quota_bytes)
    old = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    completed = set(old.get("completed", {}))
    ready = [
        v2.load_ready(marker, source_root)
        for marker in source_root.glob("data/train/*/rgb_shards/*.tar.json")
    ]
    remaining = sum(item.total for item in ready if item.relative not in completed)
    external = int(_option_default(argv, "--external-inflight-bytes", "1000000000"))
    if available - reserve_bytes - external < remaining:
        raise OSError("Full RGB upload cannot fit authorized workspace quota and reserves")
    sidecar = status_path.with_name("UPLOAD_V4_QUOTA_ADMISSION.json")
    admission = {
        "schema": "ejepa_rgb_sparse_uploader_quota_admission_v4",
        "status": "PASS",
        "source_sha256": digest(source),
        "frozen_v2_sha256": digest(v2_path),
        "atomic_status_helper_sha256": digest(atomic_status_path),
        "remote_root": remote_root,
        "quota_root": quota_root,
        "quota_bytes": quota_bytes,
        "reserve_bytes": reserve_bytes,
        "external_inflight_bytes": external,
        "remaining_verified_input_bytes": remaining,
        "initial_inventory": inventory,
        "scientific_execution_authorized": False,
    }
    if sidecar.is_file() and json.loads(sidecar.read_text(encoding="utf-8")) != admission:
        raise ValueError("Existing RGB uploader quota admission differs")
    if not sidecar.exists():
        retry_atomic_json(sidecar, admission)
    reports: list[dict[str, Any]] = []

    def quota_free(command: list[str], root: str) -> int:
        free, report = remote_inventory(command, root, quota_root, quota_bytes)
        reports.append(report)
        return free

    original_free, original_atomic, original_argv = v2.remote_free, v2.atomic_json, sys.argv
    v2.remote_free, v2.atomic_json, sys.argv = (
        quota_free,
        atomic_status.retry_writer(v2.atomic_json),
        argv,
    )
    try:
        return v2.main()
    finally:
        v2.remote_free, v2.atomic_json, sys.argv = (
            original_free,
            original_atomic,
            original_argv,
        )


if __name__ == "__main__":
    raise SystemExit(main())
