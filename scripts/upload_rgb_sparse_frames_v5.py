"""Add manifest closure and validated remote resume credit around uploader V4."""

from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

V4_SHA256 = "5756d64ffc502568053eb9f5f2a18e141ed85d8e3ccb405623b7916c900c274d"


def digest(path: Path) -> str:
    import hashlib

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _consume(argv: list[str], name: str) -> tuple[str, list[str]]:
    if argv.count(name) != 1:
        raise ValueError(f"Exactly one {name} is required")
    index = argv.index(name)
    if index + 1 >= len(argv):
        raise ValueError(f"Missing {name} value")
    return argv[index + 1], [*argv[:index], *argv[index + 2 :]]


def _option(argv: list[str], name: str, default: str | None = None) -> str:
    if name not in argv:
        if default is None:
            raise ValueError(f"Missing {name}")
        return default
    if argv.count(name) != 1 or argv.index(name) + 1 >= len(argv):
        raise ValueError(f"Exactly one valued {name} is required")
    return argv[argv.index(name) + 1]


def validate_complete_manifest(
    manifest_path: Path,
    expected_sha256: str,
    source_root: Path,
    loader: Any,
) -> tuple[list[Any], dict[str, Any]]:
    manifest_path = manifest_path.resolve(strict=True)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or digest(
        manifest_path
    ) != expected_sha256:
        raise ValueError("Complete sparse RGB manifest bytes differ")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("tars")
    physical_root = manifest.get("physical_root")
    if not (
        manifest.get("schema") == "ejepa_rgb_frame_supplement_v1"
        and manifest.get("status") == "COMPLETE_BYTE_AND_PIXEL_VERIFIED"
        and manifest.get("payload_recompressed") is False
        and manifest.get("decoded_pixel_parity") is True
        and manifest.get("optimizer_updates") == 0
        and isinstance(records, list)
        and records
        and manifest.get("tar_count") == len(records)
        and isinstance(physical_root, str)
    ):
        raise ValueError("Sparse RGB manifest is not a complete physical closure")
    if Path(physical_root).resolve(strict=True) != source_root.resolve(strict=True):
        raise ValueError("Complete sparse RGB manifest physical root differs")
    ready = [
        loader(marker, source_root)
        for marker in source_root.glob("data/train/*/rgb_shards/*.tar.json")
    ]
    actual = {item.relative: item for item in ready}
    declared = {str(record["relative"]): record for record in records}
    if actual.keys() != declared.keys() or len(declared) != len(records):
        raise ValueError("READY sparse TAR set differs from complete manifest")
    for relative, item in actual.items():
        record = declared[relative]
        if (
            record.get("sparse_tar_bytes") != item.size
            or record.get("sparse_tar_sha256") != item.sha256
            or json.loads(item.marker.read_text(encoding="utf-8")) != record
        ):
            raise ValueError("READY sparse TAR receipt differs from complete manifest")
    if sum(item.size for item in ready) != int(manifest.get("sparse_tar_bytes")):
        raise ValueError("Complete sparse RGB manifest byte total differs")
    return ready, manifest


def _safe_relative(value: str) -> str:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Unsafe staged RGB path")
    return str(relative)


def remote_staged_credit(
    command: list[str],
    remote_root: str,
    files: list[tuple[str, int]],
    *,
    runner: Any = subprocess.run,
) -> tuple[int, dict[str, Any]]:
    payload = [
        {"relative": _safe_relative(relative), "bytes": int(size)}
        for relative, size in files
    ]
    if any(item["bytes"] < 0 for item in payload):
        raise ValueError("Negative staged RGB size")
    encoded_payload = base64.b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode()
    code = f'''import base64,json,pathlib,stat
root=pathlib.Path({remote_root!r}).resolve(strict=True)
items=json.loads(base64.b64decode({encoded_payload!r}))
credit=0;partials=targets=0
for item in items:
 rel=pathlib.PurePosixPath(item["relative"]);expected=int(item["bytes"])
 if rel.is_absolute() or ".." in rel.parts or not rel.parts: raise ValueError("unsafe relative")
 target=root.joinpath(*rel.parts)
 if not target.parent.resolve().is_relative_to(root): raise ValueError("target escapes root")
 partial=target.with_name("."+target.name+".partial")
 if target.exists():
  value=target.lstat()
  if target.is_symlink() or not stat.S_ISREG(value.st_mode):
   raise ValueError("staged target type differs")
  if value.st_size!=expected: raise ValueError("staged target size differs")
  credit+=expected;targets+=1;continue
 if partial.exists():
  value=partial.lstat()
  if partial.is_symlink() or not stat.S_ISREG(value.st_mode):
   raise ValueError("staged partial type differs")
  if not 0<=value.st_size<=expected: raise ValueError("staged partial size differs")
  credit+=value.st_size;partials+=1
print(json.dumps({{"credit_bytes":credit,"target_files":targets,"partial_files":partials}}))'''
    encoded = base64.b64encode(code.encode()).decode()
    result = runner(
        command + [f'python3 -c "import base64;exec(base64.b64decode(\'{encoded}\'))"'],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    report = json.loads(result.stdout)
    credit = int(report["credit_bytes"])
    if not 0 <= credit <= sum(size for _, size in files):
        raise ValueError("Remote staged credit is outside pending byte range")
    return credit, report


def main() -> int:
    source = Path(__file__).resolve()
    v4_path = source.with_name("upload_rgb_sparse_frames_v4.py")
    if digest(v4_path) != V4_SHA256:
        raise ValueError("Frozen sparse RGB uploader V4 bytes differ")
    manifest_raw, argv = _consume(sys.argv, "--complete-manifest")
    manifest_sha, argv = _consume(argv, "--complete-manifest-sha256")
    sys.path.insert(0, str(source.parent))
    import upload_rgb_atomic_status_v1 as atomic_status
    import upload_rgb_sparse_frames_v2 as v2
    import upload_rgb_sparse_frames_v4 as v4

    source_root = Path(_option(argv, "--source")).resolve(strict=True)
    ready, manifest = validate_complete_manifest(
        Path(manifest_raw), manifest_sha, source_root, v2.load_ready
    )
    status_path = Path(_option(argv, "--status"))
    remote_root = _option(argv, "--remote-root")
    command = v2.ssh_base(
        _option(argv, "--host"),
        _option(argv, "--user", "root"),
        int(_option(argv, "--port")),
        Path(_option(argv, "--identity-file")),
    )
    original_inventory, original_argv = v4.remote_inventory, sys.argv

    def inventory(*args: Any, **kwargs: Any) -> tuple[int, dict[str, Any]]:
        free, report = original_inventory(*args, **kwargs)
        old = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        completed = set(old.get("completed", {}))
        pending = [item for item in ready if item.relative not in completed]
        files = [
            pair
            for item in pending
            for pair in (
                (item.relative, item.size),
                (item.relative + ".json", item.marker_size),
            )
        ]
        credit, staged = remote_staged_credit(command, remote_root, files)
        return free + credit, {
            **report,
            "resume_credit_bytes": credit,
            "resume_credit_scope": "regular exact-size targets plus bounded partial offsets",
            "resume_final_sha256_still_required_by_receiver": True,
            "staged": staged,
            "effective_additional_capacity_bytes": free + credit,
        }

    v4.remote_inventory, sys.argv = inventory, argv
    try:
        result = v4.main()
    finally:
        v4.remote_inventory, sys.argv = original_inventory, original_argv
    if result == 0:
        closure = {
            "schema": "ejepa_rgb_sparse_uploader_manifest_closure_v5",
            "status": "PASS",
            "source_sha256": digest(source),
            "frozen_v4_sha256": digest(v4_path),
            "complete_manifest_sha256": manifest_sha,
            "tar_count": manifest["tar_count"],
            "sparse_tar_bytes": manifest["sparse_tar_bytes"],
            "remote_final_sha256_required": True,
        }
        closure_path = status_path.with_name("UPLOAD_V5_MANIFEST_CLOSURE.json")
        if closure_path.exists() and json.loads(
            closure_path.read_text(encoding="utf-8")
        ) != closure:
            raise ValueError("Existing V5 manifest closure differs")
        if not closure_path.exists():
            atomic_status.retry_writer(v4.atomic_json)(closure_path, closure)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
