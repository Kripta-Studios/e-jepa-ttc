"""Send V5 remote resume inventory over stdin instead of the SSH command line."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

V5_SHA256 = "27d330086abff89aff4d4a33b8f6536c737f4cafb498894fc9d26814dd8157b4"
MAX_INVENTORY_FILES = 10_000
MAX_INVENTORY_JSON_BYTES = 8 * 1024 * 1024


def digest(path: Path) -> str:
    import hashlib

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def remote_staged_credit(
    command: list[str],
    remote_root: str,
    files: list[tuple[str, int]],
    *,
    runner: Any = subprocess.run,
) -> tuple[int, dict[str, Any]]:
    if len(files) > MAX_INVENTORY_FILES:
        raise ValueError("Remote staged inventory exceeds file bound")
    payload = []
    for relative, size in files:
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Unsafe staged RGB path")
        payload.append({"relative": str(path), "bytes": int(size)})
    if any(item["bytes"] < 0 for item in payload):
        raise ValueError("Negative staged RGB size")
    encoded_payload = json.dumps(payload, separators=(",", ":"))
    if len(encoded_payload.encode()) > MAX_INVENTORY_JSON_BYTES:
        raise ValueError("Remote staged inventory exceeds JSON byte bound")
    code = r'''import json,pathlib,stat,sys
root=pathlib.Path(sys.argv[1]).resolve(strict=True);items=json.load(sys.stdin)
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
print(json.dumps({"credit_bytes":credit,"target_files":targets,"partial_files":partials}))'''
    encoded_code = base64.b64encode(code.encode()).decode()
    result = runner(
        command
        + [
            f'python3 -c "import base64;exec(base64.b64decode(\'{encoded_code}\'))" '
            + remote_root
        ],
        input=encoded_payload,
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
    scripts = Path(__file__).resolve().parent
    v5_path = scripts / "upload_rgb_sparse_frames_v5.py"
    if digest(v5_path) != V5_SHA256:
        raise ValueError("Frozen sparse RGB uploader V5 bytes differ")
    sys.path.insert(0, str(scripts))
    import upload_rgb_sparse_frames_v5 as v5

    original = v5.remote_staged_credit
    v5.remote_staged_credit = remote_staged_credit
    try:
        return v5.main()
    finally:
        v5.remote_staged_credit = original


if __name__ == "__main__":
    raise SystemExit(main())
