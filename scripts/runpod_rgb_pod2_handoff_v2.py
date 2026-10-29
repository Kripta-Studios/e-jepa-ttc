"""Run the frozen Pod2 handoff with an integrity-bound 8 MiB transport chunk."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operational.runpod_migration import pod2_rgb_handoff as runtime

V1_MODULE_SHA256 = "f5d5fcb2075f01830e3a00068375e827fa3433c36433a201d00f3f382324974b"
V1_CLI_SHA256 = "950cbbbed50eaddaa6042e75c3229acaf777d04139deec0149e45a1047db38cd"
V2_CHUNK_BYTES = 8 * 1024 * 1024
OLD_GUARD = "if len(payload) > 1048576 or hashlib.sha256(payload).hexdigest() != chunk_sha:"
NEW_GUARD = "if len(payload) > 8388608 or hashlib.sha256(payload).hexdigest() != chunk_sha:"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def activate_v2() -> tuple[int, str]:
    """Verify frozen V1 bytes and apply only the bounded chunk-size substitution."""
    repository = Path(__file__).resolve().parents[1]
    module_path = repository / "operational/runpod_migration/pod2_rgb_handoff.py"
    cli_path = repository / "scripts/runpod_rgb_pod2_handoff_v1.py"
    if digest(module_path) != V1_MODULE_SHA256 or digest(cli_path) != V1_CLI_SHA256:
        raise ValueError("Frozen Pod2 handoff V1 module/CLI bytes differ")
    helper = runtime.REMOTE_HELPER
    if helper.count(OLD_GUARD) != 1 or helper.count(NEW_GUARD) != 0:
        raise ValueError("Frozen Pod2 remote 1MiB payload guard occurrence differs")
    original = (runtime.CHUNK_BYTES, helper)
    runtime.CHUNK_BYTES = V2_CHUNK_BYTES
    runtime.REMOTE_HELPER = helper.replace(OLD_GUARD, NEW_GUARD)
    if runtime.REMOTE_HELPER.count(NEW_GUARD) != 1:
        raise ValueError("Pod2 V2 remote 8MiB payload guard substitution failed")
    return original


def main() -> int:
    activate_v2()
    from scripts.runpod_rgb_pod2_handoff_v1 import main as v1_main

    return v1_main()


if __name__ == "__main__":
    raise SystemExit(main())
