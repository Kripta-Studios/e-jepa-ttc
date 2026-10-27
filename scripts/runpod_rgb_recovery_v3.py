"""Publish RGB guard-attempt evidence before delegating to frozen recovery V2."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

V2_SHA256 = "c332254ff42687bb165bab06bb1cf5cb510f13ba8a7934042b8e16944ea5a8d0"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        import hashlib

        return hashlib.file_digest(stream, "sha256").hexdigest()


def install_queue_heartbeat(module: Any, clock: Any = time.monotonic) -> None:
    """Emit entry/attempt evidence before an optional external guard can fail."""
    original = module.run_rgb_downstream

    def run(run: Path, rotation: int, **kwargs: Any) -> Any:
        root = run / "rotations" / str(rotation)
        external = kwargs.get("external_guard")
        last = float("-inf")

        def heartbeat() -> None:
            nonlocal last
            now = clock()
            if now - last >= 30:
                module._receipt(
                    root,
                    "RGB_STAGE_ACTIVE",
                    "RUNNING",
                    observation=(
                        "worker entered downstream execution or is attempting resource guard"
                    ),
                    heartbeat_event="ENTRY_OR_GUARD_ATTEMPT",
                    receipt_precedes_optional_external_guard=True,
                    external_guard_success_inferred=False,
                    optimizer_updates_inferred=False,
                )
                last = now
            if external is not None:
                external()

        heartbeat()
        return original(run, rotation, **{**kwargs, "external_guard": heartbeat})

    module.run_rgb_downstream = run


def _write_sidecar(run: Path, source: Path, v2_path: Path) -> None:
    binding = {
        "schema": "ejepa_rgb_operational_recovery_v3",
        "status": "PASS",
        "source_sha256": digest(source),
        "frozen_v2_sha256": digest(v2_path),
        "migration_binding_sha256": digest(run / "MIGRATION_BINDING.json"),
        "heartbeat_receipt_precedes_optional_external_guard": True,
        "external_guard_success_inferred": False,
        "optimizer_updates_inferred": False,
        "frozen_v2_main_delegated": True,
        "scientific_sources_rewritten": False,
    }
    target = run / "RGB_OPERATIONAL_RECOVERY_V3.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != binding:
            raise ValueError("RGB recovery V3 binding differs")
        return
    temporary = target.with_name(f".{target.name}.{os.getpid()}.pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(binding, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def main() -> None:
    source = Path(__file__).resolve()
    ops = source.parent
    v2_path = ops / "runpod_rgb_recovery_v2.py"
    if digest(v2_path) != V2_SHA256:
        raise ValueError("Frozen RGB recovery V2 bytes differ")
    sys.path.insert(0, str(ops))
    import runpod_rgb_recovery_v2 as v2

    run = Path(sys.argv[sys.argv.index("--run") + 1]).resolve(strict=True)
    _write_sidecar(run, source, v2_path)
    original = v2.install_queue_heartbeat
    v2.install_queue_heartbeat = install_queue_heartbeat
    try:
        v2.main()
    finally:
        v2.install_queue_heartbeat = original


if __name__ == "__main__":
    main()
