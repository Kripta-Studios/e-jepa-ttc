"""Load additive event migration modules without changing the sealed scientific tree."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    runtime = Path(__file__).resolve().parent
    base = Path(
        os.environ.get(
            "EJEPA_EVENT_BASE_REPO",
            "/opt/ejepa_event_bundle/workspace/e-jepa-ttc-v18-joint-measurement",
        )
    ).resolve(strict=True)
    source = (base / "src").resolve(strict=True)
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(base))
    sys.path.insert(0, str(runtime))
    import operational.runpod_migration as migration

    extension = runtime / "operational/runpod_migration"
    if not extension.is_dir():
        raise RuntimeError("Event runtime extension directory is missing")
    migration.__path__.insert(0, str(extension))
    if len(sys.argv) > 1 and sys.argv[1] == "volume-publisher":
        sys.argv.pop(1)
        runpy.run_module("operational.runpod_migration.event_volume", run_name="__main__")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "stage-bundle":
        sys.argv.pop(1)
        runpy.run_module("operational.runpod_migration.local_cache", run_name="__main__")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "seal-bundle":
        sys.argv.pop(1)
        runpy.run_module("operational.runpod_migration.immutable_seal", run_name="__main__")
        return
    runpy.run_path(str(runtime / "scripts/runpod_event_scientific_v1.py"), run_name="__main__")


if __name__ == "__main__":
    main()
