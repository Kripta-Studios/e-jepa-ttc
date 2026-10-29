"""Load frozen V19 EVENT queue with native-Path-compatible routing V2."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

QUEUE_SHA256 = "885438c69937cce791eff59b02f7557f0ee0208e82d5044fbec37fe17a41aa00"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(name: str, path: Path) -> None:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    producer = Path("/workspace/ejepa_event_operations/runtime_v12/operational/runpod_migration")
    base = Path("/opt/ejepa_event_bundle/workspace/e-jepa-ttc-v18-joint-measurement")
    queue = root / "scripts/runpod_event_queue_v1.py"
    if digest(queue) != QUEUE_SHA256:
        raise ValueError("Frozen EVENT queue bytes differ")
    sys.path[:0] = [str(base / "src"), str(base), str(root)]
    import operational.runpod_migration as migration
    import scripts as scripts_package

    migration.__path__.insert(0, str(producer))
    scripts_package.__path__ = [
        str(root / "scripts"),
        str(producer.parents[1] / "scripts"),
        *scripts_package.__path__,
    ]
    admission_path = Path(sys.argv[sys.argv.index("--supplemental-admission") + 1])
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    os.environ.update(
        {
            "EJEPA_EVENT_SUPPLEMENT_ROOT": str(admission["root"]),
            "EJEPA_EVENT_SUPPLEMENT_ADMISSION": str(admission_path.resolve()),
            "EJEPA_EVENT_AUTHORITY_SUPPLEMENT_ROOT": "/opt/ejepa_event_authority_supplement_v2",
            "EJEPA_EVENT_AUTHORITY_SUPPLEMENT_ADMISSION": (
                "/workspace/ejepa_event_operations/authority_supplement_v2/"
                "EVENT_AUTHORITY_SUPPLEMENT_ADMISSION.json"
            ),
        }
    )
    load(
        "operational.runpod_migration.event_queue",
        root / "operational/runpod_migration/event_queue.py",
    )
    from scripts.runpod_event_directory_routing_v2 import install

    install()
    runpy.run_path(str(queue), run_name="__main__")


if __name__ == "__main__":
    main()
