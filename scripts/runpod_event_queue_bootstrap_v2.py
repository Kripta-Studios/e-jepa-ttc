"""Load producer and downstream extensions without modifying the sealed bundle."""

from __future__ import annotations

import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path


def _load(name: str, path: Path) -> None:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load external EVENT module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    producer = Path("/workspace/ejepa_event_operations/runtime_v12/operational/runpod_migration")
    base = Path("/opt/ejepa_event_bundle/workspace/e-jepa-ttc-v18-joint-measurement")
    sys.path.insert(0, str(base / "src"))
    sys.path.insert(0, str(base))
    import operational.runpod_migration as migration
    import scripts as scripts_package

    if not producer.is_dir():
        raise RuntimeError("Producer event runtime extension is absent")
    migration.__path__.insert(0, str(producer))
    producer_scripts = producer.parents[1] / "scripts"
    if not producer_scripts.is_dir():
        raise RuntimeError("Producer scripts extension is absent")
    scripts_package.__path__ = [str(producer_scripts), *list(scripts_package.__path__)]
    try:
        admission_path = Path(sys.argv[sys.argv.index("--supplemental-admission") + 1])
    except (ValueError, IndexError) as error:
        raise ValueError("Queue requires --supplemental-admission") from error
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    os.environ["EJEPA_EVENT_SUPPLEMENT_ROOT"] = str(admission["root"])
    os.environ["EJEPA_EVENT_SUPPLEMENT_ADMISSION"] = str(admission_path.resolve())
    os.environ["EJEPA_EVENT_AUTHORITY_SUPPLEMENT_ROOT"] = "/opt/ejepa_event_authority_supplement_v2"
    os.environ["EJEPA_EVENT_AUTHORITY_SUPPLEMENT_ADMISSION"] = (
        "/workspace/ejepa_event_operations/authority_supplement_v2/"
        "EVENT_AUTHORITY_SUPPLEMENT_ADMISSION.json"
    )
    _load(
        "operational.runpod_migration.event_queue",
        root / "operational/runpod_migration/event_queue.py",
    )
    runpy.run_path(str(root / "scripts/runpod_event_queue_v1.py"), run_name="__main__")


if __name__ == "__main__":
    main()
