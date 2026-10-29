"""Load V18 EVENT downstream with additive split-directory routing."""

from __future__ import annotations

import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

QUEUE_V1_SHA256 = "885438c69937cce791eff59b02f7557f0ee0208e82d5044fbec37fe17a41aa00"


def _digest(path: Path) -> str:
    import hashlib

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
    queue = root / "scripts/runpod_event_queue_v1.py"
    if _digest(queue) != QUEUE_V1_SHA256:
        raise ValueError("Frozen EVENT queue V18 bytes differ")
    sys.path.insert(0, str(base / "src"))
    sys.path.insert(0, str(base))
    sys.path.insert(0, str(root))
    import operational.runpod_migration as migration
    import scripts as scripts_package

    if not producer.is_dir():
        raise RuntimeError("Producer event runtime extension is absent")
    migration.__path__.insert(0, str(producer))
    producer_scripts = producer.parents[1] / "scripts"
    if not producer_scripts.is_dir():
        raise RuntimeError("Producer scripts extension is absent")
    scripts_package.__path__ = [
        str(root / "scripts"),
        str(producer_scripts),
        *scripts_package.__path__,
    ]
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
    from scripts.runpod_event_directory_routing_v1 import install

    install()
    runpy.run_path(str(queue), run_name="__main__")


if __name__ == "__main__":
    main()
