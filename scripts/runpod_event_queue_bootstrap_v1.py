"""Load additive EVENT queue modules externally, leaving the sealed bundle untouched."""

from __future__ import annotations

import importlib.util
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
    _load(
        "operational.runpod_migration.event_queue",
        root / "operational/runpod_migration/event_queue.py",
    )
    runpy.run_path(str(root / "scripts/runpod_event_queue_v1.py"), run_name="__main__")


if __name__ == "__main__":
    main()
