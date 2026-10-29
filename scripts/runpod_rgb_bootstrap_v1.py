"""Load new RGB orchestration externally without changing the frozen bundle."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any


def load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ops = root.parent
    sys.path.insert(0, str(ops))
    import runpod_resources_v3 as resources
    import runpod_technical_overlay_v3 as technical
    from runpod_scientific_overlay_v4 import QuotaPublisher

    from operational.runpod_migration import runner
    from operational.runpod_migration.inventory import digest

    original_sources = runner.source_identity()
    sources = []
    for name in ("rgb_downstream_coverage", "rgb_queue_endpoints", "rgb_queue"):
        path = root / f"operational/runpod_migration/{name}.py"
        load(f"operational.runpod_migration.{name}", path)
        sources.append(path)
    entrypoint = root / "scripts/runpod_rgb_queue_v1.py"
    entry = load("ejepa_external_rgb_queue_entrypoint", entrypoint)
    if runner.source_identity() != original_sources:
        raise ValueError("Loading the external queue changed frozen migration sources")
    original_prepare = technical.prepare
    original_storage = resources.storage_available
    original_authority = entry._validate_authority
    publisher = None

    def prepare(bundle: Path, run: Path, budget_bytes: int) -> dict[str, Any]:
        nonlocal publisher
        setup = original_prepare(bundle, run, budget_bytes)
        if "--prepare-only" not in sys.argv:
            binding = {
                "schema": "ejepa_rgb_external_deployment_v1",
                "status": "PASS",
                "source_sha256": {
                    str(p): digest(p) for p in [Path(__file__), entrypoint, *sources]
                },
                "migration_binding_sha256": digest(run / "MIGRATION_BINDING.json"),
                "frozen_migration_sources_unchanged": True,
                "quota_refresh_seconds": 30,
                "quota_ttl_seconds": 120,
                "quota_inflight_reserve_bytes": 4 * 1024**3,
                "publisher_starts_only_when_a_head_stage_needs_resources": True,
                "checkpoint_updates": 100,
            }
            runner._write_once(run / "RGB_DEPLOYMENT_BINDING.json", binding)

            def lazy_storage(path: Path) -> tuple[int, dict[str, Any]]:
                nonlocal publisher
                if publisher is None:
                    free, report = original_storage(path)
                    publisher = QuotaPublisher(run, free, report)
                    publisher.original = original_storage
                    resources.storage_available = publisher.read
                    publisher.thread.start()
                return publisher.read(path)

            resources.storage_available = lazy_storage
        return setup

    def authority(path: Path, run: Path, rotation: int) -> tuple[datetime, dict[str, str]]:
        deadline, value = original_authority(path, run, rotation)
        value["external_deployment_binding_sha256"] = digest(run / "RGB_DEPLOYMENT_BINDING.json")
        resources.LinuxResourceMonitor.deadline_timestamp = deadline.timestamp()
        return deadline, value

    technical.prepare = prepare
    entry._validate_authority = authority
    try:
        return entry.main()
    finally:
        if publisher is not None:
            publisher.stop.set()
        resources.storage_available = original_storage
        technical.prepare = original_prepare


if __name__ == "__main__":
    raise SystemExit(main())
