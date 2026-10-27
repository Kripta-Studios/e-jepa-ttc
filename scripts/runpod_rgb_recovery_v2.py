"""Add bounded quota-race retry and truthful stage heartbeats to RGB queues."""

from __future__ import annotations

import importlib.util
import json
import os
import runpy
import sys
import time
from pathlib import Path
from typing import Any


def install_queue_heartbeat(module: Any, clock: Any = time.monotonic) -> None:
    """Publish only when this worker actually enters a stage or executes its guard."""
    original = module.run_rgb_downstream

    def run(run: Path, rotation: int, **kwargs: Any) -> Any:
        root = run / "rotations" / str(rotation)
        external = kwargs.get("external_guard")
        last = float("-inf")

        def heartbeat() -> None:
            nonlocal last
            if external is not None:
                external()
            now = clock()
            if now - last >= 30:
                module._receipt(
                    root,
                    "RGB_STAGE_ACTIVE",
                    "RUNNING",
                    observation="worker entered downstream execution or resource guard",
                    optimizer_updates_inferred=False,
                )
                last = now

        heartbeat()
        return original(run, rotation, **{**kwargs, "external_guard": heartbeat})

    module.run_rgb_downstream = run


def main() -> None:
    ops = Path(__file__).resolve().parent
    sys.path.insert(0, str(ops))
    import runpod_quota_race_overlay_v1 as quota
    import runpod_resources_v3 as resources
    import runpod_rgb_path_compat_v1 as path_compat

    expected = {
        Path(
            path_compat.__file__
        ): "4bbb2d8e5a8ec8bfead7359c9e842e7c1cb9af9526c6beb027e99e0f54f097ef",
        Path(quota.__file__): "4542d1ed8c3617a7d4b8ddcd90504b5a5df0a9bcc27f5af16a9cabbcf6ba50d2",
    }
    for path, sha in expected.items():
        if path_compat.digest(path) != sha:
            raise ValueError(f"Recovery dependency differs: {path}")
    run = Path(sys.argv[sys.argv.index("--run") + 1]).resolve(strict=True)
    binding = {
        "schema": "ejepa_rgb_operational_recovery_v2",
        "status": "PASS",
        "source_sha256": path_compat.digest(Path(__file__)),
        "dependencies": {str(path): sha for path, sha in expected.items()},
        "migration_binding_sha256": path_compat.digest(run / "MIGRATION_BINDING.json"),
        "heartbeat_emitted_by_active_worker": True,
        "heartbeat_interval_seconds": 30,
        "quota_retry_scope": "GNU-du status1 ENOENT only",
        "scientific_sources_rewritten": False,
    }
    target = run / "RGB_OPERATIONAL_RECOVERY_V2.json"
    if target.exists() and json.loads(target.read_text()) != binding:
        raise ValueError("Recovery binding differs")
    if not target.exists():
        temp = target.with_name(f".{target.name}.{os.getpid()}.pending")
        with temp.open("w") as stream:
            json.dump(binding, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
    original_storage = resources.storage_available
    resources.storage_available = quota.RetryingStorageInventory(original_storage)
    original_run_path = runpy.run_path
    bootstrap = ops / "rgb_downstream/scripts/runpod_rgb_bootstrap_v1.py"

    def run_path(path: str, **kwargs: Any) -> Any:
        if Path(path).resolve() != bootstrap.resolve():
            return original_run_path(path, **kwargs)
        spec = importlib.util.spec_from_file_location("ejepa_rgb_recovery_bootstrap", bootstrap)
        if spec is None or spec.loader is None:
            raise ImportError(str(bootstrap))
        module: Any = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original_load = module.load

        def load(name: str, source: Path) -> Any:
            loaded = original_load(name, source)
            if name == "operational.runpod_migration.rgb_queue":
                install_queue_heartbeat(loaded)
            return loaded

        module.load = load
        raise SystemExit(module.main())

    runpy.run_path = run_path
    try:
        path_compat.main()
    finally:
        runpy.run_path = original_run_path
        resources.storage_available = original_storage


if __name__ == "__main__":
    main()
