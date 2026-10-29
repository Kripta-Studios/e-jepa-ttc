"""V22 EVENT queue with eager routing for modules loaded after preparation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Q = "885438c69937cce791eff59b02f7557f0ee0208e82d5044fbec37fe17a41aa00"
R = "a7901415439166412c0a5c24ed96e3fe773e7f8cbbb6c13a9a781d96f1be3808"
P = "627c3036433309f66d561ff3834195acc57e705d9ba8735e5857157d0617c5e6"
L = "28157f567dd4f2ac30b227d6b9dd7e846913f235d3bfba1366b30137f5d662b2"


def digest(p: Path) -> str:
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load(name: str, p: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise ImportError(p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    producer = Path("/workspace/ejepa_event_operations/runtime_v12/operational/runpod_migration")
    base = Path("/opt/ejepa_event_bundle/workspace/e-jepa-ttc-v18-joint-measurement")
    queue_path = root / "scripts/runpod_event_queue_v1.py"
    routing = root / "scripts/runpod_event_directory_routing_v2.py"
    pause = root / "scripts/runpod_event_queue_pause_v1.py"
    library = root / "operational/runpod_migration/event_queue.py"
    if (digest(queue_path), digest(routing), digest(pause), digest(library)) != (Q, R, P, L):
        raise ValueError("EVENT queue runtime bytes differ")
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
    admission = json.loads(admission_path.read_text())
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
    load("operational.runpod_migration.event_queue", library)
    from scripts.runpod_event_directory_routing_v2 import install as install_paths
    from scripts.runpod_event_queue_pause_v1 import install as install_pause

    install_paths()
    install_pause()
    queue = load("ejepa_event_queue_v1_v7", queue_path)
    original_install = queue._install_rgb_routing

    def eager_install(run: Path, config: dict[str, Any], mapping: Any) -> None:
        original_install(run, config, mapping)
        # The frozen configuration names the original Windows raw-data root.
        # Relocate the whole root (metadata and all H5 queries), not one loader.
        original_raw = str(config["raw_root"]).replace("\\", "/").rstrip("/")
        admitted_originals = {
            "E:/eAP_dataset",
            "/opt/ejepa_event_bundle/drives/E/eAP_dataset",
        }
        if original_raw not in admitted_originals:
            raise ValueError("EVENT raw root is not the admitted frozen source")
        coverage_path = Path(sys.argv[sys.argv.index("--raw-coverage") + 1]).resolve(strict=True)
        raw_root = coverage_path.parent.resolve(strict=True)
        coverage = json.loads(coverage_path.read_text())
        if (
            raw_root != Path("/workspace/ejepa_event_raw")
            or coverage.get("schema") != "ejepa_event_raw_coverage_v1"
            or coverage.get("status") != "VERIFIED"
            or coverage.get("files") != 41
            or coverage.get("hashes_verified") != 41
            or coverage.get("total_bytes") != 449_003_803_674
            or not coverage.get("availability_verified")
            or not (raw_root / "data/train.parquet").is_file()
        ):
            raise ValueError("EVENT raw-root relocation is not fully admitted")
        config["raw_root"] = str(raw_root)
        # run_event_downstream reloads the frozen configuration on every stage.
        # Wrap that authority loader so every consumer (metadata and H5 queries)
        # receives the same admitted relocation.
        from operational.final_refit import scientific_authority

        original_load_config = scientific_authority.load_config

        def relocated_load_config(root: Path) -> dict[str, Any]:
            loaded = original_load_config(root)
            loaded_raw = str(loaded["raw_root"]).replace("\\", "/").rstrip("/")
            if loaded_raw not in admitted_originals:
                raise ValueError("Reloaded EVENT raw root differs from frozen authority")
            loaded["raw_root"] = str(raw_root)
            return loaded

        scientific_authority.load_config = relocated_load_config

    queue._install_rgb_routing = eager_install
    original_verify = queue._verify_producer_cloud
    budget_path = Path(sys.argv[sys.argv.index("--budget") + 1]).resolve(strict=True)
    poll = float(sys.argv[sys.argv.index("--poll-seconds") + 1])
    deadline = datetime.fromisoformat(
        str(json.loads(budget_path.read_text())["execution_deadline_utc"])
    )
    if deadline.tzinfo is None:
        raise ValueError("EVENT queue deadline must be timezone-aware")

    def verify(run: Path, rotation: int, fit: str) -> None:
        endpoint = run / "rotations" / str(rotation) / "fits" / fit
        while (endpoint / "ENDPOINT_ADMISSION.json").is_file() and not (
            endpoint / "SCIENTIFIC_CLOUD_ADMISSION.json"
        ).is_file():
            if datetime.now(UTC) >= deadline:
                raise queue.EventQueuePauseError("Authorized EVENT queue deadline reached")
            time.sleep(min(poll, max(0.1, deadline.timestamp() - time.time())))
        original_verify(run, rotation, fit)

    queue._verify_producer_cloud = verify
    result = queue.main()
    if result is not None:
        raise SystemExit(result)


if __name__ == "__main__":
    main()
