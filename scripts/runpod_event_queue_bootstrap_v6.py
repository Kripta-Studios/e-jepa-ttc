"""Run V21 queue with an exact producer cloud-admission publication gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import runpy
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

V5_SHA256 = "66ed9245b5c406c815b2a2e8ee23bda95484c1fdb13cdf2a7ef1ab061ad8643e"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    source = Path(__file__).resolve()
    v5 = source.with_name("runpod_event_queue_bootstrap_v5.py")
    if digest(v5) != V5_SHA256:
        raise ValueError("Frozen EVENT queue bootstrap V5 differs")
    spec = importlib.util.spec_from_file_location("ejepa_event_queue_bootstrap_v5_v6", v5)
    if spec is None or spec.loader is None:
        raise ImportError(v5)
    wrapper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)
    original_run_path = runpy.run_path

    def gated_run_path(
        path_name: Any, init_globals: dict[str, Any] | None = None, run_name: str | None = None
    ) -> dict[str, Any]:
        path = Path(path_name)
        if path.name != "runpod_event_queue_v1.py":
            return original_run_path(path_name, init_globals, run_name)
        queue_spec = importlib.util.spec_from_file_location("ejepa_event_queue_v1_v6", path)
        if queue_spec is None or queue_spec.loader is None:
            raise ImportError(path)
        queue = importlib.util.module_from_spec(queue_spec)
        sys.modules[queue_spec.name] = queue
        queue_spec.loader.exec_module(queue)
        original_verify = queue._verify_producer_cloud
        budget_path = Path(sys.argv[sys.argv.index("--budget") + 1]).resolve(strict=True)
        poll = float(sys.argv[sys.argv.index("--poll-seconds") + 1])
        budget = json.loads(budget_path.read_text(encoding="utf-8"))
        deadline = datetime.fromisoformat(str(budget["execution_deadline_utc"]))
        if deadline.tzinfo is None:
            raise ValueError("EVENT queue budget deadline must be timezone-aware")

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
        return {}

    runpy.run_path = gated_run_path
    wrapper.main()


if __name__ == "__main__":
    main()
