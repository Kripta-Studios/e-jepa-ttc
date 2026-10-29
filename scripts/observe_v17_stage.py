"""Observe resource-guard returns without changing the guard or training functions."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", choices=("spatial_geometry", "nonlinear", "unclipped"))
    parser.add_argument("stage")
    args = parser.parse_args()
    module = importlib.import_module("operational.representation_audit." + args.module)
    guard_code = module.resources.__code__
    snapshots = []
    calls = 0

    def observer(frame: FrameType, event: str, value: object) -> None:
        nonlocal calls
        if event == "return" and frame.f_code is guard_code and isinstance(value, dict):
            calls += 1
            if calls == 1 or not value["allowed"]:
                snapshots.append(value)

    root = Path(__file__).resolve().parents[1]
    folder = root / "artifacts/representation_audit_20261012/resource_observation"
    folder.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    sys.argv = [module.__file__, args.stage]
    try:
        sys.setprofile(observer)
        module.main()
    finally:
        sys.setprofile(None)
        result = {
            "module": args.module,
            "stage": args.stage,
            "guard_calls": calls,
            "first_and_denied_snapshots": snapshots,
            "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "training_source_sha256": hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest(),
            "scope": "Python return-event observation only; no guard/result/function mutation",
            "cost": "profiling overhead is included in normal GPU wall-time charges",
        }
        (folder / (stamp + ".json")).write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "guard_calls": calls,
                    "denied": [
                        {k: v for k, v in row.items() if k != "processes"}
                        for row in snapshots
                        if not row["allowed"]
                    ],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
