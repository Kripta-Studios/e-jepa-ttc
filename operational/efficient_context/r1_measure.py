"""R1 profiler with separately audited model startup and every family load."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import frozen_runtime, r1_profile
from .common import ROOT, Campaign, atomic_json, digest


def main() -> int:
    """Retain original paired request timings and account excluded initialization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/efficient_context_20261004/r1_20261008"
    )
    output = parser.parse_args().output.resolve()
    events = output / "measurement/runtime_initialization"
    original = frozen_runtime.FrozenRuntime

    class AuditedRuntime(original):
        """Add immutable timing receipts around the frozen runtime boundaries."""

        def __init__(self, c: Campaign, protocol: dict) -> None:
            begin = time.perf_counter()
            super().__init__(c, protocol)
            atomic_json(
                events / f"startup_{time.time_ns()}.json",
                dict(
                    kind="head_and_contract_initialization",
                    milliseconds=(time.perf_counter() - begin) * 1000,
                    excluded_from_request_latency=True,
                    optimizer_updates=0,
                ),
            )

        def load(self, query: dict) -> None:
            """Record each actual family load, including repeated chronological visits."""
            if self.family == query["family_id"]:
                return
            begin = time.perf_counter()
            super().load(query)
            atomic_json(
                events / f"family_{time.time_ns()}.json",
                dict(
                    kind="producer_family_initialization",
                    family=query["family_id"],
                    query=query["sample_token"],
                    milliseconds=(time.perf_counter() - begin) * 1000,
                    excluded_from_request_latency=True,
                    optimizer_updates=0,
                ),
            )

    atomic_json(
        output / "measurement/INITIALIZATION_AUDIT.json",
        dict(
            wrapper_sha256=digest(Path(__file__)),
            request_runner_sha256=digest(Path(r1_profile.__file__)),
            model_initialization="each initialization recorded separately, excluded from requests",
            optimizer_updates=0,
        ),
    )
    frozen_runtime.FrozenRuntime = AuditedRuntime
    try:
        r1_profile.run(
            argparse.Namespace(
                protocol=ROOT / "configs/campaign/efficient_context_v1.json",
                output=output,
                capacity_mib=64,
                cpu_only=False,
            )
        )
    finally:
        frozen_runtime.FrozenRuntime = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
