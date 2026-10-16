"""Apply the exact control-plane exception to already registered cost diagnostics."""

from __future__ import annotations

import argparse
from importlib import import_module
from unittest.mock import patch

from operational.range_transfer.common import freeze, sha, utc, write

from .common import ROOT, RUN
from .compute_guard import guard_idle
from .train import committed_sources

TARGETS = {
    "garl": ("garl_route_cost", "cost/garl_routes", "measure"),
    "shape": ("fusion_cost_shape_probe", "fusion/cost/shape_diagnostic_v1", "run"),
}
OUT = RUN / "runtime/cost_guard_overlay_v1"


def register() -> dict:
    """Bind unchanged numerical work and the narrow MCP-service guard exception."""
    sources = {
        f"operational/joint_measurement/{name}.py": sha(
            ROOT / f"operational/joint_measurement/{name}.py"
        )
        for name in (
            "cost_guard_overlay",
            "compute_guard",
            "garl_route_cost",
            "fusion_cost_shape_probe",
        )
    }
    committed_sources(sources)
    record = {
        "sources": sources,
        "registrations": {
            name: sha(RUN / path / "REGISTRATION.json") for name, (_, path, _) in TARGETS.items()
        },
        "change": "Allow exact mlflow.exe mcp run service, as in admitted graph runtime",
        "preserved": [
            "GPU owner lock",
            "project compute exclusion",
            "resource guards",
            "Garl 6GiB minimum",
            "all fixed inputs and numeric tolerances",
            "paired route timing and cost accounting",
        ],
        "optimizer_updates": 0,
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def run(name: str) -> dict:
    """Run one existing diagnostic only after all project compute has stopped."""
    if name not in TARGETS:
        raise ValueError("Unknown registered cost task")
    guard_idle()
    register()
    module, path, entrypoint = TARGETS[name]
    target = import_module("operational.joint_measurement." + module)
    freeze(
        RUN / path / "GUARD_RUNTIME.json",
        {
            "registration_sha256": sha(OUT / "REGISTRATION.json"),
            "numerical_recipe_unchanged": True,
        },
    )
    with patch.object(target, "guard_idle", guard_idle):
        result = getattr(target, entrypoint)()
    write(
        OUT / f"{name.upper()}_LATEST.json",
        {
            "utc": utc(),
            "status": result["status"],
            "registration_sha256": sha(OUT / "REGISTRATION.json"),
            "optimizer_updates": 0,
        },
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", *TARGETS))
    mode = parser.parse_args().mode
    print(register() if mode == "register" else run(mode))
