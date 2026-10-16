"""Finish B1 with the existing, parity-checked geometry export adapter."""

from __future__ import annotations

import argparse
from unittest.mock import patch

import torch

from operational.range_transfer.common import freeze, read, sha

from . import correspondence_evaluate as b1
from . import geometry_evaluate as evaluation
from .common import ROOT, RUN
from .compute_guard import guard_idle
from .data import SensorDataset
from .geometry_export import trace_portable
from .train import committed_sources

OUT = RUN / "runtime/B1_geometry_export_v1"


def register() -> dict:
    """Bind the same endpoint and five P examples before tracing either export."""
    sources = {
        f"operational/joint_measurement/{name}.py": sha(
            ROOT / f"operational/joint_measurement/{name}.py"
        )
        for name in (
            "correspondence_export",
            "geometry_export",
            "geometry_model",
            "correspondence_evaluate",
        )
    }
    committed_sources(sources)
    record = {
        "sources": sources,
        "endpoints_sha256": sha(b1.DIRECTORY / "ENDPOINTS.json"),
        "evaluation_registration_sha256": sha(b1.EVALUATION),
        "P_rows": [0, 17, 83, 159, 255],
        "failure": "Generic export traces Python set-of-shapes guard; no usable export produced",
        "correction": "Reuse original C1 PortableGeometry and trace_portable with B1 weights",
        "numerics": "Exact eager adapter parity; reloaded TorchScript rtol1e-5/atol1e-5 seconds",
        "unchanged": "Endpoints, five original examples, shape, FP32 and tolerance",
        "optimizer_updates": 0,
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def export(name: str) -> dict:
    """Use each original endpoint, preserving the generic export's intended binding."""
    if name not in (b1.FIT, b1.CONTROL):
        raise ValueError("Only frozen B1 stage2 endpoints")
    registration = register()
    endpoints = b1.freeze_endpoints()["endpoints"]
    directory = b1.DIRECTORY / "exports" / name
    binding = {
        "endpoint": endpoints[name],
        "batch_size": 1,
        "input_shape": [1, 2, 12, 128, 128],
        "delta_shape": [1, 1],
        "dtype": "float32",
        "domain": "signed native TTC",
        "test_rows": registration["P_rows"],
        "role": "P",
        "rtol": 1e-5,
        "atol_seconds": 1e-5,
        "includes_sensor_preprocessing": False,
    }
    # The original failed attempt wrote only BINDING.json. It is preserved byte-for-byte.
    freeze(
        directory / "EXPORT_RUNTIME.json",
        {
            "registration_sha256": sha(OUT / "REGISTRATION.json"),
            "adapter": "geometry_export.PortableGeometry",
            "scientific_predictions_unchanged": True,
        },
    )
    if (directory / "BINDING.json").exists() and read(directory / "BINDING.json") != binding:
        raise ValueError("Original export contract changed")
    torch.set_num_threads(1)
    dataset = SensorDataset("P")
    try:
        examples = []
        for index in registration["P_rows"]:
            batch = dataset.batch([index])
            examples.append((batch.events, batch.delta_t_s.reshape(1, 1)))
        with b1.predict_context(name):
            model = evaluation.load_model()
        return trace_portable(directory, model, examples, binding)
    finally:
        dataset.close()


def run() -> dict:
    """Resume the frozen evaluator; completed predictions and costs are reused."""
    guard_idle()
    register()
    with patch.object(b1.portable, "export", export):
        return b1.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "run"))
    mode = parser.parse_args().mode
    print(register() if mode == "register" else run())
