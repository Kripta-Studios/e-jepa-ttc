"""Full-state resume admission for the C1 capture candidate; synthetic updates only."""

from __future__ import annotations

import argparse
from unittest.mock import patch

import numpy as np
import torch

from operational.range_transfer.common import freeze, read, sha, utc

from . import geometry_train
from .common import ROOT, RUN, accounting, config, gpu_charge
from .geometry_capture import install_capture
from .geometry_capture_probe import OUT as PROBE
from .geometry_graph_probe import no_live_training_driver
from .geometry_model import GeometricPairMeasurement
from .qa import exact
from .resume_fast_proof import close
from .train import committed_sources

OUT = RUN / "qa/geometry_capture_v1"
STATE_KEYS = (
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "sampler_generator_state",
    "torch_rng_state",
    "cuda_rng_state_all",
    "numpy_random_state",
    "python_random_state",
)


class CapturedGeometry(GeometricPairMeasurement):
    """Capture lazily after the original checkpoint has restored every training state."""

    def forward(self, inputs: torch.Tensor, delta_t_s: torch.Tensor) -> dict[str, torch.Tensor]:
        """Retain the graph in the installed callable's closure, outside state_dict."""
        install_capture(self, inputs, delta_t_s)
        return self.forward(inputs, delta_t_s)


def register() -> dict:
    """Freeze thirty technical updates, exact graph resume and eager numerical comparison."""
    sources = dict(read(PROBE / "REGISTRATION.json")["sources"])
    own = "operational/joint_measurement/geometry_capture_qa.py"
    sources[own] = sha(ROOT / own)
    committed_sources(sources)
    record = {
        "sources": sources,
        "probe_registration_sha256": sha(PROBE / "REGISTRATION.json"),
        "C1_registration_sha256": sha(geometry_train.REGISTRATION),
        "scientific_updates": 0,
        "physical_technical_updates": 30,
        "population": "synthetic8_128x128",
        "trajectories": {"eager": [10], "continuous": [10], "split": [5, 10]},
        "eager_numerical_rtol": 1e-5,
        "eager_numerical_atol": 1e-7,
        "exact_resume_state_keys": list(STATE_KEYS) + ["cursor.curve"],
        "timings_excluded_from_equality": True,
        "capture_after_full_checkpoint_restore": True,
        "scope": "Synthetic model/AdamW/scheduler/sampler/RNG/dropout exact-resume proof",
        "requires_before_installation": "Passing zero-update probe and this full-state proof",
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def run() -> dict:
    """Do not spend QA updates or take the GPU unless the preceding probe passed."""
    record = register()
    if (OUT / "RESULT.json").exists():
        return read(OUT / "RESULT.json")
    proof = read(PROBE / "RESULT.json")
    if proof["status"] != "CANDIDATE_FOR_RESUME_QA" or proof["registration_sha256"] != sha(
        PROBE / "REGISTRATION.json"
    ):
        raise ValueError("Complete capture must pass its zero-update probe before resume QA")
    no_live_training_driver()
    if accounting()["technical"] + 30 > config()["caps"]["technical"]:
        raise ValueError("Insufficient remaining synthetic update budget")
    with gpu_charge("C1_complete_capture_resume_QA", reserve=900) as deadline:
        for mode, steps in record["trajectories"].items():
            model_type = GeometricPairMeasurement if mode == "eager" else CapturedGeometry
            with patch.object(geometry_train, "GeometricPairMeasurement", model_type):
                for limit in steps:
                    result = geometry_train.segment(
                        OUT / mode,
                        geometry_train.SyntheticSensors(),
                        np.full(8, 1 / 8),
                        limit=limit,
                        deadline=deadline,
                        device="cuda",
                        synthetic=True,
                    )
                    if result["completed_updates"] != limit:
                        raise RuntimeError("QA scope ended; retain and resume its checkpoints")
    payloads = {
        mode: torch.load(OUT / mode / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        for mode in record["trajectories"]
    }
    checks = {key: exact(payloads["continuous"][key], payloads["split"][key]) for key in STATE_KEYS}
    checks["curve"] = exact(
        payloads["continuous"]["cursor"]["curve"], payloads["split"]["cursor"]["curve"]
    )
    comparison = {
        key: close(
            payloads["eager"][key],
            payloads["continuous"][key],
            rtol=record["eager_numerical_rtol"],
            atol=record["eager_numerical_atol"],
        )
        for key in STATE_KEYS
    }
    comparison["curve"] = close(
        payloads["eager"]["cursor"]["curve"],
        payloads["continuous"]["cursor"]["curve"],
        rtol=record["eager_numerical_rtol"],
        atol=record["eager_numerical_atol"],
    )
    result = {
        "utc": utc(),
        "status": "PASS" if all(checks.values()) and all(comparison.values()) else "FAIL",
        "registration_sha256": sha(OUT / "REGISTRATION.json"),
        "probe_result_sha256": sha(PROBE / "RESULT.json"),
        "exact_graph_resume": checks,
        "eager_numerical_comparison": comparison,
        "physical_technical_updates": 30,
        "scientific_updates": 0,
        "remaining_before_installation": "Durable scientific origin and full-state restore witness",
    }
    freeze(OUT / "RESULT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    print(register() if parser.parse_args().register_only else run())
