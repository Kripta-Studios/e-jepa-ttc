"""Prospective full-state CUDA resume proof for the fusion engine, not a scientific fit."""

from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
import torch

from operational.range_transfer.common import freeze, read, sha, utc

from .common import ROOT, RUN, accounting, config, gpu_charge, require_resources
from .fusion_engine import FusionRecipe, segment, sources
from .fusion_engine_qa import fixture
from .geometry_graph_probe import no_live_training_driver
from .qa import exact
from .rgb_run import require_authorization
from .train import committed_sources

OUT = RUN / "qa/fusion_engine/cuda_v2"
PREVIOUS = RUN / "qa/fusion_engine/cuda"
CPU = RUN / "qa/fusion_engine/cpu"
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


def register() -> dict:
    """Bind the existing CPU proof and the two CUDA trajectories before reserving GPU."""
    proof = read(CPU / "PASS.json")
    cpu_registration = read(CPU / "REGISTRATION.json")
    if (
        proof["status"] != "PASS"
        or proof["registration_sha256"] != sha(CPU / "REGISTRATION.json")
        or set(proof["checks"]) != {"TRUE", "ZERO"}
        or any(
            set(checks) != {*STATE_KEYS, "sample_ids_losses_norms"} or not all(checks.values())
            for checks in proof["checks"].values()
        )
    ):
        raise ValueError("Complete matching CPU fusion resume proof required")
    for path, expected in cpu_registration["sources"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("CPU-admitted fusion source changed")
    binding_sources = {
        **sources(),
        **{
            f"operational/joint_measurement/{name}.py": sha(
                ROOT / "operational/joint_measurement" / f"{name}.py"
            )
            for name in ("fusion_engine_qa", "fusion_cuda_qa", "geometry_graph_probe")
        },
    }
    committed_sources(binding_sources)
    recipes = {
        mode: asdict(FusionRecipe(7, mode, 10, 8, 16, 0.03, 1e-3, 0.01, 1.0))
        for mode in ("TRUE", "ZERO")
    }
    record = {
        "supersedes_unexecuted_registration_sha256": sha(PREVIOUS / "REGISTRATION.json"),
        "revision_reason": "Ownership guard now also recognizes the admitted captured C1 driver",
        "sources": binding_sources,
        "cpu_proof_sha256": sha(CPU / "PASS.json"),
        "recipes": recipes,
        "population": "synthetic8_with_3_missing_RGB_rows_not_scientific_H",
        "device": "cuda:0",
        "precision": "FP32_no_TF32",
        "physical_updates": 40,
        "paths": "TRUE/ZERO x continuous10 versus split5+restore5",
        "checks": [*STATE_KEYS, "sample_ids_losses_norms"],
        "exclusive_driver_guard": "Wait until RGB and C1 training drivers are terminal",
        "scientific_recipe_registered": False,
        "scientific_updates": 0,
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def run() -> dict:
    """Use the original engine on resident CUDA inputs with charged synthetic updates."""
    record = register()
    if (OUT / "FAIL.json").exists():
        raise ValueError("Retained fusion CUDA resume failure requires review")
    if (OUT / "PASS.json").exists():
        result = read(OUT / "PASS.json")
        if result["registration_sha256"] != sha(OUT / "REGISTRATION.json"):
            raise ValueError("CUDA fusion proof belongs to another registration")
        return result
    # Guard before CUDA allocation or a GPU reservation, including inter-scope gaps.
    no_live_training_driver()
    require_authorization()
    require_resources(starting=True)
    if accounting()["technical"] + 40 > config()["caps"]["technical"]:
        raise RuntimeError("Insufficient technical reserve for fusion CUDA QA")
    with gpu_charge("fusion_synthetic_cuda_resume_qa", reserve=300) as deadline:
        bank = fixture()
        bank.device = torch.device(record["device"])
        bank.inputs = {key: value.to(bank.device) for key, value in bank.inputs.items()}
        target = np.array([0.6, 2.5, -3, 7, -4, 1, 5, -8], np.float32)
        masses = np.full(8, 1 / 8, np.float64)
        results = {}
        for mode, values in record["recipes"].items():
            for trajectory, limit in (("continuous", 10), ("split", 5), ("split", 10)):
                receipt = segment(
                    OUT / mode / trajectory,
                    bank,
                    target,
                    masses,
                    FusionRecipe(**values),
                    binding={"synthetic_cuda_qa_registration": sha(OUT / "REGISTRATION.json")},
                    limit=limit,
                    deadline=deadline,
                )
                if receipt["completed_updates"] < limit:
                    raise RuntimeError("Fusion CUDA QA resource pause; retain and resume state")
            payloads = [
                torch.load(
                    OUT / mode / path / "checkpoint_last.pt", map_location="cpu", weights_only=False
                )
                for path in ("continuous", "split")
            ]
            checks = {key: exact(payloads[0][key], payloads[1][key]) for key in STATE_KEYS}
            checks["sample_ids_losses_norms"] = exact(
                payloads[0]["cursor"]["curve"], payloads[1]["cursor"]["curve"]
            )
            results[mode] = checks
        passed = all(all(checks.values()) for checks in results.values())
    result = {
        "status": "PASS" if passed else "FAIL",
        "checks": results,
        "registration_sha256": sha(OUT / "REGISTRATION.json"),
        "physical_updates": 40,
        "scientific_updates": 0,
        "utc": utc(),
        "scope": "Synthetic fusion-head exact CUDA resume; no scientific parent selection",
    }
    freeze(OUT / f"{result['status']}.json", result)
    if not passed:
        raise AssertionError("CUDA fusion full-state resume differs")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    print(register() if args.register_only else run())
