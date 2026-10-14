"""Test independent parameter-gradient destinations for repeated C1 graph replay."""

from __future__ import annotations

import copy
from typing import cast

import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from operational.range_transfer.common import freeze, read, sha, utc

from .common import ROOT, RUN, gpu_charge
from .geometry_capture import install_capture
from .geometry_capture_gradient_probe import compare_gradients
from .geometry_capture_probe import ORIGIN
from .geometry_capture_probe import OUT as PROBE
from .geometry_graph_probe import no_live_training_driver
from .geometry_model import GeometricPairMeasurement
from .qa import exact, fixture, subset
from .train import backward, committed_sources, initialize, setup

OUT = RUN / "runtime/geometry_accumulation_probe_v1"


def run() -> dict:
    """Preallocate .grad before accumulation; no optimizer, new loss or precision change."""
    sources = dict(read(PROBE / "REGISTRATION.json")["sources"])
    for name in ("geometry_accumulation_probe.py", "geometry_capture_gradient_probe.py"):
        path = "operational/joint_measurement/" + name
        sources[path] = sha(ROOT / path)
    committed_sources(sources)
    registration = {
        "sources": sources,
        "prior_gradient_diagnosis_sha256": sha(
            RUN / "runtime/geometry_capture_gradient_probe_v1/RESULT.json"
        ),
        "origin_sha256": sha(ORIGIN),
        "single_factor": "Preallocate independent zeros for parameter.grad before effective batch",
        "sample_capture_input_ownership": "Dedicated clones; never overwrite comparison inputs",
        "microbatches": 8,
        "batch": 32,
        "rtol": 1e-5,
        "atol": 1e-7,
        "optimizer_updates": 0,
        "scientific_runtime_admitted": False,
    }
    freeze(OUT / "REGISTRATION.json", registration)
    if (OUT / "RESULT.json").exists():
        return read(OUT / "RESULT.json")
    no_live_training_driver()
    with gpu_charge("C1_independent_gradient_accumulation_probe", reserve=600):
        setup(7)
        initial = GeometricPairMeasurement(cast(CausalScaleTTC, initialize("A5-P2-FROZEN")))
        payload = torch.load(ORIGIN, map_location="cpu", weights_only=False)
        initial.load_state_dict(payload["model_state_dict"], strict=True)
        del payload
        data = subset(
            fixture(128),
            torch.randint(8, (32,), generator=torch.Generator().manual_seed(7)).tolist(),
        )
        batches = [
            subset(data, list(range(i, i + 4))).to(torch.device("cuda")) for i in range(0, 32, 4)
        ]
        models = {
            name: copy.deepcopy(initial).cuda().train()
            for name in ("eager", "capture", "preallocated")
        }
        cores = []
        for name in ("capture", "preallocated"):
            cores.append(
                install_capture(
                    models[name], batches[0].events.clone(), batches[0].delta_t_s.clone()
                )
            )
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        outputs = {}
        for name, model in models.items():
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            model.zero_grad(set_to_none=True)
            if name == "preallocated":
                for p in model.parameters():
                    p.grad = torch.zeros_like(p)
            losses = backward(model, batches, torch.full((32,), 1 / 8), 8)
            outputs[name] = {
                "losses": losses,
                "gradients": {
                    key: p.grad.detach().cpu().clone() if p.grad is not None else None
                    for key, p in model.named_parameters()
                },
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            }
        comparisons = {
            name: {
                "gradients": compare_gradients(
                    outputs["eager"]["gradients"], outputs[name]["gradients"]
                ),
                "losses_exact": exact(outputs["eager"]["losses"], outputs[name]["losses"]),
                "rng_exact": all(
                    exact(outputs["eager"][key], outputs[name][key])
                    for key in ("cpu_rng", "cuda_rng")
                ),
            }
            for name in ("capture", "preallocated")
        }
        result = {
            "utc": utc(),
            "comparisons": comparisons,
            "registration_sha256": sha(OUT / "REGISTRATION.json"),
            "optimizer_updates": 0,
            "scientific_runtime_admitted": False,
        }
        freeze(OUT / "RESULT.json", result)
        return result


if __name__ == "__main__":
    result = run()
    print(
        {
            name: {
                "gradients_pass": row["gradients"]["pass"],
                "losses_exact": row["losses_exact"],
                "rng_exact": row["rng_exact"],
            }
            for name, row in result["comparisons"].items()
        }
    )
