"""Quantify C1 capture gradient differences and test ownership before accumulation."""

from __future__ import annotations

import copy
from typing import cast

import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from operational.range_transfer.common import freeze, read, sha, utc

from .common import ROOT, RUN, gpu_charge
from .geometry_capture import install_capture
from .geometry_capture_probe import ORIGIN
from .geometry_capture_probe import OUT as PROBE
from .geometry_graph_probe import no_live_training_driver
from .geometry_model import GeometricPairMeasurement
from .qa import exact, fixture, subset
from .train import backward, committed_sources, initialize, setup

OUT = RUN / "runtime/geometry_capture_gradient_probe_v1"


def register() -> dict:
    """Keep numerical tolerances; test whether returned gradient buffers need ownership."""
    prior = read(PROBE / "RESULT.json")
    if prior["status"] != "REJECTED" or prior["checks"]["gradients"]:
        raise ValueError("This diagnosis requires the retained gradient discrepancy")
    sources = dict(read(PROBE / "REGISTRATION.json")["sources"])
    own = "operational/joint_measurement/geometry_capture_gradient_probe.py"
    sources[own] = sha(ROOT / own)
    committed_sources(sources)
    record = {
        "sources": sources,
        "prior_probe_sha256": sha(PROBE / "RESULT.json"),
        "origin_sha256": sha(ORIGIN),
        "variants": ["eager", "capture", "capture_owned_gradients"],
        "microbatches_per_comparison": [1, 8],
        "one_factor": "After capture, clone parameter gradients in a hook before accumulation",
        "population": "same seeded synthetic8_128x128",
        "numerical_rtol": 1e-5,
        "numerical_atol": 1e-7,
        "optimizer_updates": 0,
        "scientific_runtime_admitted": False,
        "scope": "Gradient diagnosis, not optimization acceptance or scientific training",
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def compare_gradients(left: dict, right: dict) -> dict:
    """Report magnitude and affected tensors without relaxing the registered gate."""
    rows = {}
    for key, a in left.items():
        b = right[key]
        if a is None or b is None:
            rows[key] = {"pass": a is None and b is None}
            continue
        difference = (a - b).abs()
        tolerance = 1e-7 + 1e-5 * b.abs()
        rows[key] = {
            "pass": bool(
                torch.isfinite(a).all()
                and torch.isfinite(b).all()
                and (difference <= tolerance).all()
            ),
            "max_abs": float(difference.max()),
            "reference_max_abs": float(a.abs().max()),
            "relative_l2": float(
                torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a).clamp_min(1e-30)
            ),
            "violations": int((difference > tolerance).sum()),
            "elements": a.numel(),
        }
    return {"pass": all(row["pass"] for row in rows.values()), "tensors": rows}


def run() -> dict:
    """Replay the same gradients with no optimizer and no changes to scientific files."""
    register()
    if (OUT / "RESULT.json").exists():
        return read(OUT / "RESULT.json")
    no_live_training_driver()
    with gpu_charge("C1_capture_gradient_ownership_probe", reserve=600):
        setup(7)
        parent = cast(CausalScaleTTC, initialize("A5-P2-FROZEN"))
        initial = GeometricPairMeasurement(parent)
        payload = torch.load(ORIGIN, map_location="cpu", weights_only=False)
        initial.load_state_dict(payload["model_state_dict"], strict=True)
        del payload, parent
        data = subset(
            fixture(128),
            torch.randint(8, (32,), generator=torch.Generator().manual_seed(7)).tolist(),
        )
        batches = [
            subset(data, list(range(i, i + 4))).to(torch.device("cuda")) for i in range(0, 32, 4)
        ]
        models = {
            name: copy.deepcopy(initial).cuda().train()
            for name in ("eager", "capture", "capture_owned_gradients")
        }
        cores = []
        for name in ("capture", "capture_owned_gradients"):
            cores.append(install_capture(models[name], batches[0].events, batches[0].delta_t_s))
            if name == "capture_owned_gradients":
                for parameter in models[name].parameters():
                    parameter.register_hook(lambda gradient: gradient.clone())
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        comparisons = {}
        for count in (1, 8):
            outputs = {}
            for name, model in models.items():
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state_all(cuda_rng)
                model.zero_grad(set_to_none=True)
                losses = backward(model, batches[:count], torch.full((4 * count,), 1 / 8), 8)
                outputs[name] = {
                    "losses": losses,
                    "gradients": {
                        key: p.grad.detach().cpu().clone() if p.grad is not None else None
                        for key, p in model.named_parameters()
                    },
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                }
            comparisons[str(count)] = {
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
                for name in ("capture", "capture_owned_gradients")
            }
        result = {
            "utc": utc(),
            "registration_sha256": sha(OUT / "REGISTRATION.json"),
            "comparisons": comparisons,
            "optimizer_updates": 0,
            "scientific_runtime_admitted": False,
        }
        freeze(OUT / "RESULT.json", result)
        return result


if __name__ == "__main__":
    result = run()
    print(
        {
            count: {
                name: {
                    "gradients_pass": v["gradients"]["pass"],
                    "rng_exact": v["rng_exact"],
                    "losses_exact": v["losses_exact"],
                }
                for name, v in rows.items()
            }
            for count, rows in result["comparisons"].items()
        }
    )
