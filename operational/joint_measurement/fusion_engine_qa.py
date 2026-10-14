"""Accounted CPU-only resume admission of the fusion engine, using synthetic features."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from operational.range_transfer.common import digest, freeze, read, sha, utc

from .common import RUN, accounting, config, require_resources
from .fusion_data import FusionTensorBank
from .fusion_engine import FusionRecipe, segment, sources
from .fusion_features import FeatureBlock, FeatureIdentity
from .qa import exact
from .train import committed_sources


def fixture() -> FusionTensorBank:
    """Eight synthetic queries, separate sensor clocks, and three missing RGB rows."""
    generator = np.random.default_rng(20261014)
    blocks = []
    for modality in ("rgb", "event"):
        features = generator.normal(size=(8, 6)).astype(np.float32)
        prediction = np.array([0.8, 2, -4, 8, -3, 1.5, 6, -6], dtype=np.float32)
        available = np.ones(8, dtype=np.bool_)
        times = np.stack((np.arange(8) * 100000, np.arange(8) * 100000 + 20000), 1)
        times = times.astype(np.int64)
        if modality == "rgb":
            available[[1, 4, 6]] = False
            features[~available] = np.nan
            prediction[~available] = np.nan
            times[~available] = 0
        else:
            times[:, 1] += 30000
        blocks.append(
            FeatureBlock(
                FeatureIdentity(modality, "a" * 64, "b" * 64, "c" * 64),
                "H",
                "d" * 64,
                np.array([f"synthetic-{i}" for i in range(8)]),
                features,
                prediction,
                available,
                times,
            )
        )
    return FusionTensorBank(
        *blocks,
        *(block.fit_h_normalizer() for block in blocks),
        role="H",
        manifest_sha256="d" * 64,
        sample_tokens=blocks[0].sample_tokens,
    )


def run() -> dict:
    """Freeze QA-only recipes before forty updates; compare nine restored states per arm."""
    if torch.cuda.is_initialized():
        raise ValueError("Run fusion CPU QA in a separate CPU-only process")
    root = RUN / "qa/fusion_engine/cpu"
    binding_sources = {
        **sources(),
        "operational/joint_measurement/fusion_engine_qa.py": sha(Path(__file__)),
    }
    commit = committed_sources(binding_sources)
    recipes = {
        mode: FusionRecipe(7, mode, 10, 8, 16, 0.03, 1e-3, 0.01, 1.0) for mode in ("TRUE", "ZERO")
    }
    registration = {
        "population": "synthetic8_not_scientific_H",
        "sources": binding_sources,
        "recipes": {mode: asdict(recipe) for mode, recipe in recipes.items()},
        "physical_updates": 40,
        "paths": "TRUE/ZERO x continuous10 versus split5+restore5",
        "scientific_recipe_registered": False,
        "cuda": False,
    }
    freeze(root / "REGISTRATION.json", registration)
    if (root / "FAIL.json").exists():
        raise ValueError("Retained fusion QA failure requires investigation")
    if (root / "PASS.json").exists():
        result = read(root / "PASS.json")
        if result["registration_sha256"] != sha(root / "REGISTRATION.json"):
            raise ValueError("Fusion QA binding changed")
        return result
    if accounting()["technical"] + 40 > config()["caps"]["technical"]:
        raise RuntimeError("Insufficient technical reserve for fusion resume QA")
    require_resources(starting=True)
    bank = fixture()
    target = np.array([0.6, 2.5, -3, 7, -4, 1, 5, -8], np.float32)
    masses = np.full(8, 1 / 8, np.float64)
    results = {}
    for mode, recipe in recipes.items():
        for trajectory, limit in (("continuous", 10), ("split", 5), ("split", 10)):
            receipt = segment(
                root / mode / trajectory,
                bank,
                target,
                masses,
                recipe,
                binding={"synthetic_qa_registration": digest(registration)},
                limit=limit,
                deadline=float("inf"),
            )
            if receipt["completed_updates"] < limit:
                raise RuntimeError("Fusion CPU QA resource pause; resume its durable state")
        payloads = [
            torch.load(
                root / mode / trajectory / "checkpoint_last.pt",
                map_location="cpu",
                weights_only=False,
            )
            for trajectory in ("continuous", "split")
        ]
        checks = {
            key: exact(payloads[0][key], payloads[1][key])
            for key in (
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "sampler_generator_state",
                "torch_rng_state",
                "cuda_rng_state_all",
                "numpy_random_state",
                "python_random_state",
            )
        }
        checks["sample_ids_losses_norms"] = exact(
            payloads[0]["cursor"]["curve"], payloads[1]["cursor"]["curve"]
        )
        results[mode] = checks
    passed = (
        all(all(checks.values()) for checks in results.values()) and not torch.cuda.is_initialized()
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "checks": results,
        "registration_sha256": sha(root / "REGISTRATION.json"),
        "git_commit": commit,
        "physical_updates": 40,
        "cuda_initialized": torch.cuda.is_initialized(),
        "scientific_updates": 0,
        "utc": utc(),
    }
    freeze(root / f"{result['status']}.json", result)
    if not passed:
        raise AssertionError("Fusion CPU full-state resume differs")
    return result


if __name__ == "__main__":
    print(run())
