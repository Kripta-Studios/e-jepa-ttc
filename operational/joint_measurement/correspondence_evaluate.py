"""B1 complete P/H paired evidence with frozen stages, original gates and portable exports."""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch

from e_jepa_ttc.data.object_event_v4 import weak_box_masks
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from operational.range_transfer.common import freeze, read, sha, write

from . import export as portable
from . import geometry_cost as cost
from . import geometry_evaluate as evaluation
from .bootstrap import compare
from .common import ROOT, RUN
from .correspondence_train import (
    DIRECTORY,
    MODES,
    PARENT,
    REGISTRATION,
    STEPS,
    folder,
    make_model,
    registered,
)
from .data import SensorDataset
from .evaluate import development_decision, freeze_primary
from .geometry_model import GeometricPairMeasurement
from .rgb_cost import combine_decision
from .scale_correspondence_diagnostic import matching, warp
from .train import committed_sources, initialize
from .verify_predictions import verify

EVALUATION = DIRECTORY / "evaluation/REGISTRATION.json"
FIT, CONTROL = [folder(mode, 2).name for mode in MODES]
BASELINE = "A5-P2-FROZEN"


def source_binding() -> dict:
    """Bind all reused evaluation/scorer/cost/export paths before held-out inference."""
    names = (
        "correspondence_evaluate",
        "geometry_evaluate",
        "geometry_cost",
        "export",
        "rgb_cost",
        "bootstrap",
        "verify_predictions",
        "evaluate",
        "scale_correspondence_diagnostic",
    )
    return {
        f"operational/joint_measurement/{name}.py": sha(
            ROOT / f"operational/joint_measurement/{name}.py"
        )
        for name in names
    }


def register() -> dict:
    """Admit only complete P/H cohorts, fixed endpoints and prospective scientific rules."""
    record = registered()
    committed_sources(source_binding())
    result = {
        "scientific_registration_sha256": sha(REGISTRATION),
        "sources": source_binding(),
        "cohorts": ["P", "H"],
        "stages": record["stages"],
        "fits": [FIT, CONTROL],
        "freeze": "Both stage2 and stage1 endpoints before any B1 H prediction",
        "development_gates": record["development_gates"],
        "mechanism": "Original development metric rules applied versus paired TTC control",
        "cost": "Paired128 fixed P queries,20warmups,FP32batch1;prepared GPU input,not E2E",
        "bootstrap": "Existing2000 acquisition-group resamples; no window-IID claim",
        "V": "Not opened by this evaluator; requires independent prospective H admission",
    }
    freeze(EVALUATION, result)
    return result


def freeze_endpoints() -> dict:
    """Verify all four stage receipts and durable checkpoint pointers before evaluation."""
    registered()
    endpoints = read(DIRECTORY / "ENDPOINTS.json")
    expected = {folder(mode, stage).name for stage in STEPS for mode in MODES}
    if set(endpoints) != expected:
        raise ValueError("B1 requires every stage endpoint, not a selected arm")
    for stage in STEPS:
        for mode in MODES:
            target = folder(mode, stage)
            receipt, journal, pointer, identity = [
                read(target / name)
                for name in (
                    "CHECKPOINT_RECEIPT.json",
                    "UPDATE_JOURNAL.json",
                    "CHECKPOINT_POINTER.json",
                    "IDENTITY.json",
                )
            ]
            if not (
                receipt["status"] == "COMPLETE"
                and receipt["scientific_endpoint"]
                and receipt["completed_updates"] == STEPS[stage]
                and journal["completed_updates"] == journal["durable_updates"] == STEPS[stage]
                and journal["pending_update_upper"] == 0
                and pointer["completed_updates"] == STEPS[stage]
                and identity["registration_sha256"] == sha(REGISTRATION)
                and journal["identity_sha256"] == receipt["identity_sha256"]
                and pointer["checkpoint_sha256"]
                == receipt["checkpoint_sha256"]
                == endpoints[target.name]
                == sha(target / "checkpoint_last.pt")
                == sha(target / "checkpoint_versions" / pointer["version"])
            ):
                raise ValueError("Incomplete or altered B1 stage endpoint")
    return {
        "endpoints": {
            FIT: endpoints[FIT],
            CONTROL: endpoints[CONTROL],
            BASELINE: freeze_primary()["endpoints"][BASELINE],
        },
        "registration_sha256": sha(REGISTRATION),
    }


def verify_stage(mode: str, stage: int) -> dict:
    """Regenerate all sampled IDs and inspect actual per-stage Adam, RNG and weight changes."""
    freeze_endpoints()
    target = folder(mode, stage)
    payload = torch.load(target / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    identity, journal = read(target / "IDENTITY.json"), read(target / "UPDATE_JOURNAL.json")
    model, parent_sha = make_model(mode, stage, synthetic=False)
    updates = STEPS[stage]
    if not (
        payload["identity"] == identity
        and identity["parent_sha256"] == parent_sha
        and payload["identity_sha256"] == journal["identity_sha256"]
        and payload["recovery_upper"] == journal["recovery_upper"]
        and payload["completed_updates"] == updates
        and payload["accumulation_index"] == 0
    ):
        raise ValueError("B1 stage identity/parent/cursor changed")
    initial, final = model.state_dict(), payload["model_state_dict"]
    if initial.keys() != final.keys() or any(
        initial[k].shape != final[k].shape
        or initial[k].dtype != final[k].dtype
        or not torch.isfinite(final[k]).all()
        for k in initial
    ):
        raise ValueError("B1 state topology or finite weights changed")
    changed = [name for name in initial if not torch.equal(initial[name], final[name])]
    encoder = [name for name in changed if name.startswith("encoder.")]
    readout = [name for name in changed if not name.startswith("encoder.")]
    if not encoder or (stage == 1 and readout) or (stage == 2 and not readout):
        raise ValueError("B1 stage trainability contract failed")
    curve, timings = payload["cursor"]["curve"], payload["cursor"]["timings"]
    if len(curve) != updates or len(timings) != updates:
        raise ValueError("B1 stage trajectory length changed")
    generator = torch.Generator().manual_seed(7)
    population = registered()["population_size"]
    for i, (row, elapsed) in enumerate(zip(curve, timings, strict=True), 1):
        ids = torch.randint(population, (32,), generator=generator).tolist()
        if (
            row["update"] != i
            or row["ids"] != ids
            or elapsed < 0
            or not all(
                math.isfinite(v) for k, v in {**row, "seconds": elapsed}.items() if k != "ids"
            )
        ):
            raise ValueError("B1 sample sequence, loss or timing changed")
    if not torch.equal(generator.get_state(), payload["sampler_generator_state"]):
        raise ValueError("B1 sampler final state changed")
    names = [
        name
        for prefix in (True, False)
        for name, parameter in model.named_parameters()
        if name.startswith("encoder.") == prefix and parameter.requires_grad
    ]
    optimizer = payload["optimizer_state_dict"]
    ids = [index for group in optimizer["param_groups"] for index in group["params"]]
    if len(ids) != len(names) or len(set(ids)) != len(ids) or set(ids) != set(optimizer["state"]):
        raise ValueError("B1 optimizer parameter coverage changed")
    for index, name in zip(ids, names, strict=True):
        state = optimizer["state"][index]
        if float(state["step"]) != updates or any(
            state[key].shape != final[name].shape or not torch.isfinite(state[key]).all()
            for key in ("exp_avg", "exp_avg_sq")
        ):
            raise ValueError("B1 optimizer steps or moments changed")
    if payload["scheduler_state_dict"]["last_epoch"] != updates:
        raise ValueError("B1 scheduler step changed")
    result = {
        "status": "PASS",
        "mode": mode,
        "stage": stage,
        "updates": updates,
        "sample_ids_regenerated": updates * 32,
        "parent_sha256": parent_sha,
        "checkpoint_sha256": sha(target / "checkpoint_last.pt"),
        "encoder_tensors_changed": encoder,
        "readout_tensors_changed": readout,
        "adam_parameter_states": len(ids),
        "adam_step": updates,
        "scheduler_step": updates,
        "sampler_final_state_exact": True,
        "recovery_upper": payload["recovery_upper"],
        "optimizer_updates_in_verification": 0,
    }
    freeze(DIRECTORY / "verification" / target.name / "ENDPOINT.json", result)
    return result


def model_for(name: str) -> torch.nn.Module:
    """Instantiate only the registered architecture; each caller loads its bound endpoint."""
    if name in (FIT, CONTROL):
        parent = initialize(BASELINE)
        if not isinstance(parent, CausalScaleTTC):
            raise TypeError("Native parent required")
        return GeometricPairMeasurement(parent)
    return initialize(name)


def known_scale() -> dict:
    """Retain the same24 fixed P queries and compare parent, both stages and both arms."""
    original = RUN / "diagnostics/known_scale_P/REGISTRATION.json"
    target = DIRECTORY / "diagnostics/known_scale_P"
    recipe = read(original)
    paths = {
        PARENT.name: PARENT,
        **{folder(mode, stage).name: folder(mode, stage) for stage in STEPS for mode in MODES},
    }
    registration = {
        "source_registration_sha256": sha(original),
        "samples": recipe["samples"],
        "scales": recipe["scales"],
        "support": recipe["support"],
        "null": recipe["null"],
        "endpoints": {name: sha(path / "checkpoint_last.pt") for name, path in paths.items()},
        "sources": source_binding(),
        "role": "P",
        "optimizer_updates": 0,
        "limit": "Same known-texture diagnostic; not real temporal correspondence or TTC accuracy",
    }
    freeze(target / "REGISTRATION.json", registration)
    if (target / "RESULT.json").exists():
        return read(target / "RESULT.json")
    models = {}
    for name, path in paths.items():
        model = model_for(FIT)
        if not isinstance(model, GeometricPairMeasurement):
            raise TypeError("B1 diagnostic requires the registered C1 architecture")
        payload = torch.load(path / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        models[name] = model.encoder.eval().requires_grad_(False)
        del payload, model
    source, measurements = SensorDataset("P"), []
    try:
        with torch.inference_mode():
            for sample in recipe["samples"]:
                row = source._row(sample["index"])
                pixels = torch.from_numpy(row["events"][-1:])
                box, _ = weak_box_masks(torch.from_numpy(row["boxes"])[None], height=128, width=128)
                weak = torch.nn.functional.avg_pool2d(box[0, -1:], 4)[0, 0] >= 0.5
                interior = torch.zeros(32, 32, dtype=torch.bool)
                interior[8:-8, 8:-8] = True
                permutation = torch.randperm(1024, generator=torch.Generator().manual_seed(7))
                for name, encoder in models.items():
                    previous = encoder(pixels)
                    for scale in recipe["scales"]:
                        current = encoder(warp(pixels, scale))
                        null = current.flatten(2)[:, :, permutation].reshape_as(current)
                        measurements.append(
                            {
                                **sample,
                                "model": name,
                                "scale": scale,
                                "supports": {
                                    support_name: {
                                        "warped": matching(previous, current, scale, support),
                                        "permuted": matching(previous, null, scale, support),
                                    }
                                    for support_name, support in (
                                        ("interior", interior),
                                        ("weak_bbox", interior & weak),
                                    )
                                },
                            }
                        )
    finally:
        source.close()
    result = {
        "status": "COMPLETE_DIAGNOSTIC",
        "registration_sha256": sha(target / "REGISTRATION.json"),
        "measurements": measurements,
        "optimizer_updates": 0,
        "TTC_accuracy_measured": False,
    }
    freeze(target / "RESULT.json", result)
    return result


@contextmanager
def predict_context(name: str) -> Iterator[None]:
    """Use unchanged geometry prediction code and scorer for the named B1 final stage."""
    with ExitStack() as stack:
        for key, value in {
            "FIT": name,
            "DIRECTORY": DIRECTORY,
            "REGISTRATION": REGISTRATION,
            "registered": registered,
            "freeze_endpoint": freeze_endpoints,
        }.items():
            stack.enter_context(patch.object(evaluation, key, value))
        yield


def run() -> dict:
    """Verify every stage, evaluate complete paired P/H, then export both retained endpoints."""
    registration = register()
    endpoints = freeze_endpoints()
    proofs = {
        folder(mode, stage).name: verify_stage(mode, stage) for stage in STEPS for mode in MODES
    }
    matching_diagnostic = known_scale()
    cohorts, numeric, bootstrap = {}, {}, {}
    for name in (FIT, CONTROL):
        # The reused engine reads this active-arm proof; immutable arm-specific copies remain.
        write(DIRECTORY / "verification/ENDPOINT.json", proofs[name])
        with predict_context(name):
            for role in ("P", "H"):
                while evaluation.predict(role)["status"] != "COMPLETE":
                    pass
                target = DIRECTORY / "predictions" / role / name
                numeric[f"{role}/{name}"] = verify(target)
                if numeric[f"{role}/{name}"]["status"] != "PASS":
                    raise ValueError("B1 independent metric replay failed")
                cohorts[f"{role}/{name}"] = read(target / "METRICS.json")
    for role in ("P", "H"):
        cohorts[f"{role}/{BASELINE}"] = read(RUN / "predictions" / role / BASELINE / "METRICS.json")
        for reference in (CONTROL, BASELINE):
            base = DIRECTORY if reference == CONTROL else RUN
            bootstrap[f"{role}/{reference}"] = compare(
                base / "predictions" / role / reference / "predictions.parquet",
                DIRECTORY / "predictions" / role / FIT / "predictions.parquet",
                DIRECTORY / "bootstrap" / role / reference,
            )
    with ExitStack() as stack:
        for key, value in {
            "DIRECTORY": DIRECTORY,
            "RUN": DIRECTORY,
            "FIT": FIT,
            "CONTROL": CONTROL,
            "REGISTRATION": REGISTRATION,
            "registered": registered,
            "freeze_endpoint": freeze_endpoints,
            "COST": DIRECTORY / "cost/prepared_forward",
            "sources": source_binding,
            "initialize": model_for,
        }.items():
            stack.enter_context(patch.object(cost, key, value))
        with predict_context(FIT):
            while (timing := cost.measure())["status"] != "COMPLETE":
                pass
    paired = development_decision(cohorts[f"H/{CONTROL}"], cohorts[f"H/{FIT}"])
    decisions = {}
    for name in (FIT, CONTROL):
        metric = development_decision(cohorts[f"H/{BASELINE}"], cohorts[f"H/{name}"])
        ratio = timing["results"][name]["p50_seconds"] / timing["results"][BASELINE]["p50_seconds"]
        decisions[name] = combine_decision(
            metric, ratio, registration["development_gates"]["routes"]
        )
    ranked = sorted(
        [name for name in decisions if decisions[name]["eligible"]],
        key=lambda name: (cohorts[f"H/{name}"]["overall_MiD"], name),
    )
    selection = {
        "status": "DEVELOPMENT_CANDIDATE" if ranked else "NO_CANDIDATE_B1",
        "ranked": ranked,
        "decisions": decisions,
        "paired_mechanism": paired,
        "system_promotion": False,
        "official_comparison": False,
    }
    freeze(DIRECTORY / "SELECTION_FINAL.json", selection)
    with (
        patch.object(portable, "RUN", DIRECTORY),
        patch.object(portable, "freeze_primary", freeze_endpoints),
        patch.object(portable, "initialize", model_for),
    ):
        exports = {name: portable.export(name) for name in (FIT, CONTROL)}
    result = {
        "status": "B1_READY_FOR_INTERPRETATION",
        "endpoints": endpoints,
        "evaluation_registration_sha256": sha(EVALUATION),
        "endpoint_proofs": proofs,
        "known_scale_diagnostic": matching_diagnostic,
        "cohorts": cohorts,
        "metric_proofs": numeric,
        "bootstrap": bootstrap,
        "selection": selection,
        "prepared_cost": timing,
        "exports": exports,
        "new_scientific_updates": 24000,
        "shared_parent_historical_updates": 12000,
        "system_promotion": False,
        "official_comparison": False,
    }
    freeze(DIRECTORY / "evidence/COMPLETE.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    print(register() if parser.parse_args().register_only else run())
