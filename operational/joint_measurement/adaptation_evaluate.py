"""Complete frozen A1 paired evaluation using the established scorer, costs and exports."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from operational.range_transfer.common import digest, freeze, read, sha, write

from . import adaptation_depth as a1
from . import correspondence_evaluate as reused
from . import evaluate as primary_evaluation
from . import geometry_cost as cost
from . import geometry_evaluate as evaluation
from .bootstrap import compare
from .common import ROOT, RUN
from .compute_guard import guard_idle
from .data import SensorDataset
from .evaluate import development_decision
from .geometry_export import trace_portable
from .rgb_cost import combine_decision
from .train import committed_sources
from .verify_predictions import verify

DIRECTORY = a1.DIRECTORY
EVALUATION = DIRECTORY / "evaluation/REGISTRATION.json"
FIT, CONTROL = [a1.folder(mode).name for mode in a1.MODES]
BASELINE = "A5-P2-FROZEN"
REUSED_SOURCE_BINDING = reused.source_binding


def source_binding() -> dict:
    return {
        **REUSED_SOURCE_BINDING(),
        **{
            f"operational/joint_measurement/{name}.py": sha(
                ROOT / f"operational/joint_measurement/{name}.py"
            )
            for name in (
                "adaptation_evaluate",
                "adaptation_inference",
                "adaptation_depth",
                "geometry_export",
                "geometry_capture",
                "correspondence_inference_graph",
                "prefetched_sensor",
            )
        },
    }


def folder(mode: str, stage: int) -> Path:
    if stage != 2:
        raise ValueError("A1 has one registered TTC stage")
    return a1.folder(mode)


@contextmanager
def reused_context() -> Iterator[None]:
    """Reuse immutable B1 verification, native clocks and diagnostics with A1 bindings."""
    bindings = {
        "DIRECTORY": DIRECTORY,
        "EVALUATION": EVALUATION,
        "FIT": FIT,
        "CONTROL": CONTROL,
        "REGISTRATION": a1.REGISTRATION,
        "MODES": a1.MODES,
        "STEPS": {2: a1.UPDATES},
        "PARENT": a1.PARENT,
        "folder": folder,
        "registered": a1.registered,
        "make_model": a1.make_model,
        "source_binding": source_binding,
    }
    with ExitStack() as stack:
        for key, value in bindings.items():
            stack.enter_context(patch.object(reused, key, value))
        yield


def register() -> dict:
    """Freeze the evaluation contract without reading H predictions or incomplete endpoints."""
    scientific = a1.registered()
    committed_sources(source_binding())
    record = {
        "schema": "v18_A1_evaluation_v1",
        "sources": source_binding(),
        "scientific_registration_sha256": sha(a1.REGISTRATION),
        "fits": [FIT, CONTROL],
        "freeze_rule": (
            "Both12000 endpoints with complete journals/pointers/hashes before any P/H prediction"
        ),
        "cohorts": ["P", "H"],
        "batch_size": 4,
        "tail_size": 2,
        "development_gates": scientific["development_gates"],
        "mechanism": (
            "LAST-BLOCK versus FULL under the original registered metric rules; no seed rescue"
        ),
        "ranking": ["overall_MiD", "MiDc", "fit_id"],
        "bootstrap": (
            "2000 paired acquisition-group resamples for each arm vs A5 and LAST-BLOCK vs FULL"
        ),
        "known_scale": (
            "Same24 fixed P queries and known transforms: "
            "shared B1 correspondence parent and two A1 endpoints"
        ),
        "cost": "Original128 paired P queries,20 warmups,batch1,FP32; prepared GPU input, not E2E",
        "export": {
            "P_rows": [0, 17, 83, 159, 255],
            "adapter": "geometry_export.PortableGeometry",
            "eager_parity": "exact",
            "TorchScript_rtol": 1e-5,
            "TorchScript_atol_seconds": 1e-5,
        },
        "prediction_runtime": {
            "candidate": "Previously validated inference graph and two-pack lossless prefetch",
            "probe": (
                "Both new endpoints; P first64 plus final2; exact pixels/clocks/raw phase/TTC, "
                "intact weights/RNG, invalid clock rejected"
            ),
            "speed_gate": "Median warm graph/eager <=0.90",
            "fallback": "Preserve rejected probe; use original eager reader/model if not admitted",
        },
        "V": (
            "No V predictions in this evaluator; "
            "requires prospective H admission and fixed candidate"
        ),
        "updates": 0,
        "system_promotion": False,
        "official_comparison": False,
    }
    freeze(EVALUATION, record)
    return record


def freeze_endpoints() -> dict:
    """Require the pair, never a convenient early endpoint or an incomplete control."""
    with reused_context():
        endpoints = reused.freeze_endpoints()
    for mode in a1.MODES:
        target = a1.folder(mode)
        identity = read(target / "IDENTITY.json")
        journal = read(target / "UPDATE_JOURNAL.json")
        identity_hash = identity.pop("identity_sha256")
        if not digest(identity) == identity_hash == journal["identity_sha256"]:
            raise ValueError("A1 endpoint identity content changed")
    return endpoints


def verify_endpoint(mode: str) -> dict:
    freeze_endpoints()
    with reused_context():
        result = reused.verify_stage(mode, 2)
    policy = a1.registered()["parameter_policy"][mode]
    changed = set(result["encoder_tensors_changed"] + result["readout_tensors_changed"])
    if changed.intersection(policy["frozen"]) or not any(
        name.startswith(policy["last_block_prefix"]) for name in changed
    ):
        raise ValueError("A1 actual endpoint violates its encoder adaptation boundary")
    a1.verify_change(a1.folder(mode), mode)
    return result


@contextmanager
def predict_context(name: str) -> Iterator[None]:
    if name not in (FIT, CONTROL):
        raise ValueError("Unknown A1 endpoint")
    with reused_context(), reused.predict_context(name):
        yield


def select(cohorts: dict, timing: dict, registration: dict) -> dict:
    """Keep mechanism, absolute development admission and system promotion separate."""

    def decision(reference: dict, candidate: dict) -> dict:
        with patch.object(
            primary_evaluation, "config", lambda: {"selection": registration["development_gates"]}
        ):
            return development_decision(reference, candidate)

    paired = decision(cohorts[f"H/{CONTROL}"], cohorts[f"H/{FIT}"])
    decisions = {}
    for name in (FIT, CONTROL):
        metric = decision(cohorts[f"H/{BASELINE}"], cohorts[f"H/{name}"])
        ratio = timing["results"][name]["p50_seconds"] / timing["results"][BASELINE]["p50_seconds"]
        decisions[name] = combine_decision(
            metric, ratio, registration["development_gates"]["routes"]
        )
        decisions[name]["cost_scope"] = (
            "prepared event tensor resident on GPU through signed TTC output"
        )
    ranked = sorted(
        [name for name in decisions if decisions[name]["eligible"]],
        key=lambda name: (
            cohorts[f"H/{name}"]["overall_MiD"],
            cohorts[f"H/{name}"]["bins"]["crucial"]["group_macro_mid"],
            name,
        ),
    )[: registration["development_gates"]["max_integrations"]]
    return {
        "status": "DEVELOPMENT_CANDIDATE" if ranked else "NO_CANDIDATE_A1",
        "ranked": ranked,
        "decisions": decisions,
        "paired_mechanism": paired,
        "system_promotion": False,
        "official_comparison": False,
    }


def export(name: str) -> dict:
    """Export both retained endpoints, including a negative result, with original five examples."""
    endpoints = freeze_endpoints()["endpoints"]
    if name not in (FIT, CONTROL):
        raise ValueError("Only A1 endpoints are exported here")
    record = read(EVALUATION)
    dataset = SensorDataset("P")
    try:
        examples = []
        for index in record["export"]["P_rows"]:
            batch = dataset.batch([index])
            examples.append((batch.events, batch.delta_t_s.reshape(1, 1)))
        with predict_context(name):
            model = evaluation.load_model()
        return trace_portable(
            DIRECTORY / "exports" / name,
            model,
            examples,
            {
                "endpoint": endpoints[name],
                "batch_size": 1,
                "input_shape": [1, 2, 12, 128, 128],
                "delta_shape": [1, 1],
                "dtype": "float32",
                "domain": "signed native TTC",
                "test_rows": record["export"]["P_rows"],
                "role": "P",
                "rtol": 1e-5,
                "atol_seconds": 1e-5,
                "includes_sensor_preprocessing": False,
            },
        )
    finally:
        dataset.close()


def run() -> dict:
    """Finish the complete pair without tuning against V or reducing evaluation coverage."""
    from .adaptation_inference import predict, probe

    guard_idle()
    registration = register()
    endpoints = freeze_endpoints()
    with (
        patch.object(a1.geometry_train, "DIRECTORY", DIRECTORY),
        a1.geometry_train.exclusive_driver(),
    ):
        proofs = {a1.folder(mode).name: verify_endpoint(mode) for mode in a1.MODES}
        inference = probe()
        with reused_context():
            matching = reused.known_scale()
        cohorts, numeric, bootstrap = {}, {}, {}
        for name in (FIT, CONTROL):
            write(DIRECTORY / "verification/ENDPOINT.json", proofs[name])
            for role in ("P", "H"):
                while predict(name, role)["status"] != "COMPLETE":
                    pass
                target = DIRECTORY / "predictions" / role / name
                numeric[f"{role}/{name}"] = verify(target)
                if numeric[f"{role}/{name}"]["status"] != "PASS":
                    raise ValueError("A1 independent metric replay failed")
                cohorts[f"{role}/{name}"] = read(target / "METRICS.json")
        for role in ("P", "H"):
            base = RUN / "predictions" / role / BASELINE
            if verify(base)["status"] != "PASS":
                raise ValueError("Reused A5 cohort or metrics changed")
            cohorts[f"{role}/{BASELINE}"] = read(base / "METRICS.json")
            for candidate, reference in ((FIT, CONTROL), (FIT, BASELINE), (CONTROL, BASELINE)):
                base_root = RUN if reference == BASELINE else DIRECTORY
                bootstrap[f"{role}/{candidate}_vs_{reference}"] = compare(
                    base_root / "predictions" / role / reference / "predictions.parquet",
                    DIRECTORY / "predictions" / role / candidate / "predictions.parquet",
                    DIRECTORY / "bootstrap" / role / f"{candidate}_vs_{reference}",
                )
        with reused_context(), ExitStack() as stack:
            for key, value in {
                "DIRECTORY": DIRECTORY,
                "RUN": DIRECTORY,
                "FIT": FIT,
                "CONTROL": CONTROL,
                "REGISTRATION": a1.REGISTRATION,
                "registered": a1.registered,
                "freeze_endpoint": freeze_endpoints,
                "COST": DIRECTORY / "cost/prepared_forward",
                "sources": source_binding,
                "initialize": reused.model_for,
            }.items():
                stack.enter_context(patch.object(cost, key, value))
            with predict_context(FIT):
                while (timing := cost.measure())["status"] != "COMPLETE":
                    pass
        selection = select(cohorts, timing, registration)
        freeze(DIRECTORY / "SELECTION_FINAL.json", selection)
        exports = {name: export(name) for name in (FIT, CONTROL)}
        result = {
            "status": "A1_READY_FOR_INTERPRETATION",
            "endpoints": endpoints,
            "evaluation_registration_sha256": sha(EVALUATION),
            "endpoint_proofs": proofs,
            "known_scale_diagnostic": matching,
            "cohorts": cohorts,
            "metric_proofs": numeric,
            "bootstrap": bootstrap,
            "selection": selection,
            "prepared_cost": timing,
            "exports": exports,
            "inference_runtime": inference,
            "new_scientific_updates": 24000,
            "shared_parent_historical_updates": 14000,
            "system_promotion": False,
            "official_comparison": False,
        }
        freeze(DIRECTORY / "evidence/COMPLETE.json", result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    print(register() if parser.parse_args().register_only else run())
