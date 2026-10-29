"""Complete family-D evidence through the unchanged cohort, scorer and cost engines."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from e_jepa_ttc.joint_measurement.model import PairMeasurement
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from operational.range_transfer.common import freeze, read, sha, write

from . import export as portable
from . import geometry_cost as cost
from . import geometry_evaluate as evaluation
from .bootstrap import compare
from .capacity_model import WidePairMeasurement
from .capacity_train import CONTROL, DIRECTORY, FIT, REGISTRATION, registered
from .common import ROOT, RUN
from .rgb_cost import combine_decision
from .train import committed_sources, initialize
from .verify_predictions import verify

EVALUATION = DIRECTORY / "evaluation/REGISTRATION.json"


def source_binding() -> dict:
    """Include the adapter, actual capacity model and reused evaluators."""
    return {
        f"operational/joint_measurement/{name}.py": sha(
            ROOT / f"operational/joint_measurement/{name}.py"
        )
        for name in (
            "capacity_evaluate",
            "capacity_model",
            "geometry_evaluate",
            "geometry_cost",
            "export",
            "rgb_cost",
            "bootstrap",
            "verify_predictions",
        )
    }


def register() -> dict:
    """Bind evidence execution before any D1 held-out inference, without reading outcomes."""
    record = registered()
    committed_sources(source_binding())
    result = {
        "scientific_registration_sha256": sha(REGISTRATION),
        "sources": source_binding(),
        "development_gates": record["development_gates"],
        "cohorts": ["P", "H"],
        "endpoint": 12000,
        "selection": "Original A5-P2 band/sign protections and metric/cost routes; no V rescue",
        "reused_code": (
            "Existing geometry evaluation/cost engine with explicitly scoped D1 model "
            "and paths; no geometry model is used for D1 predictions"
        ),
        "scope": "Prepared forward cost is not raw-sensor E2E; system promotion remains separate",
    }
    freeze(EVALUATION, result)
    return result


def capacity_initialize(name: str) -> CausalScaleTTC | PairMeasurement:
    """Build the registered capacity model only for its own export route."""
    if name != FIT:
        return initialize(name)
    parent = initialize("A5-P2-FROZEN")
    if not isinstance(parent, CausalScaleTTC):
        raise TypeError("Native initialization required")
    return WidePairMeasurement(parent)


@contextmanager
def context() -> Iterator[None]:
    """Keep all D1 writes separate while retaining the existing admission and scorer logic."""
    registered()
    with ExitStack() as stack:
        for module in (evaluation, cost):
            for name, value in {
                "DIRECTORY": DIRECTORY,
                "FIT": FIT,
                "CONTROL": CONTROL,
                "REGISTRATION": REGISTRATION,
                "registered": registered,
            }.items():
                stack.enter_context(patch.object(module, name, value))
        stack.enter_context(
            patch.object(evaluation, "GeometricPairMeasurement", WidePairMeasurement)
        )
        stack.enter_context(patch.object(cost, "COST", DIRECTORY / "cost/prepared_forward"))
        stack.enter_context(patch.object(cost, "sources", source_binding))
        stack.enter_context(patch.object(portable, "RUN", DIRECTORY))
        stack.enter_context(patch.object(portable, "freeze_primary", evaluation.freeze_endpoint))
        stack.enter_context(patch.object(portable, "initialize", capacity_initialize))
        yield


def run() -> dict:
    """Freeze, verify all384k IDs, evaluate full P/H, compare, time and export."""
    registration = register()
    with context():
        endpoints = evaluation.freeze_endpoint()
        proof = evaluation.verify_endpoint()
        proof["scope"] = (
            "D1 predictor-width endpoint: actual sampler, Adam and encoder; no inference"
        )
        write(DIRECTORY / "verification/ENDPOINT.json", proof)
        for role in ("P", "H"):
            while evaluation.predict(role)["status"] != "COMPLETE":
                pass
        metric = evaluation.select_metrics()
        cohorts, proofs, contrasts = {}, {}, {}
        for role in ("P", "H"):
            for name in (evaluation.BASELINE, CONTROL, FIT):
                folder = evaluation.prediction_folder(role, name)
                receipt = read(folder / "COMPLETE.json")
                if receipt["status"] != "COMPLETE" or receipt["sha256"] != sha(
                    folder / "predictions.parquet"
                ):
                    raise ValueError("Incomplete D1 comparison cohort")
                check = (
                    verify(folder)
                    if name == FIT
                    else read(RUN / "verification/predictions" / role / name / "NUMPY_RECALC.json")
                )
                if (
                    check["status"] != "PASS"
                    or check["prediction_sha256"] != receipt["sha256"]
                    or check["metrics_sha256"] != sha(folder / "METRICS.json")
                ):
                    raise ValueError("D1 native metric proof changed")
                cohorts[f"{role}/{name}"] = read(folder / "METRICS.json")
                proofs[f"{role}/{name}"] = check
            for name in (evaluation.BASELINE, CONTROL):
                contrasts[f"{role}/{name}"] = compare(
                    evaluation.prediction_folder(role, name) / "predictions.parquet",
                    evaluation.prediction_folder(role, FIT) / "predictions.parquet",
                    DIRECTORY / "bootstrap" / role / name,
                )
        while (timing := cost.measure())["status"] != "COMPLETE":
            pass
        ratio = (
            timing["results"][FIT]["p50_seconds"]
            / timing["results"][evaluation.BASELINE]["p50_seconds"]
        )
        selection = combine_decision(
            metric["decision"], ratio, registration["development_gates"]["routes"]
        )
        decision = {
            "status": "DEVELOPMENT_CANDIDATE" if selection["eligible"] else "NO_CANDIDATE_D1",
            "ranked": [FIT] if selection["eligible"] else [],
            "decision": selection,
            "metric_evidence_sha256": sha(DIRECTORY / "H_METRIC_SELECTION.json"),
            "cost_evidence_sha256": sha(DIRECTORY / "cost/prepared_forward/COMPLETE.json"),
            "system_promotion": False,
            "official_comparison": False,
        }
        freeze(DIRECTORY / "SELECTION_FINAL.json", decision)
        export = portable.export(FIT)
        p_base, p_wide = cohorts[f"P/{CONTROL}"], cohorts[f"P/{FIT}"]
        mechanism = (
            p_wide["overall_MiD"] <= 0.97 * p_base["overall_MiD"]
            and p_wide["diagnostics"]["sign_error_rate"] <= p_base["diagnostics"]["sign_error_rate"]
        )
        result = {
            "status": "D1_READY_FOR_INTERPRETATION",
            "endpoints": endpoints,
            "evaluation_registration_sha256": sha(EVALUATION),
            "cohorts": cohorts,
            "metric_proofs": proofs,
            "bootstrap": contrasts,
            "selection": decision,
            "P_capacity_signal": mechanism,
            "prepared_cost": timing,
            "export": export,
            "system_promotion": False,
            "official_comparison": False,
            "next": "V only if fixed H admission passes; otherwise close D1 hypothesis",
        }
        write(DIRECTORY / "evidence/COMPLETE.json", result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    print(register() if args.register_only else run())
