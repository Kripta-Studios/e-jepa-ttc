"""Resume registered C1 training and complete evidence, without opening V or promoting a system."""

from __future__ import annotations

import argparse
import os
import time

from operational.range_transfer.common import freeze, read, sha, utc, write

from .bootstrap import compare
from .common import RUN, accounting
from .fast_runtime import PersistentSensors
from .geometry_cost import finalize_selection, measure
from .geometry_evaluate import (
    BASELINE,
    freeze_endpoint,
    predict,
    prediction_folder,
    verify_endpoint,
)
from .geometry_export import export
from .geometry_train import (
    CONTROL,
    DIRECTORY,
    FIT,
    REGISTRATION,
    exclusive_driver,
    fit,
    qa,
    registered,
)
from .rgb_run import require_authorization
from .verify_predictions import verify


def complete_evidence() -> dict:
    """Recalculate new results; admit historical reference proofs without overwriting them."""
    endpoints = freeze_endpoint()
    cohorts, verification = {}, {}
    for role in ("P", "H"):
        for name in (BASELINE, CONTROL, FIT):
            folder = prediction_folder(role, name)
            receipt = read(folder / "COMPLETE.json")
            if (
                receipt["status"] != "COMPLETE"
                or sha(folder / "predictions.parquet") != receipt["sha256"]
            ):
                raise ValueError("Complete intact C1 P/H cohorts before interpretation")
            cohorts[f"{role}/{name}"] = receipt["sha256"]
            if name == FIT:
                proof = verify(folder)
            else:
                proof = read(RUN / "verification/predictions" / role / name / "NUMPY_RECALC.json")
            if (
                proof["status"] != "PASS"
                or proof["prediction_sha256"] != receipt["sha256"]
                or proof["metrics_sha256"] != sha(folder / "METRICS.json")
            ):
                raise ValueError("C1 reference or candidate metric proof changed")
            verification[f"{role}/{name}"] = proof
    freeze(DIRECTORY / "evidence/BINDING.json", {"endpoints": endpoints, "cohorts": cohorts})
    cost = measure()
    if cost["status"] != "COMPLETE":
        return {"status": "PARTIAL_COST", "detail": cost}
    for role in ("P", "H"):
        for name in (BASELINE, CONTROL):
            compare(
                prediction_folder(role, name) / "predictions.parquet",
                prediction_folder(role, FIT) / "predictions.parquet",
                DIRECTORY / "bootstrap" / role / name,
            )
    result = {
        "status": "C1_READY_FOR_INTERPRETATION",
        "utc": utc(),
        "registration_sha256": sha(REGISTRATION),
        "cohorts": cohorts,
        "verification": verification,
        "selection": finalize_selection(),
        "export": export(),
        "own_prediction_rows": sum(
            read(prediction_folder(role, FIT) / "COMPLETE.json")["rows"] for role in ("P", "H")
        ),
        "reused_reference_tables": 4,
        "new_scientific_fits": 1,
        "system_promotion": False,
        "official_comparison": False,
        "full_objective_complete": False,
        "next": "Interpret C1 and RGB; follow registered development and system gates",
    }
    write(DIRECTORY / "evidence/COMPLETE.json", result)
    return result


def advance(sensors: PersistentSensors) -> dict:
    """Advance one bounded stage; no H inference before endpoint and trajectory admission."""
    require_authorization()
    registered()
    qa("cpu")
    qa("cuda")
    receipt = fit(sensors)
    if receipt["status"] != "COMPLETE":
        return {"status": "TRAINING", "fit_id": FIT, "updates": receipt["completed_updates"]}
    freeze_endpoint()
    verify_endpoint()
    for role in ("H", "P"):
        outcome = predict(role)
        if outcome["status"] != "COMPLETE":
            return {"status": "EVALUATING", "role": role, "detail": outcome}
    return complete_evidence()


def run(*, once: bool = False) -> None:
    """Keep one C1 writer and one bounded input cache through GPU scopes."""
    with exclusive_driver():
        sensors = PersistentSensors()
        try:
            while True:
                try:
                    state = advance(sensors)
                except Exception as error:
                    waiting = str(error).startswith(("RESOURCE_WAIT", "GPU_OWNER_WAIT"))
                    state = {
                        "status": "RESOURCE_WAIT" if waiting else "ACTION_REQUIRED",
                        "error": str(error),
                        "utc": utc(),
                        "pid": os.getpid(),
                    }
                    write(DIRECTORY / "RUNNER_STATE.json", state)
                    if not waiting:
                        raise
                state.update(utc=utc(), pid=os.getpid(), accounting=accounting())
                write(DIRECTORY / "RUNNER_STATE.json", state)
                print(
                    {key: value for key, value in state.items() if key != "verification"},
                    flush=True,
                )
                if once or state["status"] == "C1_READY_FOR_INTERPRETATION":
                    return
                if state["status"] == "RESOURCE_WAIT":
                    time.sleep(60)
        finally:
            sensors.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    run(once=parser.parse_args().once)
