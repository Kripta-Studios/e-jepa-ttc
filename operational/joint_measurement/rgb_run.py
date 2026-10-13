"""Resume the finite registered RGB pair and its complete P/H evidence."""

from __future__ import annotations

import argparse
import os
import time

from operational.range_transfer.common import freeze, read, sha, utc, write

from .bootstrap import compare
from .common import RUN, accounting
from .rgb_cost import finalize_selection, measure
from .rgb_evaluate import BASELINE, freeze_endpoints, predict
from .rgb_export import export
from .rgb_train import REGISTRATION, fit, qa, registered
from .verify_predictions import verify


def require_authorization() -> None:
    """Require the user's applied full-program amendment before starting RGB work."""
    authorization = read(RUN / "BUDGET_AUTHORIZATION_20261014.json")
    ledger = read(RUN / "GPU_BUDGET.json")
    if (
        authorization["status"] != "APPLIED"
        or not authorization["scope"].startswith("All evidence-enabled V18 handoff branches:")
        or authorization["authorized_cap_seconds"] != ledger["cap_s"]
        or authorization["physical_update_cap"] != 1000000
    ):
        raise ValueError("RGB requires the applied full-program budget scope")


def complete_evidence() -> dict:
    """Verify all RGB cohorts, compare groups, and export before human interpretation."""
    record = registered()
    names = [BASELINE, *record["fits"]]
    cohorts = {}
    for role in ("P", "H"):
        for name in names:
            folder = RUN / "rgb/predictions" / role / name
            receipt = read(folder / "COMPLETE.json")
            if receipt["status"] != "COMPLETE" or (
                sha(folder / "predictions.parquet") != receipt["sha256"]
            ):
                raise ValueError("Finish intact RGB P/H populations before interpretation")
            cohorts[f"{role}/{name}"] = receipt["sha256"]
    directory = RUN / "rgb/evidence"
    freeze(directory / "BINDING.json", {"cohorts": cohorts, "endpoints": freeze_endpoints()})
    timing = measure()
    if timing["status"] != "COMPLETE":
        return {"status": "PARTIAL_COST", "detail": timing}
    verification = {}
    for role in ("P", "H"):
        for name in names:
            verification[f"{role}/{name}"] = verify(RUN / "rgb/predictions" / role / name)
        reference = RUN / "rgb/predictions" / role / BASELINE / "predictions.parquet"
        for name in record["fits"]:
            compare(
                reference,
                RUN / "rgb/predictions" / role / name / "predictions.parquet",
                RUN / "rgb/bootstrap" / role / name,
            )
    result = {
        "status": "RGB_READY_FOR_INTERPRETATION",
        "utc": utc(),
        "registration_sha256": sha(REGISTRATION),
        "cohorts": cohorts,
        "verification": verification,
        "selection": finalize_selection(),
        "exports": {name: export(name) for name in names},
        "system_promotion": False,
        "official_comparison": False,
        "full_objective_complete": False,
        "next": "Interpret RGB P/H and preregister the evidence-enabled continuation",
    }
    write(directory / "COMPLETE.json", result)
    return result


def advance() -> dict:
    """Advance one bounded training/evaluation scope without early H or any V search."""
    require_authorization()
    record = registered()
    qa()
    for name in record["fits"]:
        receipt = fit(name)
        if receipt["status"] != "COMPLETE":
            return {"status": "TRAINING", "fit_id": name, "updates": receipt["completed_updates"]}
    freeze_endpoints()
    for role in ("H", "P"):
        for name in [BASELINE, *record["fits"]]:
            result = predict(name, role)
            if result["status"] != "COMPLETE":
                return {"status": "EVALUATING", "fit_id": name, "role": role, "detail": result}
    return complete_evidence()


def run(*, once: bool) -> None:
    """Keep a single RGB driver, reusing existing checkpoints and resource ownership."""
    import msvcrt

    path = RUN / "rgb/DRIVER.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            while True:
                try:
                    state = advance()
                except Exception as error:
                    message = str(error)
                    waiting = message.startswith(("RESOURCE_WAIT", "GPU_OWNER_WAIT"))
                    state = {
                        "status": "RESOURCE_WAIT" if waiting else "ACTION_REQUIRED",
                        "error": message,
                        "utc": utc(),
                        "pid": os.getpid(),
                    }
                    write(RUN / "rgb/RUNNER_STATE.json", state)
                    if not waiting:
                        raise
                state.update(utc=utc(), pid=os.getpid(), accounting=accounting())
                write(RUN / "rgb/RUNNER_STATE.json", state)
                print({k: v for k, v in state.items() if k != "verification"}, flush=True)
                if once or state["status"] == "RGB_READY_FOR_INTERPRETATION":
                    return
                if state["status"] == "RESOURCE_WAIT":
                    time.sleep(60)
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    run(once=parser.parse_args().once)
