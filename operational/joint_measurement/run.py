"""Resume the registered V18 primary matrix and freeze its complete H evidence."""

from __future__ import annotations

import argparse
import time

from operational.range_transfer.common import read, utc, write

from .common import RUN, accounting, config
from .data import cache_root, prepare
from .evaluate import freeze_primary, predict, select_h
from .qa import run as run_qa
from .train import fit


def advance() -> dict:
    """Complete one durable stage; adaptive choices require their next registration."""
    if not (cache_root("P") / "COMPLETE.json").exists():
        return {"status": "WAIT_P_SENSOR", "progress": read(RUN / "PREPARE_P_PROGRESS.json")}
    run_qa("cuda")
    for fit_id in config()["fits"]:
        receipt = fit(fit_id)
        if receipt["status"] != "COMPLETE":
            return {"status": "TRAINING", "fit_id": fit_id, "updates": receipt["completed_updates"]}
    freeze_primary()
    for role in ("H", "P"):
        if not (cache_root(role) / "COMPLETE.json").exists():
            prepare(role)
        for fit_id in ["A5-P2-FROZEN", *config()["fits"]]:
            result = predict(fit_id, role)
            if result["status"] != "COMPLETE":
                return {"status": "EVALUATING", "fit_id": fit_id, "role": role, **result}
    return {"status": "PRIMARY_READY_FOR_INTERPRETATION", "selection": select_h()}


def run(*, once: bool) -> None:
    """Persist through resource waits without changing recipes or other owners."""
    while True:
        state: dict
        try:
            state = advance()
        except RuntimeError as error:
            message = str(error)
            if not message.startswith(("GPU_OWNER_WAIT", "RESOURCE_WAIT")):
                write(
                    RUN / "RUNNER_STATE.json",
                    {"status": "ACTION_REQUIRED", "error": message, "utc": utc()},
                )
                raise
            state = {"status": "RESOURCE_WAIT", "reason": message}
        state.update(utc=utc(), accounting=accounting())
        write(RUN / "RUNNER_STATE.json", state)
        print(state, flush=True)
        if once or state["status"] == "PRIMARY_READY_FOR_INTERPRETATION":
            return
        if state["status"].startswith(("WAIT", "RESOURCE")):
            time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    run(once=parser.parse_args().once)
