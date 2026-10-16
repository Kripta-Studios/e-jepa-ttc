"""Run the fixed admitted loss screen, then full endpoint evaluation and reporting."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from . import common


def state(stage: str, **details: object) -> dict:
    value = {
        "status": stage,
        "utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "pipeline_sha256": common.sha(Path(__file__)),
        "system_promotion": False,
        **details,
    }
    common.write_json(common.RUN / "PIPELINE_STATE.json", value)
    print(json.dumps(value), flush=True)
    return value


def run() -> dict:
    """Failures remain durable; resumption uses each stage's verified own receipts."""
    from . import evaluation, training
    from .registration import verify

    stage = "ADMISSION"
    try:
        with training.file_lock(training.destination("PIPELINE.lock")):
            verify()
            state(stage)
            training.prepare_registration()
            stage = "TECHNICAL_QA"
            state(stage)
            training.qa()
            stage = "SCIENTIFIC_L0_LW"
            state(stage)
            training.run()
            stage = "FULL_P_H_EVALUATION"
            for arm in evaluation.ARMS:
                previous = None
                while True:
                    state(stage, arm=arm)
                    result = evaluation.evaluate(arm)
                    if result["status"] == "COMPLETE":
                        break
                    cursor = (result.get("role"), result.get("next_row"))
                    if result["status"] != "PARTIAL" or cursor == previous:
                        raise RuntimeError("Evaluation did not make verified progress")
                    previous = cursor
            stage = "REPORT"
            state(stage)
            result = evaluation.report()
            return state(
                "INITIAL_SCREEN_COMPLETE",
                report="evaluation/REPORT.json",
                report_sha256=common.sha(common.RUN / "evaluation/REPORT.json"),
                decision=result["replication_decision"],
                next_step="Assess registered replication trigger and full system cost",
            )
    except Exception as error:
        state("STOPPED_ON_ERROR", failed_stage=stage, error=f"{type(error).__name__}: {error}")
        raise


def status() -> dict:
    path = common.RUN / "PIPELINE_STATE.json"
    return common.read_json(path) if path.exists() else {"status": "NOT_STARTED"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status"))
    args = parser.parse_args()
    print(json.dumps(run() if args.action == "run" else status(), indent=2))
