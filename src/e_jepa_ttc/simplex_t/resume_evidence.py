"""Revalidate saved real CPU resume evidence without performing optimizer work."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .model import TemporalConfig
from .training import load_checkpoint, state_digest

RESUME_FILES = frozenset(
    {
        "CONTRACT.json",
        "RESUME_QA.json",
        "TECHNICAL_JOURNAL.json",
        "continuous/checkpoint_last.pt",
        "split/checkpoint_last.pt",
    }
)


def verify_real_cpu_resume(
    root: Path,
    *,
    pins: dict[str, str],
    source_sha256: str,
    compiled_sha256: str,
    engine: Path,
    probe_script: Path,
) -> dict:
    """Bind report to actual state, 5-update boundary, source and current code.

    Pins must come from independently audited freeze inputs. This verifies one
    technical TRAIN trajectory, not all-fold loading, authority or latent parity.
    It performs no fit and does not enable any scientific stage.
    """
    if set(pins) != RESUME_FILES:
        raise ValueError("complete resume evidence file set required")
    base = root.resolve(strict=True)
    for name, digest in pins.items():
        path = (base / name).resolve(strict=True)
        if not path.is_relative_to(base) or path.stat().st_size > 16_777_216:
            raise ValueError("resume evidence path or size invalid")
        if sha256(path) != digest:
            raise ValueError("resume evidence bytes changed")

    def read(name: str) -> dict:
        return json.loads((base / name).read_text(encoding="utf-8"))

    contract, report, journal = map(
        read, ("CONTRACT.json", "RESUME_QA.json", "TECHNICAL_JOURNAL.json")
    )
    expected_contract = {
        "scope": "TECHNICAL_REAL_QUERY_CONTEXT_NOT_SCIENTIFIC_FREEZE",
        "source_sha256": source_sha256,
        "compiled_sha256": compiled_sha256,
        "script_sha256": sha256(probe_script),
        "engine_sha256": sha256(engine),
        "torch": str(torch.__version__),
        "updates_reserved": 20,
    }
    if "journal_engine_sha256" in contract:
        expected_contract["journal_engine_sha256"] = sha256(
            Path(__file__).with_name("work_budget.py")
        )
    if "model_sha256" in contract:
        expected_contract["model_sha256"] = sha256(Path(__file__).with_name("model.py"))
    if contract != expected_contract:
        raise ValueError("resume contract differs from current source, code or runtime")
    if (
        report.get("exact_complete_state_match") is not True
        or report.get("source_sha256") != source_sha256
        or report.get("executed_technical_updates") != 20
        or report.get("scientific_updates") != 0
        or report.get("nonzero_real_history") is not True
        or not 0 < report.get("train_h8_count", 0) <= report.get("train_queries", 0)
        or not 0 < report.get("max_sensor_age_seconds", 0) <= 0.35
    ):
        raise ValueError("real-history resume report incomplete")
    if (
        journal.get("schema") != "simplex_t_physical_work_v1"
        or journal.get("graph") != {"continuous": 2500, "split": 2500}
        or set(journal.get("fits", {})) != {"continuous", "split"}
    ):
        raise ValueError("resume journal graph differs")
    states = []
    expected_identity = {
        "source": source_sha256,
        "freeze": state_digest(contract),
        "config": asdict(TemporalConfig()),
        "seed": 7,
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "batch": 128,
        "endpoint": 2500,
    }
    for branch in ("continuous", "split"):
        name = f"{branch}/checkpoint_last.pt"
        row = journal["fits"][branch]
        expected_row = {
            "completed": 10,
            "uncertain_lost_upper": 0,
            "pending": None,
            "checkpoint_sha256": pins[name],
        }
        if {k: v for k, v in row.items() if k != "checkpoint_state_sha256"} != expected_row:
            raise ValueError("resume journal has incomplete or uncertain work")
        state = load_checkpoint(base / name)
        if "checkpoint_state_sha256" in row and row["checkpoint_state_sha256"] != state_digest(
            {k: v for k, v in state.items() if k != "status"}
        ):
            raise ValueError("resume journal training-state digest differs from checkpoint")
        if (
            state["completed_updates"] != 10
            or state["status"] != "TECHNICAL_PARTIAL"
            or state["identity"] != expected_identity
        ):
            raise ValueError("resume checkpoint identity or endpoint differs")
        states.append(state)
        boundaries = [
            event["completed"]
            for event in journal["events"]
            if event["operation"] == "checkpoint" and event["key"] == branch
        ]
        if boundaries != ([10] if branch == "continuous" else [5, 10]):
            raise ValueError("journal does not prove 10 versus 5+5 boundaries")
    if state_digest(states[0]) != state_digest(states[1]):
        raise ValueError("complete resume states differ")
    return {
        "status": "SAVED_REAL_CPU_RESUME_REVERIFIED",
        "source_sha256": source_sha256,
        "executed_updates_by_this_verification": 0,
        "verified_saved_technical_updates": 20,
        "scientific_stage_authorized": False,
        "state_sha256": state_digest(states[0]),
        "pins": pins,
    }
