"""Accounted real-TRAIN array resume probe; timing lineage is explicitly NOT proven."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.simplex_t.coordination import verified_ack
from e_jepa_ttc.simplex_t.current_inputs import (
    CurrentQueries,
    load_current_inputs,
    normalize_current_train,
)
from e_jepa_ttc.simplex_t.lifecycle import TechnicalBudget, admitted
from e_jepa_ttc.simplex_t.model import TemporalConfig
from e_jepa_ttc.simplex_t.training import fit, load_checkpoint, state_digest
from e_jepa_ttc.simplex_t.work_budget import EngineWorkJournal, WorkBudget


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--journal", action="store_true", help="Validate new work-journal callbacks"
    )
    args = parser.parse_args()
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    config = json.loads(Path("configs/experiment/simplex_t_coordination.json").read_text())
    ack = verified_ack(
        Path(paths["shared_coordination"]) / config["ack_filename"], config["ack_sha256"]
    )
    if not ack["resources"]["cpu_overlap_authorized"]:
        raise ValueError("CPU overlap not authorized")
    if args.output.exists():
        raise FileExistsError("existing probe must not be repeated")
    snapshot = admitted([Path(paths["worktree"])])
    if not snapshot["has_headroom"] or snapshot["host_available_bytes"] < 8 * 1024**3:
        raise RuntimeError("reserve up to 4 GiB allocation while retaining 4 GiB host availability")
    ancestry = ack["producers"]["authoritative_historical_manifest"]
    table = load_current_inputs(
        Path(ancestry["path"]).parent,
        0,
        "inner_oof",
        ancestry_sha256=ancestry["sha256"],
        allowed_sequences=set(ack["interfaces"]["role_manifest"]["roles"]["original"]),
    )
    normalizer, mass = normalize_current_train(table)
    # Technical fixture only: no claim of production observation latency calibration.
    source = CurrentQueries(table, normalizer, mass, np.zeros((len(mass), 4), dtype=np.float32))
    TechnicalBudget(Path(paths["worktree"]) / "artifacts/simplex_t/TECHNICAL_BUDGET.json").reserve(
        "current_array_journal_resume_10_vs_5_5"
        if args.journal
        else "current_array_resume_10_vs_5_5",
        20,
    )
    args.output.mkdir(parents=True)
    freeze = hashlib.sha256(
        b"technical_current_arrays_zero_timing_fixture_not_scientific_freeze"
    ).hexdigest()
    results = []
    fixture_budget = WorkBudget(
        args.output / "TECHNICAL_JOURNAL_FIXTURE.json", {"continuous": 2500, "split": 2500}, 0
    )
    for folder, count, resume in (
        ("continuous", 10, False),
        ("split", 5, False),
        ("split", 10, True),
    ):
        result = fit(
            source,
            TemporalConfig(),
            args.output / folder,
            seed=7,
            freeze_sha256=freeze,
            stop_after=count,
            resume=resume,
            resource_ok=lambda: bool(admitted([args.output])["has_headroom"]),
            journal=EngineWorkJournal(fixture_budget, folder) if args.journal else None,
        )
        results.append(result)
        if result["status"] == "PAUSED_RESOURCE":
            write_new_json(
                args.output / "PAUSED.json",
                {
                    "results": results,
                    "reserved_upper_bound": 20,
                    "resume_requires_same_probe_identity": True,
                },
            )
            return
    first = load_checkpoint(args.output / "continuous/checkpoint_last.pt")
    second = load_checkpoint(args.output / "split/checkpoint_last.pt")
    identical = state_digest(first) == state_digest(second)
    write_new_json(
        args.output / "RESUME_QA.json",
        {
            "exact_complete_state_match": identical,
            "results": results,
            "executed_technical_updates": 20,
            "scientific_updates": 0,
            "real_train_arrays": True,
            "work_journal_callbacks": args.journal,
            "journal_scope": "TECHNICAL_FIXTURE_NOT_SCIENTIFIC_PROGRESS",
            "timing": "EXPLICIT_ZERO_TECHNICAL_FIXTURE",
            "production_time_lineage_validated": False,
            "raw_expert_replay_parity": False,
            "source_sha256": source.identity_sha256,
            "admission_snapshot": snapshot,
        },
    )
    if not identical:
        raise ValueError("real-array CPU resume mismatch")
    print(json.dumps({"exact_resume": True, "technical_updates": 20}))


if __name__ == "__main__":
    main()
