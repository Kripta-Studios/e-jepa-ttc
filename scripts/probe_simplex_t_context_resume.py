"""Accounted CPU 10-versus5+5 resume using one complete real temporal TRAIN fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.simplex_t.context_sources import load_context_sources
from e_jepa_ttc.simplex_t.coordination import verified_ack
from e_jepa_ttc.simplex_t.lifecycle import ExclusiveLease, TechnicalBudget, admitted
from e_jepa_ttc.simplex_t.model import TemporalConfig
from e_jepa_ttc.simplex_t.training import fit, load_checkpoint, state_digest
from e_jepa_ttc.simplex_t.work_budget import EngineWorkJournal, WorkBudget


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--compiled-sha256", required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--dedup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.output / "RESUME_QA.json").exists():
        raise FileExistsError("completed technical probe must not be repeated")
    if args.output.exists() != args.resume:
        raise ValueError("new output or explicit resume of an existing probe required")
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    worktree = Path(paths["worktree"])
    snapshot = admitted([worktree])
    if not snapshot["has_headroom"] or snapshot["host_available_bytes"] < 12 * 1024**3:
        raise RuntimeError("RESOURCE_PAUSE: reserve4 GiB and retain8 GiB available")
    config = json.loads((worktree / "configs/experiment/simplex_t_coordination.json").read_text())
    ack = verified_ack(
        Path(paths["shared_coordination"]) / config["ack_filename"], config["ack_sha256"]
    )
    if not ack["resources"]["cpu_overlap_authorized"]:
        raise ValueError("CPU operation not acknowledged")
    ancestry = ack["producers"]["authoritative_historical_manifest"]
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    sources = load_context_sources(
        args.compiled,
        args.index,
        args.dedup,
        Path(ancestry["path"]).parent,
        compiled_manifest_sha256=args.compiled_sha256,
        ancestry_sha256=ancestry["sha256"],
        allowed_sequences=set(ack["interfaces"]["role_manifest"]["roles"]["original"]),
        feature_count=17,
    )
    source = sources["inner_oof"]
    # Real loader integration, including every TRAIN query and cold-start mask.
    history_counts = (source.history[:, -8:] >= 0).sum(1)
    max_age = 0.0
    for start in range(0, source.population, 128):
        if not admitted([worktree])["has_headroom"]:
            raise RuntimeError("RESOURCE_PAUSE during read-only source validation")
        batch = source.gather(torch.arange(start, min(start + 128, source.population)))
        if any(not torch.isfinite(value).all() for value in batch):
            raise ValueError("nonfinite production source batch")
        max_age = max(max_age, float(batch[1][:, :, 0].max()))
    if max_age <= 0 or not np.any(history_counts > 1):
        raise ValueError("actual nonzero temporal history required; no current-only fixture")
    contract = {
        "scope": "TECHNICAL_REAL_QUERY_CONTEXT_NOT_SCIENTIFIC_FREEZE",
        "source_sha256": source.identity_sha256,
        "compiled_sha256": args.compiled_sha256,
        "script_sha256": compute_file_hash(__file__),
        "engine_sha256": compute_file_hash(str(worktree / "src/e_jepa_ttc/simplex_t/training.py")),
        "journal_engine_sha256": compute_file_hash(
            str(worktree / "src/e_jepa_ttc/simplex_t/work_budget.py")
        ),
        "torch": str(torch.__version__),
        "updates_reserved": 20,
    }
    operation = "real_context_cpu_resume_10_vs_5_5_journal_state_v2"
    budget_path = worktree / "artifacts/simplex_t/TECHNICAL_BUDGET.json"
    with ExclusiveLease(args.output.with_suffix(".probe.lock")):
        if args.resume:
            if json.loads((args.output / "CONTRACT.json").read_text()) != contract:
                raise ValueError("technical resume contract changed")
            if json.loads(budget_path.read_text())["reservations"].get(operation) != 20:
                raise ValueError("technical update reservation missing")
        else:
            TechnicalBudget(budget_path).reserve(operation, 20)
            args.output.mkdir(parents=True)
            write_new_json(args.output / "CONTRACT.json", contract)
        journal_path = args.output / "TECHNICAL_JOURNAL.json"
        if journal_path.exists() and any(
            row["pending"] is not None
            for row in json.loads(journal_path.read_text())["fits"].values()
        ):
            raise RuntimeError("CRASH_ACCOUNTING_REQUIRED: preserve checkpoint and journal")
        budget = WorkBudget(journal_path, {"continuous": 2500, "split": 2500}, 0)
        freeze = state_digest(contract)
        results = []
        for folder, endpoint in (("continuous", 10), ("split", 5), ("split", 10)):
            checkpoint = args.output / folder / "checkpoint_last.pt"
            if checkpoint.exists():
                prior = load_checkpoint(checkpoint)
                if not 0 <= prior["completed_updates"] <= 10:
                    raise ValueError("technical checkpoint exceeds reserved endpoint")
                if (
                    prior["identity"]["source"] != source.identity_sha256
                    or prior["identity"]["freeze"] != freeze
                ):
                    raise ValueError("technical checkpoint identity changed")
                if prior["completed_updates"] >= endpoint:
                    continue
            result = fit(
                source,
                TemporalConfig(),
                args.output / folder,
                seed=7,
                freeze_sha256=freeze,
                stop_after=endpoint,
                resume=checkpoint.exists(),
                resource_ok=lambda: bool(admitted([worktree])["has_headroom"]),
                journal=EngineWorkJournal(budget, folder),
            )
            results.append(result)
            if result["status"] == "PAUSED_RESOURCE":
                print(json.dumps({"status": "PAUSED_RESOURCE", "resume": str(args.output)}))
                return
        first = load_checkpoint(args.output / "continuous/checkpoint_last.pt")
        second = load_checkpoint(args.output / "split/checkpoint_last.pt")
        if first["completed_updates"] != 10 or second["completed_updates"] != 10:
            raise ValueError("both technical branches must end at update10")
        identical = state_digest(first) == state_digest(second)
        write_new_json(
            args.output / "RESUME_QA.json",
            {
                "exact_complete_state_match": identical,
                "results_this_invocation": results,
                "executed_technical_updates": first["completed_updates"]
                + second["completed_updates"],
                "scientific_updates": 0,
                "source_sha256": source.identity_sha256,
                "train_queries": source.population,
                "nonzero_real_history": True,
                "max_sensor_age_seconds": max_age,
                "train_h8_count": int(np.sum(history_counts == 8)),
                "admission_snapshot": snapshot,
            },
        )
        if not identical:
            raise ValueError("actual-history CPU exact resume mismatch")
        print(json.dumps({"exact_resume": True, "technical_updates": 20}))


if __name__ == "__main__":
    main()
