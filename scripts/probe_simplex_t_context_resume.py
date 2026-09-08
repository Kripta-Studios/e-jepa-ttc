"""Accounted CPU 10-versus5+5 resume using one complete real temporal TRAIN fold."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.simplex_t.context_sources import load_context_sources
from e_jepa_ttc.simplex_t.coordination import verified_ack
from e_jepa_ttc.simplex_t.lifecycle import ExclusiveLease, TechnicalBudget, admitted
from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
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
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--feature-count", type=int, choices=(17, 145), default=17)
    parser.add_argument("--hidden", type=int, choices=(64, 160), default=64)
    args = parser.parse_args()
    if not args.benchmark and (
        args.device != "cpu" or args.feature_count != 17 or args.hidden != 64
    ):
        raise ValueError("nonhistorical device/features require explicit --benchmark")
    if args.feature_count == 145 and args.hidden != 160:
        raise ValueError("registered latent benchmark requires hidden160")
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    worktree = Path(paths["worktree"])
    lease = (
        ExclusiveLease(worktree / "artifacts/simplex_t/T1/CURRENT_REPLAY.lock")
        if args.benchmark
        else nullcontext()
    )
    with lease:
        run_probe(args)


def run_probe(args: argparse.Namespace) -> None:
    """Execute one accounted real-source probe, optionally measuring device costs."""
    started = time.perf_counter()
    if (args.output / "RESUME_QA.json").exists():
        raise FileExistsError("completed technical probe must not be repeated")
    if args.output.exists() != args.resume:
        raise ValueError("new output or explicit resume of an existing probe required")
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    worktree = Path(paths["worktree"])
    snapshot = admitted([worktree])
    if not snapshot["has_headroom"] or snapshot["host_available_bytes"] < 8 * 1024**3:
        raise RuntimeError("RESOURCE_PAUSE: reserve 4 GiB and retain 4 GiB available")
    sampled_rss = int(snapshot.get("process_tree_rss_bytes", 0))
    minimum_available = int(snapshot["host_available_bytes"])

    def resource_ok() -> bool:
        nonlocal sampled_rss, minimum_available
        current = admitted([worktree])
        sampled_rss = max(sampled_rss, int(current.get("process_tree_rss_bytes", 0)))
        minimum_available = min(minimum_available, int(current.get("host_available_bytes", 0)))
        return bool(current["has_headroom"])

    if args.device != "cpu":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("CUDA benchmark requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; do not fall back")
        torch.cuda.reset_peak_memory_stats(0)
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
        feature_count=args.feature_count,
    )
    source = sources["inner_oof"]
    load_seconds = time.perf_counter() - started
    # Real loader integration, including every TRAIN query and cold-start mask.
    history_counts = (source.history[:, -8:] >= 0).sum(1)
    max_age = 0.0
    for start in range(0, source.population, 128):
        if not resource_ok():
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
    if args.benchmark:
        contract["benchmark"] = {
            "device": args.device,
            "feature_count": args.feature_count,
            "hidden": args.hidden,
            "batch": 128,
            "precision": "FP32",
            "evaluation_role": "inner_oof",
        }
    # Renew changed source/code evidence without reusing an old budget reservation.
    operation = "real_context_cpu_resume_" + state_digest(contract)
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
        attempt_timings = []
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
            attempt_started = time.perf_counter()
            result = fit(
                source,
                TemporalConfig(feature_count=args.feature_count, hidden=args.hidden),
                args.output / folder,
                seed=7,
                freeze_sha256=freeze,
                stop_after=endpoint,
                resume=checkpoint.exists(),
                resource_ok=resource_ok,
                journal=EngineWorkJournal(budget, folder),
                device=args.device,
            )
            attempt_timings.append(
                {
                    "folder": folder,
                    "endpoint": endpoint,
                    "fit_with_checkpoint_seconds": time.perf_counter() - attempt_started,
                }
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
        benchmark = None
        if args.benchmark:
            evaluation_started = time.perf_counter()
            model = TemporalRefiner(
                TemporalConfig(feature_count=args.feature_count, hidden=args.hidden)
            ).float()
            model.load_state_dict(first["model"])
            model = model.to(args.device).eval()
            prediction_hashes = []
            with torch.inference_mode():
                for start in range(0, source.population, 128):
                    if not resource_ok():
                        raise RuntimeError("RESOURCE_PAUSE during TRAIN benchmark evaluation")
                    batch = source.gather(torch.arange(start, min(start + 128, source.population)))
                    inputs = [value.to(args.device) for value in batch[:4]]
                    predictions = model(*inputs)
                    prediction_hashes.append(state_digest(predictions))
            benchmark = {
                "load_verify_seconds": load_seconds,
                "fit_attempts_this_invocation": attempt_timings,
                "train_evaluation_seconds": time.perf_counter() - evaluation_started,
                "total_seconds_before_report": time.perf_counter() - started,
                "resumed_invocation": args.resume,
                "sampled_process_tree_rss_max_bytes": sampled_rss,
                "sampled_host_available_min_bytes": minimum_available,
                "cuda_peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated(0) if args.device != "cpu" else None
                ),
                "cuda_peak_reserved_bytes": (
                    torch.cuda.max_memory_reserved(0) if args.device != "cpu" else None
                ),
                "memory_scope": (
                    "RAM sampled at batch boundaries, not continuous peak; "
                    "CUDA peaks cover this process's PyTorch allocator, not global VRAM"
                ),
                "train_prediction_digest": state_digest(prediction_hashes),
                "note": "No accuracy scores; resumed invocation timing is not a fresh total",
            }
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
                **({"benchmark": benchmark} if args.benchmark else {}),
            },
        )
        if not identical:
            raise ValueError("actual-history CPU exact resume mismatch")
        print(json.dumps({"exact_resume": True, "technical_updates": 20}))


if __name__ == "__main__":
    main()
