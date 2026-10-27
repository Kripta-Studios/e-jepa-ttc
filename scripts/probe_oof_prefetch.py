"""Paired exact-state and split-resume tests on a disposable OOF checkpoint copy."""

from __future__ import annotations

import argparse
import contextlib
import shutil
import tempfile
import time
from pathlib import Path

import torch

from operational.final_refit import scientific_fold_producers as runner
from operational.final_refit.fold_sources import load_fold_rgb_source
from operational.final_refit.recipe import sha256
from operational.final_refit.runtime_migration import _STATE_FIELDS, _state_sha256
from operational.oof_runtime_binding import runtime_sources
from operational.oof_runtime_overlay import FoldPrefetchRuntime
from operational.rgb_port.accounting import atomic_write_json, read_json_shared
from scripts.profile_oof_loop import TechnicalGuard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    if args.steps < 4 or args.steps % 2:
        raise ValueError("An even step count >=4 is required")
    run = args.run.resolve()
    config = runner.load_config(run)
    tested_sources = {**config["source_sha256"], **{str(p): sha256(p) for p in runtime_sources()}}
    binding = runner.execution_binding(run)
    original = run / "rotations/0/fits/R_A5"
    receipt = read_json_shared(original / "CHECKPOINT_RECEIPT.json")
    if receipt["status"] != "PAUSED_RESOURCE":
        raise ValueError("Scientific pause required")
    origin_sha = sha256(original / "checkpoint_last.pt")
    runner.load_config = lambda _: config
    runner.execution_binding = lambda _: binding
    cases = []
    technical_updates = 0
    for name, optimized, chunks in [
        ("warm", False, [1]),
        ("baseline_A", False, [args.steps]),
        ("optimized_A", True, [args.steps]),
        ("optimized_B", True, [args.steps]),
        ("baseline_B", False, [args.steps]),
        ("optimized_resume", True, [args.steps // 2] * 2),
    ]:
        with tempfile.TemporaryDirectory(prefix="technical_prefetch_", dir=run) as tmp:
            scratch = Path(tmp)
            target = scratch / "rotations/0/fits/R_A5"
            shutil.copytree(original, target)
            timings, stats = [], []
            for count in chunks:
                source = load_fold_rgb_source(
                    Path(config["rgb_source"]),
                    Path(config["teacher"]),
                    Path(config["fold_contract"]),
                    0,
                )
                context = (
                    FoldPrefetchRuntime(stats_callback=stats.append)
                    if optimized
                    else contextlib.nullcontext()
                )
                try:
                    started = time.perf_counter()
                    with context:
                        result = runner.train(
                            source=source,
                            fit="R_A5",
                            config=Path(config["config"]),
                            contract=Path(config["fold_contract"]),
                            run=scratch,
                            monitor=TechnicalGuard(),
                            max_updates=count,
                        )
                    timings.append(time.perf_counter() - started)
                    technical_updates += count
                    if result["completed_updates"] != receipt["completed_updates"] + sum(
                        chunks[: len(timings)]
                    ):
                        raise RuntimeError("Technical probe paused before its requested endpoint")
                finally:
                    source.close()
            saved = torch.load(
                target / "checkpoint_last.pt", map_location="cpu", weights_only=False
            )
            hashes = {key: _state_sha256(saved[key]) for key in _STATE_FIELDS}
            case = dict(
                name=name,
                optimized=optimized,
                updates=sum(chunks),
                chunks=chunks,
                seconds=sum(timings),
                seconds_per_update=sum(timings) / sum(chunks),
                state_sha256=hashes,
                runtime_stats=stats,
                completed_updates=result["completed_updates"],
            )
            cases.append(case)
            print(name, case["seconds_per_update"], flush=True)
            atomic_write_json(run / "OOF_PREFETCH_PROBE_IN_PROGRESS.json", {"cases": cases})
    hashes = cases[1]["state_sha256"]
    parity = all(case["state_sha256"] == hashes for case in cases[2:])
    unchanged = sha256(original / "checkpoint_last.pt") == origin_sha
    sources_unchanged = all(sha256(Path(p)) == digest for p, digest in tested_sources.items())
    report = dict(
        schema="oof_prefetch_exact_state_probe_v1",
        status="PASS" if parity and unchanged and sources_unchanged else "FAIL",
        source_sha256=tested_sources,
        sources_unchanged=sources_unchanged,
        scientific_config_sha256=sha256(run / "SCIENTIFIC_CONFIG.json"),
        exact_training_state_parity=parity,
        exact_resume_parity=cases[-1]["state_sha256"] == hashes,
        original_checkpoint_unchanged=unchanged,
        original_checkpoint_sha256=origin_sha,
        scientific_optimizer_updates=0,
        technical_optimizer_updates=technical_updates,
        cases=cases,
    )
    atomic_write_json(run / "OOF_PREFETCH_PARITY.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("Exact prefetch/resume parity failed")


if __name__ == "__main__":
    main()
