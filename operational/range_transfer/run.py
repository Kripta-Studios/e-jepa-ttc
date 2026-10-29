"""Resume the bounded RANGE-TRANSFER queue without writing to V13/R1."""
# ruff: noqa: E402 -- Process-local thread limits must precede numerical imports.

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Process-local settings, established before importing numerical libraries.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import json
import subprocess
import sys
import time

import torch

from . import admission, analyze, diagnose
from .cache import HeadCache, verify_h_population
from .cached_traces import trace_v12
from .common import (
    HANDOFF,
    SCREEN,
    code_binding,
    fit_id,
    freeze,
    ownership,
    read,
    resources,
    sha,
    write,
)
from .controls import trace_h
from .figures import build
from .qa import run_qa
from .report import publish
from .train_heads import accounting, fit


def main() -> int:
    """Run viable phases once, preserving state and returning 3 for unmet dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--owner-run",
        type=Path,
        required=True,
        help="Read-only V13 artifacts/rgb_port_20261008 directory",
    )
    parser.add_argument("--v12-root", type=Path, required=True, help="Pinned V12 evidence worktree")
    parser.add_argument("--run", type=Path, default=Path("artifacts/range_transfer_20261010"))
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Execute/reuse the accounted 20-update synthetic resume test",
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Complete CPU diagnostics without scientific updates",
    )
    parser.add_argument(
        "--max-updates-this-call",
        type=int,
        help="Global scientific call limit; saves a durable frontier without shortening endpoints",
    )
    args = parser.parse_args()
    if args.max_updates_this_call is not None and args.max_updates_this_call < 1:
        parser.error("--max-updates-this-call must be positive")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run, owner, v12 = (
        args.run.resolve(),
        args.owner_run.resolve(strict=True),
        args.v12_root.resolve(strict=True),
    )
    if run.is_relative_to(owner.parent.parent) or run.is_relative_to(v12):
        raise ValueError("V14 outputs must be outside immutable V12/V13 worktrees")
    command = "& pwsh -NoProfile -File scripts/Resume-RangeTransfer.ps1"
    with ownership(run):
        started = time.perf_counter()
        active = resources(run, starting=True)
        write(run / "ACTIVE_STATE.json", active)
        write(run / f"observations/ACTIVE_STATE_{time.time_ns()}.json", active)
        observation = admission.inspect(owner, run)
        bindings = {
            "v12_commit": admission.git(v12, "rev-parse", "HEAD"),
            "v13_audited_base": "da068ebdeb2d967ee10b2afa348e1317b8720ad0",
            "owner_current_head": observation["v13_head"],
            "policy_sha256": sha(HANDOFF / "execution_policy.json"),
            "runner_sources": code_binding(),
            "owner_run": str(owner),
            "v12_root": str(v12),
            "readmenow_sha256": {
                str(path.relative_to(v12)): sha(path) for path in v12.rglob("readmenow.txt")
            },
        }
        write(run / "SOURCE_BINDINGS.json", bindings)
        if bindings["v12_commit"] != "1a686e338a1145c6794bec0704bdae4cda95f06d":
            raise ValueError("Unexpected V12 evidence commit")
        if not active["allowed"]:
            decision = {
                "status": "PAUSED_RESOURCE",
                "parent_dependencies_pending": bool(observation["pending"]),
                "phases": {
                    "N0": "COMPLETE",
                    "N1": "PUBLIC_CACHED_HEAD_TRACES_COMPLETE"
                    if (run / "v12_cached_attribution/RESULT.json").exists()
                    else "PENDING",
                    "N2": "H_METADATA_COMPLETE_FINAL_CACHE_GRADIENTS_PENDING"
                    if (run / "H_METADATA_AUDIT.json").exists()
                    else "PENDING",
                    "SCREEN": "WAITING_FINAL_PARENTS_AND_NATIVE_H",
                    "REPLICATION": "WAITING_SELECTION",
                    "TRANSFER": "WAITING_SELECTION",
                },
            }
            bundle = publish(run, decision, command)
            print(json.dumps({**decision, "bundle": str(bundle)}, indent=2))
            return 3
        for script, arguments in (
            (HANDOFF / "scripts/verify_package.py", []),
            (
                HANDOFF / "scripts/analyze_public.py",
                ["--v12-root", str(v12), "--output", str(run / "public_analysis")],
            ),
        ):
            result = subprocess.run(
                [sys.executable, "-B", str(script), *arguments],
                capture_output=True,
                text=True,
                check=True,
            )
            (run / f"{script.stem}.log").write_text(result.stdout, encoding="utf-8")
        diagnose.public_traces(v12, run)
        trace_v12(v12, run)
        raw_log = run / "qa/RAW_TRACE_WORKER.log"
        with raw_log.open("a", encoding="utf-8") as stream:
            raw_result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(HANDOFF.parents[1] / "scripts/range_transfer_v12_trace.py"),
                    "--v12-root",
                    str(v12),
                    "--run",
                    str(run),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        raw_path = run / "v12_raw_attribution/RESULT.json"
        raw_status = read(raw_path) if raw_path.exists() else {}
        write(
            run / "N1_STATUS.json",
            {
                "public_cached_head_traces": "COMPLETE",
                "public_raw_traces": raw_status,
                "raw_worker_returncode": raw_result.returncode,
                "matched_H_controls": "WAITING_FINAL_PARENTS_AND_NATIVE_H",
                "FCWD_scope": "historically_exposed_public_transfer_diagnostic",
                "external_confirmation": False,
                "universal_SOTA": False,
                "training_or_selection_on_FCWD": False,
            },
        )
        diagnose.audit_metadata(owner, run)
        if args.self_test:
            run_qa(run)
        binding = admission.admit(owner, run)
        phases = {
            "N0": "COMPLETE",
            "N1": "PUBLIC_DIAGNOSTICS_EXECUTED_MATCHED_H_CONTROLS_PENDING",
            "N2": "H_METADATA_COMPLETE_FINAL_CACHE_GRADIENTS_PENDING",
            "SCREEN": "WAITING",
            "REPLICATION": "WAITING_SELECTION",
            "TRANSFER": "WAITING_SELECTION",
            "REPORT": "COMPLETE",
        }
        if binding is None:
            write(
                run / "GRADIENT_AUDIT.json",
                {"status": "WAITING_FINAL_H_CACHE", "optimizer_updates": 0},
            )
            decision = {"status": "WAITING_FINAL_PARENTS", "phases": phases, "candidate": None}
        else:
            freeze(run / "ADMISSION.json", binding)
            cache = HeadCache(Path(binding["cache_path"]))
            verify_h_population(cache, owner)
            trace_h(cache, owner, run)
            diagnose.gradient_audit(cache, run)
            phases["N1"] = "PUBLIC_RAW_AND_MATCHED_H_CACHED_CONTROLS_COMPLETE"
            phases["N2"] = "COMPLETE"
            decision = {"status": "READY_SCREEN", "phases": phases, "candidate": None}
            if not args.diagnostics_only:
                qa = run_qa(run)
                if qa.get("status") != "PASS":
                    decision["status"] = "WAITING_TECHNICAL_ADMISSION"
                else:
                    before = accounting(run)["scientific"]
                    endpoints = {}
                    for m, loss, seed in SCREEN:
                        remaining = (
                            None
                            if args.max_updates_this_call is None
                            else (
                                args.max_updates_this_call
                                - (accounting(run)["scientific"] - before)
                            )
                        )
                        if remaining is not None and remaining <= 0:
                            break
                        result = fit(
                            cache,
                            run,
                            binding,
                            mass_id=m,
                            loss_id=loss,
                            seed=seed,
                            max_updates_this_call=remaining,
                        )
                        if result["status"] != "COMPLETE":
                            break
                        endpoints[fit_id(m, loss, seed)] = result
                    if len(endpoints) != 6:
                        decision["status"] = "SCREEN_PARTIAL_DURABLE"
                    else:
                        freeze(run / "SCREEN_ENDPOINTS.json", endpoints)
                        phases["SCREEN"] = "ENDPOINTS_FROZEN"
                        if not (owner / "features/V/E_HISTORY.json").exists():
                            decision["status"] = "WAITING_NATIVE_V_CACHE"
                        else:
                            results = analyze.evaluate_v(owner, run, endpoints, stage="screen")
                            import pandas as pd

                            predictions = run / "evaluation/screen"
                            reference = pd.read_csv(predictions / "RT-M0-L0-S7/PREDICTIONS.csv")
                            for name in endpoints:
                                if name != "RT-M0-L0-S7":
                                    analyze.bootstrap_pairs(
                                        reference,
                                        pd.read_csv(predictions / name / "PREDICTIONS.csv"),
                                        run / "bootstrap/screen" / name,
                                    )
                            selection = analyze.select(results, run)
                            decision.update(
                                status=selection["status"], candidate=selection["candidate"]
                            )
                            if selection["candidate"]:
                                replicas = {}
                                for seed in (13, 23):
                                    for m, loss in [(0, 0), tuple(selection["candidate_recipe"])]:
                                        remaining = (
                                            None
                                            if args.max_updates_this_call is None
                                            else (
                                                args.max_updates_this_call
                                                - (accounting(run)["scientific"] - before)
                                            )
                                        )
                                        if remaining is not None and remaining <= 0:
                                            break
                                        result = fit(
                                            cache,
                                            run,
                                            binding,
                                            mass_id=m,
                                            loss_id=loss,
                                            seed=seed,
                                            max_updates_this_call=remaining,
                                        )
                                        if result["status"] != "COMPLETE":
                                            break
                                        replicas[fit_id(m, loss, seed)] = result
                                if len(replicas) == 4:
                                    freeze(run / "REPLICATION_ENDPOINTS.json", replicas)
                                    analyze.evaluate_v(owner, run, replicas, stage="replication")
                                    analyze.summarize_replicas(
                                        run, tuple(selection["candidate_recipe"])
                                    )
                                    phases["REPLICATION"] = "COMPLETE"
                                    decision["status"] = "REPLICATION_COMPLETE_TRANSFER_PENDING"
                                else:
                                    decision["status"] = "REPLICATION_PARTIAL_DURABLE"
        write(
            run / "COST_SCOPE.json",
            {
                "scope": "CPU_diagnostics_or_cached_head_work_this_call",
                "seconds": time.perf_counter() - started,
                "raw_E2E": False,
                "gpu_used": False,
            },
        )
        build(run)
        bundle = publish(run, decision, command)
        print(json.dumps({**decision, "bundle": str(bundle)}, indent=2))
        return 3 if "WAITING" in decision["status"] or "PARTIAL" in decision["status"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
