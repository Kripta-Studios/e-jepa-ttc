"""Preregister and verify the separate universal-abstention CPU-fallback candidate."""

from __future__ import annotations

import argparse
import platform
import shutil
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from operational.phase_adapt_cache import atomic_receipt
from operational.phase_event_expansion.accelerated import CudaObjectiveCache
from operational.phase_event_expansion.hybrid import estimate_hybrid
from operational.phase_event_expansion.model import ExpansionResult, SolverConfig
from operational.phase_event_expansion.validation import _relative_source, validate_cached_window
from scripts.evaluate_phase_event_expansion_development import undistort_inputs
from scripts.verify_phase_event_expansion_all_cuda import (
    preflight,
    read,
    sha,
    strict_compare_results,
)


def preregister(args: argparse.Namespace, evidence: dict[str, Any]) -> None:
    """Freeze policy, tolerance, inputs and executed sources without allocating CUDA."""
    if args.output.exists():
        raise ValueError("Preregistration requires a fresh immutable directory")
    paths = {
        Path(__file__),
        Path("operational/phase_event_expansion/hybrid.py"),
        Path("tests/unit/test_phase_event_expansion_hybrid.py"),
        Path("scripts/verify_phase_event_expansion_all_cuda.py"),
        Path("tests/unit/test_phase_event_expansion_all_cuda.py"),
        Path("operational/phase_event_expansion/validation.py"),
        Path("operational/evttc_competition/geometry.py"),
    }
    paths |= {Path(p) for p in evidence["accelerator_archived_sources"]}
    receipt = {
        "schema": "phase_event_hybrid_all77_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "policy": "Run float64 CUDA; for ANY invalid GPU result rerun the original CPU solver "
        "on the same events/ROI/query/config and return the exact CPU result object unchanged",
        "rationale": "Reproduce CPU abstention diagnostics while retaining valid GPU acceleration; "
        "no accuracy claim; original GPU-only 76/77 failure remains rejected",
        "solver": evidence["recipe"],
        "queries": 77,
        "precision": "float64",
        "q_b_absolute_tolerance": 1e-5,
        "validity_reason": "exact equality",
        "result_contract": "valid requires finite q,b,TTC and q*TTC=1 within rtol1e-12; "
        "invalid requires TTC=None; missingness must match",
        "ttc_difference_bound": "abs(dq)/abs(q_cpu*q_candidate) + "
        "2e-12*(abs(1/q_cpu)+abs(1/q_candidate)); report absolute error separately",
        "optimizer_count_changes": "recorded, not asserted identical",
        "cache_policy": "one shared CudaObjectiveCache; one resident event shape; B1/B81 graphs",
        "memory_limit_bytes": 1_000_000_000,
        "labels_opened": False,
        "threads": 2,
        "source_sha256": {str(p): sha(p) for p in sorted(paths)},
        "cpu_query_sha256": evidence["cpu_query_sha256"],
        "evidence_sha256": {
            str(p): sha(p)
            for p in (
                args.cpu / "CONFIG.json",
                args.cpu / "PREDICTIONS_SEALED.json",
                args.cpu / "PREDICTIONS.npz",
                args.cache / "MANIFEST.json",
                args.cache / "CACHE_FREEZE.json",
                args.synthetic / "CONFIG.json",
                args.synthetic / "SUMMARY.json",
                args.synthetic / "CONFIRMATION_BINDING.json",
                args.profile / "CONFIG.json",
                args.profile / "SUMMARY.json",
            )
        },
        "historical_seal_limit": "CPU q/b receipts were not individually sealed; token/TTC checked "
        "against historical sealed NPZ; all receipt hashes frozen here before inference",
    }
    atomic_receipt(args.output / "CONFIG.json", receipt)
    for path in paths:
        destination = args.output / "executed_source" / path.resolve().relative_to(Path.cwd())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    atomic_receipt(
        args.output / "PREREGISTRATION_SEALED.json",
        {
            "config_sha256": sha(args.output / "CONFIG.json"),
            "inference_started": False,
            "schema": "phase_event_hybrid_preregistration_seal_v1",
        },
    )


def verify_preregistration(directory: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    """Fail before CUDA if preregistered code or any input changed."""
    if {p.name for p in directory.iterdir()} != {
        "CONFIG.json",
        "executed_source",
        "PREREGISTRATION_SEALED.json",
    }:
        raise ValueError("Expected unused preregistration; never overwrite a previous run")
    config = read(directory / "CONFIG.json")
    seal = read(directory / "PREREGISTRATION_SEALED.json")
    if (
        seal.get("config_sha256") != sha(directory / "CONFIG.json")
        or seal.get("inference_started") is not False
        or seal.get("schema") != "phase_event_hybrid_preregistration_seal_v1"
    ):
        raise ValueError("Preregistration configuration seal changed")
    if (
        config.get("schema") != "phase_event_hybrid_all77_v1"
        or config.get("solver") != evidence["recipe"]
        or config.get("cpu_query_sha256") != evidence["cpu_query_sha256"]
        or config.get("q_b_absolute_tolerance") != 1e-5
    ):
        raise ValueError("Preregistered policy/input identity differs")
    for name, expected in config["source_sha256"].items():
        if (
            sha(Path(name)) != expected
            or sha(directory / "executed_source" / _relative_source(name)) != expected
        ):
            raise ValueError("Preregistered current or archived source changed")
    for name, expected in config["evidence_sha256"].items():
        if sha(Path(name)) != expected:
            raise ValueError("Preregistered input evidence changed")
    return config


def main() -> None:
    """Use --preregister-only first, then coordinate exclusive timing before --run."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "synthetic", "cpu", "profile", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preregister-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    complete_started = time.perf_counter()
    evidence = preflight(args.cache, args.synthetic, args.cpu, args.profile)
    if args.preregister_only:
        preregister(args, evidence)
        print("PREREGISTERED; no inference or CUDA allocation")
        return
    config = verify_preregistration(args.output, evidence)
    preflight_seconds = time.perf_counter() - complete_started
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    init_started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the preregistered hybrid; no silent backend switch")
    torch.cuda.init()
    torch.cuda.synchronize()
    init_seconds = time.perf_counter() - init_started
    atomic_receipt(
        args.output / "ENVIRONMENT.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "config_sha256": sha(args.output / "CONFIG.json"),
        },
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    active = CudaObjectiveCache()
    rows = []
    loop_started = time.perf_counter()
    try:
        for i, row in enumerate(evidence["manifest"]["rows"]):
            started = time.perf_counter()
            path = args.cache / row["file"]
            if sha(path) != row["sha256"]:
                raise ValueError("Sensor changed after preflight")
            with np.load(path, allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
            validate_cached_window(arrays, row)
            read_seconds = time.perf_counter() - started
            calibration_started = time.perf_counter()
            attempt = None
            try:
                events, roi = undistort_inputs(arrays, evidence["manifest"]["calibration"])
            except ValueError as exc:
                result = ExpansionResult(
                    None,
                    None,
                    None,
                    False,
                    "calibration_inversion_failure",
                    time.perf_counter() - calibration_started,
                    {"error": str(exc)},
                )
                calibration_seconds = time.perf_counter() - calibration_started
            else:
                calibration_seconds = time.perf_counter() - calibration_started
                attempt = estimate_hybrid(
                    events, roi, int(row["query_us"]), SolverConfig(**evidence["recipe"]), active
                )
                torch.cuda.synchronize()
                result = attempt.result
            cpu = ExpansionResult(
                **{k: evidence["cpu_records"][i][k] for k in ExpansionResult.__dataclass_fields__}
            )
            record = {
                "token": row["query_id"],
                "index": row["frame_index"],
                "query_us": row["query_us"],
                "result": asdict(result),
                "hybrid_attempt": asdict(attempt) if attempt else None,
                "parity": strict_compare_results(cpu, result),
                "gpu_attempt_parity": strict_compare_results(cpu, attempt.gpu_attempt)
                if attempt
                else None,
                "cpu_query_sha256": evidence["cpu_query_sha256"][f"{i:05d}.json"],
                "read_validate_seconds": read_seconds,
                "undistort_seconds": calibration_seconds,
                "query_wall_before_receipt_seconds": time.perf_counter() - started,
                "captures_cumulative": active.capture_count,
                "replays_cumulative": active.replay_count,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            atomic_receipt(args.output / "queries" / f"{i:05d}.json", record)
            rows.append(record)
            print(
                f"QUERY {i + 1}/77 parity={record['parity']['passed']} "
                f"fallback={attempt.fallback_used if attempt else False} "
                f"wall={record['query_wall_before_receipt_seconds']:.3f}s",
                flush=True,
            )
        loop_seconds = time.perf_counter() - loop_started
        captures, replays = active.capture_count, active.replay_count
        allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    finally:
        close_started = time.perf_counter()
        active.close()
        close_seconds = time.perf_counter() - close_started
    summary = {
        "schema": "phase_event_hybrid_all77_v1",
        "rows": rows,
        "labels_opened": False,
        "all_parity_pass": len(rows) == 77 and all(r["parity"]["passed"] for r in rows),
        "all_memory_pass": max(allocated, reserved) < config["memory_limit_bytes"],
        "fallback_count": sum(
            bool(r["hybrid_attempt"] and r["hybrid_attempt"]["fallback_used"]) for r in rows
        ),
        "optimizer_count_divergences": [
            r["token"] for r in rows if not r["parity"].get("objective_count_matches", False)
        ],
        "capture_count": captures,
        "replay_count": replays,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "timing": {
            "preflight_seconds": preflight_seconds,
            "cuda_initialization_seconds": init_seconds,
            "all77_loop_with_receipts_seconds": loop_seconds,
            "cache_close_seconds": close_seconds,
            "complete_wall_before_summary_seconds": time.perf_counter() - complete_started,
        },
        "config_sha256": sha(args.output / "CONFIG.json"),
        "query_receipt_sha256": {
            p.name: sha(p) for p in sorted((args.output / "queries").glob("*.json"))
        },
    }
    atomic_receipt(args.output / "SUMMARY.json", summary)
    print(
        f"COMPLETE parity={summary['all_parity_pass']} memory={summary['all_memory_pass']}",
        flush=True,
    )
    if not summary["all_parity_pass"] or not summary["all_memory_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
