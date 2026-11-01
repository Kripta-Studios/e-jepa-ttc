"""Label-blind full CPU/CUDA solver parity, including setup and warm-cache timing."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from operational.phase_event_expansion.accelerated import CudaObjectiveCache, estimate_accelerated
from operational.phase_event_expansion.model import (
    ExpansionResult,
    SolverConfig,
    estimate_expansion,
)
from operational.phase_event_expansion.synthetic import make_fixture
from scripts.evaluate_phase_event_expansion_development import undistort_inputs


def compare_results(cpu: ExpansionResult, cuda: ExpansionResult) -> dict[str, object]:
    """Fail closed on validity/reason changes or q,b differences exceeding1e-5."""
    present = (cpu.q is None) == (cuda.q is None)
    maximum = None
    if cpu.q is not None and cuda.q is not None:
        if cpu.translation is None or cuda.translation is None:
            present = False
        else:
            maximum = float(
                np.max(
                    np.abs(
                        np.array([cpu.q, *cpu.translation]) - np.array([cuda.q, *cuda.translation])
                    )
                )
            )
    identity = cpu.valid == cuda.valid and cpu.reason == cuda.reason
    numeric = present and (maximum is None or maximum <= 1e-5)
    return {
        "passed": bool(identity and numeric),
        "validity_reason_match": identity,
        "maximum_q_b_error": maximum,
        "numeric_pass": numeric,
        "objective_count_matches": cpu.diagnostics.get("objective_evaluations")
        == cuda.diagnostics.get("objective_evaluations"),
        "ttc_absolute_error": abs(cpu.ttc_seconds - cuda.ttc_seconds)
        if cpu.ttc_seconds is not None and cuda.ttc_seconds is not None
        else None,
    }


def sha(path: Path) -> str:
    """Hash immutable evidence and source bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Evaluate synthetic and the five predeclared sensor-only Slider500 windows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new immutable output directory")
    if not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA required for this comparison")
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    config = SolverConfig(objective="temporal_ncc", reference="endpoint")
    manifest = json.loads((args.cache / "MANIFEST.json").read_text())
    if manifest["sequence_id"] != "Slider-500" or manifest["queries"] != 77:
        raise ValueError("Only the predeclared eligible Slider500 sensor cache is permitted")
    if sha(args.cache / "CACHE_FREEZE.json") != manifest["cache_freeze_sha256"]:
        raise ValueError("Changed sensor cache freeze")
    indices = [0, 10, 30, 50, 76]
    sources = [
        Path(__file__),
        Path("operational/phase_event_expansion/accelerated.py"),
        Path("operational/phase_event_expansion/torch_objective.py"),
        Path("operational/phase_event_expansion/model.py"),
        Path("operational/phase_event_expansion/synthetic.py"),
        Path("scripts/evaluate_phase_event_expansion_development.py"),
        Path("tests/unit/test_phase_event_expansion_accelerated.py"),
    ]
    receipt = {
        "schema": "phase_event_accelerated_parity_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "solver": asdict(config),
        "precision": "float64",
        "synthetic_seeds": [7, 31, 47],
        "slider500_indices": indices,
        "q_b_absolute_tolerance": 1e-5,
        "validity_reason": "exact equality",
        "labels_opened": False,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(),
        "threads": 2,
        "cache_policy": "one event shape; mutable observations; graph B1 andB81",
        "source_sha256": {str(p): sha(p) for p in sources},
        "cache_manifest_sha256": sha(args.cache / "MANIFEST.json"),
    }
    args.output.mkdir(parents=True)
    (args.output / "CONFIG.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    for path in sources:
        destination = args.output / "executed_source" / path.resolve().relative_to(Path.cwd())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    rows: list[dict[str, Any]] = []

    def evaluate(
        name: str,
        events: np.ndarray,
        roi: np.ndarray,
        query_us: int,
        preparation_seconds: float = 0.0,
    ) -> None:
        start = time.perf_counter()
        cpu = estimate_expansion(events, roi, query_us, config)
        cpu_wall = time.perf_counter() - start
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        cache = CudaObjectiveCache()
        cold = estimate_accelerated(events, roi, query_us, config, cache)
        cold_wall = time.perf_counter() - start
        start = time.perf_counter()
        warm = estimate_accelerated(events, roi, query_us, config, cache)
        warm_wall = time.perf_counter() - start
        peak_allocated, peak_reserved = (
            torch.cuda.max_memory_allocated(),
            torch.cuda.max_memory_reserved(),
        )
        row = {
            "name": name,
            "events_before_budget": len(events),
            "labels_opened": False,
            "cpu": asdict(cpu),
            "cuda_cold": asdict(cold),
            "cuda_warm": asdict(warm),
            "cold_parity": compare_results(cpu, cold),
            "warm_parity": compare_results(cpu, warm),
            "wall_seconds": {
                "cpu": cpu_wall,
                "cuda_cold": cold_wall,
                "cuda_warm": warm_wall,
                "shared_read_undistort": preparation_seconds,
                "cpu_including_shared": cpu_wall + preparation_seconds,
                "cuda_cold_including_shared": cold_wall + preparation_seconds,
                "cuda_warm_including_shared": warm_wall + preparation_seconds,
            },
            "capture_count": cache.capture_count,
            "replay_count": cache.replay_count,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "memory_pass": max(peak_allocated, peak_reserved) < 1_000_000_000,
        }
        rows.append(row)
        (args.output / f"{name}.json").write_text(
            json.dumps(row, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        cache.close()
        print(
            f"{name}: cold_match={row['cold_parity']['passed']} "
            f"warm_match={row['warm_parity']['passed']} "
            f"cpu={cpu_wall:.4f}s cold={cold_wall:.4f}s warm={warm_wall:.4f}s "
            f"q_error={row['warm_parity']['maximum_q_b_error']} reason={cpu.reason}/{warm.reason}",
            flush=True,
        )

    for seed in (7, 31, 47):
        for kind in ("expansion", "receding", "translation", "stationary", "noise", "edge"):
            fixture = make_fixture(seed, kind)
            evaluate(f"synthetic_{seed}_{kind}", fixture.events, fixture.roi, fixture.query_us)
    for index in indices:
        row = manifest["rows"][index]
        path = args.cache / row["file"]
        if sha(path) != row["sha256"]:
            raise ValueError("Changed selected sensor window")
        started = time.perf_counter()
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in ("x", "y", "t_us", "p", "roi_xyxy")}
        events, roi = undistort_inputs(arrays, manifest["calibration"])
        preparation = time.perf_counter() - started
        evaluate(f"slider500_{index:02d}", events, roi, int(row["query_us"]), preparation)
    summary = {
        "rows": rows,
        "labels_opened": False,
        "production_integrated": False,
        "all_parity_pass": all(
            r["cold_parity"]["passed"] and r["warm_parity"]["passed"] for r in rows
        ),
        "all_memory_pass": all(r["memory_pass"] for r in rows),
        "optimizer_count_divergences": [
            r["name"]
            for r in rows
            if not r["cold_parity"]["objective_count_matches"]
            or not r["warm_parity"]["objective_count_matches"]
        ],
    }
    (args.output / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"COMPLETE all_parity_pass={summary['all_parity_pass']} "
        f"memory={summary['all_memory_pass']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
