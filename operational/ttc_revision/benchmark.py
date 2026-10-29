"""Same-pass latency and numerical admission without opening TTC targets."""

# ruff: noqa: ANN401 -- frozen sensor/model records are heterogeneous.
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import torch

from operational.efficient_context.common import digest
from operational.evttc_rgb_transfer.inputs import prepare as prepare_full
from operational.evttc_rgb_transfer.model import FullGarl
from operational.evttc_transfer.inputs import prepare as prepare_legacy
from operational.evttc_transfer.models import FrozenModels
from operational.train40_system.contracts import environment
from operational.train40_system.durable_io import atomic_json
from operational.ttc_revision.inputs import EventPreparer
from operational.ttc_revision.runtime import H8Runtime


def telemetry() -> str:
    """Record desktop GPU contention rather than claiming physical exclusivity."""
    return subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout


def require_no_other_python_gpu() -> None:
    """Reject simultaneous Python GPU jobs; desktop graphics remain disclosed."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
    )
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) >= 2 and "python" in row[1].lower() and int(row[0]) != os.getpid():
            raise RuntimeError(f"another Python GPU job is active: PID {row[0]}")


def selected_rows(rows: list[dict[str, Any]], count: int = 1) -> list[dict[str, Any]]:
    """Select first chronological queries of the first sequence in each family."""
    families: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda r: (r["sequence_id"], r["anchor_us"])):
        group = families.setdefault(row["scenario_family"], [])
        if len(group) < count and (not group or row["sequence_id"] == group[0]["sequence_id"]):
            group.append(row)
    return [row for group in families.values() for row in group]


def run(
    config_path: Path, manifest_path: Path, original_cost: Path, *, cache_mib: int = 128
) -> None:
    """Compare dedicated preparation paths; record CPU, wrapper and E2E together."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not 0 <= cache_mib <= 1024:
        raise ValueError("raw cache limit must be between 0 and 1024 MiB")
    config_sha = digest(config_path)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cost = json.loads(original_cost.read_text(encoding="utf-8"))
    protocol = config["latency"]
    rows = selected_rows(manifest["rows"])
    torch.set_num_threads(protocol["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    require_no_other_python_gpu()
    freeze = {
        "config_sha256": config_sha,
        "manifest_sha256": digest(manifest_path),
        "queries": [row["query_id"] for row in rows],
        "environment": environment(),
        "telemetry_before": telemetry(),
        "status": "RUNNING",
        "source_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ("benchmark.py", "inputs.py", "runtime.py")
        },
        "retained_raw_cache_mib": cache_mib,
        "scope": "desktop WDDM; model load excluded; CPU includes raw HDF5 reads; warm filesystem",
        "system_order": "seed20261009 shuffled per query, same chronological queries per system",
    }
    freeze_path = output / "BENCHMARK.json"
    if freeze_path.exists():
        raise FileExistsError("benchmark already exists; retain it and use a new output")
    archive = output / "source_archive"
    archive.mkdir(exist_ok=True)
    for name in freeze["source_sha256"]:
        shutil.copy2(Path(__file__).with_name(name), archive / name)
    shutil.copy2(config_path, archive / "config.json")
    atomic_json(freeze_path, freeze)
    frozen = FrozenModels(Path("artifacts/train40_system_20261005"), "cuda")
    full = FullGarl(Path(cost["public_full_dir"]), Path(cost["code_root"]), "cuda")
    canonical = H8Runtime(frozen)
    fast = H8Runtime(frozen, batch=8, on_device=True)
    admission = []
    with EventPreparer() as preparer:
        for row in rows:
            old = prepare_legacy(row)
            new = preparer.prepare(row, system="both")
            for key in ("own_events", "garl_events"):
                if not np.array_equal(old[key], new[key]):
                    raise AssertionError(f"input parity failed: {row['query_id']} {key}")
            expected = np.array(list(frozen.predict(old["own_events"]).values()))
            same = canonical.predict(new["own_events"])
            compact = fast.predict(new["own_events"])
            tolerance = protocol["compact_execution_admission"]
            admitted = bool(
                np.allclose(
                    compact,
                    expected,
                    atol=tolerance["ttc_atol_seconds"],
                    rtol=tolerance["ttc_rtol"],
                )
            )
            admission.append(
                {
                    "query_id": row["query_id"],
                    "inputs_bit_exact": True,
                    "canonical_bit_exact": bool(np.array_equal(expected, same)),
                    "compact_admitted_on_query": admitted,
                    "compact_max_abs_seconds": float(np.max(np.abs(compact - expected))),
                }
            )
            if not np.array_equal(expected, same) or not admitted:
                atomic_json(output / "NUMERICAL_ADMISSION.json", {"queries": admission})
                raise AssertionError("runtime numerical admission failed")
    atomic_json(output / "NUMERICAL_ADMISSION.json", {"queries": admission})
    measurements = []
    # Historical wrapper remains a three-head system. Fast one-head is separate.
    for mode in protocol["modes"]:
        chosen = rows if mode.startswith("independent") else selected_rows(manifest["rows"], 5)
        generator = np.random.default_rng(20261009)
        for iteration in range(-protocol["warmups"], protocol["repetitions"]):
            require_no_other_python_gpu()
            with ExitStack() as stack:
                preparers = {
                    system: stack.enter_context(
                        EventPreparer(
                            cache_bytes=0 if mode.startswith("independent") else cache_mib * 1024**2
                        )
                    )
                    for system in protocol["systems"]
                }
                for row in chosen:
                    for system in generator.permutation(protocol["systems"]):
                        prep = preparers[system]
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        start = time.perf_counter()
                        if system == "h8_legacy_three":
                            prepared = prepare_legacy(row)
                        elif system == "garl_full":
                            prepared = prepare_full(row)
                        else:
                            prepared = prep.prepare(
                                row, system="garl" if system == "garl_event_only" else "h8"
                            )
                        cpu_end = time.perf_counter()
                        event_start, event_end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        event_start.record()
                        if system == "h8_legacy_three":
                            result = list(frozen.predict(prepared["own_events"]).values())
                        elif system.startswith("h8_fast"):
                            seeds = (7,) if system.endswith("one") else (7, 13, 23)
                            result = fast.predict(prepared["own_events"], seeds=seeds).tolist()
                        elif system == "garl_event_only":
                            result = [frozen.garl_predict(prepared["garl_events"])["ttc"]]
                        else:
                            result = [full.predict_sensor(prepared["sensor"])["ttc"]]
                        event_end.record()
                        torch.cuda.synchronize()
                        end = time.perf_counter()
                        measurements.append(
                            {
                                "mode": mode,
                                "system": system,
                                "query_id": row["query_id"],
                                "scenario_family": row["scenario_family"],
                                "iteration": iteration,
                                "warmup": iteration < 0,
                                "cpu_ms": (cpu_end - start) * 1000,
                                "raw_read_ms": prepared.get("diagnostics", {}).get("read_ms"),
                                "h8_voxel_ms": prepared.get("diagnostics", {}).get("h8_voxel_ms"),
                                "garl_repr_ms": prepared.get("diagnostics", {}).get("garl_repr_ms"),
                                "wrapper_ms": (end - cpu_end) * 1000,
                                "e2e_ms": (end - start) * 1000,
                                "cuda_stream_ms": event_start.elapsed_time(event_end),
                                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                                "raw_cache_hit": prepared.get("diagnostics", {}).get(
                                    "raw_cache_hit", False
                                ),
                                "prediction": json.dumps(result),
                            }
                        )
            with (output / "LATENCY_RAW.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(measurements[0]))
                writer.writeheader()
                writer.writerows(measurements)
            print(
                json.dumps({"mode": mode, "all_systems": True, "iteration": iteration}), flush=True
            )
    if digest(config_path) != config_sha:
        raise ValueError("protocol changed during timing")
    freeze.update(
        status="COMPLETE",
        telemetry_after=telemetry(),
        model_bindings=frozen.bindings,
        raw_sha256=digest(output / "LATENCY_RAW.csv"),
    )
    atomic_json(freeze_path, freeze)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment/ttc_revision_20261009.json")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/sota_campaign_20261008/dev32_expanded_rgb/QUERY_MANIFEST.json"),
    )
    parser.add_argument(
        "--original-cost",
        type=Path,
        default=Path("artifacts/sota_campaign_20261008/cost/EXECUTION_FREEZE.json"),
    )
    parser.add_argument(
        "--cache-mib", type=int, default=128, help="Bounded per-system raw-event retention"
    )
    args = parser.parse_args()
    run(args.config, args.manifest, args.original_cost, cache_mib=args.cache_mib)
