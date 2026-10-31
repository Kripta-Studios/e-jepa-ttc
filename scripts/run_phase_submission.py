"""Frozen target-blind inference of all434 official EvTTC queries with measured cost."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import platform
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch

from operational.evttc_competition.prepare import POLICY, prepare_query
from operational.evttc_competition.readers import SensorSequence
from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_adapt_training import configure_cuda
from operational.phase_robust.bundle import load_sensor_runtime
from operational.phase_robust.cache import SensorCache
from operational.phase_startup.runtime import predict_history
from operational.phase_submission.runtime import load_submission_model
from scripts.run_evttc_competition import (
    ROW_SCHEMA,
    SEQUENCES,
    append_row,
    calibration_record,
    deterministic_zip,
    expected_queries,
    freeze_json,
    identity_key,
    json_safe,
    load_existing_rows,
    preflight_inputs,
    query_identity,
    source_hashes,
    synchronize,
    write_signed_results,
)


def executed_sources() -> dict[str, str]:
    """Bind original graph/preprocessing and new router, loader and caching sources."""
    paths = [
        Path(__file__),
        Path("operational/phase_robust/cache.py"),
        Path("operational/phase_robust/bundle.py"),
        Path("operational/phase_submission/runtime.py"),
        *Path("operational/phase_startup").glob("*.py"),
    ]
    return {
        **source_hashes(),
        **{p.resolve().relative_to(Path.cwd()).as_posix(): file_sha(p) for p in paths},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "data_root", "output"):
        p.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--limit", type=int, default=0, help="Explicit partial run; no results.zip until all434"
    )
    args = p.parse_args()
    configure_cuda()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(7)
    setup_start = time.perf_counter()
    router = load_submission_model(args.bundle)
    native_runtime = load_sensor_runtime(args.bundle / "upstream", args.device)
    rows, sources, calibrations = preflight_inputs(args.data_root)
    expected = expected_queries(rows)
    frozen_sources = executed_sources()
    freeze = {
        "schema": "phase_evttc_final_replay_v1",
        "targets_read": False,
        "model_sha256": file_sha(args.bundle / "MODEL.json"),
        "sources": frozen_sources,
        "raw_sources": sources,
        "queries": {k: [query_identity(k, r) for r in v] for k, v in rows.items()},
        "calibrations": {k: calibration_record(v) for k, v in calibrations.items()},
        "original_policy": POLICY,
        "adaptation": "fixed trained inverse QA router with causal provided bbox geometry",
        "device": args.device,
        "precision": "FP32; TF32 disabled",
        "runtime_boundary": (
            "synchronized per-query raw cache read/preparation,9-component graph,"
            "geometry/router,host output; excludes one-time hashing/indexing/model "
            "load and ledger writing"
        ),
        "population": 434,
        "cache": "lossless 64MiB event chunks/16RGBframes; query-ordered",
    }
    freeze_sha = freeze_json(args.output / "RUN_FREEZE.json", freeze, resume=args.resume)
    existing = load_existing_rows(args.output / "all_rows.jsonl", freeze_sha, resume=args.resume)
    setup_seconds = time.perf_counter() - setup_start
    started = dt.datetime.now(dt.UTC).isoformat()
    completed = len(existing)
    session_queries = 0
    for sequence in SEQUENCES:
        if args.limit and session_queries >= args.limit:
            break
        with SensorSequence(args.data_root, sequence) as source:
            cached = SensorCache(source)
            boxes = np.array([r["box_xyxy"] for r in rows[sequence]])
            clocks = np.array([r["anchor_us"] for r in rows[sequence]])
            for row in rows[sequence]:
                identity = query_identity(sequence, row)
                if identity_key(identity) in existing:
                    continue
                if args.limit and session_queries >= args.limit:
                    break
                synchronize(args.device)
                start = time.perf_counter()
                prepared = prepare_query(
                    cast(SensorSequence, cached), rows[sequence], row, calibrations[sequence]
                )
                prepared_seconds = time.perf_counter() - start
                result = native_runtime.predict(prepared.events, prepared.rgb, row["anchor_us"])
                synchronize(args.device)
                native_seconds = time.perf_counter() - start - prepared_seconds
                native = np.array(
                    [
                        v if v is not None else np.nan
                        for v in (result.model_ttc, result.rgb_ttc, result.event_ttc)
                    ],
                    np.float32,
                )
                pred, physical = predict_history(router, native, boxes, clocks, row["anchor_us"])
                elapsed = time.perf_counter() - start
                if not np.isfinite(pred):
                    raise ValueError(
                        f"Unavailable query retained as failure, no replacement: {identity}"
                    )
                record = {
                    "schema": ROW_SCHEMA,
                    "freeze_sha256": freeze_sha,
                    "identity": identity,
                    "status": "ok",
                    "prediction_ttc_seconds": pred,
                    "cost_time_seconds": elapsed,
                    "native_ttc_seconds": [float(v) if np.isfinite(v) else None for v in native],
                    "preparation_seconds": prepared_seconds,
                    "native_graph_seconds": native_seconds,
                    "geometry_router_seconds": elapsed - prepared_seconds - native_seconds,
                    "geometry": json_safe(physical),
                    "prepared": json_safe(prepared.details),
                    "native_diagnostics": json_safe(dataclasses.asdict(result)),
                    "cache_hits": cached.hits,
                    "cache_misses": cached.misses,
                }
                append_row(args.output / "all_rows.jsonl", record)
                existing[identity_key(identity)] = record
                completed += 1
                session_queries += 1
                print(
                    f"{completed}/434 {sequence}:{row['index']} ttc={pred:.6f} cost={elapsed:.3f}s",
                    flush=True,
                )
    if (
        executed_sources() != frozen_sources
        or file_sha(args.bundle / "MODEL.json") != freeze["model_sha256"]
    ):
        raise ValueError("Executed sources or model changed during replay")
    load_submission_model(args.bundle)
    current_rows, current_sources, _ = preflight_inputs(args.data_root)
    if current_sources != sources or expected_queries(current_rows) != expected:
        raise ValueError("Sensor/query inputs changed during replay")
    if completed != 434:
        print(f"PARTIAL {completed}/434; resume required; no results ZIP", flush=True)
        return
    records = list(existing.values())
    validation = write_signed_results(args.output / "results", records, expected)
    deterministic_zip(args.output / "results.zip", args.output / "results", validation["files"])
    environment = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if args.device == "cuda"
        else None,
        "started_utc": started,
        "finished_utc": dt.datetime.now(dt.UTC).isoformat(),
        "setup_seconds_last_session": setup_seconds,
        "device": args.device,
    }
    atomic_receipt(args.output / "ENVIRONMENT.json", environment)
    summaries = {}
    for sequence in SEQUENCES:
        chosen = [r for r in records if r["identity"]["sequence_id"] == sequence]
        costs = np.array([r["cost_time_seconds"] for r in chosen])
        summaries[sequence] = {
            "rows": len(chosen),
            "negative": sum(r["prediction_ttc_seconds"] < 0 for r in chosen),
            "mean_cost_seconds": float(costs.mean()),
            "p95_cost_seconds": float(np.quantile(costs, 0.95)),
        }
    atomic_receipt(
        args.output / "COMPLETE.json",
        {
            "rows": completed,
            "finite_rows": completed,
            "targets_read": False,
            "by_sequence": summaries,
            "freeze_sha256": freeze_sha,
            "ledger_sha256": file_sha(args.output / "all_rows.jsonl"),
            "results_zip_sha256": file_sha(args.output / "results.zip"),
            "validation": validation,
        },
    )
    print("COMPLETE frozen434query submission", flush=True)


if __name__ == "__main__":
    main()
