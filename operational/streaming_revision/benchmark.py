"""Bounded TRAIN40 streaming pilot against native Garl, with per-query receipts."""

# ruff: noqa: ANN401 -- experiment records contain heterogeneous metadata.
from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.data.garl_official_preprocessing import official_square_box
from e_jepa_ttc.efficient_context.garl_input import native_feature, read_native_window
from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool
from operational.efficient_context.common import digest
from operational.evttc_transfer.models import FrozenModels
from operational.sota_evidence.metrics import observations
from operational.sota_evidence.test_inputs import (
    EXPOSURE_COLUMNS,
    INPUT_COLUMNS,
    describe_input,
    supported_job,
)
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.garl_predictions import release_ttc
from operational.ttc_revision.benchmark import require_no_other_python_gpu, telemetry

from .distill import load_student
from .kernels import GarlForward, GraphModule
from .packets import PacketRing, native_live_window
from .preparation import IncrementalPreparer, Query
from .runtime import StreamRuntime
from .state import Policy


def run(args: argparse.Namespace) -> None:
    """Read labels only after chronological query selection; never use official test rows."""
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        Path(__file__).parent,
        args.output / "source_archive",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    require_no_other_python_gpu()
    source = pd.read_parquet(
        args.campaign / "TRAIN40_ROWS.parquet", columns=[*INPUT_COLUMNS, "track_id", "ttc"]
    )
    student_receipt = json.loads(args.student.with_name("RESULT.json").read_text(encoding="utf-8"))
    sequences = sorted(student_receipt["validation_sequences"])[: args.sequences]
    selected = []
    for sequence in sequences:
        group = cast(pd.DataFrame, source.loc[source.sequence_id == sequence]).sort_values(
            ["track_id", "timestamp_us"]
        )
        track = group.iloc[0].track_id
        selected.extend(
            cast(pd.DataFrame, group.loc[group.track_id == track])
            .head(args.queries)
            .to_dict("records")
        )
    media = pd.read_parquet(args.raw_root / "data/train.parquet", columns=list(EXPOSURE_COLUMNS))
    exposures = {(r["sequence_id"], r["rgb_member_path"]): r for r in media.to_dict("records")}
    pool = ReaderPool()
    frozen = FrozenModels(args.campaign, "cuda")
    garl_graph = (
        GraphModule(GarlForward(cast(torch.nn.Module, frozen.garl))) if args.garl_graph else None
    )
    baselines = ["garl_event", "garl_event_graph"] if args.garl_graph else ["garl_event"]
    if args.garl_native_input:
        baselines.append("garl_event_native_input")
    policies = {
        "h8_reference": (Policy(reuse=False), {}),
        "h8_reuse": (Policy(), {}),
        "h8_vectorized": (Policy(), {"vectorized": True}),
        "h4_reuse": (Policy(history=4), {}),
        "h2_reuse": (Policy(history=2), {}),
        "h8_bf16": (Policy(), {"precision": "bf16"}),
        "h8_fp16": (Policy(), {"precision": "fp16"}),
        "h8_student": (Policy(), {"student": load_student(args.student)}),
        "h8_compiled_encoder": (Policy(), {"compile_encoder": True}),
        "h8_aggressive": (
            Policy(
                time_tolerance_us=25000, min_roi_iou=0.25, max_log_scale=0.7, max_age_us=1000000
            ),
            {},
        ),
    }
    policies.update(
        {
            "h8_warp": (Policy(), {}),
            "h8_warp_student": (Policy(), {"student": load_student(args.student)}),
            "h8_warp_graph": (Policy(), {"compile_full": True}),
            "h8_warp_graph_b2": (Policy(), {"compile_full": True, "producer_batch": 2}),
            "h8_warp_b2": (Policy(), {"producer_batch": 2}),
            "h8_warp_graph_fixed": (
                Policy(),
                {"compile_full": True, "producer_batch": 2, "pad_heads": True},
            ),
            "h8_warp_graph_student": (
                Policy(),
                {"compile_full": True, "student": load_student(args.student)},
            ),
        }
    )
    if args.variants:
        requested = args.variants.split(",")
        if set(requested) - policies.keys():
            raise ValueError("unknown requested benchmark variant")
        policies = {name: policies[name] for name in requested}
    runners = {
        name: StreamRuntime(frozen, policy, **options)
        for name, (policy, options) in policies.items()
    }
    current_path = [""]
    live_ring = PacketRing()
    live_sequence = ""

    def read(a: int, b: int) -> dict[str, np.ndarray]:
        return (
            live_ring.read_window(a, b)
            if args.live_input
            else pool.get(current_path[0]).read_window(a, b)
        )

    preparers = {
        name: IncrementalPreparer(
            read,
            raw_bytes=64 * 1024**2,
            voxel_bytes=16 * 1024**2,
            approximate_voxels="warp" in name,
            roi_reader=live_ring.read_roi if args.roi_first else None,
        )
        for name in runners
    }
    record = {
        "status": "RUNNING",
        "scope": "TRAIN40 student-heldout pilot; teacher saw these sequences",
        "selection": "first chronological queries of first track in sorted heldout sequences",
        "queries": [r["sample_token"] for r in selected],
        "sequences": sequences,
        "source_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        "model_bindings": frozen.bindings,
        "student_sha256": digest(args.student),
        "policies": {
            name: {**asdict(policy), **runners[name].configuration}
            for name, (policy, _) in policies.items()
        },
        "telemetry_before": telemetry(),
        "test_labels_read": False,
        "cold_start_included": True,
        "raw_root": str(args.raw_root),
        "timing_scope": (
            "Sensor-resident replay; ingestion and ROI/voxel CPU included; "
            "raw disk reads excluded; synchronized GPU wrapper"
            if args.live_input
            else "HDF5 replay; CPU includes reads and voxels; GPU wrapper wall-time synchronized"
        ),
        "contention": "desktop WDDM and concurrent CPU/IO campaign; no other Python GPU job",
        "garl_optimization": "same persistent reader; exact raw endpoint cache, four windows",
        "live_input": args.live_input,
        "baseline_variants": baselines,
        "roi_first": args.roi_first,
        "order_seed": args.order_seed,
        "cpu_isolation_receipt": os.environ.get("STREAMING_CPU_ISOLATION_RECEIPT"),
        "live_scope": (
            "If enabled: prerecorded events replayed into a causal packet ring; "
            "file reads excluded for BOTH systems, ring ingestion included; no physical sensor test"
        ),
    }
    atomic_json(args.output / "RESULT.json", record)
    rng = np.random.default_rng(args.order_seed)
    measurements, failures = [], []
    garl_cache: OrderedDict[tuple, dict] = OrderedDict()
    disabled: set[str] = set()
    try:
        for ordinal, row in enumerate(selected):
            job = describe_input(row, exposures, args.raw_root, split="train")
            reader = pool.get(job["path"])
            job = supported_job(
                job, int(reader.datasets["events/t"][0]), int(reader.datasets["events/t"][-1])
            )
            current_path[0] = job["path"]
            ingest_ms = 0.0
            input_refill = False
            if args.live_input:
                if live_sequence != job["sequence"]:
                    live_ring.reset()
                    live_sequence = job["sequence"]
                first = int(np.asarray(job["windows"]).min()) - 350000
                # Sparse annotation queries can jump tens of seconds. Refill
                # only the bounded required history, never allocate the gap.
                # This is an explicit cold refill, not continuous ingestion cost.
                if live_ring.end is not None and live_ring.end < first:
                    live_ring.reset()
                input_refill = live_ring.end is None
                start = (
                    max(int(reader.datasets["events/t"][0]), (first // 1000) * 1000)
                    if live_ring.end is None
                    else live_ring.end
                )
                packet = reader.read_window(start, job["anchor"])
                tick = time.perf_counter()
                live_ring.push(packet, start, job["anchor"])
                ingest_ms = (time.perf_counter() - tick) * 1000
                for i, (a, b) in enumerate(row["event_windows_us"][-2:]):
                    native = read_native_window(pool, Path(job["path"]), int(a), int(b))
                    buffered = native_live_window(live_ring, int(a), int(b))
                    if any(not np.array_equal(native[k], buffered[k]) for k in native):
                        raise ValueError("live Garl input differs from release-native raw slicing")
                    if args.roi_first:
                        boxes = [
                            (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                            for b in row["boxes_xyxy"][-2:]
                        ]
                        square = official_square_box(boxes, i)
                        cropped = native_live_window(live_ring, int(a), int(b), square)
                        if not torch.equal(
                            native_feature(native, square), native_feature(cropped, square)
                        ):
                            raise ValueError("ROI-first Garl tensor differs from native input")
            query = Query(
                job["sequence"],
                str(row["track_id"]),
                job["anchor"],
                job["available"],
                tuple((int(w[0]), int(w[1])) for w in job["windows"]),
                tuple(map(float, job["square"])),
                job["delta"],
                5.0,
                tuple(bool(v) for v in job["valid"]),
            )
            for name in rng.permutation([*runners, *baselines]):
                if name in disabled:
                    continue
                require_no_other_python_gpu()
                try:
                    if name in baselines:
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        boxes = [
                            (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                            for b in row["boxes_xyxy"][-2:]
                        ]
                        tensors = []
                        for i, (start, end) in enumerate(row["event_windows_us"][-2:]):
                            square = official_square_box(boxes, i)
                            early_crop = args.roi_first and name != "garl_event_native_input"
                            key = (
                                query.sequence,
                                int(start) // 1000,
                                int(end) // 1000,
                                square if early_crop else None,
                            )
                            if key not in garl_cache:
                                garl_cache[key] = (
                                    native_live_window(
                                        live_ring,
                                        int(start),
                                        int(end),
                                        square if early_crop else None,
                                    )
                                    if args.live_input
                                    else read_native_window(
                                        pool, Path(job["path"]), int(start), int(end)
                                    )
                                )
                            tensors.append(
                                native_feature(garl_cache[key], official_square_box(boxes, i))
                            )
                            while len(garl_cache) > 4:
                                garl_cache.popitem(last=False)
                        sensor = torch.cat(tensors).numpy()
                        prepared = time.perf_counter()
                        if name == "garl_event_graph":
                            assert garl_graph is not None
                            with torch.inference_mode():
                                heights, _ = garl_graph(
                                    torch.from_numpy(sensor[None]).to(frozen.device)
                                )
                            heights = heights.float().cpu().numpy()
                            if heights.shape != (1, 2) or not np.isfinite(heights).all():
                                raise ValueError("invalid graph Garl heights")
                            prediction = float(release_ttc(heights, float(frozen.garl.dT))[0])
                        else:
                            prediction = frozen.garl_predict(sensor)["ttc"]
                        torch.cuda.synchronize()
                        finished = time.perf_counter()
                        value = {
                            "ttc": prediction,
                            "cpu_ms": (prepared - started) * 1000,
                            "producer_ms": (finished - prepared) * 1000,
                            "total_ms": (finished - started) * 1000,
                            "computed": 1,
                            "reused": 0,
                        }
                    else:
                        value = runners[name].predict(query, preparers[name])
                    value["input_ingest_ms"] = ingest_ms
                    value["input_cold_refill"] = input_refill
                    value["packet_ring_bytes"] = live_ring.retained_bytes
                    value["total_ms"] += ingest_ms
                    value["cpu_ms"] += ingest_ms
                    measurements.append(
                        {
                            "variant": name,
                            "ordinal": ordinal,
                            "sample_token": row["sample_token"],
                            "sequence": query.sequence,
                            "truth_ttc": float(row["ttc"]),
                            **value,
                        }
                    )
                except Exception as exc:
                    disabled.add(name)
                    failures.append({"variant": name, "ordinal": ordinal, "error": repr(exc)})
                atomic_json(
                    args.output / "PROGRESS.json",
                    {
                        "ordinal": ordinal,
                        "completed_measurements": len(measurements),
                        "failures": failures,
                    },
                )
                with (args.output / "ROWS.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "variant": name,
                                "ordinal": ordinal,
                                "last_measurement": measurements[-1] if measurements else None,
                                "failures": failures,
                            },
                            default=str,
                        )
                        + "\n"
                    )
            print(
                f"query {ordinal + 1}/{len(selected)}; measurements={len(measurements)}", flush=True
            )
    except Exception as exc:
        atomic_json(
            args.output / "RESULT.json",
            {
                **record,
                "status": "FAILED",
                "error": repr(exc),
                "completed_measurements": len(measurements),
                "failures": failures,
            },
        )
        raise
    finally:
        pool.close()
    frame = pd.DataFrame(measurements)
    frame.to_csv(args.output / "PREDICTIONS.csv", index=False)
    summary: list[dict[str, Any]] = []
    for name, group in frame.groupby("variant"):
        truth, prediction = group.truth_ttc.to_numpy(float), group.ttc.to_numpy(float)
        mid = observations(truth, prediction)["garl_mid"]
        warm = group.loc[group.groupby("sequence").cumcount() > 0]
        summary.append(
            {
                "variant": name,
                "n": len(group),
                "complete": len(group) == len(selected),
                "mean_total_ms": float(group.total_ms.mean()),
                "warm_median_ms": float(warm.total_ms.median()),
                "warm_p95_ms": float(warm.total_ms.quantile(0.95)),
                "mean_cpu_ms": float(group.cpu_ms.mean()),
                "mean_producer_ms": float(group.producer_ms.mean()),
                "mean_reused": float(group.reused.mean()),
                "mean_MiD": float(np.mean(mid)),
                "MAE_s": float(np.abs(prediction - truth).mean()),
                "signed_bias_s": float((prediction - truth).mean()),
            }
        )
    pd.DataFrame(summary).to_csv(args.output / "SUMMARY.csv", index=False)
    atomic_json(
        args.output / "RESULT.json",
        {
            **record,
            "status": "COMPLETE_PILOT",
            "failures": failures,
            "summary": summary,
            "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(),
            "telemetry_after": telemetry(),
        },
    )


def main() -> None:
    """Reserve a conservative bounded budget before loading CUDA; preserve timeout evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "raw-root", "output", "student", "gpu-ledger"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=6)
    parser.add_argument("--order-seed", type=int, default=20261010)
    parser.add_argument("--sequences", type=int, default=2)
    parser.add_argument("--budget-seconds", type=float, default=480)
    parser.add_argument("--variants", default="", help="Comma-separated experimental variant names")
    parser.add_argument(
        "--live-input", action="store_true", help="Sensor-resident replay for all systems"
    )
    parser.add_argument(
        "--garl-graph", action="store_true", help="Also measure native Garl with CUDA graph capture"
    )
    parser.add_argument(
        "--garl-native-input",
        action="store_true",
        help="Also measure Garl full-sensor native input construction, without early ROI crop",
    )
    parser.add_argument(
        "--roi-first",
        action="store_true",
        help="Crop compressed packets before decoding, for both models",
    )
    args = parser.parse_args()
    if args.queries < 2 or args.sequences < 1 or not 0 < args.budget_seconds <= 600:
        parser.error("positive bounded experiment dimensions required")
    if args.roi_first and not args.live_input:
        parser.error("ROI-first requires live input")
    ledger = args.gpu_ledger / f"GPU_QA_ACCOUNTING_STREAM_{time.time_ns()}.json"
    atomic_json(
        ledger,
        {
            "seconds": args.budget_seconds,
            "status": "RESERVED_UPPER_BOUND",
            "purpose": "streaming optimization pilot",
            "output": str(args.output),
        },
    )
    start = time.monotonic()
    watchdog = threading.Timer(args.budget_seconds, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    try:
        run(args)
    finally:
        watchdog.cancel()
        atomic_json(
            ledger,
            {
                "seconds": time.monotonic() - start,
                "status": "ACCOUNTED_WALL_TIME",
                "purpose": "streaming optimization pilot",
                "output": str(args.output),
            },
        )


if __name__ == "__main__":
    main()
