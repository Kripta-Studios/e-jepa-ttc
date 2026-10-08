"""Measure exactly the first frozen FCWD query per sequence, without model or GT."""

from __future__ import annotations

import gc
import os
import threading
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import psutil
import torch

from operational.efficient_context.common import atomic_json, digest
from operational.sota_eval.fcwd_inputs import prepare
from operational.sota_eval.fcwd_run import (
    ROOT,
    read,
    source_binding,
    stamp,
    tensor_receipt,
    validate_manifest,
)


def main() -> None:
    """Emit partial receipts immediately, then a three-query timing summary."""
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    manifest_path = ROOT / "artifacts/sota_campaign_20261008/fcwd_population/QUERY_MANIFEST.json"
    output = ROOT / "artifacts/sota_campaign_20261008/fcwd_runner_qa"
    rows = validate_manifest(read(manifest_path))
    binding = source_binding(
        manifest_path,
        ROOT / "artifacts/train40_system_20261005",
        ROOT / "artifacts/evttc_rgb_transfer_20261008/public_garl",
        Path("E:/Garl-TTC"),
        "cuda",
    )
    atomic_json(output / "LOCAL_SOURCE_CLOSURE.json", binding)
    process = psutil.Process()
    receipts = []
    for sequence in ("FCWD1", "FCWD2", "FCWD3"):
        row = next(item for item in rows if item["sequence_id"] == sequence)
        gc.collect()
        baseline = process.memory_info().rss
        peak = [baseline]
        stop = threading.Event()

        def sample() -> None:
            while not stop.wait(0.02):
                peak[0] = max(peak[0], process.memory_info().rss)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        started = time.perf_counter()
        try:
            result = prepare(row)
            elapsed = time.perf_counter() - started
            peak[0] = max(peak[0], process.memory_info().rss)
        finally:
            stop.set()
            sampler.join()
        receipt = {
            "sequence_id": sequence,
            "query_id": row["query_id"],
            "elapsed_prepare_seconds": elapsed,
            "process_rss_before_bytes": baseline,
            "process_rss_peak_sampled_bytes": peak[0],
            "process_peak_wset_bytes": getattr(process.memory_info(), "peak_wset", None),
            "sampling_interval_seconds": 0.02,
            "tensors": {
                name: None if result[name] is None else tensor_receipt(result[name])
                for name in ("own_events", "garl_events", "valid")
            },
            "full_unavailable_reason": result["garl_full_unavailable_reason"],
            "gt_read": False,
            "gpu_initialized": torch.cuda.is_initialized(),
        }
        receipts.append(receipt)
        atomic_json(output / f"PREPARE_{sequence}.json", receipt)
        print(f"{sequence}: {elapsed:.3f} s; peak RSS {peak[0] / 1024**2:.1f} MiB", flush=True)
        del result
    assert binding == source_binding(
        manifest_path,
        ROOT / "artifacts/train40_system_20261005",
        ROOT / "artifacts/evttc_rgb_transfer_20261008/public_garl",
        Path("E:/Garl-TTC"),
        "cuda",
    )
    validate_manifest(read(manifest_path))
    projected = sum(
        item["elapsed_prepare_seconds"] * sum(row["sequence_id"] == item["sequence_id"] for row in rows)
        for item in receipts
    )
    atomic_json(output / "CPU_PREPARE_PROBE.json", {
        "status": "COMPLETE", "created_utc": stamp(), "threads": 2,
        "manifest_sha256": digest(manifest_path),
        "source_closure_sha256": digest(output / "LOCAL_SOURCE_CLOSURE.json"),
        "probe_script_sha256": digest(Path(__file__)),
        "query_selection": "first frozen query per sequence, no selection using GT or outputs",
        "queries": receipts, "projected_630_query_prepare_seconds": projected,
        "projection_limitation": "Only three first queries; not sustained throughput or total inference ETA. Disk cache state uncontrolled; GPU inference excluded.",
        "gt_read": False, "gpu_initialized": torch.cuda.is_initialized(), "optimizer_updates": 0,
    })


if __name__ == "__main__":
    main()
