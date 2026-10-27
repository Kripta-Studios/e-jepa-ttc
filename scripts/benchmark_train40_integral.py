"""Validate the complete TRAIN40 chain with a causal GPU event-history ring.

This is a primary-object, last-packet replay experiment. Each registered query
has its own exact clock phase; the historical ring is rebuilt outside the
last-packet clock. Raw CPU packets and RGB frames are available at its start.
All current ROI processing, ingestion, transfers and model stages are timed.
This is neither a disk-inclusive latency comparison nor a paced 20-object run.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from operational.evttc_transfer.inputs import EvTTCEventReader  # noqa: E402
from operational.rgb_port import transfer as engine  # noqa: E402
from operational.train40_runtime.integral import IntegralEventHistory  # noqa: E402
from operational.train40_runtime.measurement import (  # noqa: E402
    latency_distribution,
    prediction_parity,
    snapshot_files,
    verify_snapshot,
)
from operational.train40_runtime.predictor import PersistentTrain40Predictor  # noqa: E402
from scripts.benchmark_train40_runtime import (  # noqa: E402
    _candidate_inventory,
    _source_inventory,
    _validate_output,
    _validate_reference,
    _write,
)
from scripts.final_train40_transfer import (  # noqa: E402
    load_models,
    read,
    reject_targets,
    sha,
    verify_candidate,
    verify_frozen_runtime,
)

SCOPE = "REGISTERED_SINGLE_OBJECT_LAST_PACKET_REPLAY_WITH_PER_QUERY_HISTORY_INITIALIZATION"


def run(args: argparse.Namespace) -> None:
    """Keep full parity coverage and preserve original bytes even on failure."""
    _validate_output(args.output, [args.candidate, args.reference, args.manifest.parent])
    args.output.mkdir(parents=True, exist_ok=False)
    paths, binding = verify_candidate(args.candidate)
    frozen = verify_frozen_runtime(args.candidate)
    all_rows = read(args.manifest)["rows"]
    reject_targets(all_rows)
    reference, reference_files, reference_binding = _validate_reference(
        args.reference, args.manifest, all_rows, binding
    )
    indices = args.indices if args.indices is not None else list(range(len(all_rows)))
    if (
        not indices
        or len(indices) != len(set(indices))
        or any(not 0 <= index < len(all_rows) for index in indices)
    ):
        raise ValueError("Indices must be unique, nonempty and inside the frozen population")
    sources = [*_source_inventory(frozen, reference_binding), Path(__file__).resolve()]
    original = snapshot_files(
        [*_candidate_inventory(args.candidate), *reference_files, args.manifest, *sources]
    )
    _write(args.output / "original_snapshot.json", original)
    selected_reference = [reference[index] for index in indices]
    expected = [all_rows[index]["query_id"] for index in indices]
    records: list[dict[str, Any]] = []
    predictor = None
    reader = None
    reader_path = None
    caught: BaseException | None = None
    completed = False
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(7)
        binding.update(
            scope=SCOPE,
            indices=indices,
            manifest_sha256=sha(args.manifest),
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            sources={str(path.relative_to(ROOT)): sha(path) for path in sources},
            device=torch.cuda.get_device_name(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            python=platform.python_version(),
            precision="FP32_TF32_OFF",
            seed=7,
            rgb_workers=args.rgb_workers,
            parallel_branches=args.parallel,
            input_boundary="last_required_raw_CPU_packet_and_RGB_frames_available",
            output_boundary="final_E_CTX_and_R_CTX_TTC_available_on_host",
            history_initialization_excluded=True,
            disk_replay_excluded=True,
            original_input_tensor_check=not args.skip_input_check,
            primary_objects_per_update=1,
            target_20_objects_50hz_demonstrated=False,
        )
        _write(args.output / "binding.json", binding)
        models, normalizers = load_models(paths, "cuda")
        predictor = PersistentTrain40Predictor(
            models,
            normalizers,
            graphs=True,
            context_graphs=True,
            batch_rgb=True,
            block_graphs=True,
            reuse_encodings=True,
            parallel_branches=args.parallel,
            rgb_workers=args.rgb_workers,
        )
        history = IntegralEventHistory(device="cuda", max_objects=1, roi_batch_size=1)
        # Prime only fixed shapes on artificial startup data, once per service.
        # No query-specific model execution is excluded after observations arrive.
        synthetic_query = {"anchor_us": 1_000_000}
        synthetic_rgb = [
            (
                np.zeros((3, 3, 128, 128), dtype=np.float32),
                {
                    "delta_t_s": [0.1, 0.1],
                    "first_sensor_ts_us": 400_000 + i * 100_000,
                    "anchor_us": 600_000 + i * 100_000,
                    "available_us": 1_000_000,
                },
            )
            for i in range(5)
        ]
        prime_start = time.perf_counter()
        predictor.infer_prepared(synthetic_query, torch.zeros((4, 3, 12, 128, 128)), synthetic_rgb)
        torch.cuda.synchronize()
        _write(
            args.output / "service_startup.json",
            {
                "synthetic_shape_priming_seconds": time.perf_counter() - prime_start,
                "queries_primed": 0,
            },
        )
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="train40-rgb-request"
        ) as rgb_worker:
            for position, index in enumerate(indices):
                row = all_rows[index]
                raw_path = Path(row["raw_path"])
                if raw_path != reader_path:
                    if reader is not None:
                        reader.close()
                    reader = EvTTCEventReader(raw_path).open()
                    reader_path = raw_path
                assert reader is not None
                anchor = int(row["anchor_us"])
                startup_begin = time.perf_counter()
                startup = reader.read_window(anchor - 620_000, anchor - 20_000)
                packet = reader.read_window(anchor - 20_000, anchor)
                history.initialize(startup, anchor - 620_000, anchor - 20_000)
                del startup
                # Declare raw frames available without doing any current ROI
                # crop, normalization, model inference or event query encoding.
                predictor.inputs.preload_raw_rgb(row)
                torch.cuda.synchronize()
                startup_seconds = time.perf_counter() - startup_begin
                start_ns = time.perf_counter_ns()
                rgb_future = rgb_worker.submit(predictor.inputs.raw_rgb_observations, row)
                history.ingest(packet, anchor - 20_000, anchor)
                event_sensor = history.query(row["square_xyxy"])
                event_enqueued_ns = time.perf_counter_ns()
                rgb, reason = rgb_future.result()
                rgb_ready_ns = time.perf_counter_ns()
                result = predictor.infer_prepared(row, event_sensor, rgb, reason)
                torch.cuda.synchronize()
                end_ns = time.perf_counter_ns()
                result.update(
                    query_id=row["query_id"],
                    index=index,
                    complete_wall_ms=(end_ns - start_ns) / 1e6,
                    startup_seconds=startup_seconds,
                    event_enqueue_host_ms=(event_enqueued_ns - start_ns) / 1e6,
                    both_preprocessing_host_ms=(rgb_ready_ns - start_ns) / 1e6,
                    packet_events=len(packet["t"]),
                    ring=history.stats,
                    rgb_preprocess_detail_seconds=dict(predictor.inputs.last_profile),
                    started_ns=start_ns,
                    completed_ns=end_ns,
                )
                if not args.skip_input_check:
                    expected_input = engine.prepare_event(row)["own_events"]
                    actual_input = event_sensor.cpu().numpy()
                    difference = np.abs(actual_input - expected_input)
                    result["input_parity"] = {
                        "exact": bool(np.array_equal(actual_input, expected_input)),
                        "max_abs": float(difference.max()),
                        "different_values": int(np.count_nonzero(difference)),
                        "passed": bool(
                            np.allclose(actual_input, expected_input, rtol=1e-5, atol=1e-5)
                        ),
                    }
                records.append(result)
                with (args.output / "queries.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, allow_nan=False) + "\n")
                print(
                    json.dumps(
                        {
                            "completed": position + 1,
                            "queries": len(indices),
                            "index": index,
                            "ms": result["complete_wall_ms"],
                        }
                    ),
                    flush=True,
                )
        completed = True
    except BaseException as exc:
        caught = exc
    finally:
        cleanup_errors = []
        for resource in (predictor, reader):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as exc:
                    cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        intact = verify_snapshot(original)
        parity = prediction_parity(selected_reference, records, expected)
        input_ok = all(item.get("input_parity", {}).get("passed", False) for item in records)
        passed = completed and parity["passed"] and intact["passed"] and not cleanup_errors
        if not args.skip_input_check:
            passed = passed and input_ok
        summary = {
            "status": "COMPLETE" if passed else "FAILED",
            "passed": bool(passed),
            "scope": SCOPE,
            "parity": parity,
            "input_parity_checked": not args.skip_input_check,
            "input_parity_passed": input_ok if not args.skip_input_check else None,
            "originals": intact,
            "cleanup_errors": cleanup_errors,
            "error": f"{type(caught).__name__}: {caught}" if caught else None,
            "target_20_objects_50hz_demonstrated": False,
        }
        if records:
            summary["latency_ms"] = latency_distribution(
                [item["complete_wall_ms"] for item in records]
            )
        _write(args.output / "summary.json", summary)
    if caught is not None:
        raise caught
    if not passed:
        raise RuntimeError("Integral runtime failed parity/preservation; inspect summary.json")


def main() -> None:
    """Parse an explicitly separate last-packet measurement experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--rgb-workers", type=int, default=8)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--skip-input-check", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
