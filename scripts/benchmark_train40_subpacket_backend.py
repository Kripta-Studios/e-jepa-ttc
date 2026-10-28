"""Audit real 20ms versus 5ms ingestion without changing the TRAIN40 service.

Two histories coexist only for direct integer-state equality. Models load after
both histories are released. All 31 complete boundaries from bucket120 to240
are checked; only the final anchor uses its real ROI and original TTC oracle.
No state/RAW tensor archives are written. This is not a 50Hz latency admission.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from operational.evttc_transfer.inputs import EvTTCEventReader  # noqa: E402
from operational.rgb_port import transfer as engine  # noqa: E402
from operational.train40_runtime.context import ContextRuntime  # noqa: E402
from operational.train40_runtime.measurement import (  # noqa: E402
    prediction_parity,
    snapshot_files,
    verify_snapshot,
)
from operational.train40_runtime.temporal_prefix import TemporalPrefixEventHistory  # noqa: E402
from operational.train40_runtime.temporal_prefix_subpacket import (  # noqa: E402
    SubpacketTemporalPrefixEventHistory,
)
from scripts.benchmark_train40_producer_variants import (  # noqa: E402
    _oracle,
    _preserve_source_bytes,
    _ttc,
    _weight_values,
    tensor_comparison,
)
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

STATE_TENSORS = (
    "_counts",
    "_moments",
    "_cumulative_count",
    "_cumulative_moment",
    "_previous_count",
    "_previous_moment",
)
STATE_SCALARS = ("end_us", "_total_buckets", "_epoch_start_us", "_filled", "_ingested_packets")
BOUNDARIES = tuple(range(120, 241, 4))
CHECKPOINTS = (120, 124, 240)
PARTIAL_BUCKETS = (121, 122, 123, 237, 238, 239)
SCOPE = "REAL_RAW_SUBPACKET_STATE_AND_REPRESENTATION_DIAGNOSTIC_FINAL_TTC_ONLY"


def slice_raw(raw: Mapping[str, np.ndarray], start: int, end: int) -> dict[str, np.ndarray]:
    """Borrow the same exact half-open event interval from both backends' RAW."""
    if end <= start:
        raise ValueError("RAW interval must have positive duration")
    left, right = np.searchsorted(raw["t"], (start, end), side="left")
    return {key: value[left:right] for key, value in raw.items()}


def compare_history_state(
    original: TemporalPrefixEventHistory, candidate: TemporalPrefixEventHistory
) -> dict[str, Any]:
    """Compare every persistent history plane directly; never clone its GB ring."""
    buffers = {}
    for name in STATE_TENSORS:
        left, right = getattr(original, name), getattr(candidate, name)
        buffers[name] = {
            "exact": left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left, right),
            "shape": list(left.shape),
            "dtype": str(left.dtype),
            "bytes": left.numel() * left.element_size(),
        }
    scalars = {
        name: {
            "original": getattr(original, name),
            "candidate": getattr(candidate, name),
            "exact": getattr(original, name) == getattr(candidate, name),
        }
        for name in STATE_SCALARS
    }
    return {
        "passed": all(item["exact"] for item in buffers.values())
        and all(item["exact"] for item in scalars.values()),
        "buffers": buffers,
        "scalars": scalars,
        "scope": "ALL_PERSISTENT_RING_AND_EPOCH_STATE_EXCLUDES_RAW_STAGING_AND_QUERY_SCRATCH",
        "state_clones_or_archives_written": False,
    }


def memory_preflight(
    history_bytes: int, free_bytes: int, total_bytes: int, headroom_bytes: int
) -> dict[str, Any]:
    """Fail closed before allocating two histories when their declared budget fails."""
    values = (history_bytes, free_bytes, total_bytes, headroom_bytes)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("Memory sizes must be nonnegative integers")
    if history_bytes < 1 or total_bytes < 1 or free_bytes > total_bytes:
        raise ValueError("History/total bytes must be positive and free bytes cannot exceed total")
    required = 2 * history_bytes + headroom_bytes
    return {
        "passed": free_bytes >= required,
        "history_bytes": history_bytes,
        "two_history_bytes": 2 * history_bytes,
        "headroom_bytes": headroom_bytes,
        "required_bytes": required,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "scope": "DECLARED_RETAINED_BUFFERS_PLUS_EXPLICIT_GRAPH_AND_SCRATCH_HEADROOM",
        "peak_allocation_guaranteed": False,
        "model_checkpoints_loaded": False,
    }


def estimate_history_bytes(options: Mapping[str, Any]) -> int:
    """Use the real constructor on meta storage, without allocating GPU buffers."""
    config = {**options, "device": "meta", "graphs": False, "async_transfers": False}
    history = TemporalPrefixEventHistory(**config)
    return history.allocated_bytes


def _tensor_receipt(value: Tensor) -> dict[str, Any]:
    cpu = value.detach().cpu().contiguous()
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "sha256": hashlib.sha256(memoryview(cpu.numpy())).hexdigest(),
        "bytes": cpu.numel() * cpu.element_size(),
    }


def _partial_guard(
    candidate: SubpacketTemporalPrefixEventHistory, roi: Sequence[float], start_us: int
) -> dict[str, Any]:
    before = candidate.stats["queries"]
    checks = {}
    for name, call in (
        ("query", partial(candidate.query, roi)),
        ("query_many", partial(candidate.query_many, [roi])),
    ):
        try:
            call()
        except ValueError as error:
            checks[name] = "complete group of four" in str(error)
        else:
            checks[name] = False
    bucket = candidate._total_buckets
    passed = (
        all(checks.values())
        and not candidate.ready
        and candidate.subpacket_phase == bucket % 4
        and candidate._epoch_start_us == start_us + (bucket // 120) * 600_000
        and candidate.stats["queries"] == before
    )
    return {
        "bucket": bucket,
        "phase": candidate.subpacket_phase,
        "passed": passed,
        "rejected": checks,
        "ready": candidate.ready,
        "epoch_start_us": candidate._epoch_start_us,
        "queries_unchanged": candidate.stats["queries"] == before,
    }


def run_backend_case(
    raw: Mapping[str, np.ndarray],
    roi: Sequence[float],
    start_us: int,
    options: Mapping[str, Any],
    *,
    query_graphs: bool = False,
) -> tuple[dict[str, Any], tuple[Tensor, Tensor]]:
    """Audit one common 1.2s history, returning only two bounded CPU final tensors."""
    config = {**options, "graphs": query_graphs}
    original = TemporalPrefixEventHistory(**config)
    candidate = SubpacketTemporalPrefixEventHistory(**config)
    diagnostic_roi = (0.0, 0.0, float(original.width), float(original.height))
    checkpoints: list[dict[str, Any]] = []
    partial_guards: list[dict[str, Any]] = []
    priming = {}
    finals: tuple[Tensor, Tensor] | None = None
    try:
        for name, history in (("original", original), ("subpacket", candidate)):
            if query_graphs:
                history.prime_query_graphs([1])
            priming[name] = {
                "before_any_sensor_reset": history.end_us is None,
                "graph_count": len(history._query_graphs),
                "synthetic_calls": history.stats["synthetic_query_priming_calls"],
            }
            history.initialize(
                slice_raw(raw, start_us, start_us + 600_000), start_us, start_us + 600_000
            )
        for bucket in BOUNDARIES:
            if bucket > 120:
                left = start_us + (bucket - 4) * 5000
                original.ingest(slice_raw(raw, left, left + 20_000), left, left + 20_000)
                for sub_start in range(left, left + 20_000, 5000):
                    candidate.ingest(
                        slice_raw(raw, sub_start, sub_start + 5000), sub_start, sub_start + 5000
                    )
                    if candidate._total_buckets in PARTIAL_BUCKETS:
                        partial_guards.append(_partial_guard(candidate, diagnostic_roi, start_us))
            state = compare_history_state(original, candidate)
            current_roi = roi if bucket == 240 else diagnostic_roi
            left_output = original.query_many([current_roi])
            right_output = candidate.query_many([current_roi])
            exact = torch.equal(left_output, right_output)
            checkpoints.append(
                {
                    "bucket": bucket,
                    "explicit_checkpoint": bucket in CHECKPOINTS,
                    "state": state,
                    "representation": {
                        "passed": exact,
                        "exact": exact,
                        "original": _tensor_receipt(left_output),
                        "subpacket": _tensor_receipt(right_output),
                    },
                    "roi": list(current_roi),
                    "roi_scope": "FINAL_REAL_QUERY_ROI"
                    if bucket == 240
                    else "FIXED_DIAGNOSTIC_ROI",
                    "anchor_us": candidate.end_us,
                }
            )
            if bucket == 240:
                finals = (left_output.detach().cpu().clone(), right_output.detach().cpu().clone())
        primed = all(
            item["before_any_sensor_reset"]
            and item["graph_count"] == int(query_graphs)
            and item["synthetic_calls"] == (3 if query_graphs else 0)
            for item in priming.values()
        )
        graph_counts_stable = all(
            len(history._query_graphs) == int(query_graphs) for history in (original, candidate)
        )
        coverage = [item["bucket"] for item in checkpoints] == list(BOUNDARIES)
        guard_coverage = [item["bucket"] for item in partial_guards] == list(PARTIAL_BUCKETS)
        record = {
            "start_us": start_us,
            "anchor_us": start_us + 1_200_000,
            "initialization": "SAME_600MS_RAW_30_BULK_PACKETS_THEN_120_SUBPACKETS",
            "checkpoints": checkpoints,
            "partial_guards": partial_guards,
            "boundary_coverage_exact": coverage,
            "partial_guard_coverage_exact": guard_coverage,
            "query_priming": priming,
            "query_priming_passed": primed,
            "query_graph_count_unchanged": graph_counts_stable,
            "original_stats": original.stats,
            "subpacket_stats": candidate.stats,
            "passed": coverage
            and guard_coverage
            and primed
            and graph_counts_stable
            and all(
                item["state"]["passed"] and item["representation"]["exact"] for item in checkpoints
            )
            and all(item["passed"] for item in partial_guards),
        }
        assert finals is not None
        return record, finals
    finally:
        if original.device.type == "cuda":
            torch.cuda.synchronize(original.device)
        # Histories have no close API. Drop graph ownership before local histories
        # leave scope; callers retain only CPU outputs and perform allocator cleanup.
        original._query_graphs.clear()
        candidate._query_graphs.clear()


def _load_raw(
    reader: EvTTCEventReader, start_us: int, end_us: int, max_bytes: int
) -> dict[str, np.ndarray]:
    parts = []
    size = 0
    for left in range(start_us, end_us, 20_000):
        part = reader.read_window(left, left + 20_000)
        size += sum(value.nbytes for value in part.values())
        if size > max_bytes:
            raise ValueError(f"RAW archive exceeds explicit {max_bytes} byte budget")
        parts.append(part)
    # Peak held archive + concatenation is at most twice the explicit byte cap.
    return {key: np.concatenate([part[key] for part in parts]) for key in ("x", "y", "t", "p")}


def _raw_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def run(args: argparse.Namespace) -> None:
    """Preserve independent state/TTC gates and failures without an efficiency claim."""
    if not args.indices or len(set(args.indices)) != len(args.indices):
        raise ValueError("Query indices must be nonempty and unique")
    final_bytes = len(args.indices) * 2 * 4 * 3 * 12 * 128**2 * 4
    if final_bytes > args.max_final_output_bytes or args.max_raw_bytes < 1:
        raise ValueError("Final output storage or RAW budget is insufficient")
    _validate_output(args.output, [args.candidate, args.manifest.parent, args.reference])
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot: dict[str, Any] = {}
    raw_stats: dict[str, dict[str, int]] = {}
    records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "scope": SCOPE,
        "status": "RUNNING",
        "records": records,
        "start_time": datetime.now(UTC).isoformat(),
        "complete_update_admitted": False,
        "target_50hz_demonstrated": False,
        "latency_measured": False,
        "three_quarters_latency_saving_claimed": False,
        "objects": 1,
        "targets_read": False,
        "optimizer_updates": 0,
    }
    try:
        paths, binding = verify_candidate(args.candidate)
        frozen = verify_frozen_runtime(args.candidate)
        rows = read(args.manifest)["rows"]
        reject_targets(rows)
        if any(index < 0 or index >= len(rows) for index in args.indices):
            raise ValueError("Query index outside sealed manifest")
        reference, reference_files, reference_binding = _validate_reference(
            args.reference, args.manifest, rows, binding
        )
        sources = sorted(
            set(
                [
                    *_source_inventory(frozen, reference_binding),
                    Path(__file__).resolve(),
                    ROOT / "scripts/benchmark_train40_producer_variants.py",
                ]
            )
        )
        snapshot = snapshot_files(
            [*_candidate_inventory(args.candidate), *reference_files, args.manifest, *sources]
        )
        _write(args.output / "original_snapshot.json", snapshot)
        _preserve_source_bytes(sources, snapshot, args.output)
        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        binding.update(
            scope=SCOPE,
            config=config,
            config_hash=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            sources={str(path.relative_to(ROOT)): sha(path) for path in sources},
            manifest_sha256=sha(args.manifest),
            seed=7,
            host=platform.node(),
            python=platform.python_version(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            precision="FP32_TF32_OFF",
            objects=1,
            boundaries=list(BOUNDARIES),
            explicit_checkpoints=list(CHECKPOINTS),
            state_buffers=list(STATE_TENSORS),
            model_phase="AFTER_BOTH_HISTORIES_RELEASED",
            ttc_batch_contract="ORIGINAL_EAGER_E_FOUR_OBSERVATIONS_R_ONE_AT_A_TIME",
            final_host_tensor_bytes=final_bytes,
            raw_archive_concat_upper_bound=2 * args.max_raw_bytes,
            raw_reader_scratch_limit=256 * 1024**2,
            acceptance_tolerance={"atol": 1e-5, "rtol": 1e-5},
        )
        _write(args.output / "binding.json", binding)
        torch.set_num_threads(2)
        torch.manual_seed(7)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        options = {
            "device": "cuda",
            "width": 1280,
            "height": 720,
            "roi_size": 128,
            "max_packet_events": args.max_packet_events,
            "max_objects": 1,
            "roi_batch_size": 1,
            "normalization": "topk",
            "async_transfers": False,
        }
        history_bytes = estimate_history_bytes(options)
        finals: dict[int, tuple[Tensor, Tensor]] = {}
        with torch.inference_mode():
            for index in args.indices:
                row = rows[index]
                anchor, raw_path = int(row["anchor_us"]), Path(row["raw_path"])
                if anchor < 1_200_000:
                    raise ValueError("Final anchor cannot support the common 1.2s reset interval")
                current_stat = _raw_stat(raw_path)
                if str(raw_path) in raw_stats and current_stat != raw_stats[str(raw_path)]:
                    raise ValueError("RAW source changed between diagnostic queries")
                raw_stats.setdefault(str(raw_path), current_stat)
                reader = EvTTCEventReader(raw_path).open()
                try:
                    raw = _load_raw(reader, anchor - 1_200_000, anchor, args.max_raw_bytes)
                finally:
                    reader.close()
                gc.collect()
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info()
                budget = memory_preflight(history_bytes, free, total, args.gpu_headroom_bytes)
                _write(args.output / f"memory_preflight_{index}.json", budget)
                record = {
                    "index": index,
                    "query_id": row["query_id"],
                    "passed": False,
                    "memory_preflight": budget,
                    "raw_bytes": sum(value.nbytes for value in raw.values()),
                    "raw_array_hashes": {
                        key: hashlib.sha256(memoryview(value)).hexdigest()
                        for key, value in raw.items()
                    },
                }
                records.append(record)
                _write(args.output / "summary.json", summary)
                if not budget["passed"]:
                    raise MemoryError(
                        f"Two-history preflight needs {budget['required_bytes']} bytes; "
                        f"only {free} free"
                    )
                torch.cuda.reset_peak_memory_stats()
                backend_record, finals[index] = run_backend_case(
                    raw, row["square_xyxy"], anchor - 1_200_000, options, query_graphs=True
                )
                record.update(backend_record)
                record.update(
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
                del raw
                gc.collect()
                torch.cuda.empty_cache()
                record["allocated_after_histories_released"] = torch.cuda.memory_allocated()
                _write(args.output / "summary.json", summary)
                print(
                    json.dumps(
                        {
                            "index": index,
                            "backend_passed": record["passed"],
                            "boundaries": len(record["checkpoints"]),
                        }
                    ),
                    flush=True,
                )
            summary["allocated_before_model_loading"] = torch.cuda.memory_allocated()
            models, normalizers = load_models(paths, "cuda")
            before = _weight_values(models)
            norms_before = {
                key: [_tensor_receipt(value) for value in values]
                for key, values in normalizers.items()
            }
            context = ContextRuntime(models, normalizers, "cuda", graphs=False)
            for record in records:
                index = record["index"]
                row = rows[index]
                rgb, reason = engine.raw_rgb_observations(row)
                if reason or len(rgb) != 5:
                    raise ValueError(
                        "Final one-object oracle requires five original RGB observations"
                    )
                pixels = torch.from_numpy(np.stack([value for value, _ in rgb])).cuda()
                clock = torch.tensor(
                    [meta["delta_t_s"] for _, meta in rgb], dtype=torch.float32, device="cuda"
                )
                actuals = {}
                for label, event in zip(
                    ("original_20ms", "subpacket_5ms"), finals[index], strict=True
                ):
                    features = _oracle(
                        models,
                        (event[0].cuda(), torch.full((4, 2), 0.1, device="cuda"), pixels, clock),
                        1,
                    )
                    actual = _ttc(context, features, row, [meta for _, meta in rgb], 1)[0]
                    actual["query_id"] = row["query_id"]
                    actuals[label] = actual
                original_input = torch.from_numpy(engine.prepare_event(row)["own_events"])
                record["canonical_input_parity"] = {
                    label: tensor_comparison(event[0], original_input)
                    for label, event in zip(
                        ("original_20ms", "subpacket_5ms"), finals[index], strict=True
                    )
                }
                record["final_ttc"] = actuals
                record["final_predictions_between_backends_exact"] = (
                    actuals["original_20ms"]["predictions"]
                    == actuals["subpacket_5ms"]["predictions"]
                )
                record["final_ttc_parity"] = {
                    name: prediction_parity([reference[index]], [actual], [row["query_id"]])
                    for name, actual in actuals.items()
                }
                del finals[index]
                _write(args.output / "summary.json", summary)
            summary["weights_unchanged"] = before == _weight_values(models)
            summary["normalizers_unchanged"] = norms_before == {
                key: [_tensor_receipt(value) for value in values]
                for key, values in normalizers.items()
            }
        summary["coverage_exact"] = [record["index"] for record in records] == args.indices
        summary["all_backend_gates_passed"] = all(record["passed"] for record in records)
        summary["all_final_ttc_passed"] = all(
            all(item["passed"] for item in record["final_ttc_parity"].values())
            for record in records
        )
        summary["all_canonical_inputs_passed"] = all(
            all(item["passed"] for item in record["canonical_input_parity"].values())
            for record in records
        )
        summary["status"] = (
            "COMPLETE"
            if all(
                summary[key]
                for key in (
                    "coverage_exact",
                    "all_backend_gates_passed",
                    "all_final_ttc_passed",
                    "all_canonical_inputs_passed",
                    "weights_unchanged",
                    "normalizers_unchanged",
                )
            )
            else "COMPLETE_WITH_GATE_FAILURES"
        )
    except BaseException as error:
        summary.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["end_time"] = datetime.now(UTC).isoformat()
        summary["preservation"] = verify_snapshot(snapshot) if snapshot else {"passed": False}
        raw_checks = []
        for path, before_stat in raw_stats.items():
            try:
                after_stat = _raw_stat(Path(path))
                raw_checks.append(
                    {
                        "path": path,
                        "before": before_stat,
                        "after": after_stat,
                        "passed": before_stat == after_stat,
                    }
                )
            except OSError as error:
                raw_checks.append(
                    {
                        "path": path,
                        "before": before_stat,
                        "passed": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        summary["raw_stat_checks"] = raw_checks
        summary["raw_stats_preserved"] = bool(raw_checks) and all(
            item["passed"] for item in raw_checks
        )
        if not summary["preservation"]["passed"] or not summary["raw_stats_preserved"]:
            summary["status"] = "FAILED"
        _write(args.output / "summary.json", summary)
    if summary["status"] != "COMPLETE":
        raise SystemExit(2)


def main() -> None:
    """Require frozen inputs and a fresh directory; never run a live 200Hz feeder."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "manifest", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 137, 553])
    parser.add_argument("--max-packet-events", type=int, default=2_000_000)
    parser.add_argument("--max-raw-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--max-final-output-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--gpu-headroom-bytes", type=int, default=1024**3)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
