#!/usr/bin/env python3
"""Regenerate the detection performance diagnosis from retained local evidence."""

# The Markdown template keeps each table row on one source line so the generated
# table is directly auditable against its interpolation.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = "phase_ttc_detection_performance_diagnosis_v5"
BYTES_PER_GIB = 1024**3
LIVE_METRICS_CUTOFF_UPDATE = 3420
CAMPAIGN_TARGET_UPDATE = 8192
MONITOR_INTERVAL_UPDATES = 1024
FULL_SELECTION_FRAMES = 23958
FULL_SELECTION_INTERVAL_UPDATES = 4096

SOURCES = {
    "training_pyspy": "output/performance/live_training_after2048_v1/pyspy_raw.txt",
    "evaluation_pyspy": "output/performance/live_evaluation_update2048_v1/pyspy_raw.txt",
    "training_pyspy_transcription": (
        "output/performance/live_training_after2048_v1/tool_output_transcription.json"
    ),
    "evaluation_pyspy_transcription": (
        "output/performance/live_evaluation_update2048_v1/tool_output_transcription.json"
    ),
    "training_started": "output/performance/live_training_after2048_v1/started_at.txt",
    "training_ended": "output/performance/live_training_after2048_v1/ended_at.txt",
    "passive": (
        "output/detection_campaign/rgb_screen_v5_seed7_v1/passive_observation_update2048_v1.json"
    ),
    "passive_interpretation": (
        "output/detection_campaign/rgb_screen_v5_seed7_v1/"
        "passive_observation_update2048_interpretation_v1.json"
    ),
    "r0_1024": (
        "output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_runs_v1/r0_1024/summary.json"
    ),
    "selection_2048": (
        "output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_runs_v1/"
        "r0_8192/evaluation/selection_update_002048.json"
    ),
    "selection_3072": (
        "output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_runs_v1/"
        "r0_8192/evaluation/selection_update_003072.json"
    ),
    "tar_v1": "output/campaign_data/indexed_rgb_benchmark_v1/receipt.json",
    "tar_v1_restoration": (
        "output/campaign_data/indexed_rgb_benchmark_v1/RESTORATION_EXCEPTION.json"
    ),
    "tar_v2": "output/campaign_data/indexed_rgb_benchmark_v2/receipt.json",
    "tar_v2_code": (
        "output/campaign_data/indexed_rgb_benchmark_v2/code/benchmark_indexed_rgb_data_v2.py"
    ),
    "read_profile": "output/campaign_data/read_profile_v3_final/summary.json",
    "ap_v2": "output/evaluation_ap_fast/ap101_exact_v2/receipt.json",
    "ap_v3": "output/evaluation_ap_fast/ap101_exact_v3/receipt.json",
    "ap_v4": "output/evaluation_ap_fast/ap101_integration_v4/receipt.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def parse_folded_samples(path: Path) -> dict[str, int | float]:
    """Count py-spy folded samples; counts are sample shares, not wall time."""

    total = 0
    tarfile = 0
    unknown_empty_stack = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.search(r" (\d+)$", line)
        if match is None:
            continue
        count = int(match.group(1))
        total += count
        if not line[: match.start()].strip():
            unknown_empty_stack += count
        if "tarfile.py:" in line:
            tarfile += count
    if total <= 0:
        raise ValueError(f"no folded samples found: {path}")
    return {
        "total_samples": total,
        "tarfile_stack_samples": tarfile,
        "tarfile_sample_fraction": tarfile / total,
        "unknown_empty_stack_samples": unknown_empty_stack,
        "unknown_empty_stack_fraction": unknown_empty_stack / total,
    }


def _first_float_after(path: Path, anchor: str, field: str) -> float:
    anchored = False
    pattern = re.compile(rf'"{re.escape(field)}"\s*:\s*([-+0-9.eE]+)')
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not anchored:
                anchored = f'"{anchor}"' in line
                continue
            match = pattern.search(line)
            if match is not None:
                return float(match.group(1))
    raise ValueError(f"field {field!r} after {anchor!r} not found in {path}")


def _subset(receipt: dict[str, Any], frame_count: int) -> dict[str, Any]:
    subsets = receipt.get("subsets")
    if not isinstance(subsets, list):
        raise TypeError("AP receipt has no subsets list")
    for row in subsets:
        if isinstance(row, dict) and row.get("frame_count") == frame_count:
            return row
    raise ValueError(f"AP receipt has no {frame_count}-frame subset")


def _read_metrics_prefix(path: Path, cutoff_update: int) -> tuple[list[dict[str, Any]], str, int]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    byte_count = 0
    previous_update: int | None = None
    with path.open("rb") as handle:
        for raw_line in handle:
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise TypeError("metrics JSONL row is not an object")
            update = value.get("update")
            if not isinstance(update, int):
                raise TypeError("metrics JSONL update is not an int")
            if update > cutoff_update:
                break
            if previous_update is not None and update != previous_update + 1:
                raise ValueError("metrics JSONL prefix is not contiguous")
            rows.append(value)
            previous_update = update
            digest.update(raw_line)
            byte_count += len(raw_line)
    if not rows or rows[-1]["update"] != cutoff_update:
        raise ValueError(f"metrics JSONL does not reach requested cutoff {cutoff_update}")
    return rows, digest.hexdigest(), byte_count


def _timing_window(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    prep = [float(row["preprocessing_seconds"]) for row in rows]
    step = [float(row["forward_backward_update_seconds"]) for row in rows]
    combined = [left + right for left, right in zip(prep, step, strict=True)]
    total = sum(combined)
    return {
        "first_update": int(rows[0]["update"]),
        "last_update": int(rows[-1]["update"]),
        "update_count": len(rows),
        "median_preprocessing_seconds": statistics.median(prep),
        "median_forward_backward_seconds": statistics.median(step),
        "median_component_sum_seconds_per_update": statistics.median(combined),
        "observed_component_sum_seconds": total,
        "updates_per_component_second": len(rows) / total,
    }


def _selection_class_summary(path: Path) -> dict[str, dict[str, int | float]]:
    classes = (
        "car",
        "bicycle",
        "truck",
        "pedestrian",
        "motorcycle",
        "bus",
        "tricycle",
        "other_vehicle",
    )
    result: dict[str, dict[str, int | float]] = {}
    in_per_class = False
    current: str | None = None
    class_pattern = re.compile(r'^\s+"(' + "|".join(classes) + r')"\s*:\s*\{')
    number_pattern = re.compile(r":\s*([-+0-9.eE]+)")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not in_per_class:
                in_per_class = '"per_class": {' in line
                continue
            match = class_pattern.match(line)
            if match is not None:
                class_name = match.group(1)
                if class_name is None:
                    continue
                current = class_name
                result[class_name] = {}
                continue
            if current is None:
                continue
            if '"eligible_ground_truth_count"' in line:
                value = number_pattern.search(line)
                if value is not None:
                    result[current]["eligible_ground_truth_count"] = int(value.group(1))
            elif '"AP101_mean_thresholds"' in line:
                value = number_pattern.search(line)
                if value is not None:
                    result[current]["AP101_mean_thresholds"] = float(value.group(1))
            if len(result) == len(classes) and all(len(row) == 2 for row in result.values()):
                break
    if tuple(result) != classes or any(len(row) != 2 for row in result.values()):
        raise ValueError("could not derive complete v2 per-class selection summary")
    total = sum(int(row["eligible_ground_truth_count"]) for row in result.values())
    for row in result.values():
        row["eligible_ground_truth_fraction"] = int(row["eligible_ground_truth_count"]) / total
    return result


def _conditional_projections(
    timing_last100: dict[str, int | float],
    *,
    basis_update: int,
    target_update: int,
    monitor_inference_seconds: float,
    full_selection_frame_median_seconds: float,
) -> dict[str, Any]:
    remaining_updates = target_update - basis_update
    remaining_monitor_updates = list(
        range(
            ((basis_update // MONITOR_INTERVAL_UPDATES) + 1) * MONITOR_INTERVAL_UPDATES,
            target_update + 1,
            MONITOR_INTERVAL_UPDATES,
        )
    )
    remaining_full_selection_updates = list(
        range(
            ((basis_update // FULL_SELECTION_INTERVAL_UPDATES) + 1)
            * FULL_SELECTION_INTERVAL_UPDATES,
            target_update + 1,
            FULL_SELECTION_INTERVAL_UPDATES,
        )
    )
    component_mean = float(timing_last100["observed_component_sum_seconds"]) / int(
        timing_last100["update_count"]
    )
    training_seconds = remaining_updates * component_mean
    monitor_seconds = len(remaining_monitor_updates) * monitor_inference_seconds
    full_pass_seconds = FULL_SELECTION_FRAMES * full_selection_frame_median_seconds
    return {
        "basis_update": basis_update,
        "target_update": target_update,
        "remaining_updates": remaining_updates,
        "remaining_monitor_updates": remaining_monitor_updates,
        "last100_component_mean_seconds_per_update": component_mean,
        "training_component_seconds_using_last100_aggregate_mean": training_seconds,
        "scheduled_monitor_inference_seconds_using_update3072_total": monitor_seconds,
        "training_plus_monitor_inference_seconds": training_seconds + monitor_seconds,
        "full_selection_frames_not_executed": FULL_SELECTION_FRAMES,
        "remaining_full_selection_updates": remaining_full_selection_updates,
        "full_selection_inference_seconds_per_pass_using_update3072_frame_median": (
            full_pass_seconds
        ),
        "remaining_full_selection_inference_seconds_all_passes": (
            len(remaining_full_selection_updates) * full_pass_seconds
        ),
        "limitations": [
            "Conditional arithmetic only; not a completion promise.",
            "Training projection uses the aggregate mean of recorded preprocessing plus forward/backward/update components over the last 100 updates.",
            "Monitor projection excludes scorer, checkpoint, serialization and other overhead.",
            "Each full-selection projection is inference-only; neither remaining 23,958-frame pass has executed.",
            "No projected subtotal is an integral ETA.",
        ],
    }


def _f(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def build(root: Path) -> dict[str, Any]:
    paths = {name: root / relative for name, relative in SOURCES.items()}
    missing = [SOURCES[name] for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing performance evidence: " + ", ".join(missing))

    training_folded = parse_folded_samples(paths["training_pyspy"])
    evaluation_folded = parse_folded_samples(paths["evaluation_pyspy"])
    training_started = datetime.fromisoformat(
        paths["training_started"].read_text(encoding="utf-8").strip()
    )
    training_ended = datetime.fromisoformat(
        paths["training_ended"].read_text(encoding="utf-8").strip()
    )
    training_folded["observation_interval_seconds"] = (
        training_ended - training_started
    ).total_seconds()
    training_transcription = _json(paths["training_pyspy_transcription"])
    evaluation_transcription = _json(paths["evaluation_pyspy_transcription"])
    for name, folded, transcription, raw_path in (
        (
            "training",
            training_folded,
            training_transcription,
            paths["training_pyspy"],
        ),
        (
            "evaluation",
            evaluation_folded,
            evaluation_transcription,
            paths["evaluation_pyspy"],
        ),
    ):
        if transcription.get("schema") != "phase_ttc_pyspy_console_transcription_v1":
            raise ValueError(f"invalid {name} py-spy transcription schema")
        if transcription.get("raw_profile_sha256") != _sha256(raw_path):
            raise ValueError(f"{name} py-spy transcription is not bound to raw profile")
        if transcription.get("samples") != folded["total_samples"]:
            raise ValueError(f"{name} py-spy transcription sample count mismatch")
        folded["sampling_errors"] = int(transcription["sampling_errors"])
        folded["transcription_captured_live"] = bool(transcription["captured_live"])
    passive = _json(paths["passive"])
    passive_interpretation = _json(paths["passive_interpretation"])
    if passive_interpretation.get("observation_sha256") != _sha256(paths["passive"]):
        raise ValueError("passive interpretation is not bound to the observation")
    r0_1024 = _json(paths["r0_1024"])
    tar_v1 = _json(paths["tar_v1"])
    tar_v1_restoration = _json(paths["tar_v1_restoration"])
    tar_v2 = _json(paths["tar_v2"])
    if tar_v1_restoration.get("original_receipt_sha256") != _sha256(paths["tar_v1"]):
        raise ValueError("TAR v1 restoration exception is not bound to restored receipt")
    if tar_v2.get("schema") != "eap_indexed_rgb_benchmark_v2":
        raise ValueError("unexpected TAR v2 receipt schema")
    if tar_v2.get("status") != "PASS":
        raise ValueError("TAR v2 benchmark did not pass")
    if tar_v2["source_hashes"].get("benchmark_implementation") != _sha256(paths["tar_v2_code"]):
        raise ValueError("TAR v2 receipt is not bound to archived benchmark code")
    read_profile = _json(paths["read_profile"])
    ap_v2 = _json(paths["ap_v2"])
    ap_v3 = _json(paths["ap_v3"])
    ap_v4 = _json(paths["ap_v4"])

    ap1024 = float(r0_1024["best_selection_metric"])
    ap2048 = _first_float_after(
        paths["selection_2048"], "metrics_v2", "dev_AP101_supported_classes"
    )
    ap3072 = _first_float_after(
        paths["selection_3072"], "metrics_v2", "dev_AP101_supported_classes"
    )
    selection3072_classes = _selection_class_summary(paths["selection_3072"])
    selection3072_inference_median = _first_float_after(
        paths["selection_3072"], "metrics_v2", "inference_seconds_median"
    )
    selection3072_inference_total = _first_float_after(
        paths["selection_3072"], "metrics_v2", "inference_seconds_total"
    )
    live_metrics_path = root / (
        "output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_runs_v1/r0_8192/metrics.jsonl"
    )
    metrics_rows, metrics_prefix_sha256, metrics_prefix_bytes = _read_metrics_prefix(
        live_metrics_path, LIVE_METRICS_CUTOFF_UPDATE
    )
    timing_last100 = _timing_window(metrics_rows[-100:])
    timing_since3072 = _timing_window([row for row in metrics_rows if int(row["update"]) > 3072])
    conditional_projections = _conditional_projections(
        timing_last100,
        basis_update=LIVE_METRICS_CUTOFF_UPDATE,
        target_update=CAMPAIGN_TARGET_UPDATE,
        monitor_inference_seconds=selection3072_inference_total,
        full_selection_frame_median_seconds=selection3072_inference_median,
    )
    prep = float(r0_1024["preprocessing_seconds_median"])
    step = float(r0_1024["step_seconds_median"])
    overlap_bound = (prep + step) / max(prep, step)

    tar_seconds = tar_v1["seconds"]
    if not isinstance(tar_seconds, dict):
        raise TypeError("TAR v1 receipt has no seconds object")
    read_workers = read_profile["workers"]
    if not isinstance(read_workers, list) or len(read_workers) != 2:
        raise TypeError("read profile must contain exactly two workers")
    cache_by_sequences = {
        int(row["label_cache_sequences"]): row for row in read_workers if isinstance(row, dict)
    }
    cache2 = cache_by_sequences[2]
    cache24 = cache_by_sequences[24]

    tar_v2_trials = tar_v2["trials"]
    if not isinstance(tar_v2_trials, list) or len(tar_v2_trials) != 3:
        raise TypeError("TAR v2 must contain three trials")
    old_raw = [float(row["old"]["raw_seconds"]) for row in tar_v2_trials]
    old_prepared = [float(row["old"]["prepared_seconds"]) for row in tar_v2_trials]
    indexed_raw = [float(row["indexed"]["raw_seconds"]) for row in tar_v2_trials]
    indexed_prepared = [float(row["indexed"]["prepared_seconds"]) for row in tar_v2_trials]
    raw_hashes = {
        str(row[producer]["raw_sha256"]) for row in tar_v2_trials for producer in ("old", "indexed")
    }
    prepared_hashes = {
        str(row[producer]["prepared_sha256"])
        for row in tar_v2_trials
        for producer in ("old", "indexed")
    }
    if len(raw_hashes) != 1 or len(prepared_hashes) != 1:
        raise ValueError("TAR v2 trial hashes do not prove exact cross-trial parity")
    tar_v2_derived = {
        "old_raw_seconds": statistics.median(old_raw),
        "indexed_raw_seconds": statistics.median(indexed_raw),
        "old_prepared_seconds": statistics.median(old_prepared),
        "indexed_prepared_seconds": statistics.median(indexed_prepared),
    }
    tar_v2_derived["raw_speedup"] = (
        tar_v2_derived["old_raw_seconds"] / tar_v2_derived["indexed_raw_seconds"]
    )
    tar_v2_derived["prepared_speedup"] = (
        tar_v2_derived["old_prepared_seconds"] / tar_v2_derived["indexed_prepared_seconds"]
    )

    ap2_1024 = _subset(ap_v2, 1024)
    ap3_1024 = _subset(ap_v3, 1024)
    ap4_64 = _subset(ap_v4, 64)

    image_cache_bytes = 4096 * 3 * 384 * 640 * 4
    region_mask_bytes = 4096 * 384 * 640 * 4

    return {
        "schema": SCHEMA,
        "scope": "diagnosis only; no full-campaign speedup claim",
        "source_files": {
            name: {"path": SOURCES[name], "sha256": _sha256(path)} for name, path in paths.items()
        },
        "measured": {
            "stable_r0_1024": {
                "updates": int(r0_1024["final_update"]),
                "image_exposures": int(r0_1024["stage_image_exposures"]),
                "preprocessing_seconds_median": prep,
                "step_seconds_median": step,
                "stage_training_seconds": float(r0_1024["stage_training_seconds"]),
                "peak_allocated_bytes": int(r0_1024["peak_allocated_bytes"]),
                "peak_reserved_bytes": int(r0_1024["peak_reserved_bytes"]),
            },
            "pyspy_folded_samples": {
                "training": training_folded,
                "evaluation": evaluation_folded,
                "interpretation": "sample shares, not wall-time fractions",
            },
            "passive_update2048": {
                "elapsed_seconds": float(passive["elapsed_seconds"]),
                "raw_phase_classification": passive["phase_classification"],
                "bound_phase_inference": passive_interpretation["phase_inference"],
                "claim_boundary": passive_interpretation["claim_boundary"],
                "updates_completed": int(passive["metrics"]["updates_completed"]),
                "mean_logical_core_equivalents": float(
                    passive["trainer"]["mean_logical_core_equivalents"]
                ),
                "mean_gpu_utilization_percent": float(
                    passive["gpu"]["utilization_gpu_percent"]["mean"]
                ),
                "utilization_scope": "scheduled monitor evaluation, not training",
            },
            "ap_progression": {
                "update1024_dev_ap101_supported_classes": ap1024,
                "update2048_dev_ap101_supported_classes": ap2048,
                "update3072_dev_ap101_supported_classes": ap3072,
                "absolute_change_1024_to_3072": ap3072 - ap1024,
                "relative_change_1024_to_3072": ap3072 / ap1024 - 1.0,
                "official_score": False,
            },
            "selection3072": {
                "sample_count": 1024,
                "eligible_ground_truth_count": sum(
                    int(row["eligible_ground_truth_count"])
                    for row in selection3072_classes.values()
                ),
                "per_class": selection3072_classes,
                "inference_seconds_median_per_frame": selection3072_inference_median,
                "inference_seconds_total": selection3072_inference_total,
            },
            "live_training_prefix": {
                "path": live_metrics_path.relative_to(root).as_posix(),
                "cutoff_update": LIVE_METRICS_CUTOFF_UPDATE,
                "prefix_bytes": metrics_prefix_bytes,
                "prefix_sha256": metrics_prefix_sha256,
                "last100": timing_last100,
                "since_update3072": timing_since3072,
            },
            "data_read_profile_128": {
                "cache2_seconds": float(cache2["cold_seconds"]),
                "cache24_seconds": float(cache24["cold_seconds"]),
                "cache24_over_cache2_speedup": float(cache2["cold_seconds"])
                / float(cache24["cold_seconds"]),
                "trace_count": int(read_profile["trace_count"]),
                "parity_checks": dict(read_profile["parity"]),
            },
            "tar_reader_v1_timing_only": {
                "panel_samples": int(tar_v1["population"]["samples"]),
                "old_raw_seconds": float(tar_seconds["old_raw_32"]),
                "indexed_raw_seconds": float(tar_seconds["indexed_raw_32"]),
                "raw_panel_speedup": float(tar_seconds["old_raw_32"])
                / float(tar_seconds["indexed_raw_32"]),
                "old_prepared_seconds": float(tar_seconds["old_prepared_32"]),
                "indexed_prepared_seconds": float(tar_seconds["indexed_prepared_32"]),
                "prepared_panel_speedup": float(tar_seconds["old_prepared_32"])
                / float(tar_seconds["indexed_prepared_32"]),
                "indexed_construct_and_source_hash_seconds": float(
                    tar_seconds["indexed_construct_and_source_hash"]
                ),
                "parity_admission": "WITHHELD_PENDING_INDEXED_V2",
                "restoration_incident": tar_v1_restoration["incident"],
                "restoration_status": tar_v1_restoration["status"],
            },
            "tar_reader_v2_panel": {
                "samples": int(tar_v2["population"]["samples"]),
                "sequences": int(tar_v2["population"]["sequences"]),
                "shards": int(tar_v2["population"]["shards"]),
                "trials": len(tar_v2_trials),
                "trial_order": [row["order"] for row in tar_v2_trials],
                "old_raw_median_seconds": tar_v2_derived["old_raw_seconds"],
                "indexed_raw_median_seconds": tar_v2_derived["indexed_raw_seconds"],
                "raw_panel_speedup": tar_v2_derived["raw_speedup"],
                "old_prepared_median_seconds": tar_v2_derived["old_prepared_seconds"],
                "indexed_prepared_median_seconds": tar_v2_derived["indexed_prepared_seconds"],
                "prepared_panel_speedup": tar_v2_derived["prepared_speedup"],
                "initial_hash_and_index_seconds": float(
                    tar_v2["admission_seconds"]["indexed_full_tar_hash_and_index_construct"]
                ),
                "peak_working_set_bytes": int(tar_v2["resources"]["peak_working_set_bytes"]),
                "raw_hash_sha256": next(iter(raw_hashes)),
                "prepared_hash_sha256": next(iter(prepared_hashes)),
                "parity": dict(tar_v2["parity"]),
                "review_gate": "SHIP_REPORTED_BY_REVIEW_S0_AND_TEMPORAL_MODEL",
            },
            "ap101_profiles": {
                "v2_timing_valid_1024": {
                    key: ap2_1024[key] for key in ("original_seconds", "fast_seconds", "speedup")
                },
                "v3_contaminated_not_for_speed_claim": {
                    "frame_count": 1024,
                    "process_peak_rss_bytes": ap3_1024["process_peak_rss_bytes"],
                    "reported_speedup": ap3_1024["speedup"],
                },
                "v4_integration_64": {
                    key: ap4_64[key]
                    for key in (
                        "original_seconds",
                        "fast_seconds",
                        "speedup",
                        "process_peak_rss_bytes",
                    )
                },
            },
        },
        "bounds": {
            "perfect_preparation_compute_overlap_speedup": overlap_bound,
            "description": "median-phase arithmetic upper bound; excludes H2D and overhead",
        },
        "conditional_projections": conditional_projections,
        "estimates_from_exact_shapes": {
            "fp32_image_cache_4096_3x384x640_bytes": image_cache_bytes,
            "fp32_image_cache_gib": image_cache_bytes / BYTES_PER_GIB,
            "fp32_region_masks_4096x384x640_bytes": region_mask_bytes,
            "fp32_region_masks_gib": region_mask_bytes / BYTES_PER_GIB,
        },
        "prototypes": {
            "bounded_prefetch": {
                "status": "CPU_TESTED_NOT_INTEGRATED",
                "commit": "9526ec10",
                "capacity": [1, 2],
            }
        },
        "unknown": {
            "full_campaign_speedup": None,
            "bf16_tf32_compile_cuda_graphs_gain": None,
            "interim_campaign_summary": (
                "NOT_READY_LIVE_RUN_HAS_NO_FINAL_SUMMARY_OR_DURABLE_SEGMENT_RECEIPT"
            ),
        },
        "excluded_claims": [
            "Indexed TAR v1 logical-hash parity remains unadmitted; v2 is separate evidence.",
            "Indexed TAR v2 panel speedups are not detector-training or full-campaign speedups.",
            "py-spy error counts come from post-execution transcriptions of retained console output, not contemporaneous log files.",
            "AP101 v3 1024-frame peak/parity run is retained but contaminated and is not used for a speed claim.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    measured = report["measured"]
    stable = measured["stable_r0_1024"]
    folded = measured["pyspy_folded_samples"]
    ap = measured["ap_progression"]
    tar = measured["tar_reader_v1_timing_only"]
    tar2 = measured["tar_reader_v2_panel"]
    ap_profiles = measured["ap101_profiles"]
    live = measured["live_training_prefix"]
    selection3072 = measured["selection3072"]
    projection = report["conditional_projections"]
    shapes = report["estimates_from_exact_shapes"]
    bound = report["bounds"]
    sources = report["source_files"]

    source_rows = "\n".join(
        f"| `{value['path']}` | `{value['sha256']}` |" for value in sources.values()
    )
    class_rows = "\n".join(
        "| {name} | {count} | {fraction}% | {ap101} |".format(
            name=name,
            count=row["eligible_ground_truth_count"],
            fraction=_f(100 * row["eligible_ground_truth_fraction"], 2),
            ap101=_f(row["AP101_mean_thresholds"], 8),
        )
        for name, row in selection3072["per_class"].items()
    )
    return f"""# Diagnóstico de rendimiento del detector

Este informe se regenera exclusivamente desde artefactos locales retenidos. Separa medidas,
cotas aritméticas, prototipos y resultados desconocidos. **No acredita una aceleración de la
campaña completa.**

## Resultado ejecutivo

- El run estable de 1.024 updates midió {_f(stable["preprocessing_seconds_median"])} s de
  preparación mediana y {_f(stable["step_seconds_median"])} s de step mediano. Como están
  serializados, el solape perfecto tiene una cota aritmética de
  {_f(bound["perfect_preparation_compute_overlap_speedup"], 3)}x; no es una predicción.
- El perfil folded de entrenamiento contiene {folded["training"]["total_samples"]} muestras;
  {folded["training"]["tarfile_stack_samples"]} ({_f(100 * folded["training"]["tarfile_sample_fraction"], 1)}%)
  incluyen `tarfile.py` en {_f(folded["training"]["observation_interval_seconds"], 1)} s de observación.
  Es proporción de muestras de stack, **no porcentaje de wall time**.
- {folded["training"]["unknown_empty_stack_samples"]} muestras de entrenamiento
  ({_f(100 * folded["training"]["unknown_empty_stack_fraction"], 2)}%) no contienen stack y se
  clasifican como desconocidas; no se reasignan a TAR, CPU ni GPU.
- AP local-v2 pasó de {_f(ap["update1024_dev_ap101_supported_classes"], 6)} en update 1.024 a
  {_f(ap["update2048_dev_ap101_supported_classes"], 6)} en 2.048 y
  {_f(ap["update3072_dev_ap101_supported_classes"], 6)} en 3.072. No es score oficial.
- El panel TAR v2 de {tar2["samples"]} muestras midió {_f(tar2["raw_panel_speedup"], 2)}x
  en lectura raw y {_f(tar2["prepared_panel_speedup"], 3)}x en preparación, con hashes
  exactos entre productores y trials. El lector v2 cerró revisión `SHIP`; esto no amplía
  el alcance más allá del panel CPU.
- El speedup integral de campaña continúa **desconocido**.

## Evidencia medida

| Evidencia | Población | Resultado | Alcance |
|---|---:|---:|---|
| R0 estable | 1.024 updates / 4.096 exposiciones | prep {_f(stable["preprocessing_seconds_median"])} s; step {_f(stable["step_seconds_median"])} s | Medianas del runner congelado |
| py-spy entrenamiento | {folded["training"]["total_samples"]} samples / {folded["training"]["sampling_errors"]} errors | TAR {_f(100 * folded["training"]["tarfile_sample_fraction"], 1)}% | Sample share; transcripción posterior, no wall time |
| py-spy evaluación | {folded["evaluation"]["total_samples"]} samples / {folded["evaluation"]["sampling_errors"]} errors | TAR {_f(100 * folded["evaluation"]["tarfile_sample_fraction"], 1)}% | Sample share; transcripción posterior, no wall time |
| Cache de labels | {measured["data_read_profile_128"]["trace_count"]} frames | cache2 {_f(measured["data_read_profile_128"]["cache2_seconds"])} s; cache24 {_f(measured["data_read_profile_128"]["cache24_seconds"])} s | Perfil CPU retenido |
| TAR indexado v1 | {tar["panel_samples"]} samples | raw {_f(tar["raw_panel_speedup"], 2)}x; prepared {_f(tar["prepared_panel_speedup"], 2)}x | Timing de panel solamente; paridad exacta no admitida |
| TAR indexado v2 | {tar2["samples"]} samples / {tar2["sequences"]} secuencias / {tar2["trials"]} trials | raw {_f(tar2["raw_panel_speedup"], 2)}x; prepared {_f(tar2["prepared_panel_speedup"], 3)}x | Panel CPU alternado; review `SHIP` |
| AP101 v2 | 1.024 frames | {_f(ap_profiles["v2_timing_valid_1024"]["speedup"], 2)}x | Timing válido |
| AP101 v4 | 64 frames | {_f(ap_profiles["v4_integration_64"]["speedup"], 2)}x | Integración medida |

El AP101 v3 conserva un pico de {ap_profiles["v3_contaminated_not_for_speed_claim"]["process_peak_rss_bytes"]:,}
bytes, pero su corrida de 1.024 frames está contaminada y no se usa para afirmar speedup.
La interpretación firmada de la observación pasiva en update 2.048 es
`{measured["passive_update2048"]["bound_phase_inference"]}` y completó
{measured["passive_update2048"]["updates_completed"]} updates. Su CPU y 5,1% medio de GPU
corresponden a la evaluación programada, no al entrenamiento ni a throughput estable.

## Progreso RGB en el corte 3.420

El prefijo inmutable usado por este informe termina en update
`{live["cutoff_update"]}` y tiene SHA-256 `{live["prefix_sha256"]}`. La AP local-v2 del
monitor de 1.024 frames en update 3.072 es
`{_f(ap["update3072_dev_ap101_supported_classes"], 8)}`. El run sigue vivo; no existe todavía
un summary final ni un segmento durable apto para `summarize_detection_campaign.py`.

| Ventana | Updates | Prep mediana (s) | Step mediano (s) | Suma componentes mediana (s/update) | Updates/s de componentes |
|---|---:|---:|---:|---:|---:|
| Últimos 100 | {live["last100"]["update_count"]} | {_f(live["last100"]["median_preprocessing_seconds"])} | {_f(live["last100"]["median_forward_backward_seconds"])} | {_f(live["last100"]["median_component_sum_seconds_per_update"])} | {_f(live["last100"]["updates_per_component_second"], 4)} |
| Desde 3.072 | {live["since_update3072"]["update_count"]} | {_f(live["since_update3072"]["median_preprocessing_seconds"])} | {_f(live["since_update3072"]["median_forward_backward_seconds"])} | {_f(live["since_update3072"]["median_component_sum_seconds_per_update"])} | {_f(live["since_update3072"]["updates_per_component_second"], 4)} |

| Clase | GT elegible | Fracción GT | AP101 media |
|---|---:|---:|---:|
{class_rows}

## Proyecciones condicionales, no promesas

- Quedan `{projection["remaining_updates"]}` updates hasta 8.192 en el corte fijado. Aplicar la
  media agregada de los componentes observados en los últimos 100
  (`{_f(projection["last100_component_mean_seconds_per_update"], 6)}` s/update) produce
  `{_f(projection["training_component_seconds_using_last100_aggregate_mean"] / 3600, 2)}` horas
  de componentes de entrenamiento. La mediana 2,736 s/update de la tabla anterior describe un
  update típico, pero no se usa para totalizar trabajo.
- Quedan monitores programados en `{projection["remaining_monitor_updates"]}`. Escalar el total
  de inferencia observado en update 3.072 a esos monitores añade condicionalmente
  `{_f(projection["scheduled_monitor_inference_seconds_using_update3072_total"] / 3600, 2)}` horas.
- El subtotal condicional de esos dos componentes es
  `{_f(projection["training_plus_monitor_inference_seconds"] / 3600, 2)}` horas. Excluye scorer,
  checkpoint, serialización, H2D no separado, full-selection y otros overheads.
- Quedan dos pasadas full-selection, en updates
  `{projection["remaining_full_selection_updates"]}`; ninguna se ha ejecutado. Multiplicar 23.958
  frames por la mediana actual de
  `{_f(selection3072["inference_seconds_median_per_frame"], 6)}` s/frame da
  `{_f(projection["full_selection_inference_seconds_per_pass_using_update3072_frame_median"] / 3600, 2)}`
  horas por pasada y
  `{_f(projection["remaining_full_selection_inference_seconds_all_passes"] / 3600, 2)}` horas para
  ambas, solo de inferencia. Ningún subtotal es una ETA integral ni un compromiso de finalización.

## Memoria por formas exactas

| Materialización | Cálculo | Bytes | GiB |
|---|---|---:|---:|
| Imágenes FP32 | 4096×3×384×640×4 | {shapes["fp32_image_cache_4096_3x384x640_bytes"]:,} | {_f(shapes["fp32_image_cache_gib"], 2)} |
| Region masks FP32 BHW | 4096×384×640×4 | {shapes["fp32_region_masks_4096x384x640_bytes"]:,} | {_f(shapes["fp32_region_masks_gib"], 2)} |

Estas cifras son cálculos de tamaño, no picos RSS medidos. No se infiere memoria adicional.

## Estado de optimizaciones

| Intervención | Clasificación | Estado |
|---|---|---|
| Prefetch acotado CPU | Prototipo | Tests CPU; no integrado (`9526ec10`) |
| Solape prep/compute | Cota | {_f(bound["perfect_preparation_compute_overlap_speedup"], 3)}x ideal |
| TAR indexado | Medida parcial | Panel v2 exacto: {_f(tar2["prepared_panel_speedup"], 3)}x prepared; no end-to-end |
| AP101 optimizado | Medida parcial | suffix-max + búsqueda binaria Python; v2 timing válido; v4 integración |
| BF16 / TF32 / compile / CUDA Graphs | Desconocido | Sin experimento admitido |
| Campaña completa | Desconocido | Sin benchmark end-to-end |

No se admite el receipt v1 del lector indexado como prueba de paridad exacta: su hash lógico es
ambiguo. El receipt original fue restaurado byte a byte tras una sobrescritura accidental; la
excepción de restauración está hash-bound y v1 sigue siendo timing-only. TAR v2 es evidencia
separada, write-once, y su 1,776x no se interpreta como ganancia del entrenamiento. Los 102/22
errores de muestreo proceden de transcripciones posteriores del output de
la herramienta, hash-bound al perfil raw; no son logs contemporáneos ni miden tiempo GPU.

## Reproducción

```text
python scripts/build_detection_performance_diagnosis.py
```

El comando escribe `output/performance/performance_diagnosis_v5.json` y este documento.
Las rutas siguientes son relativas al repositorio; el Markdown no conserva rutas de usuario.

| Artefacto | SHA-256 |
|---|---|
{source_rows}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("output/performance/performance_diagnosis_v5.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/detection_sota_20261029/PERFORMANCE_DIAGNOSIS.md"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    report = build(root)
    output_json = root / args.output_json
    output_md = root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output_json.exists():
        if output_json.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(
                f"write-once diagnosis differs from regenerated evidence: {output_json}"
            )
    else:
        output_json.write_text(serialized, encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(output_json.relative_to(root).as_posix())
    print(output_md.relative_to(root).as_posix())


if __name__ == "__main__":
    main()
