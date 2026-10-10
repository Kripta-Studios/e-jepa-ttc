"""Validation and report-extension helpers for pipeline V2 receipts."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file

from .contracts import EVENT_FIT_IDS

RUNTIME_SCHEMA = "rgb_port_pipeline_runtime_v2"
PENDING_SCHEMA = "rgb_port_pipeline_pending_v2"
TIMINGS_SCHEMA = "rgb_port_pipeline_timings_v2"
RECEIPT_SCHEMA = "rgb_port_pipeline_checkpoint_receipt_v2"
CANARY_SCHEMA = "rgb_port_pipeline_canary_v2"


def _bindings(value: Mapping[str, Any], freeze: Mapping[str, Any], freeze_file_sha256: str) -> bool:
    return (
        value.get("pipeline_freeze_sha256") == freeze["identity_sha256"]
        and value.get("pipeline_freeze_file_sha256") == freeze_file_sha256
        and value.get("original_acceleration_freeze_sha256")
        == freeze["original_acceleration_freeze_sha256"]
        and value.get("original_acceleration_freeze_identity_sha256")
        == freeze["original_acceleration_freeze_identity_sha256"]
        and value.get("source_sha256") == freeze["source_sha256"]
    )


def validate_fit_lineage(
    run: Path,
    freeze: Mapping[str, Any],
    freeze_path: Path,
    fit_id: str,
    *,
    require_endpoint: bool = False,
) -> None:
    """Validate every retained pipeline checkpoint without retroactive attribution."""
    if fit_id not in EVENT_FIT_IDS:
        raise ValueError(f"Unscoped pipeline fit: {fit_id}")
    fit = run / "fits" / fit_id
    pointer_path = fit / "CHECKPOINT_POINTER.json"
    origin = freeze["origins"][fit_id]
    if not pointer_path.is_file():
        if origin.get("mode") == "FRESH_UNSTARTED":
            return
        raise FileNotFoundError(f"Pipeline fit lost its checkpoint pointer: {fit_id}")
    pointer = read_json_shared(pointer_path)
    completed = int(pointer.get("completed_updates", -1))
    origin_update = int(origin["completed_updates"])
    if completed < origin_update:
        raise RuntimeError(f"Pipeline checkpoint moved behind its origin: {fit_id}")
    version = fit / "checkpoint_versions" / str(pointer.get("version", ""))
    if not version.is_file() or sha256_file(version) != pointer.get("checkpoint_sha256"):
        raise RuntimeError(f"Pipeline pointer does not bind checkpoint bytes: {fit_id}")
    runtime_path = fit / "PIPELINE_RUNTIME.json"
    receipt_root = fit / "pipeline_checkpoints"
    receipt_path = receipt_root / f"checkpoint_{completed:06d}.json"
    if fit_id == EVENT_FIT_IDS[1] and not runtime_path.is_file():
        a5_decision_path = run / "fits" / EVENT_FIT_IDS[0] / "PIPELINE_CANARY.json"
        if a5_decision_path.is_file():
            a5_decision = read_json_shared(a5_decision_path)
            if (
                a5_decision.get("schema") == CANARY_SCHEMA
                and a5_decision.get("status") == "ROLLED_BACK_V1"
                and _bindings(a5_decision, freeze, sha256_file(freeze_path))
            ):
                return
    if completed == origin_update and not runtime_path.is_file() and not receipt_path.is_file():
        if origin.get("mode") != "PAUSED_FULL_CHECKPOINT":
            raise RuntimeError("Only a sealed paused origin may predate pipeline receipts")
        if pointer.get("checkpoint_sha256") != origin.get("checkpoint_sha256") or pointer.get(
            "identity_sha256"
        ) != origin.get("identity_sha256"):
            raise RuntimeError("Current native checkpoint differs from the pipeline origin")
        return
    freeze_file_sha256 = sha256_file(freeze_path)
    if not runtime_path.is_file():
        pending_path = receipt_root / "PENDING_EXECUTION_RECEIPT.json"
        if not pending_path.is_file():
            raise FileNotFoundError(f"Pipeline progress lacks runtime or pending proof: {fit_id}")
        pending = read_json_shared(pending_path)
        if (
            pending.get("schema") != PENDING_SCHEMA
            or pending.get("status") != "PENDING"
            or pending.get("fit_id") != fit_id
            or pending.get("mode") not in {"DEPTH1_EVENT_PREFETCH", "V1_LINEAGE_ONLY"}
            or int(pending.get("end_update", -1)) != completed
            or pending.get("checkpoint_version") != pointer.get("version")
            or pending.get("checkpoint_identity_sha256") != pointer.get("identity_sha256")
            or not _bindings(pending, freeze, freeze_file_sha256)
        ):
            raise RuntimeError("Pipeline pending proof cannot repair the durable checkpoint")
        return
    runtime = read_json_shared(runtime_path)
    runtime_update = int(runtime.get("last_saved_update", runtime.get("completed_updates", -1)))
    runtime_sha = runtime.get("last_checkpoint_sha256", runtime.get("checkpoint_sha256"))
    if (
        runtime.get("schema") != RUNTIME_SCHEMA
        or runtime.get("status") not in {"READY", "ACTIVE", "CANARY_DECISION_PENDING"}
        or runtime.get("fit_id") != fit_id
        or runtime.get("mode") not in {"CANARY", "ENABLED", "ROLLED_BACK_V1"}
        or runtime.get("optimizer_updates") != 0
        or runtime_update != completed
        or runtime_sha != pointer.get("checkpoint_sha256")
        or not _bindings(runtime, freeze, freeze_file_sha256)
    ):
        raise RuntimeError(f"Pipeline runtime does not bind the checkpoint: {fit_id}")
    pending_path = receipt_root / "PENDING_EXECUTION_RECEIPT.json"
    if not receipt_path.is_file():
        if not pending_path.is_file():
            raise FileNotFoundError(f"Pipeline checkpoint lacks receipt or pending proof: {fit_id}")
        pending = read_json_shared(pending_path)
        if (
            pending.get("schema") != PENDING_SCHEMA
            or pending.get("status") != "PENDING"
            or pending.get("fit_id") != fit_id
            or pending.get("mode") not in {"DEPTH1_EVENT_PREFETCH", "V1_LINEAGE_ONLY"}
            or int(pending.get("end_update", -1)) != completed
            or pending.get("checkpoint_version") != pointer.get("version")
            or pending.get("checkpoint_identity_sha256") != pointer.get("identity_sha256")
            or not _bindings(pending, freeze, freeze_file_sha256)
        ):
            raise RuntimeError("Pipeline pending proof differs")
        return
    for checkpoint in (fit / "checkpoint_versions").glob("checkpoint_*.pt"):
        update = int(checkpoint.stem.removeprefix("checkpoint_"))
        if update < origin_update:
            continue
        item_path = receipt_root / f"checkpoint_{update:06d}.json"
        timing_path = receipt_root / f"timings_{update:06d}.json"
        if not item_path.is_file() or not timing_path.is_file():
            raise FileNotFoundError(f"Pipeline checkpoint evidence is incomplete: {update}")
        item = read_json_shared(item_path)
        timing = read_json_shared(timing_path)
        if (
            item.get("schema") != RECEIPT_SCHEMA
            or item.get("status") not in {"COMPLETE", "RECOVERED_FROM_PENDING", "ORIGIN_BOUND"}
            or (item.get("status") == "ORIGIN_BOUND" and update != origin_update)
            or item.get("fit_id") != fit_id
            or int(item.get("completed_updates", -1)) != update
            or item.get("checkpoint_version") != checkpoint.name
            or item.get("checkpoint_sha256") != sha256_file(checkpoint)
            or item.get("checkpoint_identity_sha256") != pointer.get("identity_sha256")
            or item.get("mode") not in {"DEPTH1_EVENT_PREFETCH", "V1_LINEAGE_ONLY"}
            or item.get("timings_snapshot") != timing_path.name
            or item.get("timings_sha256") != sha256_file(timing_path)
            or timing.get("schema") != TIMINGS_SCHEMA
            or timing.get("fit_id") != fit_id
            or timing.get("mode") not in {"DEPTH1_EVENT_PREFETCH", "V1_LINEAGE_ONLY"}
            or int(timing.get("completed_updates", -1)) != update
            or not _bindings(item, freeze, freeze_file_sha256)
            or not _bindings(timing, freeze, freeze_file_sha256)
        ):
            raise RuntimeError(f"Pipeline checkpoint evidence changed: {update}")
    canary = freeze["canary"]
    boundary = origin_update + int(canary["scheduled_updates"])
    decision_path = fit / "PIPELINE_CANARY.json"
    if fit_id == EVENT_FIT_IDS[0] and completed >= boundary:
        if not decision_path.is_file():
            if completed == boundary and runtime.get("status") == "CANARY_DECISION_PENDING":
                return
            raise RuntimeError("Pipeline crossed its canary boundary without a decision")
        decision = read_json_shared(decision_path)
        boundary_receipt_path = receipt_root / f"checkpoint_{boundary:06d}.json"
        boundary_receipt = read_json_shared(boundary_receipt_path)
        boundary_timing_path = receipt_root / str(boundary_receipt.get("timings_snapshot", ""))
        boundary_timing = read_json_shared(boundary_timing_path)
        boundary_evidence_valid = (
            boundary_receipt.get("schema") == RECEIPT_SCHEMA
            and boundary_receipt.get("status") in {"COMPLETE", "RECOVERED_FROM_PENDING"}
            and boundary_receipt.get("fit_id") == fit_id
            and int(boundary_receipt.get("completed_updates", -1)) == boundary
            and boundary_receipt.get("checkpoint_sha256") == decision.get("checkpoint_sha256")
            and boundary_receipt.get("timings_snapshot") == boundary_timing_path.name
            and boundary_receipt.get("timings_sha256") == sha256_file(boundary_timing_path)
            and boundary_timing.get("schema") == TIMINGS_SCHEMA
            and boundary_timing.get("status") == "SNAPSHOT"
            and boundary_timing.get("fit_id") == fit_id
            and int(boundary_timing.get("completed_updates", -1)) == boundary
            and _bindings(boundary_receipt, freeze, freeze_file_sha256)
            and _bindings(boundary_timing, freeze, freeze_file_sha256)
        )
        marks = boundary_timing.get("canary_marks")
        expected_marks = [origin_update + 100, origin_update + 200, origin_update + 300]
        marks_valid = (
            isinstance(marks, list)
            and [item.get("completed_updates") for item in marks] == expected_marks
        )
        mark_times = (
            [float(item["science_seconds"]) for item in marks]
            if isinstance(marks, list) and marks_valid
            else []
        )
        derived_windows = (
            [
                6000.0 / mark_times[0],
                6000.0 / (mark_times[1] - mark_times[0]),
                6000.0 / (mark_times[2] - mark_times[1]),
            ]
            if len(mark_times) == 3
            and mark_times[0] > 0
            and mark_times[1] > mark_times[0]
            and mark_times[2] > mark_times[1]
            else []
        )
        gates = decision.get("integrity_gates", {})
        admission_gates = decision.get("admission_gates", {})
        gpu_admission = read_json_shared(Path(freeze["admissions"]["gpu"]["path"]))
        journal_baseline = decision.get("journal_baseline", {})
        journal_end = decision.get("journal_end", {})
        journals_present = isinstance(journal_baseline, dict) and isinstance(journal_end, dict)
        derived_recovery_delta = (
            max(
                0,
                int(journal_end.get("recovery_upper", -1))
                - int(journal_baseline.get("recovery_upper", -1)),
            )
            if journals_present
            else -1
        )
        journal_state_exact = (
            journals_present
            and int(journal_baseline.get("completed_updates", -1)) == origin_update
            and int(journal_baseline.get("durable_updates", -1)) == origin_update
            and int(journal_end.get("completed_updates", -1)) == boundary
            and int(journal_end.get("durable_updates", -1)) == boundary
            and int(journal_end.get("pending_update_upper", -1)) == 0
            and journal_baseline.get("identity_sha256") == origin.get("identity_sha256")
            and journal_end.get("identity_sha256") == origin.get("identity_sha256")
        )
        journal_failures = (
            derived_recovery_delta
            + int(journal_baseline.get("pending_update_upper", -1))
            + int(journal_end.get("pending_update_upper", -1))
            if journals_present
            else -1
        )
        expected_recovery_failures = int(gpu_admission.get("recovery_failures", -1)) + max(
            journal_failures, 0
        )
        expected_resource_failures = int(gpu_admission.get("resource_failures", -1)) + (
            0 if decision.get("boundary_checkpoint_status") == "PAUSED_REQUESTED" else 1
        )
        admission_passed = (
            admission_gates.get("rng_cursor_unchanged") is True
            and admission_gates.get("deferred_oserror_passed") is True
            and admission_gates.get("checkpoint_integrity") is True
        )
        counters = decision.get("prefetch_counters", {})
        measured = boundary_timing.get("timings", {})
        counters_bound = isinstance(counters, dict) and all(
            counters.get(name) == measured.get(name) for name in counters
        )
        counters_passed = (
            counters.get("submitted") == counters.get("consumed")
            and int(counters.get("outstanding_max", 2)) <= 1
            and counters.get("outstanding_current") == 0
            and counters.get("errors") == 0
            and counters.get("cancelled") == 0
            and counters.get("order_mismatches") == 0
            and counters_bound
        )
        raw_window_rates = decision.get("subwindow_updates_per_minute")
        windows_valid = (
            isinstance(raw_window_rates, list)
            and len(raw_window_rates) == canary["subwindow_count"]
            and all(isinstance(rate, (float, int)) and rate > 0 for rate in raw_window_rates)
        )
        window_rates = (
            [float(rate) for rate in raw_window_rates]
            if isinstance(raw_window_rates, list) and windows_valid
            else []
        )
        windows_bound = len(window_rates) == len(derived_windows) and all(
            math.isclose(actual, derived, rel_tol=1e-12)
            for actual, derived in zip(window_rates, derived_windows, strict=True)
        )
        median_rate = statistics.median(window_rates) if windows_valid else -1.0
        minimum_rate = min(window_rates) if windows_valid else -1.0
        windows_passed = (
            windows_valid
            and median_rate > canary["minimum_updates_per_minute_exclusive"]
            and minimum_rate >= canary["minimum_subwindow_rate_inclusive"]
            and windows_bound
        )
        safety_passed = (
            gates.get("byte_parity") is True
            and gates.get("schedule_order_unchanged") is True
            and gates.get("finite_outputs") is True
            and gates.get("recovery_failures") == expected_recovery_failures == 0
            and gates.get("resource_failures") == expected_resource_failures == 0
            and admission_passed
            and counters_passed
            and windows_passed
            and journal_state_exact
        )
        observed_rate = float(decision.get("observed_updates_per_minute", -1.0))
        observed_seconds = float(decision.get("observed_seconds", -1.0))
        derived_rate = (
            float(canary["scheduled_updates"]) * 60.0 / observed_seconds
            if observed_seconds > 0.0
            else -1.0
        )
        rate_passed = observed_rate > float(canary["minimum_updates_per_minute_exclusive"])
        expected_status = "PASSED" if rate_passed and safety_passed else "ROLLED_BACK_V1"
        expected_runtime_mode = "ENABLED" if expected_status == "PASSED" else "ROLLED_BACK_V1"
        runtime_mode_valid = runtime.get("mode") == expected_runtime_mode or (
            completed == boundary and runtime.get("mode") == "CANARY"
        )
        if (
            not boundary_evidence_valid
            or decision.get("schema") != CANARY_SCHEMA
            or decision.get("status") != expected_status
            or decision.get("fit_id") != fit_id
            or decision.get("origin_update") != origin_update
            or decision.get("boundary_update") != boundary
            or decision.get("scheduled_updates") != canary["scheduled_updates"]
            or decision.get("minimum_updates_per_minute_exclusive")
            != canary["minimum_updates_per_minute_exclusive"]
            or decision.get("subwindow_floor_updates_per_minute")
            != canary["minimum_subwindow_rate_inclusive"]
            or decision.get("subwindow_gate_passed") is not windows_passed
            or not math.isclose(
                float(decision.get("subwindow_median_updates_per_minute", -1.0)),
                float(median_rate),
                rel_tol=1e-12,
            )
            or not math.isclose(
                float(decision.get("subwindow_minimum_updates_per_minute", -1.0)),
                float(minimum_rate),
                rel_tol=1e-12,
            )
            or decision.get("foreground_wait_ratio_gate") != "DIAGNOSTIC_ONLY_NO_THRESHOLD"
            or not math.isclose(
                float(decision.get("overlap_fraction", -1.0)),
                1.0 - float(decision.get("foreground_wait_ratio", 2.0)),
                rel_tol=1e-12,
            )
            or decision.get("strict_rate_gate_passed") is not rate_passed
            or not math.isclose(observed_rate, derived_rate, rel_tol=1e-12)
            or not math.isclose(
                observed_seconds,
                float(boundary_timing.get("total_seconds", -1.0)),
                rel_tol=1e-12,
            )
            or decision.get("safety_gates_passed") is not safety_passed
            or decision.get("journal_recovery_delta") != derived_recovery_delta
            or decision.get("journal_state_exact") is not journal_state_exact
            or decision.get("boundary_checkpoint_status") != "PAUSED_REQUESTED"
            or (decision.get("status") == "PASSED" and not safety_passed)
            or not runtime_mode_valid
            or not _bindings(decision, freeze, freeze_file_sha256)
        ):
            raise RuntimeError("Pipeline canary decision differs")
    if require_endpoint and fit_id == EVENT_FIT_IDS[0] and not decision_path.is_file():
        raise RuntimeError("Complete event fit lacks a pipeline canary decision")


def write_report_extension(run: Path, freeze: Mapping[str, Any], freeze_path: Path) -> Path:
    """Write additive genealogy and the real resume command for final reporting."""
    fits: dict[str, Any] = {}
    for fit_id in EVENT_FIT_IDS:
        fit = run / "fits" / fit_id
        runtime = fit / "PIPELINE_RUNTIME.json"
        decision = fit / "PIPELINE_CANARY.json"
        fits[fit_id] = {
            "origin": freeze["origins"][fit_id],
            "runtime": read_json_shared(runtime) if runtime.is_file() else None,
            "canary": read_json_shared(decision) if decision.is_file() else None,
        }
    value = {
        "schema": "rgb_port_pipeline_report_extension_v2",
        "status": "CURRENT",
        "pipeline_freeze_sha256": freeze["identity_sha256"],
        "pipeline_freeze_file_sha256": sha256_file(freeze_path),
        "original_acceleration_freeze_sha256": freeze["original_acceleration_freeze_sha256"],
        "event_fits": fits,
        "resume_command": (f"python -m operational.rgb_port_pipeline_v2.queue resume --run {run}"),
    }
    path = run / "pipeline_v2/REPORT_EXTENSION.json"
    atomic_write_json(path, value)
    return path
