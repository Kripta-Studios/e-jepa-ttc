"""Pipeline V2 host-prefetch sidecar for frozen RGB-PORT event producers."""

# ruff: noqa: ANN401

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from torch import Tensor

import operational.rgb_port.train_producers as training
import operational.rgb_port_acceleration.producer as acceleration
from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file
from operational.rgb_port_pipeline_v2.prefetch import DepthOneEventPrefetch

EVENT_FITS = {"E_A5_MATCHED", "E_C2F_MATCHED"}


class _TimedComponents(dict[str, Tensor]):
    def __init__(self, values: Mapping[str, Tensor], runtime: Runtime) -> None:
        super().__init__(values)
        self._values = values
        self._runtime = runtime

    def items(self) -> Any:
        started = time.perf_counter()
        try:
            return self._values.items()
        finally:
            self._runtime.add_time("metrics_seconds", time.perf_counter() - started)


class Runtime:
    """Own cursor binding, in-memory timings and supplemental checkpoint lineage."""

    def __init__(
        self,
        source: training.ProducerSource,
        recipe: Any,
        run: Path,
        freeze_path: Path,
        freeze: Mapping[str, Any],
        *,
        prefetch_enabled: bool = True,
    ) -> None:
        self.fit_id = str(recipe.fit_id)
        if self.fit_id not in EVENT_FITS or str(recipe.modality) != "event":
            raise ValueError("Pipeline V2 is restricted to E_A5/E_C2F event producers")
        self.prefetch_enabled = prefetch_enabled
        self.original_source = source
        self.prefetch = DepthOneEventPrefetch(
            source,
            effective_batch_size=int(recipe.effective_batch_size),
            microbatch_size=int(recipe.microbatch_size),
        )
        self.training_source = self.prefetch if prefetch_enabled else source
        self.run = run
        self.freeze = dict(freeze)
        self.freeze_sha256 = str(freeze["identity_sha256"])
        self.freeze_file_sha256 = sha256_file(freeze_path)
        self.v1_freeze_sha256 = str(freeze["original_acceleration_freeze_sha256"])
        self.v1_freeze_identity = str(freeze["original_acceleration_freeze_identity_sha256"])
        self.fit = run / "fits" / self.fit_id
        decision_path = run / "fits/E_A5_MATCHED/PIPELINE_CANARY.json"
        decision = read_json_shared(decision_path) if decision_path.is_file() else {}
        self.runtime_mode = (
            "ROLLED_BACK_V1"
            if not prefetch_enabled
            else "ENABLED"
            if self.fit_id == "E_C2F_MATCHED" or decision.get("status") == "PASSED"
            else "CANARY"
        )
        self.runtime_path = self.fit / "PIPELINE_RUNTIME.json"
        self.receipt_dir = self.fit / "pipeline_checkpoints"
        self.pending_path = self.receipt_dir / "PENDING_EXECUTION_RECEIPT.json"
        journal_path = self.fit / "UPDATE_JOURNAL.json"
        self.journal_baseline = read_json_shared(journal_path) if journal_path.is_file() else None
        self.science_started: float | None = None
        self.canary_marks: list[dict[str, float | int]] = []
        self._last_snapshot_total = 0.0
        self._last_snapshot_timings: dict[str, float | int] = {}
        self.timings: dict[str, float | int] = {
            "h2d_seconds": 0.0,
            "loss_seconds": 0.0,
            "metrics_seconds": 0.0,
            "fast_guard_seconds": 0.0,
            "full_guard_seconds": 0.0,
            "save_seconds": 0.0,
            "h2d_calls": 0,
            "loss_calls": 0,
            "fast_guard_calls": 0,
            "full_guard_calls": 0,
            "save_calls": 0,
        }

    def add_time(self, name: str, elapsed: float) -> None:
        self.timings[name] = float(self.timings.get(name, 0.0)) + elapsed
        calls = name.removesuffix("_seconds") + "_calls"
        if calls in self.timings:
            self.timings[calls] = int(self.timings[calls]) + 1

    def _common(self) -> dict[str, Any]:
        return {
            "fit_id": self.fit_id,
            "pipeline_freeze_sha256": self.freeze_sha256,
            "pipeline_freeze_file_sha256": self.freeze_file_sha256,
            "original_acceleration_freeze_sha256": self.v1_freeze_sha256,
            "original_acceleration_freeze_identity_sha256": self.v1_freeze_identity,
            "source_sha256": self.freeze["source_sha256"],
            "mode": "DEPTH1_EVENT_PREFETCH" if self.prefetch_enabled else "V1_LINEAGE_ONLY",
            "optimizer_updates": 0,
        }

    def _validate_bindings(self, value: Mapping[str, Any]) -> None:
        expected = self._common()
        if any(value.get(name) != item for name, item in expected.items() if name != "mode"):
            raise RuntimeError("Pipeline V2 artifact bindings differ")

    def _timing_value(self, state: Any) -> dict[str, Any]:
        total = (
            time.perf_counter() - self.science_started if self.science_started is not None else 0.0
        )
        timings = {**self.timings, **self.prefetch.timing_snapshot()}
        interval = {
            name: value - self._last_snapshot_timings.get(name, 0)
            for name, value in timings.items()
            if isinstance(value, (float, int))
        }
        return {
            "schema": "rgb_port_pipeline_timings_v2",
            "status": "SNAPSHOT",
            **self._common(),
            "completed_updates": state.completed,
            "checkpoint_sha256": sha256_file(state.path),
            "total_seconds": total,
            "interval_seconds": total - self._last_snapshot_total,
            "timings": timings,
            "interval_timings": interval,
            "canary_marks": list(self.canary_marks),
            "prefetch": {"depth": 1, "workers": 1, "pinned": False, "gpu_stream": False},
        }

    def before_save(self, state: Any) -> None:
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.pending_path,
            {
                "schema": "rgb_port_pipeline_pending_v2",
                "status": "PENDING",
                **self._common(),
                "start_update": state.durable,
                "end_update": state.completed,
                "checkpoint_version": f"checkpoint_{state.completed:06d}.pt",
                "checkpoint_identity_sha256": state.identity_sha256,
            },
        )

    def _seal(self, state: Any, *, status: str) -> None:
        timing_path = self.receipt_dir / f"timings_{state.completed:06d}.json"
        value = self._timing_value(state)
        if timing_path.is_file():
            existing = read_json_shared(timing_path)
            self._validate_bindings(existing)
            if (
                existing.get("schema") != "rgb_port_pipeline_timings_v2"
                or existing.get("status") != "SNAPSHOT"
                or existing.get("completed_updates") != state.completed
                or existing.get("checkpoint_sha256") != sha256_file(state.path)
            ):
                raise RuntimeError("Immutable Pipeline V2 timing snapshot differs")
        else:
            atomic_write_json(timing_path, value)
            self._last_snapshot_total = float(value["total_seconds"])
            self._last_snapshot_timings = dict(value["timings"])
        receipt = {
            "schema": "rgb_port_pipeline_checkpoint_receipt_v2",
            "status": status,
            **self._common(),
            "completed_updates": state.completed,
            "checkpoint_version": f"checkpoint_{state.completed:06d}.pt",
            "checkpoint_sha256": sha256_file(state.path),
            "checkpoint_identity_sha256": state.identity_sha256,
            "timings_snapshot": timing_path.name,
            "timings_sha256": sha256_file(timing_path),
        }
        atomic_write_json(self.receipt_dir / f"checkpoint_{state.completed:06d}.json", receipt)
        if self.pending_path.is_file():
            self.pending_path.unlink()

    def after_restore(self, state: Any, cursor: Mapping[str, Any]) -> None:
        if self.science_started is None:
            self.science_started = time.perf_counter()
        if self.prefetch_enabled:
            self.prefetch.bind(cursor)
        pointer = read_json_shared(state.pointer)
        receipt = self.receipt_dir / f"checkpoint_{state.completed:06d}.json"
        if self.pending_path.is_file() and not receipt.is_file():
            pending = read_json_shared(self.pending_path)
            self._validate_bindings(pending)
            if (
                pending.get("schema") != "rgb_port_pipeline_pending_v2"
                or pending.get("status") != "PENDING"
                or pending.get("fit_id") != self.fit_id
                or pending.get("mode") != self._common()["mode"]
                or int(pending.get("start_update", -1)) > state.completed
                or pending.get("end_update") != state.completed
                or pending.get("checkpoint_version") != pointer.get("version")
                or pending.get("checkpoint_identity_sha256") != state.identity_sha256
                or pointer.get("checkpoint_sha256") != sha256_file(state.path)
            ):
                raise RuntimeError("Pipeline V2 pending proof cannot repair restored checkpoint")
            self._seal(state, status="RECOVERED_FROM_PENDING")
        elif receipt.is_file():
            value = read_json_shared(receipt)
            timing = self.receipt_dir / str(value.get("timings_snapshot", ""))
            self._validate_bindings(value)
            if (
                value.get("schema") != "rgb_port_pipeline_checkpoint_receipt_v2"
                or value.get("status") not in {"COMPLETE", "RECOVERED_FROM_PENDING"}
                or value.get("fit_id") != self.fit_id
                or value.get("completed_updates") != state.completed
                or value.get("checkpoint_version") != pointer.get("version")
                or value.get("checkpoint_sha256") != sha256_file(state.path)
                or value.get("checkpoint_identity_sha256") != state.identity_sha256
                or not timing.is_file()
                or value.get("timings_sha256") != sha256_file(timing)
            ):
                raise RuntimeError("Pipeline V2 checkpoint receipt differs")
            timing_value = read_json_shared(timing)
            self._validate_bindings(timing_value)
            if (
                timing_value.get("schema") != "rgb_port_pipeline_timings_v2"
                or timing_value.get("fit_id") != self.fit_id
                or timing_value.get("completed_updates") != state.completed
                or timing_value.get("checkpoint_sha256") != sha256_file(state.path)
            ):
                raise RuntimeError("Pipeline V2 checkpoint timing snapshot differs")
        else:
            origin = self.freeze["origins"][self.fit_id]
            if (
                origin.get("mode") != "PAUSED_FULL_CHECKPOINT"
                or origin.get("completed_updates") != state.completed
                or origin.get("checkpoint_sha256") != sha256_file(state.path)
                or origin.get("identity_sha256") != state.identity_sha256
            ):
                raise RuntimeError("Pipeline V2 restored checkpoint lacks supplemental lineage")
            self._seal(state, status="COMPLETE")
        self._write_runtime(state, "READY")

    def _write_runtime(self, state: Any, status: str) -> None:
        origin = int(self.freeze["origins"][self.fit_id]["completed_updates"])
        boundary = origin + int(self.freeze["canary"]["scheduled_updates"])
        if self.fit_id == "E_A5_MATCHED" and state.completed == boundary:
            status = "CANARY_DECISION_PENDING"
        atomic_write_json(
            self.runtime_path,
            {
                "schema": "rgb_port_pipeline_runtime_v2",
                "status": status,
                **self._common(),
                "mode": self.runtime_mode,
                "completed_updates": state.completed,
                "last_saved_update": state.completed,
                "checkpoint_sha256": sha256_file(state.path),
                "last_checkpoint_sha256": sha256_file(state.path),
                "timings": {**self.timings, **self.prefetch.timing_snapshot()},
            },
        )

    def after_save(self, state: Any, cursor: Mapping[str, Any]) -> None:
        if self.science_started is None:
            self.science_started = time.perf_counter()
        if self.prefetch_enabled:
            self.prefetch.bind(cursor)
        self._write_runtime(state, "ACTIVE")
        self._seal(state, status="COMPLETE")

    def after_commit(self, state: Any) -> None:
        if self.runtime_mode != "CANARY" or self.science_started is None:
            return
        origin = int(self.freeze["origins"][self.fit_id]["completed_updates"])
        distance = state.completed - origin
        if distance in {100, 200, 300}:
            self.canary_marks.append(
                {
                    "completed_updates": state.completed,
                    "science_seconds": time.perf_counter() - self.science_started,
                }
            )

    def close(self) -> None:
        self.prefetch.close()

    def finalize_canary(self, receipt: Mapping[str, Any]) -> None:
        """Record the preregistered boundary decision from measured science time."""
        if self.fit_id != "E_A5_MATCHED":
            return
        canary = self.freeze["canary"]
        origin = int(self.freeze["origins"][self.fit_id]["completed_updates"])
        boundary = origin + int(canary["scheduled_updates"])
        completed = int(receipt["completed_updates"])
        if completed != boundary:
            return
        timing = read_json_shared(self.receipt_dir / f"timings_{boundary:06d}.json")
        elapsed = float(timing["total_seconds"])
        if elapsed <= 0.0:
            raise RuntimeError("Pipeline canary science clock is invalid")
        rate = float(canary["scheduled_updates"]) * 60.0 / elapsed
        measured = timing["timings"]
        source_prepare = float(measured.get("source_prepare_seconds", 0.0))
        source_wait = float(measured.get("source_wait_seconds", 0.0))
        wait_ratio = source_wait / source_prepare if source_prepare > 0.0 else 0.0
        marks = timing.get("canary_marks", [])
        expected_marks = [origin + 100, origin + 200, origin + 300]
        marks_valid = [item.get("completed_updates") for item in marks] == expected_marks
        mark_times = [float(item["science_seconds"]) for item in marks] if marks_valid else []
        window_durations = (
            [mark_times[0], mark_times[1] - mark_times[0], mark_times[2] - mark_times[1]]
            if marks_valid
            else []
        )
        window_rates = [100.0 * 60.0 / duration for duration in window_durations]
        threshold = float(canary["minimum_updates_per_minute_exclusive"])
        floor = float(canary["baseline_updates_per_minute"]) * 0.95
        windows_pass = (
            len(window_rates) == 3
            and statistics.median(window_rates) > threshold
            and min(window_rates) >= floor
        )
        counters_ok = (
            measured.get("submitted") == measured.get("consumed")
            and int(measured.get("outstanding_max", 2)) <= 1
            and measured.get("outstanding_current") == 0
            and measured.get("errors") == 0
            and measured.get("cancelled") == 0
            and measured.get("order_mismatches") == 0
        )
        cpu = read_json_shared(Path(self.freeze["admissions"]["cpu"]["path"]))
        gpu = read_json_shared(Path(self.freeze["admissions"]["gpu"]["path"]))
        journal_end = read_json_shared(self.fit / "UPDATE_JOURNAL.json")
        if self.journal_baseline is None:
            raise RuntimeError("Pipeline canary lacks its initial update journal")
        recovery_delta = max(
            0,
            int(journal_end["recovery_upper"]) - int(self.journal_baseline["recovery_upper"]),
        )
        journal_failures = (
            recovery_delta
            + int(self.journal_baseline["pending_update_upper"])
            + int(journal_end["pending_update_upper"])
        )
        journal_state_exact = (
            int(self.journal_baseline["completed_updates"]) == origin
            and int(self.journal_baseline["durable_updates"]) == origin
            and int(journal_end["completed_updates"]) == boundary
            and int(journal_end["durable_updates"]) == boundary
            and int(journal_end["pending_update_upper"]) == 0
            and self.journal_baseline.get("identity_sha256") == journal_end.get("identity_sha256")
        )
        gates = {
            "byte_parity": cpu["batch_byte_parity"],
            "schedule_order_unchanged": cpu["schedule_order_unchanged"],
            "finite_outputs": gpu["finite_outputs"],
            "recovery_failures": gpu["recovery_failures"] + journal_failures,
            "resource_failures": gpu["resource_failures"]
            + (0 if receipt.get("status") == "PAUSED_REQUESTED" else 1),
        }
        admission_gates = {
            "rng_cursor_unchanged": cpu["rng_cursor_unchanged"],
            "deferred_oserror_passed": cpu["deferred_oserror_passed"],
            "checkpoint_integrity": gpu["checkpoint_integrity"],
        }
        safety = (
            gates["byte_parity"] is True
            and gates["schedule_order_unchanged"] is True
            and gates["finite_outputs"] is True
            and gates["recovery_failures"] == 0
            and gates["resource_failures"] == 0
            and all(value is True for value in admission_gates.values())
            and counters_ok
            and windows_pass
            and journal_state_exact
        )
        strict_rate = rate > float(canary["minimum_updates_per_minute_exclusive"])
        passed = strict_rate and safety
        decision = {
            "schema": "rgb_port_pipeline_canary_v2",
            "status": "PASSED" if passed else "ROLLED_BACK_V1",
            **self._common(),
            "origin_update": origin,
            "boundary_update": boundary,
            "scheduled_updates": canary["scheduled_updates"],
            "observed_seconds": elapsed,
            "observed_updates_per_minute": rate,
            "subwindow_updates_per_minute": window_rates,
            "subwindow_median_updates_per_minute": (
                statistics.median(window_rates) if window_rates else None
            ),
            "subwindow_minimum_updates_per_minute": min(window_rates) if window_rates else None,
            "subwindow_floor_updates_per_minute": floor,
            "subwindow_gate_passed": windows_pass,
            "minimum_updates_per_minute_exclusive": canary["minimum_updates_per_minute_exclusive"],
            "strict_rate_gate_passed": strict_rate,
            "integrity_gates": gates,
            "admission_gates": admission_gates,
            "prefetch_counters": {
                name: measured.get(name)
                for name in (
                    "submitted",
                    "consumed",
                    "cancelled",
                    "errors",
                    "order_mismatches",
                    "outstanding_max",
                    "outstanding_current",
                    "cache_reads",
                    "cache_hits",
                    "cache_entries",
                    "cache_bytes",
                )
            },
            "safety_gates_passed": safety,
            "foreground_wait_ratio": wait_ratio,
            "overlap_fraction": 1.0 - wait_ratio,
            "foreground_wait_ratio_gate": "DIAGNOSTIC_ONLY_NO_THRESHOLD",
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "boundary_checkpoint_status": receipt["status"],
            "journal_baseline": self.journal_baseline,
            "journal_end": journal_end,
            "journal_recovery_delta": recovery_delta,
            "journal_state_exact": journal_state_exact,
        }
        atomic_write_json(self.fit / "PIPELINE_CANARY.json", decision)
        runtime = read_json_shared(self.runtime_path)
        runtime.update(
            status="ACTIVE",
            mode="ENABLED" if passed else "ROLLED_BACK_V1",
            canary_status=decision["status"],
        )
        self.runtime_mode = cast(str, runtime["mode"])
        atomic_write_json(self.runtime_path, runtime)


@contextmanager
def installed(runtime: Runtime) -> Iterator[None]:
    original_restore = training.ProducerCheckpoint.restore
    original_save = training.ProducerCheckpoint.save
    original_commit = training.ProducerCheckpoint.commit
    original_to_device = training._to_device
    original_loss = training._producer_loss
    original_fast = training._fast_resource_guard
    original_full = training._resource_guard

    def restore(state: Any, *args: Any, **kwargs: Any) -> Any:
        cursor = original_restore(state, *args, **kwargs)
        runtime.after_restore(state, cursor)
        return cursor

    def save(state: Any, *args: Any, **kwargs: Any) -> None:
        cursor = cast(Mapping[str, Any], args[4])
        runtime.before_save(state)
        started = time.perf_counter()
        original_save(state, *args, **kwargs)
        runtime.add_time("save_seconds", time.perf_counter() - started)
        runtime.after_save(state, cursor)

    def commit(state: Any) -> None:
        original_commit(state)
        runtime.after_commit(state)

    def to_device(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_to_device(*args, **kwargs)
        finally:
            runtime.add_time("h2d_seconds", time.perf_counter() - started)

    def loss(*args: Any, **kwargs: Any) -> tuple[Tensor, Mapping[str, Tensor]]:
        started = time.perf_counter()
        total, components = original_loss(*args, **kwargs)
        runtime.add_time("loss_seconds", time.perf_counter() - started)
        return total, _TimedComponents(components, runtime)

    def timed_guard(name: str, function: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            runtime.add_time(name, time.perf_counter() - started)

    cast(Any, training.ProducerCheckpoint).restore = restore
    cast(Any, training.ProducerCheckpoint).save = save
    if runtime.runtime_mode == "CANARY":
        cast(Any, training.ProducerCheckpoint).commit = commit
    if runtime.prefetch_enabled:
        cast(Any, training)._to_device = to_device
        cast(Any, training)._producer_loss = loss
        cast(Any, training)._fast_resource_guard = lambda *a, **k: timed_guard(
            "fast_guard_seconds", original_fast, *a, **k
        )
        cast(Any, training)._resource_guard = lambda *a, **k: timed_guard(
            "full_guard_seconds", original_full, *a, **k
        )
    try:
        yield
    finally:
        cast(Any, training.ProducerCheckpoint).restore = original_restore
        cast(Any, training.ProducerCheckpoint).save = original_save
        if runtime.runtime_mode == "CANARY":
            cast(Any, training.ProducerCheckpoint).commit = original_commit
        if runtime.prefetch_enabled:
            cast(Any, training)._to_device = original_to_device
            cast(Any, training)._producer_loss = original_loss
            cast(Any, training)._fast_resource_guard = original_fast
            cast(Any, training)._resource_guard = original_full


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-freeze", type=Path, required=True)
    parser.add_argument("forward", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.forward or args.forward[0] != "--":
        parser.error("V1 acceleration arguments must follow --")
    forwarded = args.forward[1:]
    if "--fit-id" not in forwarded:
        parser.error("nested producer command lacks --fit-id")
    fit_id = forwarded[forwarded.index("--fit-id") + 1]
    if fit_id not in EVENT_FITS:
        parser.error("Pipeline V2 is restricted to E_A5/E_C2F")
    from operational.rgb_port_pipeline_v2.contracts import validate_pipeline_freeze

    freeze_path = args.pipeline_freeze.resolve(strict=True)
    freeze = validate_pipeline_freeze(freeze_path)
    if "--acceleration-freeze" not in forwarded:
        parser.error("nested V1 command lacks --acceleration-freeze")
    v1_path = Path(forwarded[forwarded.index("--acceleration-freeze") + 1]).resolve(strict=True)
    if sha256_file(v1_path) != freeze["original_acceleration_freeze_sha256"]:
        raise ValueError("Nested V1 acceleration freeze differs from Pipeline V2 parent")
    original_fit = training.fit_producer

    def fit(
        source: training.ProducerSource, recipe: Any, run: Path, **kwargs: Any
    ) -> dict[str, Any]:
        decision_path = run / "fits/E_A5_MATCHED/PIPELINE_CANARY.json"
        decision = read_json_shared(decision_path) if decision_path.is_file() else None
        if decision is not None:
            from operational.rgb_port_pipeline_v2.receipts import validate_fit_lineage

            validate_fit_lineage(
                run,
                freeze,
                freeze_path,
                "E_A5_MATCHED",
                require_endpoint=True,
            )
        if recipe.fit_id == "E_C2F_MATCHED" and (
            decision is None or decision.get("status") != "PASSED"
        ):
            return original_fit(source, recipe, run, **kwargs)
        prefetch_enabled = decision is None or decision.get("status") == "PASSED"
        runtime = Runtime(
            source,
            recipe,
            run,
            freeze_path,
            freeze,
            prefetch_enabled=prefetch_enabled,
        )
        if recipe.fit_id == "E_A5_MATCHED" and decision is None:
            pointer = read_json_shared(run / "fits/E_A5_MATCHED/CHECKPOINT_POINTER.json")
            current = int(pointer["completed_updates"])
            origin = int(freeze["origins"]["E_A5_MATCHED"]["completed_updates"])
            boundary = origin + int(freeze["canary"]["scheduled_updates"])
            remaining = boundary - current
            if remaining < 0:
                raise RuntimeError("Pipeline canary boundary was crossed without a decision")
            if remaining == 0:
                try:
                    receipt = read_json_shared(run / "fits/E_A5_MATCHED/CHECKPOINT_RECEIPT.json")
                    runtime.finalize_canary(receipt)
                    return receipt
                finally:
                    runtime.close()
            runtime.prefetch.limit_effective_batches(remaining)
            requested = kwargs.get("max_updates_this_call")
            kwargs["max_updates_this_call"] = (
                remaining if requested is None else min(int(requested), remaining)
            )
        try:
            with installed(runtime):
                receipt = original_fit(runtime.training_source, recipe, run, **kwargs)
            runtime.finalize_canary(receipt)
            return receipt
        finally:
            runtime.close()

    training.fit_producer = fit
    try:
        return acceleration.main(forwarded)
    finally:
        training.fit_producer = original_fit


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EVENT_FITS", "Runtime", "installed", "main"]
