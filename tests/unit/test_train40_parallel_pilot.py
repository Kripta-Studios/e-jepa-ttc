from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from operational.train40_system import (
    parallel_controller,
    parallel_engine,
    resource_monitor,
)
from operational.train40_system.parallel_runtime import (
    LEDGER_NAME,
    PILOT_UPDATES_PER_ARM,
    REGISTRY_NAME,
    ParallelResourceMonitor,
    arm_target_complete,
    physical_budget,
    publish_upfront_reservation,
)


def write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def journal(committed: int, *, durable: int | None = None, recovery: int = 0) -> dict[str, Any]:
    return {
        "contract": {"updates_limit": 49_932},
        "committed_updates": committed,
        "durable_updates": committed if durable is None else durable,
        "recovery_upper": recovery,
        "pending_update_upper": 0,
    }


def prepared_output(tmp_path: Path) -> Path:
    write(tmp_path / "AUTHORIZATION.json", {"previous_physical_execution_upper": 100})
    write(tmp_path / "TECHNICAL_ACCOUNTING.json", {"synthetic_optimizer_updates": 3})
    write(tmp_path / "fits/a5_seed7/UPDATE_JOURNAL.json", journal(27_500, recovery=30))
    write(tmp_path / "fits/c2f_seed7/UPDATE_JOURNAL.json", journal(400))
    write(
        tmp_path / REGISTRY_NAME,
        {"token": "nonce", "phase": "BOTH_AT_BARRIER", "children": {}},
    )
    return tmp_path


def test_upfront_reservation_uses_post_restore_baselines_and_balanced_targets(
    tmp_path: Path,
) -> None:
    output = prepared_output(tmp_path)
    ledger = publish_upfront_reservation(output, "nonce")
    assert ledger["arms"]["a5"] == {
        "baseline_committed": 27_500,
        "baseline_durable": 27_500,
        "baseline_recovery_upper": 30,
        "target_committed": 28_500,
    }
    assert ledger["arms"]["c2f"]["target_committed"] == 1_400
    assert ledger["updates_per_arm"] == PILOT_UPDATES_PER_ARM
    assert ledger["combined_updates"] == 2_000
    assert ledger["physical_updates_reserved_upper"] == physical_budget(output)["physical"] + 2_000
    assert json.loads((output / LEDGER_NAME).read_text())["token"] == "nonce"


def test_reservation_fails_closed_for_pending_or_physical_cap(tmp_path: Path) -> None:
    output = prepared_output(tmp_path)
    pending = journal(400)
    pending["pending_update_upper"] = 1
    write(output / "fits/c2f_seed7/UPDATE_JOURNAL.json", pending)
    with pytest.raises(ValueError, match="pending"):
        publish_upfront_reservation(output, "nonce")

    write(output / "fits/c2f_seed7/UPDATE_JOURNAL.json", journal(400))
    write(output / "AUTHORIZATION.json", {"previous_physical_execution_upper": 239_000})
    with pytest.raises(ValueError, match="physical update cap"):
        publish_upfront_reservation(output, "nonce")


def test_completed_arm_requires_exact_target_full_checkpoint(tmp_path: Path) -> None:
    output = prepared_output(tmp_path)
    ledger = publish_upfront_reservation(output, "nonce")
    target = ledger["arms"]["a5"]["target_committed"]
    write(output / "fits/a5_seed7/UPDATE_JOURNAL.json", journal(target, recovery=30))
    checkpoint = output / "fits/a5_seed7/checkpoint_last.pt"
    checkpoint.write_bytes(b"full checkpoint")
    from operational.efficient_context.common import digest

    write(
        output / "fits/a5_seed7/CHECKPOINT_RECEIPT.json",
        {
            "status": "PAUSED_RESOURCE",
            "committed_updates": target,
            "recovery_upper": 30,
            "sha256": digest(checkpoint),
            "full_optimizer_scheduler_sampler_and_all_RNG": True,
        },
    )
    assert arm_target_complete(output, "a5", "nonce")
    value = journal(target, durable=target - 1)
    write(output / "fits/a5_seed7/UPDATE_JOURNAL.json", value)
    assert not arm_target_complete(output, "a5", "nonce")


def test_reservation_requires_durable_baselines_and_exact_a5_recovery(tmp_path: Path) -> None:
    output = prepared_output(tmp_path)
    write(
        output / "fits/a5_seed7/UPDATE_JOURNAL.json",
        journal(27_500, durable=27_400, recovery=30),
    )
    with pytest.raises(ValueError, match="full durable"):
        publish_upfront_reservation(output, "nonce")
    write(output / "fits/a5_seed7/UPDATE_JOURNAL.json", journal(27_500, recovery=29))
    with pytest.raises(ValueError, match="exactly accounted as 30"):
        publish_upfront_reservation(output, "nonce")


def test_repeated_guards_do_not_consume_quota_and_target_cannot_overshoot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = prepared_output(tmp_path)
    ledger = publish_upfront_reservation(output, "nonce")
    monitor = object.__new__(ParallelResourceMonitor)
    monitor.output = output
    monitor.arm = "a5"
    monitor.token = "nonce"
    monitor._guard_invocations = 0
    monitor._parallel_guard_total_seconds = 0.0
    monitor._paired_sample = lambda required_arms: {
        "aggregate_tree_rss_bytes": 1,
        "host_available_bytes": 4 * 1024**3,
    }
    monkeypatch.setattr(
        resource_monitor.ResourceMonitor,
        "guard",
        lambda self, path: (True, {"reasons": []}),
    )
    allowed, _ = ParallelResourceMonitor.guard(monitor, output)
    assert allowed  # A5 prewarm admission consumes no quota.
    allowed, _ = ParallelResourceMonitor.guard(monitor, output)
    assert allowed
    assert json.loads((output / LEDGER_NAME).read_text()) == ledger
    target = ledger["arms"]["a5"]["target_committed"]
    write(output / "fits/a5_seed7/UPDATE_JOURNAL.json", journal(target, recovery=30))
    allowed, telemetry = ParallelResourceMonitor.guard(monitor, output)
    assert not allowed
    assert "PARALLEL_ARM_TARGET_REACHED" in telemetry["reasons"]


def test_failure_cleanup_waits_for_live_peer_full_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = prepared_output(tmp_path)
    checkpoint = output / "fits/a5_seed7/checkpoint_last.pt"
    checkpoint.write_bytes(b"paused")
    from operational.efficient_context.common import digest

    write(output / "fits/a5_seed7/UPDATE_JOURNAL.json", journal(27_500, recovery=30))
    write(
        output / "fits/a5_seed7/CHECKPOINT_RECEIPT.json",
        {"committed_updates": 27_500, "sha256": digest(checkpoint)},
    )

    class Process:
        calls = 0

        def poll(self) -> int | None:
            self.calls += 1
            return None if self.calls == 1 else 0

    pauses: list[str] = []
    monkeypatch.setattr(
        parallel_controller,
        "request_peer_pause",
        lambda out, token, arm: pauses.append(arm),
    )
    parallel_controller.wait_for_paused_survivors(
        output, "nonce", cast(Any, {"a5": Process()})
    )
    assert pauses == ["coordinator_failure"]


def test_frozen_heavy_predicate_keeps_parallel_route_explicitly_separate() -> None:
    predicate = resource_monitor._is_heavy_command
    assert not predicate(["python", "-m", "operational.train40_system.parallel_engine"])
    assert predicate(["python", "-m", "operational.train40_system.engine_host_fast_4"])


def test_parallel_engine_restores_monitor_binding_on_delegate_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from operational.train40_system import engine_host_fast_4

    original = resource_monitor.ResourceMonitor
    monkeypatch.setattr(parallel_engine, "verify_admission", lambda output: {})
    monkeypatch.setattr(parallel_engine, "wait_for_registry", lambda *args: None)

    def fail(output: Path, arm: str) -> None:
        assert resource_monitor.ResourceMonitor is not original
        raise RuntimeError("delegate failed")

    monkeypatch.setattr(engine_host_fast_4, "run", fail)
    with pytest.raises(RuntimeError, match="delegate failed"):
        parallel_engine.run(tmp_path, "a5", "nonce")
    assert resource_monitor.ResourceMonitor is original
    runtime = json.loads((tmp_path / "fits/a5_seed7/PARALLEL_PILOT_RUNTIME.json").read_text())
    assert runtime == {"arm": "a5", "status": "FAILED", "token": "nonce"}


def test_engine_wrapper_calls_run_directly_and_never_frozen_main() -> None:
    source = Path(parallel_engine.__file__).read_text(encoding="utf-8")
    assert "engine_host_fast_4.run(output, arm)" in source
    assert "engine_host_fast_4.main" not in source
    assert 'Lease(output / "fits" / f"{args.arm}_seed7")' in source


def test_parallel_constants_match_explicit_authorization() -> None:
    from operational.train40_system import parallel_runtime

    assert parallel_runtime.AGGREGATE_RSS_LIMIT_BYTES == 24 * 1024**3
    assert parallel_runtime.HOST_AVAILABLE_MIN_BYTES == 2 * 1024**3
    assert parallel_runtime.DISK_FREE_MIN_BYTES == 20_000_000_000
    assert parallel_runtime.PHYSICAL_UPDATE_LIMIT == 240_000
