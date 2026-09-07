"""Actual accounting and transport readers on small non-training evidence fixtures."""

import hashlib
import json

import pytest

from e_jepa_ttc.simplex_t.technical_bundle import technical_bundle_members


@pytest.mark.parametrize("fault", ["none", "payload", "counter", "pause"])
def test_underlying_failed_suite_counters_are_included(tmp_path, fault):
    def save(name, obj):
        path = tmp_path / name
        path.write_text(json.dumps(obj), encoding="utf-8")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    ledger, ledger_hash = save(
        "ledger.json", dict(schema="simplex_t_technical_budget_v1", reservations={"probe": 10})
    )
    counters, counters_hash = save(
        "UPDATE_PROGRESS.json", dict(completed_optimizer_updates=7, attempted_optimizer_updates=7)
    )
    evidence, evidence_hash = save(
        "evidence.json",
        dict(
            scientific_updates=0, completed_optimizer_updates=7, failed_nodeids=["fixture_failure"]
        ),
    )
    audit, audit_hash = save(
        "audit.json",
        dict(
            status="TECHNICAL_EXECUTION_RECORDS_RECONCILED_WITH_EVIDENCE_CLASSES",
            ledger_sha256=ledger_hash,
            reserved_updates=10,
            recorded_executed_updates=7,
            historical_noninstrumented_reconciliation_updates=0,
            engine_journal_or_instrumented_receipt_updates=7,
            unknown_or_unmatched_reservations=[],
            scientific_updates_in_these_records=0,
            operations=[
                dict(
                    operations=["probe"],
                    reserved_updates=10,
                    recorded_executed_updates=7,
                    evidence=dict(path=evidence.name, sha256=evidence_hash),
                    evidence_class="INSTRUMENTED_SUITE_COUNTER_INCLUDING_FAILED_RUNS",
                    counters_sha256=counters_hash,
                    failed_nodeids=["fixture_failure"],
                )
            ],
        ),
    )
    if fault == "payload":
        evidence.write_text("{}")
    elif fault == "counter":
        counters.write_text("{}")
    kwargs = dict(
        reconciliation_sha256=audit_hash,
        ledger=ledger,
        ledger_sha256=ledger_hash,
        work_root=tmp_path,
        resource_ok=lambda: fault != "pause",
    )
    if fault != "none":
        with pytest.raises((ValueError, InterruptedError)):
            technical_bundle_members(audit, **kwargs)
        return
    members = technical_bundle_members(audit, **kwargs)
    assert set(members) == {
        "technical/audit.json",
        "technical/ledger.json",
        "technical/evidence.json",
        "technical/UPDATE_PROGRESS.json",
    }
    for member in members.values():
        assert hashlib.sha256(member.path.read_bytes()).hexdigest() == member.sha256
    assert "fixture_failure" in evidence.read_text()
