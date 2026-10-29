"""Technical reservations are not execution; evidence tampering is rejected."""

import hashlib
import json

import pytest

from e_jepa_ttc.simplex_t.technical_accounting import verify_technical_accounting


@pytest.mark.parametrize("fault", ["none", "pending", "upper", "completed"])
def test_unsettled_reservation_keeps_conservative_upper(tmp_path, fault):
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    ledger, ledger_hash = save(
        "ledger.json",
        {
            "schema": "simplex_t_technical_budget_v1",
            "reservations": {"failed": 20},
        },
    )
    evidence, evidence_hash = save(
        "journal.json",
        {
            "fits": {
                "continuous": {
                    "completed": 1 if fault == "completed" else 0,
                    "pending": None if fault == "pending" else [0, 100],
                }
            }
        },
    )
    audit, audit_hash = save(
        "audit.json",
        {
            "status": "TECHNICAL_EXECUTION_RECORDS_RECONCILED_WITH_EVIDENCE_CLASSES",
            "ledger_sha256": ledger_hash,
            "reserved_updates": 20,
            "recorded_executed_updates": 0,
            "uncertain_updates_upper": 20,
            "historical_noninstrumented_reconciliation_updates": 0,
            "engine_journal_or_instrumented_receipt_updates": 0,
            "unknown_or_unmatched_reservations": [],
            "scientific_updates_in_these_records": 0,
            "operations": [
                {
                    "operations": ["failed"],
                    "reserved_updates": 20,
                    "recorded_executed_updates": 0,
                    "evidence": {"path": evidence.name, "sha256": evidence_hash},
                    "evidence_class": "UNSETTLED_TECHNICAL_RESERVATION_CONSERVATIVE_UPPER",
                    "uncertain_updates_upper": 0 if fault == "upper" else 20,
                    "failure_id": "preserved_failure",
                }
            ],
        },
    )
    kwargs = dict(
        reconciliation_sha256=audit_hash,
        ledger=ledger,
        ledger_sha256=ledger_hash,
        work_root=tmp_path,
        resource_ok=lambda: True,
    )
    if fault != "none":
        with pytest.raises(ValueError):
            verify_technical_accounting(audit, **kwargs)
    else:
        result = verify_technical_accounting(audit, **kwargs)
        assert result["recorded_executed_updates"] == 0
        assert result["uncertain_updates_upper"] == 20
        assert result["failed_nodeids"] == ["preserved_failure"]


@pytest.mark.parametrize(
    "fault", ["none", "duplicate", "execution", "reservation", "payload", "pause"]
)
def test_underlying_execution_receipt(tmp_path, fault):
    def save(name, content):
        path = tmp_path / name
        path.write_text(json.dumps(content))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    ledger, ledger_hash = save(
        "ledger.json", dict(schema="simplex_t_technical_budget_v1", reservations={"probe": 10})
    )
    evidence, evidence_hash = save(
        "evidence.json",
        dict(scientific_fits=0, result=dict(scientific_endpoint=False, completed_updates=7)),
    )
    row = dict(
        operations=["probe"],
        reserved_updates=10,
        recorded_executed_updates=7,
        evidence=dict(path=evidence.name, sha256=evidence_hash),
        evidence_class="ENGINE_COMPLETION_RECEIPT",
    )
    audit = dict(
        status="TECHNICAL_EXECUTION_RECORDS_RECONCILED_WITH_EVIDENCE_CLASSES",
        ledger_sha256=ledger_hash,
        operations=[row],
        reserved_updates=10,
        recorded_executed_updates=7,
        historical_noninstrumented_reconciliation_updates=0,
        engine_journal_or_instrumented_receipt_updates=7,
        unknown_or_unmatched_reservations=[],
        scientific_updates_in_these_records=0,
    )
    if fault == "duplicate":
        audit["operations"].append(row)
    elif fault == "execution":
        row["recorded_executed_updates"] = 10
    elif fault == "reservation":
        row["reserved_updates"] = 11
    path, digest = save("audit.json", audit)
    if fault == "payload":
        evidence.write_text("{}")
    kwargs = dict(
        reconciliation_sha256=digest,
        ledger=ledger,
        ledger_sha256=ledger_hash,
        work_root=tmp_path,
        resource_ok=lambda: fault != "pause",
    )
    if fault != "none":
        with pytest.raises((ValueError, InterruptedError)):
            verify_technical_accounting(path, **kwargs)
        return
    result = verify_technical_accounting(path, **kwargs)
    assert result["reserved_updates"] == 10
    assert result["recorded_executed_updates"] == 7
    assert result["new_optimizer_updates"] == 0
