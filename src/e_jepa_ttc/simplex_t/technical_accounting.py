"""Revalidate recorded technical optimizer work, preserving evidence limitations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path


def verify_technical_accounting(
    reconciliation: Path,
    *,
    reconciliation_sha256: str,
    ledger: Path,
    ledger_sha256: str,
    work_root: Path,
    resource_ok: Callable[[], bool],
) -> dict:
    """Rehash operation evidence and derive counts without summing resume endpoints.

    The input reconciliation is an explicit pinned audit, not inferred execution
    from reservation size. Its historical noninstrumented evidence remains marked
    as such. This neither reads scientific scores nor executes an optimizer.
    """
    work = work_root.resolve(strict=True)

    def read(path: Path, expected: str, *, decode: bool = True) -> dict:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: technical accounting")
        target = path.resolve(strict=True)
        if not target.is_relative_to(work):
            raise ValueError("technical evidence outside companion worktree")
        with target.open("rb") as stream:
            payload = stream.read(8_388_609)
        if len(payload) > 8_388_608 or hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("technical evidence hash or size differs")
        result = json.loads(payload) if decode else {}
        if not isinstance(result, dict):
            raise ValueError("technical evidence object required")
        return result

    def count(value: object) -> int:
        if type(value) is not int or value < 0:
            raise ValueError("nonnegative integer execution count required")
        return value

    budget = read(ledger, ledger_sha256)
    audit = read(reconciliation, reconciliation_sha256)
    if (
        budget.get("schema") != "simplex_t_technical_budget_v1"
        or audit.get("ledger_sha256") != ledger_sha256
        or audit.get("status") != "TECHNICAL_EXECUTION_RECORDS_RECONCILED_WITH_EVIDENCE_CLASSES"
    ):
        raise ValueError("technical reconciliation contract differs")
    reservations = budget["reservations"]
    reserved = sum(count(value) for value in reservations.values())
    if reserved > 1025:
        raise ValueError("technical reservation cap exceeded")
    seen, classes, failures = set(), {}, []
    uncertain_upper = 0
    consumed_evidence: set[tuple[Path, str]] = set()
    for row in audit["operations"]:
        operations = row["operations"]
        if (
            not operations
            or len(set(operations)) != len(operations)
            or seen.intersection(operations)
            or set(operations) - set(reservations)
        ):
            raise ValueError("missing, unknown or duplicate technical operation")
        seen.update(operations)
        allocation = sum(reservations[name] for name in operations)
        if row["reserved_updates"] != allocation:
            raise ValueError("operation reservation differs")
        evidence_path = work / row["evidence"]["path"]
        evidence = read(evidence_path, row["evidence"]["sha256"])
        category = row["evidence_class"]
        evidence_key = (
            evidence_path.resolve(strict=True),
            row["supporting_junit"]["path"]
            if category == "CONTEMPORANEOUS_RECONCILIATION_NOT_INSTRUMENTED_COUNTER"
            else "",
        )
        if evidence_key in consumed_evidence:
            raise ValueError("technical execution evidence counted more than once")
        consumed_evidence.add(evidence_key)
        if category == "CONTEMPORANEOUS_RECONCILIATION_NOT_INSTRUMENTED_COUNTER":
            support = row["supporting_junit"]
            read(evidence_path.parent / support["path"], support["sha256"], decode=False)
            matches = [
                entry["updates"]
                for entry in evidence["technical_update_receipts"]
                if entry["evidence"] == support["path"]
            ]
            if len(matches) != 1:
                raise ValueError("historical technical receipt mapping differs")
            executed = count(matches[0])
        elif category == "ENGINE_COMPLETION_RECEIPT":
            if (
                evidence["scientific_fits"] != 0
                or evidence["result"]["scientific_endpoint"] is not False
            ):
                raise ValueError("profile is not technical-only")
            executed = count(evidence["result"]["completed_updates"])
        elif category == "ENGINE_RESUME_RECEIPT_CUMULATIVE_ENDPOINTS_NOT_SUMMED":
            if (
                evidence["scientific_updates"] != 0
                or evidence["exact_complete_state_match"] is not True
            ):
                raise ValueError("resume receipt is not exact technical evidence")
            executed = count(evidence["executed_technical_updates"])
        elif category == "SETTLED_TECHNICAL_JOURNAL_NOT_CAMPAIGN_SCIENTIFIC_FITS":
            if set(evidence["fits"]) != {"continuous", "split"} or any(
                fit["pending"] is not None or fit["uncertain_lost_upper"] != 0
                for fit in evidence["fits"].values()
            ):
                raise ValueError("technical journal has unsettled work")
            executed = sum(count(fit["completed"]) for fit in evidence["fits"].values())
        elif category == "UNSETTLED_TECHNICAL_RESERVATION_CONSERVATIVE_UPPER":
            if (
                set(evidence["fits"]) != {"continuous"}
                or evidence["fits"]["continuous"]["completed"] != 0
                or evidence["fits"]["continuous"]["pending"] != [0, 100]
                or row["uncertain_updates_upper"] != allocation
            ):
                raise ValueError("failed technical reservation evidence differs")
            executed = 0
            uncertain_upper += allocation
            failures.append(row["failure_id"])
        elif category == "INSTRUMENTED_SUITE_COUNTER_INCLUDING_FAILED_RUNS":
            counters = read(evidence_path.parent / "UPDATE_PROGRESS.json", row["counters_sha256"])
            executed = count(counters["completed_optimizer_updates"])
            if (
                counters["attempted_optimizer_updates"] != executed
                or evidence["completed_optimizer_updates"] != executed
                or evidence["scientific_updates"] != 0
                or row["failed_nodeids"] != evidence["failed_nodeids"]
            ):
                raise ValueError("instrumented suite count or failure IDs differ")
            failures.extend(evidence["failed_nodeids"])
        else:
            raise ValueError("unrecognized technical evidence class")
        if executed != count(row["recorded_executed_updates"]) or executed > allocation:
            raise ValueError(
                "recorded execution differs from underlying evidence or exceeds reservation"
            )
        classes[category] = classes.get(category, 0) + executed
    total = sum(classes.values())
    legacy = classes.get("CONTEMPORANEOUS_RECONCILIATION_NOT_INSTRUMENTED_COUNTER", 0)
    if (
        seen != set(reservations)
        or audit["unknown_or_unmatched_reservations"]
        or audit["reserved_updates"] != reserved
        or audit["recorded_executed_updates"] != total
        or audit["historical_noninstrumented_reconciliation_updates"] != legacy
        or audit["engine_journal_or_instrumented_receipt_updates"] != total - legacy
        or audit["scientific_updates_in_these_records"] != 0
        or audit.get("uncertain_updates_upper", 0) != uncertain_upper
    ):
        raise ValueError("technical reconciliation totals or coverage differ")
    return dict(
        status="TECHNICAL_RECORDED_EXECUTION_REVERIFIED_NOT_SCIENTIFIC_COMPLETION",
        ledger_sha256=ledger_sha256,
        reconciliation_sha256=reconciliation_sha256,
        reserved_updates=reserved,
        recorded_executed_updates=total,
        uncertain_updates_upper=uncertain_upper,
        evidence_class_updates=classes,
        historical_noninstrumented_updates=legacy,
        failed_nodeids=sorted(set(failures)),
        operations=len(seen),
        new_optimizer_updates=0,
        scientific_updates_in_these_records=0,
    )
