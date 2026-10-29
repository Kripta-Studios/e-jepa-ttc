"""Extend the pinned 885-update audit with real device probes; no training."""

import argparse
import json
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.lifecycle import admitted
from e_jepa_ttc.simplex_t.technical_accounting import verify_technical_accounting
from e_jepa_ttc.simplex_t.training import load_checkpoint, state_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = args.worktree.resolve(strict=True)
    base = work / "artifacts/simplex_t"
    old = base / "T0/TECHNICAL_RECONCILIATION_885.json"
    if sha256(old) != "cfe6a569cec45b17708e09e3ddcb822516eb9acb5ba115a37109a14c9410911f":
        raise ValueError("prior accounting pin differs")
    audit = json.loads(old.read_text(encoding="utf-8"))
    ledger = base / "TECHNICAL_BUDGET.json"
    reservations = json.loads(ledger.read_text(encoding="utf-8"))["reservations"]
    added_updates = 0
    uncertain_upper = 0
    for name in (
        "cpu_scalar_c160",
        "cuda_scalar_c160",
        "cuda_scalar_c160_medianfix",
        "cpu_latent_c160",
        "cuda_latent_c160",
    ):
        root = base / "T0/device_benchmark" / name
        contract = json.loads((root / "CONTRACT.json").read_text(encoding="utf-8"))
        operation = "real_context_cpu_resume_" + state_digest(contract)
        if reservations.get(operation) != 20:
            raise ValueError("device probe reservation differs")
        continuous = load_checkpoint(root / "continuous/checkpoint_last.pt")
        failed = name == "cuda_scalar_c160"
        if failed:
            if continuous["completed_updates"] != 0 or continuous["optimizer"]["state"]:
                raise ValueError("failed probe checkpoint is not update0")
            evidence = root / "TECHNICAL_JOURNAL.json"
            category = "UNSETTLED_TECHNICAL_RESERVATION_CONSERVATIVE_UPPER"
            uncertain_upper += reservations[operation]
        else:
            split = load_checkpoint(root / "split/checkpoint_last.pt")
            if continuous["completed_updates"] != 10 or state_digest(continuous) != state_digest(
                split
            ):
                raise ValueError("device probe complete state mismatch")
            evidence = root / "RESUME_QA.json"
            category = "ENGINE_RESUME_RECEIPT_CUMULATIVE_ENDPOINTS_NOT_SUMMED"
            added_updates += continuous["completed_updates"] + split["completed_updates"]
        row = {
            "operations": [operation],
            "reserved_updates": 20,
            "recorded_executed_updates": 0 if failed else 20,
            "evidence": {"path": str(evidence.relative_to(work)), "sha256": sha256(evidence)},
            "evidence_class": category,
        }
        if failed:
            row.update(
                uncertain_updates_upper=20,
                failure_id="CUDA_MEDIAN_INDICES_REJECTED_BY_STRICT_DETERMINISM",
            )
        audit["operations"].append(row)
    audit.update(
        ledger_sha256=sha256(ledger),
        reserved_updates=sum(reservations.values()),
        recorded_executed_updates=audit["recorded_executed_updates"] + added_updates,
        uncertain_updates_upper=uncertain_upper,
        engine_journal_or_instrumented_receipt_updates=(
            audit["engine_journal_or_instrumented_receipt_updates"] + added_updates
        ),
    )
    write_new_json(args.output, audit)
    result = verify_technical_accounting(
        args.output,
        reconciliation_sha256=sha256(args.output),
        ledger=ledger,
        ledger_sha256=sha256(ledger),
        work_root=work,
        resource_ok=lambda: bool(admitted([work])["has_headroom"]),
    )
    write_new_json(args.output.with_suffix(".verified.json"), result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
