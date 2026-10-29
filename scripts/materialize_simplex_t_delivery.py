"""Gather actual phase/attempt references for the existing T6 analysis worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.coordination import shared_write_admission
from e_jepa_ttc.simplex_t.delivery_launch_materialization import delivery_launch
from e_jepa_ttc.simplex_t.lifecycle import admitted


def read(path: Path, limit: int = 1_048_576) -> dict:
    """Bound all metadata reads; never decode prediction tables here."""
    if path.stat().st_size > limit:
        raise ValueError("delivery metadata exceeds bound")
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("delivery metadata object required")
    return value


def main() -> int:
    """Prepare an immutable v3 launch; actual T6 admission remains mandatory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--reconciliation-sha256", required=True)
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    work = Path(__file__).resolve().parents[1]
    root, run_root = args.campaign_root.resolve(), args.run_root.resolve()
    if any(
        path == work / "artifacts" or not path.is_relative_to(work / "artifacts")
        for path in (root, run_root)
    ):
        raise ValueError("campaign and logs must remain inside companion artifacts")
    if args.other_reserved_bytes < 0:
        parser.error("nonnegative reservation required")
    snapshot = admitted([work])
    if not snapshot["has_headroom"] or not shared_write_admission(
        snapshot["written_volume_free_bytes"][0], args.other_reserved_bytes + 8_388_608
    ):
        return 3
    if sha256(args.reconciliation) != args.reconciliation_sha256:
        raise ValueError("technical reconciliation changed")
    base = read(root / "launches/T2.json")
    frozen = read(Path(base["freeze"]), 8_388_608)
    if sha256(Path(base["freeze"])) != base["freeze_sha256"]:
        raise ValueError("scientific freeze changed")
    flags = frozen["source_contract"]["availability"]
    stages = ["T2"] + (["T4"] if flags["latent"] else [])
    for stage in ("T3", "T5"):
        decision = read(root / "launches" / f"{stage}.decision.json")
        if (
            decision.get("schema") != "simplex_t_practical_launch_decision_v1"
            or decision.get("stage") != stage
            or decision.get("freeze_sha256") != base["freeze_sha256"]
            or type(decision.get("eligible")) is not bool
        ):
            raise ValueError("canonical practical decision differs")
        if decision["eligible"]:
            stages.append(stage)
    templates = {stage: read(root / "launches" / f"{stage}.json") for stage in stages}
    attempts, missing, seen = [], [], set()
    for report_path in sorted(run_root.glob("*.run.log.report.json")):
        report = read(report_path)
        if report.get("schema") != "simplex_t_phase_attempt_v1":
            continue
        launch_path = Path(report["launch"]).resolve(strict=True)
        if (
            not launch_path.is_relative_to(work / "artifacts")
            or sha256(launch_path) != report["launch_sha256"]
        ):
            raise ValueError("attempt launch changed or outside campaign artifacts")
        launch = read(launch_path)
        stage = launch["stage"]
        if stage not in templates or launch["freeze_sha256"] != base["freeze_sha256"]:
            raise ValueError("observed attempt belongs to another scientific campaign")
        if {k: v for k, v in launch.items() if k != "resource_receipt"} != {
            k: v for k, v in templates[stage].items() if k != "resource_receipt"
        }:
            raise ValueError("attempt changed scientific phase template")
        receipt = Path(launch["resource_receipt"]).resolve()
        if not receipt.is_relative_to(work / "artifacts"):
            raise ValueError("resource receipt outside companion")
        if not receipt.is_file():
            missing.append(
                {
                    "report": str(report_path),
                    "sha256": sha256(report_path),
                    "stage": stage,
                    "reason": "NO_TERMINAL_RESOURCE_RECEIPT_NOT_ZERO_WORK",
                }
            )
            continue
        if launch_path in seen:
            raise ValueError("duplicate phase attempt launch")
        seen.add(launch_path)
        attempts.append(
            dict(
                launch=str(launch_path),
                launch_sha256=report["launch_sha256"],
                receipt=str(receipt),
                receipt_sha256=sha256(receipt),
            )
        )
    ledger = work / "artifacts/simplex_t/TECHNICAL_BUDGET.json"
    journal = Path(base["execution"]) / "PHYSICAL_WORK.json"
    accounting = dict(
        journal=str(journal),
        journal_sha256=sha256(journal),
        reconciliation=str(args.reconciliation.resolve()),
        reconciliation_sha256=args.reconciliation_sha256,
        ledger=str(ledger),
        ledger_sha256=sha256(ledger),
    )
    value = delivery_launch(
        base=base,
        stage_templates=templates,
        accounting=accounting,
        resource_attempts=attempts,
        analysis_commit=args.analysis_commit,
        attempt_root=root / "T6",
    )
    discovery = dict(
        schema="simplex_t_resource_attempt_discovery_v1",
        run_root=str(run_root),
        observed_terminal_attempts=len(attempts),
        missing_terminal_receipts=missing,
        limitation=(
            "Only reports in this orchestration run root are enumerated; "
            "missing receipts are not evidence of zero updates."
        ),
    )
    for path, payload in (
        (root / "launches/T6.json", value),
        (root / "launches/T6_resource_discovery.json", discovery),
    ):
        if path.exists():
            if read(path) != payload:
                raise ValueError("existing T6 materialization differs; no overwrite")
        elif args.verify_only:
            return 10
        else:
            write_new_json(path, payload)
    print(
        json.dumps(
            {
                "status": "T6_LAUNCH_PREPARED_NOT_ANALYSIS_OR_DELIVERY",
                "path": str(root / "launches/T6.json"),
                "sha256": sha256(root / "launches/T6.json"),
                "missing_terminal_receipts": len(missing),
                "optimizer_updates": 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
