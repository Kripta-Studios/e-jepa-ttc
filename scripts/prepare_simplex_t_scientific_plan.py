"""Append frozen scientific phases and actual T6 delivery to a pinned prerequisite plan."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json


def build_plan(
    *,
    work: Path,
    prefix: dict,
    freeze_launch: Path,
    freeze_hash: str,
    campaign_root: Path,
    run_root: Path,
    reconciliation: Path,
    reconciliation_hash: str,
    analysis_commit: str,
    other_reserved: int,
    own_reserved: int,
) -> dict:
    """Bind the finite graph; gates are resolved by existing scientific workers."""
    if prefix.get("schema") != "simplex_t_orchestration_plan_v1" or not prefix.get("steps"):
        raise ValueError("nonempty pinned prerequisite plan required")
    if any(
        step["id"] in {"scientific_freeze", "T2", "T3", "T4", "T5", "T6"}
        for step in prefix["steps"]
    ):
        raise ValueError("prerequisite plan already contains scientific stage IDs")
    if other_reserved < 0 or own_reserved < 8_388_608:
        raise ValueError("nonnegative other and at least 8388608 own reservation required")
    for root in (campaign_root, run_root):
        if root == work / "artifacts" or not root.is_relative_to(work / "artifacts"):
            raise ValueError("campaign and logs must remain inside companion artifacts")
    if len(analysis_commit) != 40 or set(analysis_commit) - set("0123456789abcdef"):
        raise ValueError("full analysis commit required")

    def pin(name: str) -> str:
        return sha256(work / "scripts" / name)

    def command(name: str, arguments: list[str], report: bool = False) -> dict:
        value: dict = dict(script="scripts/" + name, sha256=pin(name), arguments=arguments)
        if report:
            value["attempt_report"] = True
        return value

    result = copy.deepcopy(prefix)
    result["scope"] = (
        "Prerequisite plan followed by scientific freeze, T2, independent technical T4, "
        "canonical practical T3/T5 and T6 analysis/archive verification. "
        "Plan construction is not prerequisite validation or scientific completion."
    )
    result["required_run_root"] = str(run_root)
    result["scientific_tail"] = ["scientific_freeze", "T2", "T4", "T3", "T5", "T6"]
    reserve = [
        "--other-reserved-bytes",
        str(other_reserved),
        "--own-reserved-bytes",
        str(own_reserved),
    ]
    freeze_args = ["--launch", str(freeze_launch), "--launch-sha256", freeze_hash, *reserve]
    freeze_command = command("freeze_simplex_t_campaign.py", freeze_args)
    result["steps"].append(
        dict(
            id="scientific_freeze",
            run=freeze_command,
            resume=copy.deepcopy(freeze_command),
            verify=command("freeze_simplex_t_campaign.py", [*freeze_args, "--verify-only"]),
        )
    )
    phase_args = [
        "--freeze-launch",
        str(freeze_launch),
        "--freeze-launch-sha256",
        freeze_hash,
        "--campaign-root",
        str(campaign_root),
        "--materializer-sha256",
        pin("materialize_simplex_t_technical_phases.py"),
        "--attempt-sha256",
        pin("execute_simplex_t_phase_attempt.py"),
        "--worker-sha256",
        pin("execute_simplex_t_frozen_phase.py"),
        *reserve,
    ]
    for stage in ("T2", "T4", "T3", "T5"):
        args = [*phase_args, "--stage", stage]
        result["steps"].append(
            dict(
                id=stage,
                run=command("run_simplex_t_materialized_stage.py", args, True),
                resume=command("run_simplex_t_materialized_stage.py", [*args, "--resume"], True),
                verify=command("run_simplex_t_materialized_stage.py", [*args, "--verify-only"]),
            )
        )
    delivery_args = [
        "--campaign-root",
        str(campaign_root),
        "--run-root",
        str(run_root),
        "--reconciliation",
        str(reconciliation),
        "--reconciliation-sha256",
        reconciliation_hash,
        "--analysis-commit",
        analysis_commit,
        "--materializer-sha256",
        pin("materialize_simplex_t_delivery.py"),
        "--attempt-sha256",
        pin("execute_simplex_t_postprocessing_attempt.py"),
        "--worker-sha256",
        pin("postprocess_simplex_t_campaign.py"),
        *reserve,
    ]
    result["steps"].append(
        dict(
            id="T6",
            run=command("run_simplex_t_materialized_delivery.py", delivery_args),
            resume=command("run_simplex_t_materialized_delivery.py", [*delivery_args, "--resume"]),
            verify=command(
                "run_simplex_t_materialized_delivery.py", [*delivery_args, "--verify-only"]
            ),
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prerequisites", "freeze-launch", "reconciliation"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    for name in ("campaign-root", "run-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--own-reserved-bytes", type=int, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    work = Path(__file__).resolve().parents[1]
    for name in ("prerequisites", "freeze_launch", "reconciliation"):
        path = getattr(args, name)
        if path.stat().st_size > 1_048_576 or sha256(path) != getattr(args, name + "_sha256"):
            raise ValueError("scientific plan input pin changed")
    if not args.output.resolve().is_relative_to(work / "artifacts"):
        raise ValueError("plan output must remain inside companion artifacts")
    launch = json.loads(args.freeze_launch.read_text("utf-8"))
    if (
        launch.get("schema") != "simplex_t_freeze_launch_v1"
        or launch["code_commit"] != args.analysis_commit
    ):
        raise ValueError("actual freeze launch with matching analysis commit required")
    prefix = json.loads(args.prerequisites.read_text("utf-8"))
    # Verify every inherited executable; do not silently refresh a stale plan.
    for step in prefix["steps"]:
        for mode in ("run", "resume", "verify"):
            command = step[mode]
            path = (work / command["script"]).resolve(strict=True)
            if not path.is_relative_to(work / "scripts") or sha256(path) != command["sha256"]:
                raise ValueError("prerequisite worker changed")
    value = build_plan(
        work=work,
        prefix=prefix,
        freeze_launch=args.freeze_launch.resolve(),
        freeze_hash=args.freeze_launch_sha256,
        campaign_root=args.campaign_root.resolve(),
        run_root=args.run_root.resolve(),
        reconciliation=args.reconciliation.resolve(),
        reconciliation_hash=args.reconciliation_sha256,
        analysis_commit=args.analysis_commit,
        other_reserved=args.other_reserved_bytes,
        own_reserved=args.own_reserved_bytes,
    )
    value["prerequisites"] = dict(
        path=str(args.prerequisites.resolve()), sha256=args.prerequisites_sha256
    )
    if args.output.exists():
        if json.loads(args.output.read_text("utf-8")) != value:
            raise ValueError("existing campaign plan differs; no overwrite")
    elif args.verify_only:
        return 10
    else:
        write_new_json(args.output, value)
    print(
        json.dumps(
            dict(
                status="SCIENTIFIC_PLAN_BOUND_NOT_EXECUTED",
                path=str(args.output),
                sha256=sha256(args.output),
                steps=len(value["steps"]),
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
