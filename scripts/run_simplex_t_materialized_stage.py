"""Bridge synchronous orchestration to verified, late-materialized phase launches."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256


def main() -> int:
    """Resolve phase metadata, then delegate exact resume; never implement fitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-launch", type=Path, required=True)
    parser.add_argument("--freeze-launch-sha256", required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("T2", "T3", "T4", "T5"), required=True)
    parser.add_argument("--materializer-sha256", required=True)
    parser.add_argument("--attempt-sha256", required=True)
    parser.add_argument("--worker-sha256", required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--own-reserved-bytes", type=int, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    work = Path(__file__).resolve().parents[1]
    root = args.campaign_root.resolve()
    if not root.is_relative_to(work / "artifacts") or root == work / "artifacts":
        raise ValueError("campaign must remain inside companion artifacts")
    if args.other_reserved_bytes < 0 or args.own_reserved_bytes < 8_388_608:
        parser.error("nonnegative other and at least 8388608 own reservation required")
    if args.verify_only and (args.resume or args.report is not None):
        parser.error("verification must not request execution or a report")
    if not args.verify_only and args.report is None:
        parser.error("execution requires a fresh attempt report")
    scripts = work / "scripts"
    pins = {
        scripts / "materialize_simplex_t_technical_phases.py": args.materializer_sha256,
        scripts / "execute_simplex_t_phase_attempt.py": args.attempt_sha256,
        scripts / "execute_simplex_t_frozen_phase.py": args.worker_sha256,
        args.freeze_launch: args.freeze_launch_sha256,
    }

    def check_pins() -> None:
        for path, expected in pins.items():
            if path.stat().st_size > 1_048_576 or sha256(path) != expected:
                raise ValueError("stage orchestration input changed")

    check_pins()
    metadata = [
        sys.executable,
        "-B",
        str(scripts / "materialize_simplex_t_technical_phases.py"),
        "--freeze-launch",
        str(args.freeze_launch),
        "--freeze-launch-sha256",
        args.freeze_launch_sha256,
        "--campaign-root",
        str(root),
        "--stage",
        args.stage,
        "--other-reserved-bytes",
        str(args.other_reserved_bytes),
    ]
    if args.verify_only:
        metadata.append("--verify-only")
    code = subprocess.run(metadata, check=False, cwd=work).returncode
    if code:
        return code
    check_pins()
    if args.stage in {"T3", "T5"} or (
        args.stage == "T4" and not (root / "launches/T4.json").exists()
    ):
        decision = root / "launches" / f"{args.stage}.decision.json"
        state = json.loads(decision.read_text("utf-8"))
        if (
            state.get("schema")
            != (
                "simplex_t_technical_launch_decision_v1"
                if args.stage == "T4"
                else "simplex_t_practical_launch_decision_v1"
            )
            or state.get("stage") != args.stage
            or state.get("optimizer_updates") != 0
            or state.get("scientific_completion") is not False
        ):
            raise ValueError("canonical decision schema or stage differs")
        if state["eligible"] is False:
            print(
                json.dumps(
                    {
                        "status": "CANONICAL_STAGE_NOT_ENABLED",
                        "stage": args.stage,
                        "decision_sha256": sha256(decision),
                        "optimizer_updates": 0,
                    }
                )
            )
            return 0
        if state["eligible"] is not True:
            raise ValueError("explicit canonical boolean decision required")
    template = root / "launches" / f"{args.stage}.json"
    command = [
        sys.executable,
        "-B",
        str(scripts / "execute_simplex_t_phase_attempt.py"),
        "--template",
        str(template),
        "--template-sha256",
        sha256(template),
        "--worker-sha256",
        args.worker_sha256,
        "--other-reserved-bytes",
        str(args.other_reserved_bytes),
        "--own-reserved-bytes",
        str(args.own_reserved_bytes),
    ]
    if args.verify_only:
        command.append("--verify-only")
    else:
        command.extend(["--report", str(args.report)])
        if args.resume:
            command.append("--resume")
    check_pins()
    return subprocess.run(command, check=False, cwd=work).returncode


if __name__ == "__main__":
    raise SystemExit(main())
