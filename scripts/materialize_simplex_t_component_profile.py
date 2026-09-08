"""Bind finished real QA outputs into the mandatory pre-freeze evidence profile."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.component_verification import verify_component_profile
from e_jepa_ttc.simplex_t.coordination import shared_write_admission
from e_jepa_ttc.simplex_t.lifecycle import admitted
from e_jepa_ttc.simplex_t.unit_qa_evidence import UNIT_QA_FILES


def main() -> int:
    """Only publish after all semantic verifiers pass; preserve failed candidates."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "local-paths",
        "base-profile",
        "unit-root",
        "static-root",
        "types-report",
        "powershell-report",
        "h16-root",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--base-profile-sha256", required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    work = Path(__file__).resolve().parents[1]
    if args.other_reserved_bytes < 0:
        parser.error("nonnegative outstanding reservation required")
    for path in (
        args.unit_root,
        args.static_root,
        args.types_report,
        args.powershell_report,
        args.h16_root,
        args.output,
    ):
        if not path.resolve().is_relative_to(work / "artifacts/simplex_t/T0"):
            raise ValueError("QA outputs must remain inside companion T0")
    if (
        args.base_profile.stat().st_size > 1_048_576
        or sha256(args.base_profile) != args.base_profile_sha256
    ):
        raise ValueError("base historical replay/resume profile changed")
    old = json.loads(args.base_profile.read_text("utf-8"))
    if old.get("schema") != "simplex_t_component_evidence_profile_v1":
        raise ValueError("recognized component profile required")
    # Historical replay and exact resume retain their original pins. No stale
    # static/unit evidence is inherited or relabelled as current evidence.
    profile = {
        key: old[key]
        for key in ("schema", "ack_sha256", "historical_replay", "coherent_replay", "resume")
    }
    unit_files = sorted(UNIT_QA_FILES)
    static_files = ("COMPARISON.json", "BASELINE_RUFF.json", "CURRENT_RUFF.json")
    required = [
        *(args.unit_root / name for name in unit_files),
        *(args.static_root / name for name in static_files),
        args.types_report,
        args.powershell_report,
        args.h16_root / "QA.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print(
            json.dumps(
                dict(
                    status="WAITING_COMPLETE_REAL_QA", missing=missing, scientific_completion=False
                )
            )
        )
        return 10 if args.verify_only else 2

    def relative(path: Path) -> str:
        return path.resolve(strict=True).relative_to(work).as_posix()

    def resource_ok() -> bool:
        state = admitted([work])
        return state["has_headroom"] and shared_write_admission(
            state["written_volume_free_bytes"][0], args.other_reserved_bytes + 8_388_608
        )

    if not resource_ok():
        return 3
    profile.update(
        unit_qa=dict(
            root=relative(args.unit_root),
            pins={name: sha256(args.unit_root / name) for name in unit_files},
            technical_ledger_sha256=sha256(work / "artifacts/simplex_t/TECHNICAL_BUDGET.json"),
        ),
        static_qa=dict(
            root=relative(args.static_root),
            pins={name: sha256(args.static_root / name) for name in static_files},
        ),
        types_qa=dict(path=relative(args.types_report), sha256=sha256(args.types_report)),
        powershell_qa=dict(
            path=relative(args.powershell_report), sha256=sha256(args.powershell_report)
        ),
        h16_replay=dict(
            root=relative(args.h16_root), report_sha256=sha256(args.h16_root / "QA.json")
        ),
    )
    if args.output.exists():
        if (
            args.output.stat().st_size > 1_048_576
            or json.loads(args.output.read_text("utf-8")) != profile
        ):
            raise ValueError("existing QA profile differs; no overwrite")
        candidate = args.output
    elif args.verify_only:
        return 10
    else:
        candidate = args.output.with_name(
            args.output.stem + ".candidate_" + uuid.uuid4().hex + ".json"
        )
        write_new_json(candidate, profile)
    try:
        verify_component_profile(
            args.local_paths,
            candidate,
            sha256(candidate),
            resource_ok=resource_ok,
            require_h16=True,
            require_unit_qa=True,
            require_static_qa=True,
            require_types=True,
            require_powershell=True,
        )
    except InterruptedError:
        return 3
    if sha256(args.base_profile) != args.base_profile_sha256:
        raise ValueError("historical profile changed during verification")
    if not resource_ok():
        return 3
    if not args.output.exists():
        write_new_json(args.output, profile)
    print(
        json.dumps(
            dict(
                status="ALL_COMPONENTS_VERIFIED_NOT_SCIENTIFIC_FREEZE",
                path=str(args.output),
                sha256=sha256(args.output),
                optimizer_updates=0,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
