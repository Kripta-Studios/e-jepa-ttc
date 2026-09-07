"""Reusable real evidence verification for scientific prerequisite callbacks."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .ancestry_evidence import verify_acknowledged_producers
from .h16_qa_evidence import verify_h16_execution_identity, verify_h16_replay
from .replay_evidence import verify_coherent_replay
from .resume_evidence import verify_real_cpu_resume
from .unit_qa_evidence import verify_companion_unit_qa


def verify_component_profile(
    local_paths: Path,
    profile_path: Path,
    profile_sha256: str,
    *,
    resource_ok: Callable[[], bool],
    require_h16: bool,
    require_unit_qa: bool = False,
) -> dict:
    """Reopen actual ancestry, replay arrays and resume states, without fitting.

    Scientific callers must require H16. The diagnostic CLI may inspect partial
    evidence, but cannot promote its report to scientific admission. This is
    only one part of admission: complete source preparation, availability,
    repository QA, code freeze and practical stage gates remain separate.
    """
    if type(require_h16) is not bool or type(require_unit_qa) is not bool:
        raise ValueError("explicit evidence verification policies required")
    if profile_path.stat().st_size > 1_048_576 or sha256(profile_path) != profile_sha256:
        raise ValueError("component evidence profile changed")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("schema") != "simplex_t_component_evidence_profile_v1":
        raise ValueError("unrecognized evidence profile")
    if require_h16 and "h16_replay" not in profile:
        raise ValueError(
            "WAITING_H16_REPLAY_EVIDENCE: scientific prerequisites require real H16 QA"
        )
    if require_unit_qa and "unit_qa" not in profile:
        raise ValueError("WAITING_CURRENT_UNIT_QA: complete pinned unit evidence required")
    paths_hash = sha256(local_paths)
    paths = json.loads(local_paths.read_text(encoding="utf-8"))
    work = Path(paths["worktree"]).resolve(strict=True)

    def resolve(relative: str) -> Path:
        value = Path(relative)
        if value.is_absolute() or value.drive or ".." in value.parts:
            raise ValueError("evidence path must be relative to companion")
        result = (work / value).resolve(strict=True)
        if not result.is_relative_to(work):
            raise ValueError("evidence path escapes companion")
        return result

    def boundary() -> None:
        if sha256(local_paths) != paths_hash or sha256(profile_path) != profile_sha256:
            raise ValueError("evidence configuration changed during verification")
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: component evidence verification")

    boundary()
    ancestry = verify_acknowledged_producers(
        Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json",
        profile["ack_sha256"],
        resource_ok=resource_ok,
    )
    boundary()
    historical, coherent = profile["historical_replay"], profile["coherent_replay"]
    replay = verify_coherent_replay(
        work,
        resolve(historical["path"]),
        historical["sha256"],
        resolve(coherent["path"]),
        coherent["sha256"],
    )
    boundary()
    resume = profile["resume"]
    proof = verify_real_cpu_resume(
        resolve(resume["root"]),
        pins=resume["pins"],
        source_sha256=resume["source_sha256"],
        compiled_sha256=resume["compiled_sha256"],
        engine=resolve("src/e_jepa_ttc/simplex_t/training.py"),
        probe_script=resolve("scripts/probe_simplex_t_context_resume.py"),
    )
    boundary()
    h16 = None
    if "h16_replay" in profile:
        entry = profile["h16_replay"]
        h16 = verify_h16_replay(
            work,
            resolve(entry["root"]),
            entry["report_sha256"],
            validate_execution_identity=lambda record: verify_h16_execution_identity(
                work, local_paths, record
            ),
            resource_ok=resource_ok,
        )
        boundary()
    unit_qa = None
    if "unit_qa" in profile:
        entry = profile["unit_qa"]
        unit_qa = verify_companion_unit_qa(
            work,
            resolve(entry["root"]),
            pins=entry["pins"],
            technical_ledger_sha256=entry["technical_ledger_sha256"],
            resource_ok=resource_ok,
        )
        boundary()
    return {
        "status": "REAL_COMPONENT_EVIDENCE_VERIFIED_NOT_SCIENTIFIC_ADMISSION",
        "profile_sha256": profile_sha256,
        "local_paths_sha256": paths_hash,
        "ancestry": ancestry,
        "replay": replay,
        "resume": proof,
        "h16_replay": h16,
        "unit_qa": unit_qa,
        "optimizer_updates_executed": 0,
        "scientific_admission": False,
        "not_covered": [
            *([] if h16 is not None else ["Production H16 extraction numerical parity"]),
            "All-fold and expanded-pool source integration and normalization",
            "Supplementary expanded time authority",
            "Complete baseline-versus-new repository QA and frozen code inventory",
            "Scientific freeze, practical gates, fits or final analyses",
        ],
    }
