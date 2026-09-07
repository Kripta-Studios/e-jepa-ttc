"""Verify pinned real lineage/replay/resume components; never authorize fits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.ancestry_evidence import verify_acknowledged_producers
from e_jepa_ttc.simplex_t.coordination import shared_write_admission
from e_jepa_ttc.simplex_t.h16_qa_evidence import verify_h16_execution_identity, verify_h16_replay
from e_jepa_ttc.simplex_t.lifecycle import admitted
from e_jepa_ttc.simplex_t.replay_evidence import verify_coherent_replay
from e_jepa_ttc.simplex_t.resume_evidence import verify_real_cpu_resume


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("component evidence report already exists")
    if args.other_reserved_bytes < 0:
        raise ValueError("nonnegative pending output reservation required")
    if args.profile.stat().st_size > 1_048_576 or sha256(args.profile) != args.profile_sha256:
        raise ValueError("component evidence profile changed")
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if profile["schema"] != "simplex_t_component_evidence_profile_v1":
        raise ValueError("unrecognized evidence profile")
    paths_hash = sha256(args.local_paths)
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    work = Path(paths["worktree"]).resolve(strict=True)
    if not args.output.resolve().is_relative_to(work):
        raise ValueError("report must remain in the companion worktree")

    def resolve(relative: str) -> Path:
        value = Path(relative)
        if value.is_absolute() or value.drive or ".." in value.parts:
            raise ValueError("evidence path must be relative to companion")
        result = (work / value).resolve(strict=True)
        if not result.is_relative_to(work):
            raise ValueError("evidence path escapes companion")
        return result

    def resources() -> bool:
        snapshot = admitted([work])
        return snapshot["has_headroom"] and shared_write_admission(
            snapshot["written_volume_free_bytes"][0], args.other_reserved_bytes + 1_048_576
        )

    def boundary() -> None:
        if sha256(args.local_paths) != paths_hash or sha256(args.profile) != args.profile_sha256:
            raise ValueError("evidence configuration changed during verification")
        if not resources():
            raise InterruptedError("PAUSED_RESOURCE: component evidence verification")

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    started = time.monotonic()
    boundary()
    ancestry = verify_acknowledged_producers(
        Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json",
        profile["ack_sha256"],
        resource_ok=resources,
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
                work, args.local_paths, record
            ),
            resource_ok=resources,
        )
        boundary()
    result = {
        "status": "REAL_COMPONENT_EVIDENCE_VERIFIED_NOT_SCIENTIFIC_ADMISSION",
        "profile_sha256": args.profile_sha256,
        "local_paths_sha256": paths_hash,
        "ancestry": ancestry,
        "replay": replay,
        "resume": proof,
        "h16_replay": h16,
        "optimizer_updates_executed": 0,
        "scientific_admission": False,
        "not_covered": [
            *([] if h16 is not None else ["Production H16 extraction numerical parity"]),
            "All-fold and expanded-pool source integration and normalization",
            "Supplementary expanded time authority",
            "Complete baseline-versus-new repository QA and frozen code inventory",
            "Scientific freeze, practical gates, fits or final analyses",
        ],
        "observed_seconds_excluding_imports": time.monotonic() - started,
    }
    write_new_json(args.output, result)
    print(json.dumps({"status": result["status"], "optimizer_updates_executed": 0}))


if __name__ == "__main__":
    main()
