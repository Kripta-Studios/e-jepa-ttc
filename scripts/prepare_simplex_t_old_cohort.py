"""Publish acknowledged OLD evaluation identities, without scores or training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.coordination import shared_write_admission, verified_ack
from e_jepa_ttc.simplex_t.lifecycle import admitted
from e_jepa_ttc.simplex_t.old_cohort import load_old_evaluation_cohort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    args = parser.parse_args()
    local_hash = sha256(args.local_paths)
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    work = Path(paths["worktree"]).resolve(strict=True)
    if args.output.exists() or not args.output.resolve().is_relative_to(
        work / "artifacts/simplex_t/T0"
    ):
        raise ValueError("new companion T0 cohort output directory required")
    if args.other_reserved_bytes < 0:
        raise ValueError("explicit nonnegative outstanding output reservation required")
    ack_path = Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json"
    ack_hash = "3e55ab3c6e9a57eecd862ad05e999627ea90957e58e329b2eb3652120e953318"
    ack = verified_ack(ack_path, ack_hash)
    ancestry = ack["producers"]["authoritative_historical_manifest"]

    def resources() -> bool:
        snapshot = admitted([work])
        return snapshot["has_headroom"] and shared_write_admission(
            snapshot["written_volume_free_bytes"][0], args.other_reserved_bytes + 8_388_608
        )

    frame, report = load_old_evaluation_cohort(
        Path(ancestry["path"]).parent,
        ancestry_sha256=ancestry["sha256"],
        table_index_sha256="c4969b5354ceb4c87c483d94f7b9de1b9983b0c5e8af81a952332c846c4b9c4f",
        allowed_sequences=set(ack["interfaces"]["role_manifest"]["roles"]["original"]),
        resource_ok=resources,
    )
    if sha256(args.local_paths) != local_hash or verified_ack(ack_path, ack_hash) != ack:
        raise ValueError("cohort authority changed during preparation")
    if not resources():
        raise InterruptedError("PAUSED_RESOURCE before cohort publication")
    args.output.mkdir(parents=True)
    cohort = args.output / "OLD8192_COHORT.parquet"
    frame.to_parquet(cohort, index=False)
    report.update(
        cohort_sha256=sha256(cohort),
        ack_sha256=ack_hash,
        local_paths_sha256=local_hash,
        loader_sha256=sha256(work / "src/e_jepa_ttc/simplex_t/old_cohort.py"),
        script_sha256=sha256(Path(__file__)),
        scientific_admission=False,
    )
    write_new_json(args.output / "COHORT.json", report)
    print(json.dumps({"queries": len(frame), "source_role": "outer_dev", "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
