"""Publish the scientific freeze only through real source authority and complete QA."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.coordination import shared_write_admission
from e_jepa_ttc.simplex_t.freeze_integrity import FrozenFile
from e_jepa_ttc.simplex_t.lifecycle import admitted
from e_jepa_ttc.simplex_t.scientific_admission import validate_scientific_admission
from e_jepa_ttc.simplex_t.scientific_freeze import publish_scientific_freeze


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--launch-sha256", required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--own-reserved-bytes", type=int, required=True)
    args = parser.parse_args()
    if args.other_reserved_bytes < 0 or args.own_reserved_bytes < 8_388_608:
        parser.error("nonnegative other and at least 8388608 own reserved bytes required")
    if args.launch.stat().st_size > 1_048_576 or sha256(args.launch) != args.launch_sha256:
        raise ValueError("freeze launch bytes changed or exceed bound")
    config = json.loads(args.launch.read_text(encoding="utf-8"))
    required = {
        "schema",
        "local_paths",
        "source_configuration",
        "source_configuration_sha256",
        "evidence_profile",
        "evidence_profile_sha256",
        "roots",
        "files",
        "preparation",
        "technical_ledger",
        "code_commit",
        "output",
    }
    if set(config) != required or config["schema"] != "simplex_t_freeze_launch_v1":
        raise ValueError("exact freeze launch schema required")
    roots = {key: Path(value).resolve(strict=True) for key, value in config["roots"].items()}
    output = Path(config["output"]).resolve()
    if not output.is_relative_to(roots["work"] / "artifacts") or output.exists():
        raise ValueError("new companion artifact freeze path required")
    files = [FrozenFile(**pin) for pin in config["files"]]
    preparation = FrozenFile(**config["preparation"])
    ledger = FrozenFile(**config["technical_ledger"])
    prepared_path = (roots[preparation.root] / preparation.relative_path).resolve(strict=True)
    if not prepared_path.is_relative_to(roots[preparation.root]):
        raise ValueError("preparation escapes its declared root")

    def resource_ok() -> bool:
        if sha256(args.launch) != args.launch_sha256:
            raise ValueError("freeze launch changed during admission")
        snapshot = admitted([roots["work"]])
        return snapshot["has_headroom"] and shared_write_admission(
            snapshot["written_volume_free_bytes"][0],
            args.other_reserved_bytes + args.own_reserved_bytes,
        )

    def validate() -> None:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: scientific freeze publication")
        if prepared_path.stat().st_size > 8_388_608 or sha256(prepared_path) != preparation.sha256:
            raise ValueError("prepared source identities changed")
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        # Admission needs these two prospective fields. The publisher separately
        # reconstructs and validates the entire canonical freeze and code commit.
        validate_scientific_admission(
            {"files": [asdict(pin) for pin in files], "source_contract": prepared["contract"]},
            roots=roots,
            local_paths=Path(config["local_paths"]),
            source_configuration=Path(config["source_configuration"]),
            source_configuration_sha256=config["source_configuration_sha256"],
            evidence_profile=Path(config["evidence_profile"]),
            evidence_profile_sha256=config["evidence_profile_sha256"],
            resource_ok=resource_ok,
        )

    try:
        publish_scientific_freeze(
            output,
            files=files,
            roots=roots,
            preparation=preparation,
            technical_ledger=ledger,
            code_commit=config["code_commit"],
            validate_prerequisites=validate,
        )
    except InterruptedError as error:
        if not str(error).startswith("PAUSED_RESOURCE"):
            raise
        print(json.dumps({"status": "PAUSED_RESOURCE", "reason": str(error)}))
        return 3
    print(
        json.dumps(
            {
                "status": "FROZEN_INPUTS_STAGE_GATES_REQUIRED",
                "path": str(output),
                "sha256": sha256(output),
                "optimizer_updates_executed": 0,
                "campaign_complete": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
