"""Execute one frozen CPU phase through real QA, practical gates and OLD publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.configured_execution import execute_configured_phase
from e_jepa_ttc.simplex_t.coordination import shared_write_admission
from e_jepa_ttc.simplex_t.lifecycle import admitted
from e_jepa_ttc.simplex_t.stage_gate import CanonicalPublication


def main() -> int:
    """Require explicit pinned launch metadata and absolute output reservations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--launch-sha256", required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--own-reserved-bytes", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.other_reserved_bytes < 0 or args.own_reserved_bytes < 1:
        parser.error("nonnegative other reservations and positive own output reservation required")
    if args.launch.stat().st_size > 1_048_576 or sha256(args.launch) != args.launch_sha256:
        raise ValueError("launch configuration bytes changed")
    config = json.loads(args.launch.read_text(encoding="utf-8"))
    required = {
        "schema",
        "local_paths",
        "source_configuration",
        "source_configuration_sha256",
        "evidence_profile",
        "evidence_profile_sha256",
        "freeze",
        "freeze_sha256",
        "roots",
        "execution",
        "publication",
        "stage",
        "availability",
        "publications",
    }
    if set(config) != required or config["schema"] != "simplex_t_frozen_phase_launch_v1":
        raise ValueError("unrecognized launch schema")
    roots = {key: Path(value).resolve(strict=True) for key, value in config["roots"].items()}
    publications = {
        key: CanonicalPublication(
            publication=Path(value["publication"]),
            publication_sha256=value["publication_sha256"],
            endpoints=Path(value["endpoints"]),
            endpoints_sha256=value["endpoints_sha256"],
            checkpoint_root=Path(value["checkpoint_root"]),
            availability=value["availability"],
        )
        for key, value in config["publications"].items()
    }
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

    def resource_ok() -> bool:
        if sha256(args.launch) != args.launch_sha256:
            raise ValueError("launch metadata changed during execution")
        snapshot = admitted([roots["work"]])
        return snapshot["has_headroom"] and shared_write_admission(
            snapshot["written_volume_free_bytes"][0],
            args.other_reserved_bytes + args.own_reserved_bytes,
        )

    result = execute_configured_phase(
        local_paths=Path(config["local_paths"]),
        source_configuration=Path(config["source_configuration"]),
        source_configuration_sha256=config["source_configuration_sha256"],
        evidence_profile=Path(config["evidence_profile"]),
        evidence_profile_sha256=config["evidence_profile_sha256"],
        freeze=Path(config["freeze"]),
        freeze_sha256=config["freeze_sha256"],
        roots=roots,
        execution=Path(config["execution"]),
        publication=Path(config["publication"]),
        stage=config["stage"],
        availability=config["availability"],
        publications=publications,
        resource_ok=resource_ok,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2))
    return 3 if result["status"] == "PAUSED_RESOURCE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
