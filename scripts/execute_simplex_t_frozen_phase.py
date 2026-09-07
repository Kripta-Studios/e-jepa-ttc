"""Execute one frozen CPU phase through real QA, practical gates and OLD publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.configured_execution import execute_configured_phase
from e_jepa_ttc.simplex_t.coordination import shared_write_admission
from e_jepa_ttc.simplex_t.lifecycle import admitted
from e_jepa_ttc.simplex_t.resource_observations import ResourceObservations
from e_jepa_ttc.simplex_t.stage_gate import CanonicalPublication


def main() -> int:
    """Require explicit pinned launch metadata and absolute output reservations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--launch-sha256", required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--own-reserved-bytes", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.other_reserved_bytes < 0 or args.own_reserved_bytes < 65536:
        parser.error(
            "nonnegative other reservations and at least 65536 own reserved bytes required"
        )
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
        "resource_receipt",
    }
    if set(config) != required or config["schema"] != "simplex_t_frozen_phase_launch_v2":
        raise ValueError("unrecognized launch schema")
    roots = {key: Path(value).resolve(strict=True) for key, value in config["roots"].items()}
    receipt = Path(config["resource_receipt"]).resolve()
    receipt_root = roots["work"] / "artifacts/simplex_t/resource_observations"
    if (
        not receipt.is_relative_to(receipt_root)
        or receipt.suffix != ".json"
        or (receipt.exists() and not args.verify_only)
    ):
        raise ValueError("new resource receipt under companion resource_observations required")
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
    observations = ResourceObservations()

    def resource_ok() -> bool:
        if sha256(args.launch) != args.launch_sha256:
            raise ValueError("launch metadata changed during execution")
        snapshot = admitted([roots["work"]])
        allowed = snapshot["has_headroom"] and shared_write_admission(
            snapshot["written_volume_free_bytes"][0],
            args.other_reserved_bytes + args.own_reserved_bytes,
        )
        observations.observe(snapshot, allowed=allowed)
        return allowed

    outcome, failure = "ERROR_BEFORE_RETURN", None
    try:
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
            verify_only=args.verify_only,
        )
        outcome = result["status"]
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)[:4096]}
        if (
            args.verify_only
            and isinstance(error, (InterruptedError, RuntimeError))
            and (
                str(error).startswith(("PAUSED_RESOURCE:", "RESOURCE_PAUSE:"))
                or str(error) in {
                    "prediction integrity resource pause",
                    "prediction table resource pause",
                    "endpoint validation resource pause",
                }
            )
        ):
            result = {"status": "PAUSED_RESOURCE", "reason": str(error), "optimizer_updates": 0}
        else:
            raise
    finally:
        # A small reserved receipt is also needed on safe resource pauses. It
        # neither starts new work nor claims a checkpoint exists after an error.
        if not args.verify_only:
            receipt.parent.mkdir(parents=True, exist_ok=True)
            write_new_json(
                receipt,
                dict(
                    **observations.summary(),
                    launch_sha256=args.launch_sha256,
                    freeze_sha256=config["freeze_sha256"],
                    stage=config["stage"],
                    resume_requested=args.resume,
                    execution_result_status=outcome,
                    failure=failure,
                    other_reserved_bytes=args.other_reserved_bytes,
                    own_reserved_bytes=args.own_reserved_bytes,
                    disk_floor_after_reservations_bytes=40000000000,
                    campaign_complete=False,
                ),
            )
    print(json.dumps(result, indent=2))
    if result["status"] == "PHASE_INCOMPLETE":
        return 10
    return 3 if result["status"] == "PAUSED_RESOURCE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
