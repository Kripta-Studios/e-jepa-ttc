"""Assemble frozen SIMPLEX-T analyses and compact weights; never train or open holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.campaign_accounting import AccountingPins
from e_jepa_ttc.simplex_t.configured_postprocessing import postprocess_configured_campaign
from e_jepa_ttc.simplex_t.coordination import shared_write_admission
from e_jepa_ttc.simplex_t.lifecycle import admitted
from e_jepa_ttc.simplex_t.stage_gate import CanonicalPublication


def main() -> int:
    """Require a pinned launch and a new output; retry analysis never refits heads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--launch-sha256", required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--own-reserved-bytes", type=int, required=True)
    args = parser.parse_args()
    if args.other_reserved_bytes < 0 or args.own_reserved_bytes < 1:
        parser.error("nonnegative other and positive own output reservations required")
    if args.launch.stat().st_size > 1_048_576 or sha256(args.launch) != args.launch_sha256:
        raise ValueError("postprocessing launch bytes changed")
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
        "output",
        "publications",
        "accounting",
    }
    if set(config) != required or config["schema"] != "simplex_t_postprocessing_launch_v2":
        raise ValueError("unrecognized postprocessing launch schema")
    evidence = config["accounting"]
    if set(evidence) != {
        "journal",
        "journal_sha256",
        "reconciliation",
        "reconciliation_sha256",
        "ledger",
        "ledger_sha256",
    }:
        raise ValueError("exact physical and technical accounting pins required")
    accounting = AccountingPins(
        journal=Path(evidence["journal"]),
        journal_sha256=evidence["journal_sha256"],
        reconciliation=Path(evidence["reconciliation"]),
        reconciliation_sha256=evidence["reconciliation_sha256"],
        ledger=Path(evidence["ledger"]),
        ledger_sha256=evidence["ledger_sha256"],
    )
    stages = set(config["publications"])
    if "T2" not in stages or not stages <= {"T2", "T3", "T4", "T5"}:
        raise ValueError("registered sealed publications including T2 required")
    roots = {key: Path(value).resolve(strict=True) for key, value in config["roots"].items()}
    phases = {
        stage: CanonicalPublication(
            publication=Path(value["publication"]),
            publication_sha256=value["publication_sha256"],
            endpoints=Path(value["endpoints"]),
            endpoints_sha256=value["endpoints_sha256"],
            checkpoint_root=Path(value["checkpoint_root"]),
            availability=value["availability"],
        )
        for stage, value in config["publications"].items()
    }
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

    def resource_ok() -> bool:
        if sha256(args.launch) != args.launch_sha256:
            raise ValueError("postprocessing launch changed during execution")
        snapshot = admitted([roots["work"]])
        return snapshot["has_headroom"] and shared_write_admission(
            snapshot["written_volume_free_bytes"][0],
            args.other_reserved_bytes + args.own_reserved_bytes,
        )

    try:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: postprocessing launch")
        result = postprocess_configured_campaign(
            Path(config["output"]),
            local_paths=Path(config["local_paths"]),
            source_configuration=Path(config["source_configuration"]),
            source_configuration_sha256=config["source_configuration_sha256"],
            evidence_profile=Path(config["evidence_profile"]),
            evidence_profile_sha256=config["evidence_profile_sha256"],
            freeze=Path(config["freeze"]),
            freeze_sha256=config["freeze_sha256"],
            roots=roots,
            phases=phases,
            accounting_pins=accounting,
            resource_ok=resource_ok,
        )
    except (InterruptedError, RuntimeError) as error:
        if not str(error).startswith(("PAUSED_RESOURCE", "RESOURCE_PAUSE")):
            raise
        print(
            json.dumps(
                {
                    "status": "PAUSED_RESOURCE",
                    "reason": str(error),
                    "launch_sha256": args.launch_sha256,
                    "output": config["output"],
                    "optimizer_updates_executed": 0,
                    "campaign_complete": False,
                    "retry": "Keep partial analysis; use a new pinned output path. No refitting.",
                }
            )
        )
        return 3
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
