"""Materialize immutable T2/T4 templates from a real freeze without model work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.coordination import shared_write_admission
from e_jepa_ttc.simplex_t.lifecycle import admitted
from e_jepa_ttc.simplex_t.phase_launch_materialization import technical_phase_launch
from e_jepa_ttc.simplex_t.scientific_admission import validate_scientific_admission
from e_jepa_ttc.simplex_t.scientific_freeze import read_scientific_freeze


def main() -> int:
    """Retain identical templates on resume; reject changed metadata, never refit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-launch", type=Path, required=True)
    parser.add_argument("--freeze-launch-sha256", required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.other_reserved_bytes < 0:
        parser.error("nonnegative reservation required")
    if (
        args.freeze_launch.stat().st_size > 1_048_576
        or sha256(args.freeze_launch) != args.freeze_launch_sha256
    ):
        raise ValueError("freeze launch changed")
    launch = json.loads(args.freeze_launch.read_text("utf-8"))
    freeze = Path(launch["output"])
    if not freeze.is_file():
        print(json.dumps({"status": "WAITING_SCIENTIFIC_FREEZE", "path": str(freeze)}))
        return 10 if args.verify_only else 2
    digest = sha256(freeze)
    roots = {key: Path(value).resolve(strict=True) for key, value in launch["roots"].items()}

    def resource_ok() -> bool:
        snapshot = admitted([roots["work"]])
        return snapshot["has_headroom"] and shared_write_admission(
            snapshot["written_volume_free_bytes"][0], args.other_reserved_bytes + 8_388_608
        )

    def validate() -> None:
        if sha256(args.freeze_launch) != args.freeze_launch_sha256 or sha256(freeze) != digest:
            raise ValueError("materialization inputs changed")
        validate_scientific_admission(
            json.loads(freeze.read_text("utf-8")),
            roots=roots,
            local_paths=Path(launch["local_paths"]),
            source_configuration=Path(launch["source_configuration"]),
            source_configuration_sha256=launch["source_configuration_sha256"],
            evidence_profile=Path(launch["evidence_profile"]),
            evidence_profile_sha256=launch["evidence_profile_sha256"],
            resource_ok=resource_ok,
        )

    try:
        frozen = read_scientific_freeze(
            freeze, expected_sha256=digest, roots=roots, validate_prerequisites=validate
        )
        stages = ["T2"] + (["T4"] if frozen["source_contract"]["availability"]["latent"] else [])
        outputs = []
        for stage in stages:
            template = technical_phase_launch(
                stage=stage,
                freeze_launch=launch,
                frozen=frozen,
                freeze_sha256=digest,
                campaign_root=args.campaign_root,
            )
            path = args.campaign_root / "launches" / f"{stage}.json"
            if path.exists():
                if json.loads(path.read_text("utf-8")) != template:
                    raise ValueError("existing phase template changed; no overwrite")
            elif args.verify_only:
                return 10
            else:
                if not resource_ok():
                    return 3
                write_new_json(path, template)
            outputs.append({"stage": stage, "path": str(path), "sha256": sha256(path)})
        validate()
        print(
            json.dumps(
                {
                    "status": "TECHNICAL_STAGE_TEMPLATES_VERIFIED_NOT_FITS",
                    "templates": outputs,
                    "optimizer_updates": 0,
                }
            )
        )
        return 0
    except InterruptedError:
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
