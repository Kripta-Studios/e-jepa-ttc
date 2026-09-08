"""Materialize phase templates from a real freeze and canonical practical gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.configured_execution import execute_configured_phase
from e_jepa_ttc.simplex_t.coordination import shared_write_admission
from e_jepa_ttc.simplex_t.lifecycle import admitted
from e_jepa_ttc.simplex_t.phase_launch_materialization import technical_phase_launch
from e_jepa_ttc.simplex_t.practical_launch_materialization import resolve_practical_launch
from e_jepa_ttc.simplex_t.scientific_admission import validate_scientific_admission
from e_jepa_ttc.simplex_t.scientific_freeze import read_scientific_freeze
from e_jepa_ttc.simplex_t.stage_gate import CanonicalPublication


def main() -> int:
    """Retain identical templates on resume; reject changed metadata, never refit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-launch", type=Path, required=True)
    parser.add_argument("--freeze-launch-sha256", required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--stage", choices=("T2", "T3", "T4", "T5"))
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
        if args.stage is not None:
            stages = [args.stage]
        outputs = []
        for stage in stages:
            if stage == "T4" and not frozen["source_contract"]["availability"]["latent"]:
                decision = {
                    "schema": "simplex_t_technical_launch_decision_v1",
                    "stage": "T4",
                    "freeze_sha256": digest,
                    "eligible": False,
                    "reason": "LATENT_TECHNICALLY_UNAVAILABLE_IN_FREEZE",
                    "optimizer_updates": 0,
                    "scientific_completion": False,
                }
                decision_path = args.campaign_root / "launches/T4.decision.json"
                if (args.campaign_root / "launches/T4.json").exists():
                    raise ValueError("technically unavailable stage has an existing template")
                if decision_path.exists():
                    if json.loads(decision_path.read_text("utf-8")) != decision:
                        raise ValueError("technical availability decision changed")
                elif args.verify_only:
                    return 10
                else:
                    if not resource_ok():
                        return 3
                    write_new_json(decision_path, decision)
                outputs.append(
                    {
                        "stage": stage,
                        "eligible": False,
                        "decision": str(decision_path),
                        "sha256": sha256(decision_path),
                    }
                )
                continue
            template = technical_phase_launch(
                stage=stage if stage in {"T2", "T4"} else "T2",
                freeze_launch=launch,
                frozen=frozen,
                freeze_sha256=digest,
                campaign_root=args.campaign_root,
            )
            if stage in {"T3", "T5"}:

                def publication(family: str) -> dict:
                    prior = "T2" if family == "TPR" else "T4"
                    expected = technical_phase_launch(
                        stage=prior,
                        freeze_launch=launch,
                        frozen=frozen,
                        freeze_sha256=digest,
                        campaign_root=args.campaign_root,
                    )
                    saved_path = args.campaign_root / "launches" / f"{prior}.json"
                    saved = json.loads(saved_path.read_text("utf-8"))
                    if saved != expected:
                        raise ValueError("prior phase template differs from frozen campaign")
                    published = Path(saved["publication"]) / f"{prior}_PREDICTIONS.json"
                    endpoints = Path(saved["execution"]) / f"{prior}_ENDPOINTS.json"
                    return dict(
                        publication=str(published),
                        publication_sha256=sha256(published),
                        endpoints=str(endpoints),
                        endpoints_sha256=sha256(endpoints),
                        checkpoint_root=str(Path(saved["execution"]) / "fits"),
                        availability=saved["availability"],
                    )

                def check_gate(probe: dict) -> dict:
                    return execute_configured_phase(
                        local_paths=Path(probe["local_paths"]),
                        source_configuration=Path(probe["source_configuration"]),
                        source_configuration_sha256=probe["source_configuration_sha256"],
                        evidence_profile=Path(probe["evidence_profile"]),
                        evidence_profile_sha256=probe["evidence_profile_sha256"],
                        freeze=freeze,
                        freeze_sha256=digest,
                        roots=roots,
                        execution=Path(probe["execution"]),
                        publication=Path(probe["publication"]),
                        stage=probe["stage"],
                        availability=probe["availability"],
                        publications={
                            family: CanonicalPublication(
                                Path(pin["publication"]),
                                pin["publication_sha256"],
                                Path(pin["endpoints"]),
                                pin["endpoints_sha256"],
                                Path(pin["checkpoint_root"]),
                                pin["availability"],
                            )
                            for family, pin in probe["publications"].items()
                        },
                        resource_ok=resource_ok,
                        resume=False,
                        verify_only=True,
                    )

                template, decision = resolve_practical_launch(
                    stage,
                    base=template,
                    frozen_availability=frozen["source_contract"]["availability"],
                    campaign_root=args.campaign_root,
                    publication=publication,
                    check_gate=check_gate,
                )
                decision_path = args.campaign_root / "launches" / f"{stage}.decision.json"
                if decision_path.exists():
                    if json.loads(decision_path.read_text("utf-8")) != decision:
                        raise ValueError("practical decision changed; no silent reselection")
                elif args.verify_only:
                    return 10
                else:
                    if not resource_ok():
                        return 3
                    write_new_json(decision_path, decision)
                if template is None:
                    if (args.campaign_root / "launches" / f"{stage}.json").exists():
                        raise ValueError("ineligible stage has an existing template")
                    outputs.append(
                        {
                            "stage": stage,
                            "decision": str(decision_path),
                            "sha256": sha256(decision_path),
                            "eligible": False,
                        }
                    )
                    continue
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
                    "status": "STAGE_TEMPLATES_AND_DECISIONS_VERIFIED_NOT_FITS",
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
