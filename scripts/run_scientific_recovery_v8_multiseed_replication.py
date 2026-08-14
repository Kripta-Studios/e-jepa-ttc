#!/usr/bin/env python
# ruff: noqa: E501
"""Plan only the preregistered V8 seeds 13/23 after one signed nomination."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash  # noqa: E402
from e_jepa_ttc.evaluation.scientific_recovery_v8_runner import (  # noqa: E402
    V8IntegrityError,
    assert_multiseed_replication_candidate,
    verify_frozen_inputs,
)
from e_jepa_ttc.training.scientific_recovery_v8_jobs import (  # noqa: E402
    V8JobIntegrityError,
    build_fold_jobs,
    clone_multiseed_configs,
    execute_jobs,
    signed_derived_multiseed_manifest,
)


def _nomination(results_root: Path, candidate: str) -> dict[str, object]:
    matches: list[dict[str, object]] = []
    for path in results_root.rglob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or not verify_artifact_hash(value):
            continue
        if int(value.get("seed", -1)) != 7:
            continue
        identity = str(value.get("candidate_id", value.get("arm", value.get("model_name", ""))))
        nominated = value.get("multiseed_replication_candidate") is True
        gate = value.get("gate_decision", value.get("gates", {}))
        nominated = nominated or (
            isinstance(gate, dict) and gate.get("multiseed_replication_candidate") is True
        )
        if identity == candidate and nominated:
            matches.append(value)
    if len(matches) != 1:
        raise V8IntegrityError(f"expected exactly one signed seed-7 nomination for {candidate!r}")
    return matches[0]


def _candidate_configs(frozen: object, candidate: str) -> list[Path]:
    entries = frozen.manifest["enabled_seed7_configs"]
    configs = [
        ROOT / str(entry["path"])
        for name, entry in entries.items()
        if name.startswith(f"{candidate}_fold") and name.endswith("_seed7")
    ]
    if len(configs) != 3:
        raise V8IntegrityError(
            f"frozen manifest lacks exactly three seed-7 configs for {candidate!r}"
        )
    return sorted(configs)


def _jepa_control_sources(source: dict[str, object]) -> list[Path]:
    """Require the D0/D1/best-D2-or-D3/D4 replication set for a positive JEPA claim."""

    positive = source.get("jepa_causally_positive") is True or source.get("jepa_positive") is True
    if not positive:
        return []
    controls = source.get("jepa_control_replication_set")
    if not isinstance(controls, dict):
        raise V8IntegrityError(
            "JEPA-positive nomination lacks its signed D0/D1/best-D2-or-D3/D4 replication set"
        )
    required = {"D0", "D1", "D4"}
    if not required.issubset(controls) or not ({"D2", "D3"} & set(controls)):
        raise V8IntegrityError("JEPA-positive replication set must include D0, D1, D4 and D2 or D3")
    paths: list[Path] = []
    for arm, entry in controls.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("config_paths"), list):
            raise V8IntegrityError(f"JEPA control {arm} lacks frozen config_paths")
        paths.extend(ROOT / str(value) for value in entry["config_paths"])
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/protocol/scientific_recovery_v8_temporal.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs/experiment/scientific_recovery_v8_fold_chain/frozen_manifest.json",
    )
    parser.add_argument(
        "--results-root", type=Path, default=ROOT / "artifacts/scientific_recovery_v8"
    )
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        frozen = verify_frozen_inputs(args.protocol, args.manifest)
        assert_multiseed_replication_candidate(args.results_root, args.candidate)
        source = _nomination(args.results_root, args.candidate)
        derived_dir = args.results_root / "multiseed_replication"
        source_configs = _candidate_configs(frozen, args.candidate)
        control_configs = _jepa_control_sources(source)
        source_configs = [*source_configs, *control_configs]
        config_dir = derived_dir / "configs"
        if args.dry_run:
            planned = {
                "status": "planned",
                "candidate": args.candidate,
                "seeds": [13, 23],
                "source_configs": [str(path) for path in source_configs],
                "jepa_positive_controls_replicated": bool(control_configs),
                "derived_manifest": str(derived_dir / "derived_manifest.json"),
                "no_tuning": True,
                "no_reselection": True,
            }
            print(json.dumps(planned, indent=2, sort_keys=True))
            return 0
        configs = clone_multiseed_configs(
            candidate=args.candidate,
            source_configs=_candidate_configs(frozen, args.candidate),
            output_dir=config_dir,
        )
        for control in control_configs:
            parsed = yaml.safe_load(control.read_text(encoding="utf-8"))
            arm = parsed.get("experiment", {}).get("arm") if isinstance(parsed, dict) else None
            if not isinstance(arm, str):
                raise V8IntegrityError(f"JEPA control config has no arm: {control}")
            configs.extend(
                clone_multiseed_configs(
                    candidate=arm, source_configs=[control], output_dir=config_dir / "jepa_controls"
                )
            )
        signed_derived_multiseed_manifest(
            candidate=args.candidate,
            source=source,
            output_path=derived_dir / "derived_manifest.json",
        )
        jobs = build_fold_jobs(
            configs=configs,
            output_root=derived_dir / "runs",
            device=args.device,
            max_parallel=args.max_parallel,
            allowed_seeds=(13, 23),
        )
        outputs = execute_jobs(
            jobs,
            protocol_hash=str(frozen.protocol["artifact_sha256"]),
            manifest_hash=str(frozen.manifest["artifact_sha256"]),
            dry_run=False,
            max_parallel=args.max_parallel,
        )
        print(json.dumps({"status": "executed", "jobs": outputs}, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, V8IntegrityError, V8JobIntegrityError) as error:
        parser.exit(2, f"V8 multiseed stage failed closed: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
