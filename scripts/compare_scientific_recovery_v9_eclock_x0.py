#!/usr/bin/env python
"""Build the signed primary E-Clock X0 DYN-U versus BASE-U comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jsonschema

from e_jepa_ttc.evaluation.collision_clock_cross_arm import compare_dyn_vs_base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-root", type=Path, required=True)
    parser.add_argument("--dyn-run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config_root = repo / "configs/experiment/scientific_recovery_v9_eclock"
    comparison, gate = compare_dyn_vs_base(
        base_run_root=args.base_run_root.resolve(),
        dyn_run_root=args.dyn_run_root.resolve(),
        base_config_path=config_root / "x0_base_u.yaml",
        dyn_config_path=config_root / "x0_dyn_u.yaml",
        protocol_path=args.protocol.resolve(),
        reference_path=args.reference.resolve(),
        schema_root=repo / "schemas",
    )
    comparison_schema = json.loads(
        (repo / "schemas/scientific_recovery_v9_eclock_cross_arm_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(comparison_schema).validate(comparison)
    for path, payload in ((args.output, comparison), (args.gate_output, gate)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"comparison": comparison, "gate": gate}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
