"""Run the fixed independent rotation-0 E_A5 supervised producer."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from operational.independent_refit.producer_contract import build_producer_plan
from operational.independent_refit.producer_train import build_run_contract, run_e_a5


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train40-protocol", type=Path, required=True)
    parser.add_argument("--rgb-recipe", type=Path, required=True)
    parser.add_argument("--fold-contract", type=Path, required=True)
    parser.add_argument("--raw-history", type=Path, required=True)
    parser.add_argument("--h5-preflight", type=Path, required=True)
    parser.add_argument("--h5-admission", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--teacher-role-receipt", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--prepare-freeze", type=Path, required=True)
    parser.add_argument("--crosswalk-receipt", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    args = parser.parse_args(argv)

    plan = build_producer_plan(
        component="E_A5",
        rotation=0,
        train40_protocol_path=args.train40_protocol,
        rgb_recipe_path=args.rgb_recipe,
        fold_contract_path=args.fold_contract,
        raw_history_receipt_path=args.raw_history,
        h5_preflight_path=args.h5_preflight,
        teacher_manifest_path=args.teacher_manifest,
        teacher_role_receipt_path=args.teacher_role_receipt,
    )
    run_contract = build_run_contract(
        plan=plan,
        h5_admission_path=args.h5_admission,
        input_manifest_path=args.input_manifest,
        prepare_freeze_path=args.prepare_freeze,
        crosswalk_receipt_path=args.crosswalk_receipt,
        crosswalk_path=args.crosswalk,
        teacher_manifest_path=args.teacher_manifest,
        teacher_role_receipt_path=args.teacher_role_receipt,
    )
    summary = run_e_a5(
        output_dir=args.output,
        plan=plan,
        run_contract=run_contract,
        source_kwargs={
            "source_manifest_path": args.input_manifest,
            "source_root": args.input_root,
            "prepare_freeze_path": args.prepare_freeze,
            "crosswalk_receipt_path": args.crosswalk_receipt,
            "crosswalk_path": args.crosswalk,
            "h5_admission_path": args.h5_admission,
            "plan": plan,
            "teacher_manifest_path": args.teacher_manifest,
            "teacher_role_receipt_path": args.teacher_role_receipt,
            "rotation": 0,
            "max_cached_shards": 2,
        },
        teacher_manifest_path=args.teacher_manifest,
        fold_contract_path=args.fold_contract,
        teacher_role_receipt_path=args.teacher_role_receipt,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
