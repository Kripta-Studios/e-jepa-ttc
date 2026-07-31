"""Pretrain EvTTC-compatible encoders on public eAP events and geometry."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e_jepa_ttc.training.eap_jepa import (  # noqa: E402
    EAPJEPATrainerConfig,
    inspect_eap_jepa_windows,
    pretrain_eap_jepa,
)


def _profile(name: str, objective: str) -> EAPJEPATrainerConfig:
    geometry_weight = 0.25 if objective in {"geo", "geo2"} else 0.0
    ttc_weight = 0.5 if objective == "ttc" else 0.0
    if name == "smoke":
        return EAPJEPATrainerConfig(
            epochs=1,
            batch_size=4,
            gradient_accumulation=1,
            num_workers=0,
            horizons_ms=(100,),
            max_windows_per_sequence=2,
            max_train_samples=12,
            max_validation_samples=4,
            early_stopping_min_epochs=1,
            early_stopping_patience=0,
            geometry_loss_weight=geometry_weight,
            geometry_target_version="v2" if objective == "geo2" else "v1",
            geometry_sampling_strategy=(
                "balanced_tracks" if objective == "geo2" else "nearest"
            ),
            ttc_loss_weight=ttc_weight,
        )
    if name == "pilot":
        return EAPJEPATrainerConfig(
            epochs=3,
            batch_size=24,
            gradient_accumulation=2,
            num_workers=8,
            max_train_samples=1024,
            max_validation_samples=256,
            early_stopping_min_epochs=2,
            early_stopping_patience=1,
            geometry_loss_weight=geometry_weight,
            geometry_target_version="v2" if objective == "geo2" else "v1",
            geometry_sampling_strategy=(
                "balanced_tracks" if objective == "geo2" else "nearest"
            ),
            ttc_loss_weight=ttc_weight,
        )
    return EAPJEPATrainerConfig(
        geometry_loss_weight=geometry_weight,
        geometry_target_version="v2" if objective == "geo2" else "v1",
        geometry_sampling_strategy="balanced_tracks" if objective == "geo2" else "nearest",
        ttc_loss_weight=ttc_weight,
    )


def main() -> int:
    """Run a resource-aware, sequence-disjoint eAP SSL or geometry pilot."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=("ssl", "geo", "geo2", "ttc"), required=True)
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="smoke")
    parser.add_argument("--root", type=Path, default=Path(r"E:\eAP_dataset"))
    parser.add_argument("--garlttc-root", type=Path, default=Path(r"E:\GarlTTC_dataset"))
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("data/manifests/eap_train40_inventory_v1.json"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        help="Signed split override; defaults to pilot-12 or train-40 by profile.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-windows-per-sequence", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--geometry-loss-weight", type=float)
    parser.add_argument("--geometry-target-version", choices=("v1", "v2"))
    parser.add_argument(
        "--geometry-sampling-strategy",
        choices=("nearest", "balanced_tracks"),
    )
    parser.add_argument("--corridor-half-width", type=float)
    parser.add_argument("--patch-objectness-weight", type=float)
    parser.add_argument("--ttc-loss-weight", type=float)
    parser.add_argument(
        "--expected-garlttc-train-rows",
        type=int,
        default=88_744,
    )
    parser.add_argument(
        "--allow-garlttc-version-change",
        action="store_true",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count projected windows without opening event payloads, RGB, or the GPU.",
    )
    args = parser.parse_args()

    config = _profile(args.profile, args.objective)
    split = args.split or Path(
        "data/splits/eap_train40_v1.json"
        if args.profile == "full"
        else "data/splits/eap_pilot12_v1.json"
    )
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "num_workers": args.workers,
        "prefetch_factor": args.prefetch_factor,
        "seed": args.seed,
        "max_windows_per_sequence": args.max_windows_per_sequence,
    }
    if args.max_train_samples is not None:
        overrides["max_train_samples"] = args.max_train_samples
    if args.max_validation_samples is not None:
        overrides["max_validation_samples"] = args.max_validation_samples
    if args.geometry_loss_weight is not None:
        overrides["geometry_loss_weight"] = args.geometry_loss_weight
    if args.geometry_target_version is not None:
        overrides["geometry_target_version"] = args.geometry_target_version
    if args.geometry_sampling_strategy is not None:
        overrides["geometry_sampling_strategy"] = args.geometry_sampling_strategy
    if args.corridor_half_width is not None:
        overrides["corridor_half_width"] = args.corridor_half_width
    if args.patch_objectness_weight is not None:
        overrides["patch_objectness_weight"] = args.patch_objectness_weight
    if args.ttc_loss_weight is not None:
        overrides["ttc_loss_weight"] = args.ttc_loss_weight

    overrides["expected_garlttc_train_rows"] = args.expected_garlttc_train_rows
    overrides["allow_garlttc_version_change"] = args.allow_garlttc_version_change

    config = replace(
        config,
        **{key: value for key, value in overrides.items() if value is not None},
    )
    if args.objective == "ssl" and config.geometry_loss_weight != 0.0:
        raise ValueError("The SSL control must keep geometry-loss-weight at zero.")
    if args.objective in {"geo", "geo2"} and config.geometry_loss_weight <= 0.0:
        raise ValueError("The Geo objective requires a positive geometry-loss-weight.")
    if args.objective == "geo2" and (
        config.geometry_target_version != "v2"
        or config.geometry_sampling_strategy != "balanced_tracks"
    ):
        raise ValueError("geo2 requires v2 targets and balanced track sampling.")
    if args.objective == "ttc" and config.ttc_loss_weight <= 0.0:
        raise ValueError("The TTC objective requires a positive ttc-loss-weight.")
    output = args.output or Path(
        f"artifacts/runs/eap_{args.objective}_{args.profile}_seed{config.seed}"
    )
    if args.dry_run:
        result = inspect_eap_jepa_windows(
            root=args.root,
            inventory_path=args.inventory,
            split_path=split,
            config=config,
        )
    else:
        if args.objective == "ttc":
            from e_jepa_ttc.data.garlttc_audit import audit

            audit_res = audit(
                eap_root=args.root,
                garlttc_root=args.garlttc_root,
                eap_split_path=split,
                expected_train_rows=(args.expected_garlttc_train_rows),
                allow_dataset_version_change=(args.allow_garlttc_version_change),
            )
            audit_path = output / "garlttc_eap_audit.json"
            output.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(json.dumps(audit_res, indent=2), encoding="utf-8")
            if audit_res.get("result") != "PASS":
                err_msg = json.dumps(audit_res, indent=2)
                raise ValueError(f"Mandatory GarlTTC ↔ eAP linkage audit FAILED:\n{err_msg}")

            from e_jepa_ttc.training.eap_ttc import pretrain_eap_jepa_ttc

            result = pretrain_eap_jepa_ttc(
                eap_root=args.root,
                garlttc_root=args.garlttc_root,
                inventory_path=args.inventory,
                split_path=split,
                output_dir=output,
                config=config,
                audit_json_path=audit_path,
                audit_result=audit_res.get("result", "FAIL"),
                device_name=args.device,
                resume=args.resume,
            )
        else:
            result = pretrain_eap_jepa(
                root=args.root,
                inventory_path=args.inventory,
                split_path=split,
                output_dir=output,
                config=config,
                device_name=args.device,
                resume=args.resume,
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
