"""Preregistered joint native support/context experiment with frozen informational references."""

from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_campaign.data import read
from operational.phase_native_support.joint_training import fit, prepare_fold, select_candidate
from operational.phase_native_support.training import EPOCHS, SEEDS
from scripts.train_phase_native_support_fusion import (
    load_experiment_data,
    load_iq_reference,
    predict,
)
from scripts.train_phase_robust import load_arrays, scored

PLAN = Path("docs/superpowers/plans/2026-10-31-phase-support-context-joint.md")
COUNTS = {"DEVELOPMENT": 2608, "RESTARTS": 447, "SOURCE_TRAIN40": 88744, "TEST12": 6762}
INDICES = list(range(165, 170)) + list(range(335, 340))


def load_informational_reference(
    root: Path, clean: dict, names: tuple[str, ...], population: Path
) -> dict:
    """Bind prior outcomes to the same population; these references cannot select a model."""
    config, selection, verification = [
        read(root / f"{name}.json") for name in ("CONFIG", "SELECTION", "VERIFICATION")
    ]
    if (
        verification.get("status") != "PASS"
        or verification.get("fits_verified") != 48
        or selection.get("config_sha256") != file_sha(root / "CONFIG.json")
        or selection.get("oof_sha256") != file_sha(root / "OOF_PREDICTIONS.npz")
        or verification.get("selection_sha256") != file_sha(root / "SELECTION.json")
        or config["inputs"].get(str(population)) != file_sha(population)
        or config.get("seeds") != list(SEEDS)
        or config.get("epochs") != EPOCHS
    ):
        raise ValueError("Informational reference seal or population differs")
    values = load_arrays(root / "OOF_PREDICTIONS.npz")
    if any(not np.array_equal(values[k], clean[k]) for k in ("token", "group")):
        raise ValueError("Informational reference identities differ")
    result = {
        key: values[key] for name in names for key in (name, *[f"{name}_seed{s}" for s in SEEDS])
    }
    if any(v.shape != clean["token"].shape or not np.isfinite(v).all() for v in result.values()):
        raise ValueError("Incomplete informational reference")
    for name in names:
        if not np.array_equal(
            result[name], np.mean([result[f"{name}_seed{s}"] for s in SEEDS], axis=0)
        ):
            raise ValueError("Informational reference seed arithmetic differs")
    return result


def run(args: argparse.Namespace) -> None:
    """Freeze sources before48 fits and report all gates, including unsuccessful candidates."""
    if args.output.exists():
        raise ValueError("Use a fresh immutable experiment directory")
    clean, restarts, indices, admission = load_experiment_data(args)
    baseline = load_iq_reference(args.baseline, clean, args.population, args.original_config)
    baseline.update(load_informational_reference(args.baseline, clean, ("IQn",), args.population))
    baseline.update(
        load_informational_reference(args.support_reference, clean, ("F2", "Fcap"), args.population)
    )
    original_support = read(args.support_reference / "CONFIG.json")
    original_source_hashes = {
        Path(k).resolve(): v for k, v in original_support["source_hashes"].items()
    }
    for relative in (
        "operational/phase_native_support/model.py",
        "operational/phase_native_support/training.py",
        "scripts/train_phase_native_support_fusion.py",
    ):
        path = Path(relative)
        if original_source_hashes.get(path.resolve()) != file_sha(path):
            raise ValueError("Completed support model/training/runner source changed")
    if args.preflight_only:
        print(
            "PREFLIGHT PASS: support2608/restart447/auxiliary tables, "
            "native seals and exact IQ reference; no fits"
        )
        return
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    sources = [
        Path(__file__),
        Path("tests/unit/test_phase_support_context_joint.py"),
        PLAN,
        *Path("operational/phase_native_support").glob("*.py"),
        Path("scripts/train_phase_native_context_fusion.py"),
        Path("scripts/train_phase_native_support_fusion.py"),
        Path("scripts/train_phase_startup.py"),
        Path("operational/phase_campaign/data.py"),
        Path("operational/evttc_transfer_audit/common.py"),
        Path("scripts/train_phase_robust.py"),
        Path("scripts/audit_phase_native_source_closure.py"),
        Path("operational/phase_observation_cache/integrity.py"),
        Path("operational/phase_adapt_cache.py"),
        Path("operational/phase_robust/model.py"),
        Path("operational/phase_startup/model.py"),
        Path("operational/phase_campaign/evaluation.py"),
    ]
    input_paths = [
        *[
            args.support_reference / name
            for name in (
                "CONFIG.json",
                "SELECTION.json",
                "VERIFICATION.json",
                "OOF_PREDICTIONS.npz",
            )
        ],
        args.population,
        args.original_config,
        args.source_closure,
        args.support_cache / "COMPLETE.json",
        args.support_cache / "FREEZE.json",
        args.native_cache / "COMPLETE.json",
        args.native_cache / "AUGMENTATIONS.npz",
        *[args.support_cache / f"{key}.npz" for key in COUNTS],
        *[
            args.baseline / name
            for name in (
                "CONFIG.json",
                "SELECTION.json",
                "VERIFICATION.json",
                "IQ_REPRODUCTION.json",
                "OOF_PREDICTIONS.npz",
            )
        ],
    ]
    config = {
        "schema": "phase_support_context_joint_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seeds": list(SEEDS),
        "folds": sorted(set(clean["family"])),
        "epochs": EPOCHS,
        "fits": 48,
        "parameters": {"F2n": 868, "Fcapn": 878},
        "new_support_order": "RGB clock4,count/8; event clock4,count/8",
        "native_feature_indices": INDICES,
        "constant_support_mask": "train clean+restart union variance<1e-12 =>0; F2n only",
        "training": "0.5clean+0.5actual native context restarts; no QA corruption; "
        "equal family/query and inverse admitted variant count",
        "selection": "All4 clean metric gates versus IQ and fullcoverage; mean4ratios; tie F2n",
        "attribution": "F2n<=Fcapn onall4 and<onatleast1,fullcoverage; association,notcausalproof",
        "normalization": "base33 and MAE scale clean train; "
        "support10 admitted clean+restart train union",
        "fit_groups": sorted(set(clean["group"])),
        "source_hashes": {str(p): file_sha(p) for p in sources},
        "inputs": {str(p): file_sha(p) for p in input_paths},
        "admission": admission,
        "original_support_numerical_sources_unchanged": True,
        "device": "cpu",
        "threads": 2,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
        "scope": "eligible development; auxiliary TRAIN40/Test12 support verified, "
        "no candidate evaluation there",
        "limitations": "Architecture and support normalization differ; association is not "
        "causal proof or independent generalization evidence; Fcapn has10more parameters. "
        "F2QA to F2n changes both objective and support normalization population.",
    }
    atomic_receipt(args.output / "CONFIG.json", config)
    for path in sources:
        destination = args.output / "executed_source" / path.resolve().relative_to(Path.cwd())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    oof = {
        **baseline,
        **{
            f"{mode}_seed{s}": np.full(len(clean["token"]), np.nan, np.float32)
            for mode in ("F2n", "Fcapn")
            for s in SEEDS
        },
    }
    restart_oof = {
        f"{mode}_seed{s}": np.full(len(restarts["token"]), np.nan, np.float32)
        for mode in ("F2n", "Fcapn")
        for s in SEEDS
    }
    masks_by_fold, completed, started = {}, 0, time.perf_counter()
    for family in config["folds"]:
        fold = prepare_fold(clean, restarts, indices, family)
        train, test, restart_test = fold["train"], fold["test"], fold["aug_test"]
        masks_by_fold[family] = {}
        for mode in ("F2n", "Fcapn"):
            for seed in SEEDS:
                destination = args.output / "folds" / family / mode / str(seed)
                atomic_receipt(
                    destination / "ADMISSION.json",
                    {
                        "fit_groups": sorted(set(clean["group"][train])),
                        "test_groups": sorted(set(clean["group"][test])),
                        "fit_token_sha256": hashlib.sha256(
                            clean["token"][train].tobytes()
                        ).hexdigest(),
                        "test_token_sha256": hashlib.sha256(
                            clean["token"][test].tobytes()
                        ).hexdigest(),
                        "restart_rows_used_for_training": int(fold["aug_mask"].sum()),
                    },
                )
                try:
                    model = fit(clean, restarts, fold, mode, seed, destination / "fit")
                    masks_by_fold[family][f"{mode}_seed{seed}"] = (
                        model.support_mask.tolist() if mode == "F2n" else None
                    )
                    key = f"{mode}_seed{seed}"
                    oof[key][test] = predict(model, clean, test)
                    restart_oof[key][restart_test] = predict(model, restarts, restart_test)
                    save_npz(
                        destination / "OOF.npz",
                        {
                            "prediction": oof[key][test],
                            "token": clean["token"][test],
                            "group": clean["group"][test],
                        },
                    )
                    save_npz(
                        destination / "RESTART_OOF.npz",
                        {
                            "prediction": restart_oof[key][restart_test],
                            "token": restarts["token"][restart_test],
                            "keep": restarts["keep"][restart_test],
                        },
                    )
                except Exception as error:
                    atomic_receipt(
                        destination / "FAILED.json",
                        {"error": repr(error), "mode": mode, "seed": seed, "family": family},
                    )
                    raise
                completed += 1
                print(
                    f"FIT {completed}/48 {family} {mode} seed={seed} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    for mode in ("F2n", "Fcapn"):
        oof[mode] = np.mean([oof[f"{mode}_seed{s}"] for s in SEEDS], axis=0)
        restart_oof[mode] = np.mean([restart_oof[f"{mode}_seed{s}"] for s in SEEDS], axis=0)
    save_npz(
        args.output / "OOF_PREDICTIONS.npz",
        {**oof, "token": clean["token"], "group": clean["group"]},
    )
    save_npz(
        args.output / "RESTART_OOF_PREDICTIONS.npz",
        {
            **restart_oof,
            "token": restarts["token"],
            "group": restarts["group"],
            "keep": restarts["keep"],
        },
    )
    masks = {
        "all": np.ones(len(clean["token"]), bool),
        "road": clean["family"] != "Slider",
        "slider": clean["family"] == "Slider",
    }
    reports = {
        key: {domain: scored(clean, pred, mask) for domain, mask in masks.items()}
        for key, pred in oof.items()
    }
    secondary = {
        key: {str(keep): scored(restarts, pred, restarts["keep"] == keep) for keep in (1, 2, 3)}
        for key, pred in restart_oof.items()
    }
    selection = select_candidate(reports)
    for path, digest in {**config["inputs"], **config["source_hashes"]}.items():
        if file_sha(Path(path)) != digest:
            raise ValueError("Frozen input/source changed during campaign")
    atomic_receipt(
        args.output / "SELECTION.json",
        {
            **selection,
            "report": reports,
            "secondary_restart_report": secondary,
            "active_native_mask_by_fold_seed": masks_by_fold,
            "fits_completed": completed,
            "config_sha256": file_sha(args.output / "CONFIG.json"),
            "oof_sha256": file_sha(args.output / "OOF_PREDICTIONS.npz"),
            "restart_oof_sha256": file_sha(args.output / "RESTART_OOF_PREDICTIONS.npz"),
            "elapsed_seconds": time.perf_counter() - started,
            "eap_evaluated": False,
            "test12_evaluated": False,
            "benchmark_evaluated": False,
        },
    )
    print(f"COMPLETE selected={selection['selected']}; no automatic refit/promotion", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "population",
        "original-config",
        "support-cache",
        "native-cache",
        "source-closure",
        "baseline",
        "support-reference",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
