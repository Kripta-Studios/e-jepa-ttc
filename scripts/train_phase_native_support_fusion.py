"""Preregistered F2 versus Fcap support experiment with a sealed reproduced IQ reference."""

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
from operational.phase_native_support.training import EPOCHS, SEEDS, fit, select_candidate
from scripts.train_phase_native_context_fusion import load_training_data
from scripts.train_phase_robust import load_arrays, scored

PLAN = Path("docs/superpowers/plans/2026-10-31-phase-native-support-fusion.md")
COUNTS = {"DEVELOPMENT": 2608, "RESTARTS": 447, "SOURCE_TRAIN40": 88744, "TEST12": 6762}
INDICES = list(range(165, 170)) + list(range(335, 340))


def validate_support_table(table: dict, rows: int, *, restarts: bool = False) -> None:
    """Validate exact schema, causal clock algebra and explicit missing contexts."""
    required = {"token", "group", "quality10", "knownmask2", "status", "reason"}
    if restarts:
        required.add("keep")
    if set(table) != required:
        raise ValueError("Support table schema differs")
    for key in ("token", "group", "status", "reason"):
        if table[key].shape != (rows,) or table[key].dtype.kind not in "US":
            raise ValueError("Invalid textual support identity columns")
    quality, known = table["quality10"], table["knownmask2"]
    if quality.shape != (rows, 10) or quality.dtype != np.float32 or not np.isfinite(quality).all():
        raise ValueError("Support must be finite FP32[N,10]")
    if known.shape != (rows, 2) or known.dtype != np.bool_:
        raise ValueError("Expected explicit RGB/event context-executed masks")
    if restarts:
        if table["keep"].shape != (rows,) or table["keep"].dtype.kind not in "iu":
            raise ValueError("Invalid restart length")
        pairs = list(zip(table["token"], table["keep"], strict=True))
        if len(set(pairs)) != rows or not np.isin(table["keep"], [1, 2, 3]).all():
            raise ValueError("Missing/duplicate restart identity")
        for token in set(table["token"]):
            mask = table["token"] == token
            if sorted(table["keep"][mask].tolist()) != [1, 2, 3]:
                raise ValueError("Incomplete query restart variants")
    elif len(set(table["token"])) != rows:
        raise ValueError("Duplicate support token")
    for modality in (0, 1):
        values = quality[:, modality * 5 : modality * 5 + 5]
        if np.any(values[~known[:, modality]] != 0):
            raise ValueError("Unknown context must use explicit zero placeholders")
        active = values[known[:, modality]]
        if len(active) and (
            np.any(active < 0)
            or np.any(active[:, 4] < 0.125)
            or np.any(active[:, 4] > 1)
            or not np.allclose(active[:, 4] * 8, np.rint(active[:, 4] * 8), atol=1e-6, rtol=0)
            or not np.allclose(active[:, 0], active[:, 1] + active[:, 3], atol=1e-6, rtol=1e-5)
        ):
            raise ValueError("Invalid native count or causal clock algebra")


def attach_support(arrays: dict, table: dict, *, restarts: bool = False) -> None:
    """Join by complete ordered identity; no nearest-time matching or silent reordering."""
    keys = ("token", "group", "keep") if restarts else ("token", "group")
    if any(not np.array_equal(arrays[k], table[k]) for k in keys):
        raise ValueError("Support token/group/variant identity differs")
    arrays["support"] = table["quality10"].copy()


def family_masks(arrays: dict, family: str) -> tuple[np.ndarray, np.ndarray]:
    """Hold out a complete family, including every member of its acquisition groups."""
    test = arrays["family"] == family
    train = ~test
    if (
        not test.any()
        or not train.any()
        or set(arrays["group"][train]) & set(arrays["group"][test])
    ):
        raise ValueError("Empty fold or acquisition-group leakage")
    return train, test


def verify_support_cache(root: Path, population: Path) -> dict[str, dict]:
    """Verify all four label-free support tables and source hashes before target joining."""
    complete = read(root / "COMPLETE.json")
    if (
        complete.get("schema") != "phase_native_support_cache_v1"
        or complete.get("status") != "COMPLETE"
        or complete.get("targets_read") is not False
        or complete.get("optimizer_updates") != 0
        or complete.get("counts") != COUNTS
        or complete.get("feature_indices") != INDICES
        or complete.get("freeze_sha256") != file_sha(root / "FREEZE.json")
    ):
        raise ValueError("Support cache is incomplete or uses a different contract")
    freeze = read(root / "FREEZE.json")
    if complete["inputs"] != freeze["inputs"] or complete["inputs"].get(
        str(population)
    ) != file_sha(population):
        raise ValueError("Support cache source population differs")
    if not {f"{key}.npz" for key in COUNTS}.issubset(complete["artifacts"]):
        raise ValueError("Missing support tables in seal")
    for relative, digest in complete["artifacts"].items():
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or not path.is_file()
            or file_sha(path) != digest
        ):
            raise ValueError("Changed, missing or escaping support artifact")
    for name, digest in complete["inputs"].items():
        if file_sha(Path(name)) != digest:
            raise ValueError("Support source input changed")
    tables = {}
    for key, count in COUNTS.items():
        tables[key] = load_arrays(root / f"{key}.npz")
        validate_support_table(tables[key], count, restarts=key == "RESTARTS")
    return tables


def load_iq_reference(root: Path, arrays: dict, population: Path, original_config: Path) -> dict:
    """Require the completed previous experiment and exact historical IQ reproduction."""
    config, selection = read(root / "CONFIG.json"), read(root / "SELECTION.json")
    verification, reproduction = (
        read(root / "VERIFICATION.json"),
        read(root / "IQ_REPRODUCTION.json"),
    )
    if (
        config.get("seeds") != list(SEEDS)
        or config.get("epochs") != EPOCHS
        or config.get("folds") != sorted(set(arrays["family"]))
        or verification.get("status") != "PASS"
        or verification.get("fits_verified") != 48
        or reproduction.get("exact") is not True
        or reproduction.get("maximum_absolute_difference_seconds") != 0
        or selection.get("config_sha256") != file_sha(root / "CONFIG.json")
        or selection.get("oof_sha256") != file_sha(root / "OOF_PREDICTIONS.npz")
        or verification.get("selection_sha256") != file_sha(root / "SELECTION.json")
    ):
        raise ValueError("IQ reference is not the verified exact original control")
    for path in (population, original_config):
        if config["inputs"].get(str(path)) != file_sha(path):
            raise ValueError("IQ reference population differs")
    values = load_arrays(root / "OOF_PREDICTIONS.npz")
    for key in ("token", "group"):
        if not np.array_equal(values[key], arrays[key]):
            raise ValueError("IQ reference ordered identities differ")
    result = {k: values[k] for k in ["IQ", *[f"IQ_seed{s}" for s in SEEDS]]}
    if any(v.shape != arrays["token"].shape or not np.isfinite(v).all() for v in result.values()):
        raise ValueError("IQ reference is incomplete")
    if not np.array_equal(result["IQ"], np.mean([result[f"IQ_seed{s}"] for s in SEEDS], axis=0)):
        raise ValueError("IQ reference mean differs from original seed arithmetic")
    return result


def predict(model: torch.nn.Module, arrays: dict, mask: np.ndarray) -> np.ndarray:
    """Evaluate a fixed model on explicitly selected rows."""
    with torch.inference_mode():
        return model(
            *(
                torch.from_numpy(arrays[k][mask])
                for k in ("native", "geometry", "quality", "support")
            )
        ).numpy()


def load_experiment_data(args: argparse.Namespace) -> tuple[dict, dict, np.ndarray, dict]:
    """Check support identities against sensor-side identities before opening targets."""
    tables = verify_support_cache(args.support_cache, args.population)
    with np.load(args.population, allow_pickle=False) as data:
        clean_identity = {k: data[k] for k in ("token", "group")}
    attach_support(clean_identity, tables["DEVELOPMENT"])
    with np.load(args.native_cache / "AUGMENTATIONS.npz", allow_pickle=False) as data:
        restart_identity = {k: data[k] for k in ("token", "group", "keep")}
    attach_support(restart_identity, tables["RESTARTS"], restarts=True)
    clean, restarts, indices, admission = load_training_data(
        args.population, args.original_config, args.native_cache, args.source_closure
    )
    attach_support(clean, tables["DEVELOPMENT"])
    attach_support(restarts, tables["RESTARTS"], restarts=True)
    return clean, restarts, indices, admission


def run(args: argparse.Namespace) -> None:
    """Freeze sources before48 fits and report all gates, including unsuccessful candidates."""
    if args.output.exists():
        raise ValueError("Use a fresh immutable experiment directory")
    clean, restarts, indices, admission = load_experiment_data(args)
    baseline = load_iq_reference(args.baseline, clean, args.population, args.original_config)
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
        Path("tests/unit/test_phase_native_support_fusion.py"),
        PLAN,
        *Path("operational/phase_native_support").glob("*.py"),
        Path("scripts/train_phase_native_context_fusion.py"),
        Path("scripts/train_phase_robust.py"),
        Path("scripts/audit_phase_native_source_closure.py"),
        Path("operational/phase_observation_cache/integrity.py"),
        Path("operational/phase_adapt_cache.py"),
        Path("operational/phase_robust/model.py"),
        Path("operational/phase_startup/model.py"),
        Path("operational/phase_campaign/evaluation.py"),
    ]
    input_paths = [
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
        "schema": "phase_native_support_fusion_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seeds": list(SEEDS),
        "folds": sorted(set(clean["family"])),
        "epochs": EPOCHS,
        "fits": 48,
        "parameters": {"F2": 868, "Fcap": 878},
        "new_support_order": "RGB clock4,count/8; event clock4,count/8",
        "native_feature_indices": INDICES,
        "constant_support_mask": "train variance<1e-12 =>0 after normalization; F2 only",
        "training": "Original QA:0.5clean+0.5single-expert corruption; "
        "25%rows,-3/-1/.25/4; unchanged support",
        "selection": "All4 clean metric gates versus IQ and fullcoverage; mean4ratios; tie F2",
        "attribution": "F2<=Fcap onall4 and<onatleast1,fullcoverage; association,notcausalproof",
        "normalization": "admitted clean fit rows only; first33 untouched",
        "fit_groups": sorted(set(clean["group"])),
        "source_hashes": {str(p): file_sha(p) for p in sources},
        "inputs": {str(p): file_sha(p) for p in input_paths},
        "admission": admission,
        "device": "cpu",
        "threads": 2,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
        "scope": "eligible development; auxiliary TRAIN40/Test12 support verified, "
        "no candidate evaluation there",
        "limitations": "Different architecture initializations consume different RNG draws; "
        "same seeds and QA algorithm do not imply paired identical corruption realizations. "
        "Support association is not causal proof; Fcap has10more parameters.",
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
            for mode in ("F2", "Fcap")
            for s in SEEDS
        },
    }
    restart_oof = {
        f"{mode}_seed{s}": np.full(len(restarts["token"]), np.nan, np.float32)
        for mode in ("F2", "Fcap")
        for s in SEEDS
    }
    masks_by_fold, completed, started = {}, 0, time.perf_counter()
    for family in config["folds"]:
        train, test = family_masks(clean, family)
        restart_test = test[indices]
        masks_by_fold[family] = {}
        for mode in ("F2", "Fcap"):
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
                        "restart_rows_used_for_training": 0,
                    },
                )
                try:
                    model = fit(clean, train, mode, seed, destination / "fit")
                    masks_by_fold[family][f"{mode}_seed{seed}"] = (
                        model.support_mask.tolist() if mode == "F2" else None
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
    for mode in ("F2", "Fcap"):
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
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
