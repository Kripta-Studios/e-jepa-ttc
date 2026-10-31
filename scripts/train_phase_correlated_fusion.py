"""One preregistered correlated-expert QA control; no downstream test evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_campaign.data import read
from operational.phase_robust.model import fit_normalization
from operational.phase_startup.model import ExpertRouter
from operational.phase_startup.runtime import RouterEnsemble
from scripts.train_phase_robust import load_arrays, scored

SEEDS = (7, 13, 21)
EPOCHS = 300
PLAN = Path("docs/superpowers/plans/2026-10-31-phase-correlated-failure-augmentation.md")


def corrupt_native(
    native: torch.Tensor, *, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Corrupt 25% of rows using uniform single, F0+RGB, F0+event or triple patterns."""
    if native.ndim != 2 or native.shape[1] != 3 or native.device.type != "cpu":
        raise ValueError("Expected CPU native estimates [N,3]")
    output = native.clone()
    indices = (torch.rand(len(native), generator=generator) < 0.25).nonzero().flatten()
    patterns = torch.randint(0, 4, (len(indices),), generator=generator)
    single = torch.randint(0, 3, (len(indices),), generator=generator)
    factors = torch.tensor([-3.0, -1.0, 0.25, 4.0], dtype=native.dtype)[
        torch.randint(0, 4, (len(indices),), generator=generator)
    ]
    membership = torch.zeros((len(indices), 3), dtype=torch.bool)
    slots = torch.arange(len(indices))
    membership[slots[patterns == 0], single[patterns == 0]] = True
    membership[patterns != 0, 0] = True
    membership[(patterns == 1) | (patterns == 3), 1] = True
    membership[(patterns == 2) | (patterns == 3), 2] = True
    output[indices] *= torch.where(membership, factors[:, None], torch.ones_like(factors[:, None]))
    return output


def predict(model: torch.nn.Module, arrays: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    """Evaluate a frozen model on explicitly supplied rows."""
    with torch.inference_mode():
        return model(
            *(torch.from_numpy(arrays[k][mask]) for k in ("native", "geometry", "quality"))
        ).numpy()


def valid_training_rows(
    native: torch.Tensor, geometry: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Use the original IQ admission rule, including geometry-only observations."""
    return torch.isfinite(target) & (torch.isfinite(native).any(1) | torch.isfinite(geometry))


def fit_control(
    arrays: dict[str, np.ndarray], mask: np.ndarray, seed: int, destination: Path
) -> ExpertRouter:
    """Fit the unchanged inverse router for exactly 300 full-batch CPU updates."""
    destination.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = ExpertRouter("quality", "inverse")
    if sum(p.numel() for p in model.parameters()) != 688:
        raise ValueError("Preregistered model capacity changed")
    native, geometry, quality, target = [
        torch.from_numpy(arrays[k][mask]) for k in ("native", "geometry", "quality", "target")
    ]
    valid = valid_training_rows(native, geometry, target)
    native, geometry, quality, target = [
        value[valid] for value in (native, geometry, quality, target)
    ]
    families = arrays["family"][mask][valid.numpy()]
    weights = np.array([1 / np.sum(families == family) for family in families], np.float32)
    mass = torch.from_numpy(weights / weights.sum())
    center, scale = fit_normalization(model.inputs(native, geometry, quality).detach().numpy())
    model.center.copy_(torch.from_numpy(center))
    model.scale.copy_(torch.from_numpy(scale))
    seconds_scale = target.abs().median().clamp(min=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    history = []
    started = time.perf_counter()
    fit_binding = {
        "seed": seed,
        "mode": "QAc",
        "coordinate": "inverse",
        "epochs": EPOCHS,
        "fit_groups": sorted(set(arrays["group"][mask])),
        "rows": len(target),
        "parameters": 688,
        "seconds_scale_train_only": float(seconds_scale),
    }
    atomic_receipt(destination / "FIT_CONFIG.json", fit_binding)

    def objective(prediction: torch.Tensor) -> torch.Tensor:
        error = (prediction - target).abs()
        return (mass * (error / target.abs().clamp(min=0.2) + error / seconds_scale)).sum()

    try:
        for epoch in range(EPOCHS):
            clean = objective(model(native, geometry, quality))
            corrupted = objective(model(corrupt_native(native), geometry, quality))
            loss = 0.5 * clean + 0.5 * corrupted
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite correlated-control objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(loss.detach()),
                    "clean": float(clean.detach()),
                    "corrupted": float(corrupted.detach()),
                }
            )
    except Exception as error:
        torch.save(
            {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
            destination / "failed_checkpoint.pt",
        )
        atomic_receipt(
            destination / "FAILED.json",
            {
                **fit_binding,
                "error": repr(error),
                "loss": history,
                "seconds": time.perf_counter() - started,
            },
        )
        raise
    torch.save(
        {"state_dict": model.state_dict(), "mode": "QAc", "seed": seed, "coordinate": "inverse"},
        destination / "checkpoint.pt",
    )
    atomic_receipt(
        destination / "COMPLETE.json",
        {
            **fit_binding,
            "loss": history,
            "seconds": time.perf_counter() - started,
            "checkpoint_sha256": file_sha(destination / "checkpoint.pt"),
        },
    )
    return model.eval()


def run(args: argparse.Namespace) -> None:
    """Run only the fixed development experiment and its conditional final refit."""
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.output.exists():
        raise ValueError("Use a fresh directory; previous runs are immutable")
    arrays = load_arrays(args.population)
    baseline_config = read(args.baseline / "CONFIG.json")
    original = read(args.original_config)
    excluded = set(original["excluded_road"] + original["excluded_slider"])
    if (
        len(arrays["target"]) != 2608
        or len(set(arrays["group"])) != 25
        or set(arrays["group"]) & excluded
        or sorted(set(arrays["group"])) != original["fit_groups"]
    ):
        raise ValueError("Population violates fixed group or benchmark exclusion contract")
    folds = sorted(set(arrays["family"]))
    if (
        len(folds) != 8
        or folds != baseline_config["folds"]
        or list(SEEDS) != baseline_config["seeds"]
        or EPOCHS != baseline_config["epochs"]
    ):
        raise ValueError("Baseline folds/seeds/budget do not match")
    for p in (args.population, args.original_config):
        if baseline_config["inputs"].get(str(p)) != file_sha(p):
            raise ValueError("Baseline input identity changed")
    baseline_path = args.baseline / "OOF_PREDICTIONS.npz"
    baseline = load_arrays(baseline_path)["IQ"]
    if baseline.shape != arrays["target"].shape or not np.isfinite(baseline).all():
        raise ValueError("Invalid matched IQ OOF baseline")
    sources = [
        Path(__file__),
        PLAN,
        Path("operational/phase_startup/model.py"),
        Path("operational/phase_startup/runtime.py"),
        Path("operational/phase_robust/model.py"),
        Path("operational/phase_campaign/evaluation.py"),
        Path("operational/phase_adapt_cache.py"),
        Path("scripts/train_phase_robust.py"),
    ]
    source_hashes = {str(p): file_sha(p) for p in sources}
    input_paths = (
        args.population,
        args.original_config,
        baseline_path,
        args.baseline / "CONFIG.json",
        args.baseline / "SELECTION.json",
    )
    config = {
        "schema": "phase_correlated_control_v1",
        "mode": "QAc",
        "coordinate": "inverse",
        "started_utc": dt.datetime.now(dt.UTC).isoformat(),
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": source_hashes,
        "inputs": {str(p): file_sha(p) for p in input_paths},
        "fit_groups": sorted(set(arrays["group"])),
        "excluded": sorted(excluded),
        "folds": folds,
        "parameters": 688,
        "device": "cpu",
        "threads": 2,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
        "augmentation": {
            "row_probability": 0.25,
            "uniform_patterns": ["single_random", "F0_RGB", "F0_event", "all_three"],
            "shared_factor_support": [-3.0, -1.0, 0.25, 4.0],
            "clean_loss_weight": 0.5,
        },
        "selection": (
            "full coverage and road/Slider500 macro MAE/RAE <= matched IQ; no alternatives"
        ),
        "scope": "historically exposed eligible development only; no eAP or competition evaluation",
    }
    atomic_receipt(args.output / "CONFIG.json", config)
    for p in sources:
        destination = args.output / "executed_source" / p.resolve().relative_to(Path.cwd())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, destination)
    by_seed = {f"QAc_seed{seed}": np.full(len(baseline), np.nan, np.float32) for seed in SEEDS}
    started = time.perf_counter()
    completed = 0
    for family in folds:
        test = arrays["family"] == family
        train = ~test
        if set(arrays["group"][train]) & set(arrays["group"][test]):
            raise ValueError("Acquisition group leakage")
        for seed in SEEDS:
            model = fit_control(arrays, train, seed, args.output / "folds" / family / str(seed))
            by_seed[f"QAc_seed{seed}"][test] = predict(model, arrays, test)
            save_npz(
                args.output / "folds" / family / str(seed) / "OOF.npz",
                {"prediction": by_seed[f"QAc_seed{seed}"][test], "token": arrays["token"][test]},
            )
            completed += 1
            elapsed = time.perf_counter() - started
            print(
                f"FIT {completed}/24 family={family} seed={seed} elapsed={elapsed:.1f}s "
                f"eta={elapsed / completed * (24 - completed):.1f}s",
                flush=True,
            )
    mean = np.mean([by_seed[f"QAc_seed{s}"] for s in SEEDS], axis=0)
    predictions = {**by_seed, "QAc": mean, "IQ": baseline, "token": arrays["token"]}
    save_npz(args.output / "OOF_PREDICTIONS.npz", predictions)
    masks = {
        "all": np.ones(len(baseline), bool),
        "road": arrays["family"] != "Slider",
        "slider": arrays["family"] == "Slider",
    }
    reports = {
        name: {domain: scored(arrays, pred, mask) for domain, mask in masks.items()}
        for name, pred in (("IQ", baseline), ("QAc", mean))
    }
    gates = {
        f"{domain}_{key}": reports["QAc"][domain][key] <= reports["IQ"][domain][key]
        for domain in ("road", "slider")
        for key in ("macro_record_mae_seconds", "macro_record_relative_error_pct")
    }
    full_coverage = all(reports["QAc"][domain]["coverage"] == 1.0 for domain in masks)
    selected = "QAc" if full_coverage and all(gates.values()) else None
    for p, digest in config["inputs"].items():
        if file_sha(Path(p)) != digest:
            raise ValueError("Frozen input changed during campaign")
    if any(file_sha(Path(p)) != digest for p, digest in source_hashes.items()):
        raise ValueError("Implementation changed during campaign")
    atomic_receipt(
        args.output / "SELECTION.json",
        {
            "selected": selected,
            "gates": gates,
            "full_coverage": full_coverage,
            "report": reports,
            "elapsed_seconds": time.perf_counter() - started,
            "config_sha256": file_sha(args.output / "CONFIG.json"),
            "oof_sha256": file_sha(args.output / "OOF_PREDICTIONS.npz"),
            "reason": "all preregistered gates pass"
            if selected
            else "fixed control rejected; no tuning",
        },
    )
    if selected:
        final = [
            fit_control(arrays, masks["all"], seed, args.output / "final" / str(seed))
            for seed in SEEDS
        ]
        ensemble = RouterEnsemble(final).eval()
        scripted = torch.jit.script(ensemble)
        scripted.save(str(args.output / "router.ts"))
        reloaded = torch.jit.load(str(args.output / "router.ts"))
        actual = predict(ensemble, arrays, masks["all"])
        np.testing.assert_array_equal(actual, predict(reloaded, arrays, masks["all"]))
        save_npz(args.output / "FIT_PREDICTIONS.npz", {"prediction": actual})
        atomic_receipt(
            args.output / "FINAL_FREEZE.json",
            {
                "selected": selected,
                "coordinate": "inverse",
                "config_sha256": file_sha(args.output / "CONFIG.json"),
                "selection_sha256": file_sha(args.output / "SELECTION.json"),
                "checkpoints": {
                    str(s): file_sha(args.output / "final" / str(s) / "checkpoint.pt")
                    for s in SEEDS
                },
                "export_sha256": file_sha(args.output / "router.ts"),
                "export_exact_rows": len(actual),
                "eap_evaluated": False,
                "competition_evaluated": False,
            },
        )
    print(f"COMPLETE selected={selected} elapsed={time.perf_counter() - started:.1f}s", flush=True)


def main() -> None:
    """Read explicit frozen inputs and start the single fixed control."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("population", "original-config", "baseline", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
