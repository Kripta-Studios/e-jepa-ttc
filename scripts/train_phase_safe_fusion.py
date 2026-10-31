"""Preregistered family-held-out sign-safe fusion campaign on eligible data only."""

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
from operational.phase_safe_fusion.model import SafeFusion
from scripts.train_phase_robust import load_arrays, scored
from scripts.train_phase_startup import fit_router

MODES = ("IQ", "LG", "LA", "GR")
SEEDS = (7, 13, 21)


def predict(model: torch.nn.Module, a: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    """Evaluate any three-input fusion without changing its numerical convention."""
    with torch.inference_mode():
        return model(
            *(torch.from_numpy(a[k][mask]) for k in ("native", "geometry", "quality"))
        ).numpy()


def save_progress(root: Path, family: str, mode: str, predictions: dict[str, np.ndarray]) -> None:
    """Write a distinct immutable snapshot after each completed fold/control."""
    save_npz(root / "progress" / family / f"{mode}.npz", predictions)


def select_candidate(reports: dict) -> str | None:
    """Require coverage and both-domain MAE/RAE nonregression against OOF IQ."""
    keys = ("macro_record_relative_error_pct", "macro_record_mae_seconds")
    passing = [
        mode
        for mode in ("LG", "LA", "GR")
        if all(
            reports[mode][domain]["coverage"] == 1.0
            and all(reports[mode][domain][key] <= reports["IQ"][domain][key] for key in keys)
            for domain in ("road", "slider")
        )
    ]
    return min(passing, key=lambda mode: reports[mode]["all"][keys[0]]) if passing else None


def fit_safe(
    a: dict[str, np.ndarray], mask: np.ndarray, mode: str, seed: int, dest: Path, epochs: int = 300
) -> SafeFusion:
    """Train on cached features with fixed budget; no validation-driven stopping."""
    torch.manual_seed(seed)
    model = SafeFusion("residual" if mode == "GR" else "log")
    native, geometry, quality, target = [
        torch.from_numpy(a[k][mask]) for k in ("native", "geometry", "quality", "target")
    ]
    if not torch.isfinite(target).all():
        raise ValueError("Nonfinite training target")
    families = a["family"][mask]
    weights = np.array([1 / np.sum(families == f) for f in families], np.float32)
    mass = torch.from_numpy(weights / weights.sum())
    center, scale = fit_normalization(model.inputs(native, geometry, quality).detach().numpy())
    model.features.center.copy_(torch.from_numpy(center))
    model.features.scale.copy_(torch.from_numpy(scale))
    seconds_scale = target.abs().median().clamp(min=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    history = []
    start = time.perf_counter()

    def objective(prediction: torch.Tensor) -> torch.Tensor:
        error = (prediction - target).abs()
        return (mass * (error / target.abs().clamp(min=0.2) + error / seconds_scale)).sum()

    for epoch in range(epochs):
        loss = objective(model(native, geometry, quality))
        if mode != "LG":
            corrupted = native.clone()
            indices = (torch.rand(len(native)) < 0.25).nonzero().flatten()
            experts = torch.randint(0, 3, (len(indices),))
            factors = torch.tensor([-3.0, -1.0, 0.25, 4.0])[torch.randint(0, 4, (len(indices),))]
            corrupted[indices, experts] *= factors
            loss = 0.5 * loss + 0.5 * objective(model(corrupted, geometry, quality))
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite objective; preserve failed run")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if epoch % 30 == 0 or epoch == epochs - 1:
            history.append({"epoch": epoch + 1, "loss": float(loss.detach())})
    dest.mkdir(parents=True, exist_ok=False)
    torch.save(
        {"state_dict": model.state_dict(), "mode": mode, "seed": seed}, dest / "checkpoint.pt"
    )
    atomic_receipt(
        dest / "COMPLETE.json",
        {
            "mode": mode,
            "seed": seed,
            "epochs": epochs,
            "rows": len(target),
            "fit_groups": sorted(set(a["group"][mask])),
            "loss": history,
            "parameters": sum(p.numel() for p in model.parameters()),
            "seconds_scale_train_only": float(seconds_scale),
            "seconds": time.perf_counter() - start,
            "checkpoint_sha256": file_sha(dest / "checkpoint.pt"),
        },
    )
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--original-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.output.exists():
        raise ValueError("Use a fresh output directory; prior experiments are immutable")
    a = load_arrays(args.population)
    original = read(args.original_config)
    excluded = set(original["excluded_road"] + original["excluded_slider"])
    if set(a["group"]) & excluded or sorted(set(a["group"])) != original["fit_groups"]:
        raise ValueError("Population conflicts with benchmark exclusions")
    if len(a["target"]) != 2608 or len(set(a["group"])) != 25:
        raise ValueError("Expected audited 2608-row eligible population")
    sources = [
        Path(__file__),
        Path("operational/phase_safe_fusion/model.py"),
        Path("scripts/train_phase_startup.py"),
        Path("operational/phase_startup/model.py"),
        Path("operational/phase_robust/model.py"),
        Path("scripts/train_phase_robust.py"),
        Path("docs/superpowers/plans/2026-10-31-phase-sign-safe-fusion.md"),
    ]
    config = {
        "schema": "phase_sign_safe_campaign_v1",
        "started_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {str(p): file_sha(p) for p in sources},
        "inputs": {str(p): file_sha(p) for p in (args.population, args.original_config)},
        "seeds": list(SEEDS),
        "modes": list(MODES),
        "epochs": 300,
        "fit_groups": sorted(set(a["group"])),
        "excluded": sorted(excluded),
        "folds": sorted(set(a["family"])),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
        "device": "cpu",
        "scope": "historically exposed grouped development; benchmark not opened for selection",
        "selection": (
            "both road and Slider500 macro MAE/RAE <=IQ with full coverage; min overall macro RAE"
        ),
    }
    atomic_receipt(args.output / "CONFIG.json", config)
    for path in sources:
        relative = path.resolve().relative_to(Path.cwd())
        dst = args.output / "executed_source" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dst)
    predictions = {mode: np.full(len(a["target"]), np.nan, np.float32) for mode in MODES}
    start = time.perf_counter()
    done = 0
    total = len(MODES) * len(config["folds"]) * len(SEEDS)
    for family in config["folds"]:
        test = a["family"] == family
        train = ~test
        if set(a["group"][train]) & set(a["group"][test]):
            raise ValueError("Acquisition leakage across fold")
        for mode in MODES:
            values = []
            for seed in SEEDS:
                dest = args.output / "folds" / family / mode / str(seed)
                model = (
                    fit_router(a, train, "QA", seed, dest, "inverse")
                    if mode == "IQ"
                    else fit_safe(a, train, mode, seed, dest)
                )
                values.append(predict(model, a, test))
                done += 1
                elapsed = time.perf_counter() - start
                print(
                    f"FIT {done}/{total} family={family} mode={mode} seed={seed} "
                    f"elapsed={elapsed:.1f}s eta={elapsed / done * (total - done):.1f}s",
                    flush=True,
                )
            predictions[mode][test] = np.mean(values, axis=0)
            save_progress(args.output, family, mode, predictions)
    geometry_ok = (
        np.isfinite(a["geometry"])
        & np.isfinite(a["quality"]).all(1)
        & (a["quality"][:, 9] > 0)
        & (a["quality"][:, 11] <= 0.15)
    )
    predictions["geometry_F0"] = np.where(geometry_ok, a["geometry"], a["native"][:, 0])
    predictions["F0"] = a["native"][:, 0]
    masks = {
        "all": np.ones(len(a["target"]), bool),
        "road": a["family"] != "Slider",
        "slider": a["family"] == "Slider",
    }
    report = {
        mode: {name: scored(a, pred, mask) for name, mask in masks.items()}
        for mode, pred in predictions.items()
    }
    selected = select_candidate(report)
    atomic_receipt(
        args.output / "SELECTION.json",
        {
            "selected": selected,
            "scope": config["scope"],
            "report": report,
            "elapsed_seconds": time.perf_counter() - start,
            "config_sha256": file_sha(args.output / "CONFIG.json"),
            "reason": "preregistered grouped nonregression gate"
            if selected
            else "all new controls rejected",
        },
    )
    save_npz(args.output / "OOF_PREDICTIONS.npz", predictions)
    if selected:
        mask = masks["all"]
        final = [
            fit_safe(a, mask, selected, seed, args.output / "final" / str(seed)) for seed in SEEDS
        ]
        save_npz(
            args.output / "FIT_PREDICTIONS.npz",
            {"prediction": np.mean([predict(m, a, mask) for m in final], axis=0)},
        )
        atomic_receipt(
            args.output / "FINAL_FREEZE.json",
            {
                "selected": selected,
                "selection_sha256": file_sha(args.output / "SELECTION.json"),
                "checkpoints": {
                    str(seed): file_sha(args.output / "final" / str(seed) / "checkpoint.pt")
                    for seed in SEEDS
                },
            },
        )
    print(f"COMPLETE selected={selected} seconds={time.perf_counter() - start:.1f}", flush=True)


if __name__ == "__main__":
    main()
