"""Fixed v2 controls: reliability supervision, median seeds, geometry augmentation."""

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
from operational.phase_safe_fusion.runtime import SafeEnsemble, combine_seeds
from operational.phase_safe_fusion.training import gate_risk
from scripts.train_phase_robust import load_arrays, scored
from scripts.train_phase_safe_fusion import SEEDS, predict, save_progress


def fit_supervised(
    a: dict[str, np.ndarray], mask: np.ndarray, augmented: dict, mode: str, seed: int, dest: Path
) -> SafeFusion:
    """Train a fixed endpoint with targets restricted to the fit mask."""
    torch.manual_seed(seed)
    model = SafeFusion("log")
    native, geometry, quality, target = [
        torch.from_numpy(a[k][mask]) for k in ("native", "geometry", "quality", "target")
    ]
    aug_geo = torch.from_numpy(augmented["geometry"][:, mask])
    aug_quality = torch.from_numpy(augmented["quality"][:, mask])
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

    def objective(n: torch.Tensor, g: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        error = (model(n, g, q) - target).abs()
        cost = error / target.abs().clamp(min=0.2) + error / seconds_scale
        return (mass * (cost + 0.5 * gate_risk(model, n, g, q, target, seconds_scale))).sum()

    for epoch in range(300):
        loss = objective(native, geometry, quality)
        corrupted = native.clone()
        indices = (torch.rand(len(native)) < 0.25).nonzero().flatten()
        experts = torch.randint(0, 3, (len(indices),))
        factors = torch.tensor([-3.0, -1.0, 0.25, 4.0])[torch.randint(0, 4, (len(indices),))]
        corrupted[indices, experts] *= factors
        g, q = geometry, quality
        if mode == "SGA":
            variant = torch.randint(0, len(aug_geo), (len(native),))
            rows = torch.arange(len(native))
            g, q = aug_geo[variant, rows], aug_quality[variant, rows]
        loss = 0.5 * loss + 0.5 * objective(corrupted, g, q)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if epoch % 30 == 0 or epoch == 299:
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
            "epochs": 300,
            "rows": len(target),
            "fit_groups": sorted(set(a["group"][mask])),
            "loss": history,
            "seconds": time.perf_counter() - start,
            "checkpoint_sha256": file_sha(dest / "checkpoint.pt"),
            "seconds_scale_train_only": float(seconds_scale),
        },
    )
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("population", "original-config", "augmentations", "baseline", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.output.exists():
        raise ValueError("Use fresh output directory")
    a = load_arrays(args.population)
    augmented = load_arrays(args.augmentations)
    if not np.array_equal(a["token"], augmented["token"]):
        raise ValueError("Augmented geometry row identity differs")
    original = read(args.original_config)
    if set(a["group"]) & set(original["excluded_road"] + original["excluded_slider"]):
        raise ValueError("Benchmark labels in fit population")
    if sorted(set(a["group"])) != original["fit_groups"] or len(a["target"]) != 2608:
        raise ValueError("Eligible population differs")
    baseline_config = read(args.baseline / "CONFIG.json")
    if file_sha(args.population) not in baseline_config["inputs"].values():
        raise ValueError("Baseline uses a different population")
    baseline = read(args.baseline / "SELECTION.json")["report"]["IQ"]
    modes = ("SG", "SGA")
    sources = [
        Path(__file__),
        *Path("operational/phase_safe_fusion").glob("*.py"),
        Path("scripts/train_phase_safe_fusion.py"),
        Path("scripts/train_phase_startup.py"),
        Path("docs/superpowers/plans/2026-10-31-phase-sign-safe-fusion.md"),
    ]
    config = {
        "schema": "phase_sign_safe_supervised_v2",
        "started_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {str(p): file_sha(p) for p in sources},
        "inputs": {
            str(p): file_sha(p)
            for p in (
                args.population,
                args.original_config,
                args.augmentations,
                args.baseline / "SELECTION.json",
            )
        },
        "seeds": list(SEEDS),
        "modes": modes,
        "epochs": 300,
        "gate_loss_weight": 0.5,
        "ensemble": "median3 with exact history fallback",
        "device": "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
        "scope": "historically exposed grouped development; no benchmark-dependent selection",
    }
    atomic_receipt(args.output / "CONFIG.json", config)
    for path in sources:
        dst = args.output / "executed_source" / path.resolve().relative_to(Path.cwd())
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dst)
    seed_predictions = {mode: np.full((3, len(a["target"])), np.nan, np.float32) for mode in modes}
    predictions = {mode: np.full(len(a["target"]), np.nan, np.float32) for mode in modes}
    start = time.perf_counter()
    done = 0
    for family in sorted(set(a["family"])):
        test = a["family"] == family
        train = ~test
        if set(a["group"][train]) & set(a["group"][test]):
            raise ValueError("Acquisition leakage")
        for mode in modes:
            for i, seed in enumerate(SEEDS):
                model = fit_supervised(
                    a,
                    train,
                    augmented,
                    mode,
                    seed,
                    args.output / "folds" / family / mode / str(seed),
                )
                seed_predictions[mode][i, test] = predict(model, a, test)
                done += 1
                elapsed = time.perf_counter() - start
                print(
                    f"FIT {done}/48 family={family} mode={mode} seed={seed} "
                    f"elapsed={elapsed:.1f}s eta={elapsed / done * (48 - done):.1f}s",
                    flush=True,
                )
            predictions[mode][test] = combine_seeds(
                torch.from_numpy(seed_predictions[mode][:, test]),
                torch.from_numpy(a["native"][test]),
                torch.from_numpy(a["quality"][test]),
            ).numpy()
            save_progress(args.output, family, mode, {mode: predictions[mode]})
    masks = {
        "all": np.ones(len(a["target"]), bool),
        "road": a["family"] != "Slider",
        "slider": a["family"] == "Slider",
    }
    report = {
        mode: {name: scored(a, pred, mask) for name, mask in masks.items()}
        for mode, pred in predictions.items()
    }
    keys = ("macro_record_relative_error_pct", "macro_record_mae_seconds")
    passing = [
        m
        for m in modes
        if all(
            report[m][d]["coverage"] == 1.0 and all(report[m][d][k] <= baseline[d][k] for k in keys)
            for d in ("road", "slider")
        )
    ]
    selected = min(passing, key=lambda m: report[m]["all"][keys[0]]) if passing else None
    atomic_receipt(
        args.output / "SELECTION.json",
        {
            "selected": selected,
            "report": report,
            "baseline_IQ": baseline,
            "scope": config["scope"],
            "config_sha256": file_sha(args.output / "CONFIG.json"),
            "reason": "fixed nonregression gates" if selected else "no variant passes all gates",
            "seed_metrics": {
                m: [scored(a, p, masks["all"]) for p in seed_predictions[m]] for m in modes
            },
            "seed_sign_disagreement": {
                m: int((np.ptp(np.sign(p), axis=0) > 0).sum()) for m, p in seed_predictions.items()
            },
        },
    )
    save_npz(args.output / "OOF_PREDICTIONS.npz", predictions)
    save_npz(args.output / "SEED_PREDICTIONS.npz", seed_predictions)
    if selected:
        models = [
            fit_supervised(
                a, masks["all"], augmented, selected, seed, args.output / "final" / str(seed)
            )
            for seed in SEEDS
        ]
        ensemble = SafeEnsemble(models).eval()
        scripted = torch.jit.script(ensemble)
        scripted.save(str(args.output / "model.ts"))
        inputs = [torch.from_numpy(a[k]) for k in ("native", "geometry", "quality")]
        with torch.inference_mode():
            output = ensemble(*inputs)
            torch.testing.assert_close(output, scripted(*inputs), rtol=0, atol=0)
        save_npz(args.output / "FIT_PREDICTIONS.npz", {"prediction": output.numpy()})
        atomic_receipt(
            args.output / "FINAL_FREEZE.json",
            {
                "selected": selected,
                "selection_sha256": file_sha(args.output / "SELECTION.json"),
                "model_sha256": file_sha(args.output / "model.ts"),
                "checkpoints": {
                    str(s): file_sha(args.output / "final" / str(s) / "checkpoint.pt")
                    for s in SEEDS
                },
                "source_preservation": "pending active eAP gate",
            },
        )
    print(f"COMPLETE selected={selected} seconds={time.perf_counter() - start:.1f}", flush=True)


if __name__ == "__main__":
    main()
