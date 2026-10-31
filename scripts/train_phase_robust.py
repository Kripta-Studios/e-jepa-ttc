"""Fixed-budget grouped quality-fusion campaign; reserves never select weights."""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_campaign.data import read
from operational.phase_campaign.evaluation import metrics, paired_group_bootstrap
from operational.phase_robust.model import QualityFusion, fit_normalization


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {k: source[k] for k in source.files}


def population(diagnosis: Path, campaign: Path) -> dict[str, np.ndarray]:
    """Align existing exposed data; original eight overlap recordings stay out."""
    road, slider = load_arrays(diagnosis / "road.npz"), load_arrays(diagnosis / "slider500.npz")
    folds = read(campaign / "F1_grouped" / "CONFIG.json")["folds"]
    families = {g: f["family"] for f in folds for g in f["test"]}
    mask = np.isin(road["group"], list(families))
    return {
        "native": np.concatenate(
            (
                np.stack([road[k][mask] for k in ("F0", "RGB_context", "event_context")], 1),
                np.stack([slider[k] for k in ("F0", "RGB", "event")], 1),
            )
        ).astype(np.float32),
        "target": np.concatenate((road["target"][mask], slider["target"])).astype(np.float32),
        "geometry": np.concatenate((road["geometry"][mask], slider["geometry"])).astype(np.float32),
        "quality": np.concatenate((road["quality"][mask], slider["quality"])).astype(np.float32),
        "group": np.concatenate(
            (road["group"][mask], np.repeat("Slider-500", len(slider["target"])))
        ),
        "family": np.array(
            [families[g] for g in road["group"][mask]] + ["Slider"] * len(slider["target"])
        ),
        "token": np.concatenate(
            (
                road["sample_token"][mask],
                np.array([f"Slider-500:{i}" for i in range(len(slider["target"]))]),
            )
        ),
    }


def fit(
    a: dict[str, np.ndarray], mask: np.ndarray, mode: str, seed: int, dest: Path
) -> QualityFusion:
    """Fixed 200 full-batch AdamW updates; every normalization sees only fit rows."""
    torch.manual_seed(seed)
    model = QualityFusion(mode)
    native, geo, quality, target = [
        torch.from_numpy(a[k][mask]) for k in ("native", "geometry", "quality", "target")
    ]
    valid = torch.isfinite(target) & (torch.isfinite(native[:, 0]) | torch.isfinite(geo))
    # Train loss cannot learn from queries where no estimate is possible.
    native, geo, quality, target = [x[valid] for x in (native, geo, quality, target)]
    families = a["family"][mask][valid.numpy()]
    weights = np.array([1 / np.sum(families == f) for f in families], np.float32)
    weights /= weights.sum()
    weight = torch.from_numpy(weights)
    center, scale = fit_normalization(model.inputs(native, quality).detach().numpy())
    model.center.copy_(torch.from_numpy(center))
    model.scale.copy_(torch.from_numpy(scale))
    seconds_scale = float(torch.median(target.abs()).clamp(min=0.2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    history = []
    start = time.perf_counter()
    for epoch in range(200):
        pred = model(native, geo, quality)
        error = (pred - target).abs()
        loss = (weight * (error / target.abs().clamp(min=0.2) + error / seconds_scale)).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 20 == 0 or epoch == 199:
            history.append({"epoch": epoch + 1, "loss": float(loss.detach())})
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "mode": mode, "seed": seed}, dest / "checkpoint.pt"
    )
    atomic_receipt(
        dest / "COMPLETE.json",
        {
            "seed": seed,
            "mode": mode,
            "rows": len(target),
            "fit_groups": sorted(set(a["group"][mask])),
            "epochs": 200,
            "loss": history,
            "seconds_scale_train_only": seconds_scale,
            "duration_seconds": time.perf_counter() - start,
            "checkpoint_sha256": file_sha(dest / "checkpoint.pt"),
        },
    )
    return model.eval()


def predict(model: QualityFusion, a: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        return model(
            *(torch.from_numpy(a[k][mask]) for k in ("native", "geometry", "quality"))
        ).numpy()


def scored(a: dict[str, np.ndarray], pred: np.ndarray, mask: np.ndarray) -> dict:
    result = metrics(a["target"][mask], pred[mask])
    by_group = {
        g: metrics(a["target"][mask & (a["group"] == g)], pred[mask & (a["group"] == g)])
        for g in np.unique(a["group"][mask])
    }
    result["macro_record_relative_error_pct"] = float(
        np.mean([r["relative_absolute_error_pct"] for r in by_group.values()])
    )
    result["macro_record_mae_seconds"] = float(
        np.mean([r["mae_seconds"] for r in by_group.values()])
    )
    result["by_record"] = by_group
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("diagnosis", "campaign", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    a = population(args.diagnosis, args.campaign)
    development = ~np.isin(a["family"], ["CCRm", "CPNAO"])
    calibration, confirmation = a["family"] == "CPNAO", a["family"] == "CCRm"
    config = {
        "schema": "phase_robust_quality_v1",
        "start_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {
            str(p): file_sha(p)
            for p in [Path(__file__), *Path("operational/phase_robust").glob("*.py")]
        },
        "inputs": {str(p): file_sha(p) for p in args.diagnosis.glob("*.npz")},
        "torch": torch.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
        "device": "cpu",
        "seeds": [7, 13, 21],
        "epochs": 200,
        "optimizer": "AdamW",
        "lr": 0.01,
        "weight_decay": 0.01,
        "loss": (
            "family-balanced absolute_relative_error(floor .2s) + "
            "absolute_error/train_median_abs_TTC"
        ),
        "selection": (
            "CV road macro RAE and macro MAE both below F0; Slider RAE and MAE "
            "both below F0; select lowest road macro RAE among passing "
            "controls; otherwise retain F0"
        ),
        "controls": (
            "quality23->16->1 versus native3->80->1; 401 parameters each; BOTH outputs use geometry"
        ),
        "development_groups": sorted(set(a["group"][development])),
        "calibration_groups": sorted(set(a["group"][calibration])),
        "confirmation_groups": sorted(set(a["group"][confirmation])),
        "exposure": (
            "All datasets historically exposed, including current geometry "
            "diagnostic. Reserves exclude fitting/selection in this gate "
            "experiment only."
        ),
        "upstream": "original F0 frozen, no all-data F1 predictions used as model inputs",
        "source_protection": (
            "exact F0 fallback for absent bbox history; no claim about source "
            "eAP full-history behavior"
        ),
        "history_us": 650000,
        "geometry_clip_seconds": [-60, 60],
        "external_reserve": (
            "public Slider750/1000 if exact sensor/label contract passes; FCWD "
            "full fusion unavailable without audited RGB mapping"
        ),
    }
    atomic_receipt(args.output / "CONFIG.json", config)
    save_npz(args.output / "POPULATION.npz", a)
    base = a["native"][:, 0]
    road = development & (a["family"] != "Slider")
    slider = development & (a["family"] == "Slider")
    report = {
        "baseline": {"road": scored(a, base, road), "slider": scored(a, base, slider)},
        "controls": {},
    }
    predictions = {"F0": base}
    for mode in ("quality", "native"):
        by_seed = []
        for seed in (7, 13, 21):
            oof = np.full(len(base), np.nan, np.float32)
            for family in np.unique(a["family"][development]):
                test = development & (a["family"] == family)
                train = development & ~test
                model = fit(
                    a, train, mode, seed, args.output / "cv" / mode / str(seed) / str(family)
                )
                oof[test] = predict(model, a, test)
            by_seed.append(oof)
            predictions[f"{mode}_{seed}"] = oof
            print(f"CV {mode} seed {seed} complete", flush=True)
        ensemble = np.mean(by_seed, axis=0)
        predictions[mode] = ensemble
        report["controls"][mode] = {
            "road": scored(a, ensemble, road),
            "slider": scored(a, ensemble, slider),
            "seeds": [
                {"road": scored(a, p, road), "slider": scored(a, p, slider)} for p in by_seed
            ],
        }
    passing = []
    for mode, result in report["controls"].items():
        if all(
            result["road"][k] < report["baseline"]["road"][k]
            for k in ("macro_record_relative_error_pct", "macro_record_mae_seconds")
        ) and all(
            result["slider"][k] < report["baseline"]["slider"][k]
            for k in ("relative_absolute_error_pct", "mae_seconds")
        ):
            passing.append(mode)
    selected = (
        min(passing, key=lambda m: report["controls"][m]["road"]["macro_record_relative_error_pct"])
        if passing
        else None
    )
    report["selected"] = selected
    save_npz(args.output / "CV_PREDICTIONS.npz", predictions)
    atomic_receipt(args.output / "SELECTION.json", report)
    if selected is None:
        print("No candidate passed the fixed development gates; F0 retained", flush=True)
        return
    # The selected mode and fixed budget are now immutable, before reserve scoring.
    models = [
        fit(a, development, selected, seed, args.output / "final" / str(seed))
        for seed in (7, 13, 21)
    ]
    all_mask = np.ones(len(base), bool)
    final_predictions = np.stack([predict(m, a, all_mask) for m in models])
    final = final_predictions.mean(0)
    save_npz(
        args.output / "FINAL_PREDICTIONS.npz", {"prediction": final, "seeds": final_predictions}
    )
    freeze = {
        "selection_sha256": file_sha(args.output / "SELECTION.json"),
        "config_sha256": file_sha(args.output / "CONFIG.json"),
        "checkpoints": {
            str(s): file_sha(args.output / "final" / str(s) / "checkpoint.pt") for s in (7, 13, 21)
        },
        "prediction_sha256": file_sha(args.output / "FINAL_PREDICTIONS.npz"),
        "selected": selected,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    atomic_receipt(args.output / "FINAL_FREEZE.json", freeze)
    # Separate calibration groups; target labels were never passed to fit().
    residual = np.abs(final[calibration] - a["target"][calibration])
    radius = float(np.quantile(residual[np.isfinite(residual)], 0.8, method="higher"))
    confirmation_report = {
        "F0": scored(a, base, confirmation),
        "candidate": scored(a, final, confirmation),
        "calibration": {
            "groups": config["calibration_groups"],
            "radius_seconds_80": radius,
            "confirmation_coverage_80": float(
                np.mean(np.abs(final[confirmation] - a["target"][confirmation]) <= radius)
            ),
            "claim": (
                "descriptive split-calibration; no finite-sample guarantee with "
                "correlated historically exposed groups"
            ),
        },
        "fit_resubstitution": {"road": scored(a, final, road), "slider": scored(a, final, slider)},
    }
    paired = confirmation & np.isfinite(base) & np.isfinite(final)
    confirmation_report["paired_record_bootstrap_relative_pp"] = paired_group_bootstrap(
        100 * np.abs(base[paired] - a["target"][paired]) / np.abs(a["target"][paired]),
        100 * np.abs(final[paired] - a["target"][paired]) / np.abs(a["target"][paired]),
        a["group"][paired],
    )
    atomic_receipt(args.output / "CONFIRMATION.json", confirmation_report)
    print(f"COMPLETE selected={selected}; confirmation frozen and scored", flush=True)


if __name__ == "__main__":
    main()
