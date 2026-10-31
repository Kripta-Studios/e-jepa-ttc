"""Fixed startup-router experiment with historically exposed grouped controls."""

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
from operational.phase_robust.data import histories
from operational.phase_robust.model import fit_normalization
from operational.phase_robust.runtime import load_fusion
from operational.phase_startup.geometry import startup_geometry
from operational.phase_startup.model import ExpertRouter
from scripts.train_phase_robust import load_arrays, scored


def build_population(previous: Path, data: Path, campaign: Path) -> dict[str, np.ndarray]:
    """Use exact audited sensor clocks; geometry never receives target columns."""
    old = load_arrays(previous / "all_data_refit/PREDICTIONS.npz")
    a = {k: old[k] for k in ("native", "target", "group", "family", "token")}
    population = read(data / "POPULATION.json")["rows"]
    metadata = {"evttc:" + r["query_id"]: r for r in population}
    tracks = histories(population)
    slider_tracks = {}
    for speed in (500, 750, 1000):
        if speed == 500:
            box_path = campaign / "Slider500/inputs/Slider-500/bbox.csv"
            clocks = np.array(
                [r["anchor_us"] for r in read(campaign / "Slider500/FREEZE.json")["queries"]]
            )
        else:
            box_path = previous / f"external_contract/{speed}_bbox.csv"
            clocks = load_arrays(previous / f"external/Slider-{speed}/PREDICTIONS.npz")["clocks"]
        raw = np.loadtxt(box_path, delimiter=",")
        if len(raw) != len(clocks) or np.max(np.abs(raw[:, 1] * 1e6 - clocks)) > 2:
            raise ValueError("Slider box/sensor clocks differ")
        slider_tracks[f"Slider-{speed}"] = raw[:, 2:6], clocks
    geometry, clocks_out = [], []
    for token in a["token"]:
        if str(token).startswith("evttc:"):
            row = metadata[str(token)]
            boxes, clocks = tracks[row["sequence_id"]]
            clock = row["anchor_us"]
        else:
            group, index = str(token).split(":")
            boxes, clocks = slider_tracks[group]
            clock = int(clocks[int(index)])
        geometry.append(startup_geometry(boxes, clocks, int(clock)))
        clocks_out.append(clock)
    a["geometry"] = np.array(
        [g["ttc"] if g["valid"] else np.nan for g in geometry], np.float32
    ).clip(-60, 60)
    a["quality"] = np.array([g["quality"] for g in geometry], np.float32)
    a["clocks"] = np.array(clocks_out, np.int64)
    a["geometry_reason"] = np.array([g["reason"] for g in geometry])
    with torch.inference_mode():
        prior = load_fusion(previous / "training")
        a["previous_quality_frozen"] = prior(
            *(torch.from_numpy(old[k]) for k in ("native", "geometry", "quality"))
        ).numpy()
    old_pop = load_arrays(previous / "training/POPULATION.npz")
    old_cv = load_arrays(previous / "training/CV_PREDICTIONS.npz")
    lookup = {t: v for t, v in zip(old_pop["token"], old_cv["quality"], strict=True)}
    a["previous_quality_oof"] = np.array([lookup.get(t, np.nan) for t in a["token"]], np.float32)
    return a


def fit_router(
    a: dict[str, np.ndarray],
    mask: np.ndarray,
    mode: str,
    seed: int,
    dest: Path,
    coordinate: str = "seconds",
) -> ExpertRouter:
    """Train only explicitly supplied fit rows; preserve every fixed-budget endpoint."""
    torch.manual_seed(seed)
    model = ExpertRouter("native" if mode == "NAcap" else "quality", coordinate=coordinate)
    native, geo, quality, target = [
        torch.from_numpy(a[k][mask]) for k in ("native", "geometry", "quality", "target")
    ]
    valid = torch.isfinite(target) & (torch.isfinite(native).any(1) | torch.isfinite(geo))
    native, geo, quality, target = [t[valid] for t in (native, geo, quality, target)]
    families = a["family"][mask][valid.numpy()]
    weights = np.array([1 / np.sum(families == f) for f in families], np.float32)
    weights /= weights.sum()
    mass = torch.from_numpy(weights)
    center, scale = fit_normalization(model.inputs(native, geo, quality).detach().numpy())
    model.center.copy_(torch.from_numpy(center))
    model.scale.copy_(torch.from_numpy(scale))
    seconds_scale = target.abs().median().clamp(min=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    history = []
    start = time.perf_counter()

    def objective(prediction: torch.Tensor) -> torch.Tensor:
        error = (prediction - target).abs()
        return (mass * (error / target.abs().clamp(min=0.2) + error / seconds_scale)).sum()

    for epoch in range(300):
        loss = objective(model(native, geo, quality))
        if mode != "Q":
            corrupted = native.clone()
            chosen = torch.rand(len(native)) < 0.25
            indices = chosen.nonzero().flatten()
            experts = torch.randint(0, 3, (len(indices),))
            factors = torch.tensor([-3.0, -1.0, 0.25, 4.0])[torch.randint(0, 4, (len(indices),))]
            corrupted[indices, experts] *= factors
            loss = 0.5 * loss + 0.5 * objective(model(corrupted, geo, quality))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 30 == 0 or epoch == 299:
            history.append({"epoch": epoch + 1, "loss": float(loss.detach())})
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "mode": mode, "seed": seed, "coordinate": coordinate},
        dest / "checkpoint.pt",
    )
    atomic_receipt(
        dest / "COMPLETE.json",
        {
            "mode": mode,
            "coordinate": coordinate,
            "seed": seed,
            "fit_groups": sorted(set(a["group"][mask])),
            "rows": len(target),
            "epochs": 300,
            "loss": history,
            "seconds": time.perf_counter() - start,
            "checkpoint_sha256": file_sha(dest / "checkpoint.pt"),
        },
    )
    return model.eval()


def predict(model: ExpertRouter, a: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        return model(
            *(torch.from_numpy(a[k][mask]) for k in ("native", "geometry", "quality"))
        ).numpy()


def uncertainty_scale(a: dict[str, np.ndarray]) -> np.ndarray:
    """Inference-only dispersion of four available experts in seconds."""
    experts = np.column_stack((a["native"], a["geometry"])).astype(np.float64)
    center = np.nanmedian(experts, axis=1)
    return np.maximum(0.1, np.nanmedian(np.abs(experts - center[:, None]), axis=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "data", "campaign", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--coordinate", choices=("seconds", "inverse"), default="seconds")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    a = build_population(args.previous, args.data, args.campaign)
    development = ~np.isin(a["family"], ["CCRm", "CPNAO"])
    road = development & (a["family"] != "Slider")
    slider = development & (a["family"] == "Slider")
    calibration, confirmation = a["family"] == "CPNAO", a["family"] == "CCRm"
    source_paths = [Path(__file__), *Path("operational/phase_startup").glob("*.py")]
    config = {
        "schema": "phase_startup_router_v1",
        "coordinate": args.coordinate,
        "started_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "host": platform.node(),
            "device": "cpu",
        },
        "source_hashes": {str(p): file_sha(p) for p in source_paths},
        "inputs": {
            str(p): file_sha(p)
            for p in [
                args.previous / "all_data_refit/PREDICTIONS.npz",
                args.data / "POPULATION.json",
                args.campaign / "Slider500/FREEZE.json",
                *args.previous.glob("external_contract/*bbox.csv"),
            ]
        },
        "modes": ["Q", "QA", "NAcap"],
        "seeds": [7, 13, 21],
        "epochs": 300,
        "lr": 0.01,
        "weight_decay": 0.01,
        "development_groups": sorted(set(a["group"][development])),
        "calibration_groups": sorted(set(a["group"][calibration])),
        "confirmation_groups": sorted(set(a["group"][confirmation])),
        "scope": (
            "all data historically exposed; no new independent confirmation; "
            "masks exclude current fit/selection"
        ),
        "augmentation": (
            "QA and NAcap: half clean/half augmented loss,25%rows,one native expert "
            "times random{-3,-1,.25,4};no label change"
        ),
        "selection": (
            "road macro RAE/MAE both below F0 and Slider macro RAE/MAE both below prior "
            "frozen quality; lowest road macro RAE among passing"
        ),
        "interval": (
            "CPNAO: max(constant80%residual quantile,80%scaled residual quantile times "
            "inference expert MAD); predeclared, no test selection"
        ),
    }
    atomic_receipt(args.output / "CONFIG.json", config)
    for p in source_paths:
        relative = p.resolve().relative_to(Path.cwd())
        dest = args.output / "executed_source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    save_npz(args.output / "POPULATION.npz", a)
    baseline = a["native"][:, 0]
    report = {
        "F0": {"road": scored(a, baseline, road), "slider": scored(a, baseline, slider)},
        "prior_quality": {
            "road_oof": scored(a, a["previous_quality_oof"], road),
            "slider_frozen_exposed": scored(a, a["previous_quality_frozen"], slider),
        },
        "controls": {},
    }
    predictions = {"F0": baseline}
    for mode in config["modes"]:
        seeds = []
        for seed in config["seeds"]:
            oof = np.full(len(baseline), np.nan, np.float32)
            for family in np.unique(a["family"][development]):
                test = development & (a["family"] == family)
                model = fit_router(
                    a,
                    development & ~test,
                    mode,
                    seed,
                    args.output / "cv" / mode / str(seed) / str(family),
                    args.coordinate,
                )
                oof[test] = predict(model, a, test)
            seeds.append(oof)
            predictions[f"{mode}_{seed}"] = oof
            print(f"CV {mode} seed{seed} complete", flush=True)
        ensemble = np.mean(seeds, axis=0)
        predictions[mode] = ensemble
        report["controls"][mode] = {
            "road": scored(a, ensemble, road),
            "slider": scored(a, ensemble, slider),
            "seeds": [{"road": scored(a, s, road), "slider": scored(a, s, slider)} for s in seeds],
        }
    passing = [
        m
        for m, r in report["controls"].items()
        if all(
            r["road"][k] < report["F0"]["road"][k]
            for k in ("macro_record_relative_error_pct", "macro_record_mae_seconds")
        )
        and all(
            r["slider"][k] < report["prior_quality"]["slider_frozen_exposed"][k]
            for k in ("macro_record_relative_error_pct", "macro_record_mae_seconds")
        )
    ]
    selected = (
        min(passing, key=lambda m: report["controls"][m]["road"]["macro_record_relative_error_pct"])
        if passing
        else None
    )
    report["selected"] = selected
    save_npz(args.output / "CV_PREDICTIONS.npz", predictions)
    atomic_receipt(args.output / "SELECTION.json", report)
    if selected is None:
        print("No candidate passes; preserve previous weights", flush=True)
        return
    models = [
        fit_router(a, development, selected, s, args.output / "final" / str(s), args.coordinate)
        for s in config["seeds"]
    ]
    all_rows = np.ones(len(baseline), bool)
    final = np.stack([predict(m, a, all_rows) for m in models]).mean(0)
    save_npz(args.output / "FINAL_PREDICTIONS.npz", {"prediction": final})
    atomic_receipt(
        args.output / "FINAL_FREEZE.json",
        {
            "selected": selected,
            "coordinate": args.coordinate,
            "config_sha256": file_sha(args.output / "CONFIG.json"),
            "selection_sha256": file_sha(args.output / "SELECTION.json"),
            "checkpoints": {
                str(s): file_sha(args.output / "final" / str(s) / "checkpoint.pt")
                for s in config["seeds"]
            },
        },
    )
    residual = np.abs(final[calibration] - a["target"][calibration])
    scale = uncertainty_scale(a)
    constant = float(np.quantile(residual, 0.8, method="higher"))
    factor = float(np.quantile(residual / scale[calibration], 0.8, method="higher"))
    radius = np.maximum(constant, factor * scale)
    atomic_receipt(
        args.output / "CONFIRMATION.json",
        {
            "F0": scored(a, baseline, confirmation),
            "previous_quality": scored(a, a["previous_quality_frozen"], confirmation),
            "candidate": scored(a, final, confirmation),
            "calibration": {
                "constant_radius": constant,
                "scaled_factor": factor,
                "groups": config["calibration_groups"],
                "confirmation_coverage80": float(
                    np.mean(
                        np.abs(final[confirmation] - a["target"][confirmation])
                        <= radius[confirmation]
                    )
                ),
                "confirmation_mean_width_seconds": float(np.mean(2 * radius[confirmation])),
                "claim": (
                    "descriptive separate-fit calibration, historically exposed/correlated "
                    "groups, no certified guarantee"
                ),
            },
            "fit_resubstitution": {
                "road": scored(a, final, road),
                "slider": scored(a, final, slider),
            },
        },
    )
    print(f"COMPLETE selected={selected}", flush=True)


if __name__ == "__main__":
    main()
