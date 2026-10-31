"""Test one matched grouped router refit after sealed canonical PAIR extraction."""

import argparse
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
from operational.phase_startup.runtime import RouterEnsemble
from operational.phase_timewarp.screen_metrics import compare_screen
from scripts.audit_phase_eap_router_sensitivity import verified_parent
from scripts.report_phase_canonical_population import verified_inputs
from scripts.train_phase_robust import load_arrays
from scripts.train_phase_startup import fit_router

PLAN = Path("docs/research_log_canonical_population2608/CANONICAL_ROUTER_REFIT_PROTOCOL.md")
SEEDS = (7, 13, 21)


def joined_population(source: dict, records: list[dict], config: dict) -> dict:
    """Bind recaptured states to all development rows without admitting benchmark groups."""
    tokens = [str(t) for t in source["token"]]
    lookup = {r["token"]: r for r in records}
    if len(tokens) != len(set(tokens)) or len(records) != len(lookup) or set(tokens) != set(lookup):
        raise ValueError("Development token inventory differs")
    groups = set(source["group"])
    if groups != set(config["fit_groups"]) or groups & set(
        config["excluded_road"] + config["excluded_slider"]
    ):
        raise ValueError("Benchmark exclusion or development groups differ")
    a = {k: source[k].copy() for k in ("token", "group", "family", "target", "geometry", "quality")}
    if not np.isfinite(a["target"]).all():
        raise ValueError("Missing development target")
    for token, group in zip(tokens, source["group"], strict=True):
        if lookup[token]["group"] != group or not lookup[token]["baseline_exact"]:
            raise ValueError("Recaptured group or original identity differs")
    for arm in ("original", "canonical"):
        a[arm] = np.asarray([lookup[t]["results"][arm]["native"] for t in tokens], np.float32)
        if a[arm].shape != (len(tokens), 3):
            raise ValueError("Expected F0/RGB/event native vector")
    for group in groups:
        if len(set(a["family"][a["group"] == group])) != 1:
            raise ValueError("Recording crosses family folds")
    return a


def comparison(a: dict, old: np.ndarray, new: np.ndarray, mask: np.ndarray) -> dict:
    """Apply the existing coverage and macro error rules to both fixed domains."""
    domains = {
        name: compare_screen(a["target"][m], old[m], new[m], a["group"][m])
        for name, m in (
            ("road", mask & (a["group"] != "Slider-500")),
            ("slider", mask & (a["group"] == "Slider-500")),
        )
    }
    return {
        "domains": domains,
        "passes": all(d["nonworsening"] for d in domains.values())
        and any(d["any_strict_improvement"] for d in domains.values()),
    }


def main() -> None:
    """Train paired fixed-budget folds, seal OOF predictions, then assess development only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    for name, default in (
        ("population", "output/phase_timewarp/population2608_v1"),
        ("evaluation", "output/phase_timewarp/population2608_evaluation_v1"),
        ("screen", "output/phase_timewarp/screen149_v1"),
        ("training", "output/phase_submission/training"),
    ):
        parser.add_argument("--" + name, type=Path, default=Path(default))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    verified_inputs(args.population, args.evaluation)
    screen = verified_parent(args.screen, {"FREEZE.json", "PREDICTIONS.json"})
    if screen["queries"] != 149 or not screen["baseline_exact"]:
        raise ValueError("Expected original fixed screen")
    pop_freeze = read(args.population / "FREEZE.json")
    population_path = args.training / "POPULATION.npz"
    if file_sha(population_path) != pop_freeze["inputs"][str(population_path)]:
        raise ValueError("Training population binding differs")
    records = read(args.population / "PREDICTIONS.json")["records"]
    a = joined_population(
        load_arrays(population_path), records, read(args.training / "CONFIG.json")
    )
    screen_tokens = read(args.screen / "FREEZE.json")["tokens"]
    screen_mask = np.isin(a["token"], screen_tokens)
    families = sorted(set(a["family"]))
    if len(a["token"]) != 2608 or len(families) != 8 or int(screen_mask.sum()) != 149:
        raise ValueError("Fixed fold/population counts differ")
    sources = [
        Path(__file__),
        PLAN,
        *[
            Path(p)
            for p in (
                "scripts/train_phase_startup.py",
                "scripts/train_phase_robust.py",
                "scripts/report_phase_canonical_population.py",
                "scripts/audit_phase_eap_router_sensitivity.py",
                "operational/phase_startup/model.py",
                "operational/phase_startup/runtime.py",
                "operational/phase_robust/model.py",
                "operational/phase_timewarp/screen_metrics.py",
            )
        ],
    ]
    inputs = [
        args.population / "COMPLETE.json",
        args.screen / "COMPLETE.json",
        population_path,
        args.training / "CONFIG.json",
    ]
    pins = {str(p): file_sha(p) for p in [*sources, *inputs]}
    freeze = {
        "schema": "phase_canonical_router_refit_v1",
        "sources_and_inputs": pins,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seeds": list(SEEDS),
        "folds": families,
        "steps_per_fit": 300,
        "arms": ["original", "canonical"],
        "fits": 48,
        "threads": 1,
        "device": "cpu",
        "host": platform.node(),
        "python_version": platform.python_version(),
        "start_time": datetime.now(UTC).isoformat(),
        "torch": torch.__version__,
        "optimizer_updates_per_fit": 300,
        "scope": "historically exposed grouped development; no Test12 targets or benchmark fit",
        "promotion_admitted": False,
        "prior_frozen_router_screen_reclassified": False,
    }
    atomic_receipt(args.output / "FREEZE.json", freeze)
    for path in sources:
        target = args.output / "executed_source" / path.resolve().relative_to(Path.cwd())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    predictions = {arm: np.full((3, len(a["token"])), np.nan, np.float32) for arm in freeze["arms"]}
    ensembles = {arm: np.full(len(a["token"]), np.nan, np.float32) for arm in freeze["arms"]}
    started, done = time.perf_counter(), 0
    try:
        for family in families:
            held = a["family"] == family
            fit = ~held
            if set(a["group"][held]) & set(a["group"][fit]):
                raise ValueError("Recording leakage across fold")
            for arm in freeze["arms"]:
                data = {**a, "native": a[arm]}
                models = []
                tensors = [
                    torch.from_numpy(data[k][held]) for k in ("native", "geometry", "quality")
                ]
                for i, seed in enumerate(SEEDS):
                    model = fit_router(
                        data,
                        fit,
                        "QA",
                        seed,
                        args.output / "folds" / family / arm / str(seed),
                        "inverse",
                    )
                    models.append(model)
                    with torch.inference_mode():
                        predictions[arm][i, held] = RouterEnsemble([model])(*tensors).numpy()
                    done += 1
                    elapsed = time.perf_counter() - started
                    print(
                        f"FIT {done}/48 {family}/{arm}/{seed} elapsed={elapsed:.1f}s "
                        f"eta={elapsed / done * (48 - done):.1f}s",
                        flush=True,
                    )
                with torch.inference_mode():
                    ensembles[arm][held] = RouterEnsemble(models)(*tensors).numpy()
        for path, digest in pins.items():
            if file_sha(Path(path)) != digest:
                raise ValueError("Source or parent changed during refit")
        saved = {
            "token": a["token"],
            "group": a["group"],
            **ensembles,
            **{arm + "_seeds": pred for arm, pred in predictions.items()},
        }
        save_npz(args.output / "OOF_PREDICTIONS.npz", saved)
        atomic_receipt(
            args.output / "PREDICTIONS_SEALED.json",
            {
                "sha256": file_sha(args.output / "OOF_PREDICTIONS.npz"),
                "freeze_sha256": file_sha(args.output / "FREEZE.json"),
                "scoring_started": False,
                "fits": done,
            },
        )
        views = {"population2608": np.ones(len(a["token"]), bool), "screen149": screen_mask}
        report: dict = {
            name: comparison(a, ensembles["original"], ensembles["canonical"], mask)
            for name, mask in views.items()
        }
        report["per_seed"] = {
            str(seed): {
                name: comparison(a, predictions["original"][i], predictions["canonical"][i], mask)
                for name, mask in views.items()
            }
            for i, seed in enumerate(SEEDS)
        }
        report["admit_further_validation"] = all(report[name]["passes"] for name in views)
        report["promotion_admitted"] = False
        report["test12_accuracy_evaluated"] = False
        report["prior_frozen_router_screen_reclassified"] = False
        atomic_receipt(args.output / "REPORT.json", report)
        artifacts = {
            str(p.relative_to(args.output)): file_sha(p)
            for p in args.output.rglob("*")
            if p.is_file()
        }
        atomic_receipt(
            args.output / "COMPLETE.json",
            {
                "status": "COMPLETE",
                "artifacts": artifacts,
                "elapsed_seconds": time.perf_counter() - started,
                "fits": done,
            },
        )
        print({k: report[k] for k in ("admit_further_validation", "promotion_admitted")})
    except Exception as exc:
        atomic_receipt(args.output / "FAILED.json", {"error": repr(exc), "completed_fits": done})
        raise


if __name__ == "__main__":
    main()
