"""Join development targets only after verifying all fixed-screen prediction seals."""

import argparse
from pathlib import Path

import numpy as np

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_campaign.data import read
from operational.phase_timewarp.screen_metrics import compare_screen
from scripts.build_phase_native_context_augmentations import verify_artifacts


def main() -> None:
    """Report both routes, all missing rows and fixed continuation decisions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument(
        "--population", type=Path, default=Path("output/phase_submission/training/POPULATION.npz")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh evaluation directory")
    complete, freeze = read(args.screen / "COMPLETE.json"), read(args.screen / "FREEZE.json")
    if (
        complete["status"] != "COMPLETE"
        or not complete["baseline_exact"]
        or complete["targets_read"]
        or complete["optimizer_updates"] != 0
        or complete["queries"] != 149
        or file_sha(args.screen / "FREEZE.json") != complete["freeze_sha256"]
    ):
        raise ValueError("Require a complete target-blind identity-verified screen")
    verify_artifacts(args.screen, complete["artifacts"])
    if file_sha(args.population) != freeze["inputs"][str(args.population)]:
        raise ValueError("Target population binding differs")
    records = read(args.screen / "PREDICTIONS.json")["records"]
    if (
        [r["token"] for r in records] != freeze["tokens"]
        or [r["group"] for r in records] != freeze["groups"]
        or len(set(freeze["tokens"])) != 149
    ):
        raise ValueError("Prediction identity/order differs")
    # This is the first target access; all prediction artifacts already passed hashes.
    with np.load(args.population, allow_pickle=False) as source:
        tokens, groups, target = source["token"], source["group"], source["target"]
    index = {str(token): i for i, token in enumerate(tokens)}
    if len(index) != len(tokens):
        raise ValueError("Duplicate target token")
    indices = [index[r["token"]] for r in records]
    selected_target, selected_groups = target[indices].astype(np.float64), groups[indices]
    if list(selected_groups) != freeze["groups"]:
        raise ValueError("Target group identity differs")
    arrays = {
        "token": np.asarray(freeze["tokens"]),
        "group": selected_groups,
        "target": selected_target,
    }
    report = {
        "scope": "historically exposed development screen, not official or confirmation",
        "screen_complete_sha256": file_sha(args.screen / "COMPLETE.json"),
        "population_sha256": file_sha(args.population),
        "evaluator_sources": {
            str(p): file_sha(p)
            for p in (Path(__file__), Path("operational/phase_timewarp/screen_metrics.py"))
        },
        "routes": {},
        "producer_counts": {
            name: {
                "accepted_observations": {
                    modality: sum(
                        r["results"][name]["accepted_observations"][modality] for r in records
                    )
                    for modality in ("event", "rgb")
                },
                "rejected_calls": sum(len(r["results"][name]["rejected_calls"]) for r in records),
            }
            for name in ("original", "canonical")
        },
    }
    for route in ("F0", "router"):
        for name in ("original", "canonical"):
            values = [
                r["results"][name]["native"][0] if route == "F0" else r["results"][name]["router"]
                for r in records
            ]
            arrays[route + "_" + name] = np.asarray(
                [np.nan if v is None else v for v in values], np.float64
            )
        domains = {}
        for domain, mask in (
            ("road", selected_groups != "Slider-500"),
            ("slider", selected_groups == "Slider-500"),
        ):
            domains[domain] = compare_screen(
                selected_target[mask],
                arrays[route + "_original"][mask],
                arrays[route + "_canonical"][mask],
                selected_groups[mask],
            )
        report["routes"][route] = {
            "domains": domains,
            "continue_full_population": all(v["nonworsening"] for v in domains.values())
            and any(v["any_strict_improvement"] for v in domains.values()),
        }
    save_npz(args.output / "SCORED.npz", arrays)
    report["scored_sha256"] = file_sha(args.output / "SCORED.npz")
    atomic_receipt(args.output / "REPORT.json", report)
    print({route: data["continue_full_population"] for route, data in report["routes"].items()})


if __name__ == "__main__":
    main()
