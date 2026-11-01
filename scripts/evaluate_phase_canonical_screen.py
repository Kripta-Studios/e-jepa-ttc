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
    parser.add_argument("--expected-queries", type=int, choices=(149, 2608), default=149)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh evaluation directory")
    complete, freeze = read(args.screen / "COMPLETE.json"), read(args.screen / "FREEZE.json")
    if (
        complete["status"] != "COMPLETE"
        or not complete["baseline_exact"]
        or complete["targets_read"]
        or complete["optimizer_updates"] != 0
        or complete["queries"] != args.expected_queries
        or freeze.get("schema")
        != (
            "phase_canonical_screen149_v1"
            if args.expected_queries == 149
            else "phase_canonical_population2608_v1"
        )
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
        or len(set(freeze["tokens"])) != args.expected_queries
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
        "population_size": args.expected_queries,
        "primary_route": "F0" if args.expected_queries == 2608 else "separate_F0_and_router",
        "router_prior_screen": "failed; diagnostic only" if args.expected_queries == 2608 else None,
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
        passes = all(v["nonworsening"] for v in domains.values()) and any(
            v["any_strict_improvement"] for v in domains.values()
        )
        report["routes"][route] = {
            "domains": domains,
            "continue_full_population": passes if args.expected_queries == 149 else False,
            "passes_fixed_criteria": passes,
            "eligible_for_eap_validation": args.expected_queries == 2608
            and route == "F0"
            and passes,
        }
        if args.expected_queries == 2608:
            report["routes"][route]["road_bootstrap"] = group_bootstrap(domains["road"])
    save_npz(args.output / "SCORED.npz", arrays)
    report["scored_sha256"] = file_sha(args.output / "SCORED.npz")
    atomic_receipt(args.output / "REPORT.json", report)
    print({route: data["passes_fixed_criteria"] for route, data in report["routes"].items()})


def group_bootstrap(domain: dict) -> dict:
    """Describe paired changes by resampling whole recordings, never individual windows."""
    rows = list(domain["by_recording"].values())
    if len(rows) < 2 or any(row["paired"] == 0 for row in rows):
        return {"available": False, "reason": "need at least two fully represented recordings"}
    rng = np.random.default_rng(7)
    indices = rng.integers(0, len(rows), size=(10000, len(rows)))
    result: dict = {
        "available": True,
        "seed": 7,
        "resamples": 10000,
        "recordings": len(rows),
        "difference": "canonical minus original; negative is better",
        "scope": "descriptive historically exposed development, not independent confirmation",
    }
    for metric in ("mae_seconds", "relative_error_pct"):
        changes = np.asarray([r["canonical"][metric] - r["original"][metric] for r in rows])
        distribution = changes[indices].mean(1)
        result[metric] = {
            "mean_change": float(changes.mean()),
            "ci95": np.quantile(distribution, [0.025, 0.975]).tolist(),
        }
    return result


if __name__ == "__main__":
    main()
