"""Verify saved scientific contracts before packaging a completed local campaign."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.data import read
from operational.phase_campaign.evaluation import metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign
    report = read(root / "F1_grouped" / "REPORT.json")
    if report["status"] != "COMPLETE":
        raise ValueError("Incomplete grouped experiment")
    fits = 0
    tokens = None
    for seed in (7, 13, 21):
        for fold in report["config"]["folds"]:
            if set(fold["train"]) & set(fold["test"]):
                raise ValueError("Family leakage")
            fit = root / "F1_grouped" / f"seed_{seed}" / fold["family"] / "fit"
            receipt = read(fit / "COMPLETE.json")
            if file_sha(fit / "checkpoint.pt") != receipt["checkpoint_sha256"]:
                raise ValueError("Changed grouped checkpoint")
            fits += 1
        with np.load(root / "F1_grouped" / f"seed_{seed}" / "heldout_predictions.npz") as p:
            current = p["sample_token"]
            if len(set(current)) != len(current):
                raise ValueError("Duplicate held-out query")
            if tokens is not None:
                np.testing.assert_array_equal(tokens, current)
            tokens = current
            for model in ("F0", "F1"):
                calculated = metrics(p["target"], p[model])
                saved = report["seeds"][str(seed)]["models"][model]
                for key, value in calculated.items():
                    if value != saved[key]:
                        raise ValueError(f"Changed grouped score: {seed}/{model}/{key}")
    external = root / "Slider500"
    seal = read(external / "PREDICTIONS_SEALED.json")
    if seal["sha256"] != file_sha(external / "predictions_sealed.npz"):
        raise ValueError("Changed external predictions")
    if seal["freeze_sha256"] != file_sha(external / "FREEZE.json") or seal["labels_opened"]:
        raise ValueError("External prelabel freeze invalid")
    external_report = read(external / "REPORT.json")
    with np.load(external / "targets.npz") as y, np.load(external / "predictions_sealed.npz") as p:
        for name in p.files:
            if metrics(y["target"], p[name]) != external_report["metrics"][name]:
                raise ValueError("Changed external metric")
    for item in external_report["physical"]:
        np.testing.assert_allclose(item["original"], item["clock_shift"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(item["original"], item["future_append"], rtol=0, atol=0)
    robust = read(root / "robustness" / "REPORT.json")
    failures = [d for d in robust["details"] if d["failure"]]
    clean_failures = [d for d in failures if d["variant"] == "clean"]
    if clean_failures:
        raise ValueError("Clean robustness query did not reproduce saved state")
    summary = read(root / "report" / "SUMMARY.json")
    for source in summary["inputs"].values():
        if file_sha(root / source["path"]) != source["sha256"]:
            raise ValueError("Report source changed")
    atomic_receipt(
        root / "VERIFICATION.json",
        {
            "status": "PASS",
            "grouped_fits": fits,
            "same_heldout_queries_all_seeds": len(tokens),
            "external_prediction_seal": "verified",
            "external_metrics": "recomputed",
            "physical_actual_producer_checks": len(external_report["physical"]),
            "robustness_attempts": len(robust["details"]),
            "robustness_failures_retained": len(failures),
            "clean_state_parity": "verified_by_runner_for_all_clean_queries",
            "claim": "artifact integrity only; no scientific promotion implied",
        },
    )
    print("PASS: grouped fits, population, seals, score recomputation and causal controls")


if __name__ == "__main__":
    main()
