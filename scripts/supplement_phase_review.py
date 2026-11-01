"""Complete comparator audit details without replacing sealed experiment evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.data import read
from operational.phase_campaign.review_controls import shared_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign
    diagnosis = read(root / "diagnosis" / "REPORT.json")
    with np.load(root / "diagnosis" / "predictions.npz") as source:
        eligible = ~np.isin(source["group"], diagnosis["excluded_recordings"])
        names = list(diagnosis["models"])
        paired = {
            name: shared_scores(
                source["target"][eligible],
                {"F0": source["F0"][eligible], name: source[name][eligible]},
            )
            for name in names
            if name != "F0"
        }
    geometry = read(root / "geometry" / "REPORT.json")
    missing = []
    with np.load(root / "geometry" / "predictions.npz") as source:
        for index, prediction in zip(source["indices"], source["STRTTC_linear"], strict=True):
            detail = next(r for r in geometry["details"] if r["query"] == int(index))
            if not np.isfinite(prediction) and "STRTTC_linear" not in detail["failures"]:
                if "all" in detail["failures"]:
                    continue
                missing.append(
                    {
                        "query": int(index),
                        "reason": "nonpositive_or_nonfinite_endpoint_inverse_ttc",
                        "basis": "inferred from the only non-exception NaN branch of executed code",
                        "prediction_changed": False,
                    }
                )
    payload = {
        "road_shared_eligible24": paired,
        "strttc_missing_reason_addendum": missing,
        "original_predictions_unchanged": True,
        "sources": {
            str(p.relative_to(root)): file_sha(p)
            for p in (
                root / "diagnosis" / "predictions.npz",
                root / "diagnosis" / "REPORT.json",
                root / "geometry" / "predictions.npz",
                root / "geometry" / "REPORT.json",
            )
        },
    }
    atomic_receipt(root / "REVIEW_SUPPLEMENT.json", payload)
    print(f"Saved {len(paired)} paired comparisons and {len(missing)} missing-reason addenda")


if __name__ == "__main__":
    main()
