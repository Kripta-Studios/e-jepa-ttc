"""Regenerate compact original-producer parity evidence from a sealed RGB capture."""

import argparse
from pathlib import Path

import numpy as np

from operational.evttc_transfer_audit.common import file_sha
from operational.final_refit.test12 import TRUSTED_INPUTS_SHA256
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.data import read
from scripts.capture_phase_eap_rgb_pair import parity, reference_observations


def main() -> None:
    """Verify all capture arrays and recompute every stored original-parity verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh report path")
    complete = read(args.capture / "COMPLETE.json")
    freeze = read(args.capture / "FREEZE.json")
    results = read(args.capture / "RESULTS.json")
    if (
        complete["status"] != "COMPLETE"
        or complete["freeze_sha256"] != file_sha(args.capture / "FREEZE.json")
        or complete["results_sha256"] != file_sha(args.capture / "RESULTS.json")
        or freeze["baseline_sha256"] != file_sha(args.baseline_audit / "COMPLETE.json")
        or file_sha(args.inputs) != TRUSTED_INPUTS_SHA256
        or set(results) != set(freeze["selected"])
        or len(results) != complete["observations"]
        or complete["targets_read"]
        or complete["candidate_evaluated"]
    ):
        raise ValueError("Capture seal, identities, or scope differ")
    reference = reference_observations(
        read(args.baseline_audit / "COMPLETE.json"), read(args.inputs), freeze["selected"]
    )
    maxima = {"raw17": np.zeros(17), "expert3": np.zeros(3)}
    failed = {"raw17": 0, "expert3": 0}
    exact = {"raw17": 0, "expert3": 0}
    for identity, row in results.items():
        path = args.capture / "observations" / f"{identity}.npz"
        if file_sha(path) != row["sha256"]:
            raise ValueError("Captured observation changed")
        with np.load(path, allow_pickle=False) as values:
            for key in maxima:
                actual, expected = values[f"call0.{key}"], reference[identity][key]
                check = parity(actual, expected)
                if check != row["checks"][key]:
                    raise ValueError("Stored parity verdict does not reproduce")
                failed[key] += not check["pass"]
                exact[key] += check["exact"]
                maxima[key] = np.maximum(maxima[key], np.abs(actual - expected)[0])
    failed_rows = sum(not all(c["pass"] for c in r["checks"].values()) for r in results.values())
    if failed_rows != complete["failed_observations"]:
        raise ValueError("Aggregate failure count differs")
    atomic_receipt(
        args.output,
        {
            "schema": "phase_eap_rgb_original_parity_report_v1",
            "capture_complete_sha256": file_sha(args.capture / "COMPLETE.json"),
            "capture": str(args.capture),
            "mode": complete["mode"],
            "observations": len(results),
            "failed_observations": failed_rows,
            "fields": {
                key: {
                    "failed": failed[key],
                    "exact": exact[key],
                    "max_abs_by_column": maxima[key].tolist(),
                    "max_abs": float(maxima[key].max()),
                }
                for key in maxima
            },
            "rtol": freeze["rtol"],
            "atol": freeze["atol"],
            "targets_read": False,
            "candidate_evaluated": False,
            "scope": "Original RGB raw17 and phase experts; these differences are not TTC seconds",
            "full_capture_admitted": complete["mode"] == "pilot" and failed_rows == 0,
            "reporter_sha256": file_sha(Path(__file__)),
        },
    )
    print({"failed_observations": failed_rows, "observations": len(results)})


if __name__ == "__main__":
    main()
