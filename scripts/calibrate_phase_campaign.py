"""Training-only interval scaling; no held-out labels enter the scale estimate."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_adapt_training import load_head
from operational.phase_campaign.data import load_data, read
from operational.phase_campaign.evaluation import phase_interval_metrics
from operational.phase_campaign.protocol import fold_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "campaign", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    values, _, _ = load_data(args.data)
    config = read(args.campaign / "CONFIG.json")
    report = {
        "scope": "training residual scaling; not split conformal; no coverage guarantee",
        "target_coverage": 0.8,
        "seeds": {},
    }
    for seed in config["seeds"]:
        collected = []
        scales = []
        for fold in config["folds"]:
            train, test = fold_indices(values, fold)
            calibration = train[values["domain"][train] == "evttc"]
            root = args.campaign / f"seed_{seed}" / fold["family"]
            receipt = read(root / "fit" / "COMPLETE.json")
            checkpoint = root / "fit" / "checkpoint.pt"
            model = load_head(checkpoint, receipt["checkpoint_sha256"])
            with torch.inference_mode():
                fit = model(
                    torch.from_numpy(values["features"][calibration]),
                    torch.from_numpy(values["rgb_width_logits"][calibration]),
                )
                heldout = model(
                    torch.from_numpy(values["features"][test]),
                    torch.from_numpy(values["rgb_width_logits"][test]),
                )
            truth = values["target_phase"][calibration]
            location = fit["raw_location"].numpy()
            width = np.where(
                truth < location, location - fit["q10"].numpy(), fit["q90"].numpy() - location
            )
            score = np.abs(truth - location) / np.maximum(width, 1e-8)
            factor = float(np.quantile(score, 0.8, method="higher"))
            center = heldout["raw_location"].numpy()
            lower = center - factor * (center - heldout["q10"].numpy())
            upper = center + factor * (heldout["q90"].numpy() - center)
            collected.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "target": values["target_phase"][test],
                    "indices": test,
                }
            )
            scales.append(
                {
                    "family": fold["family"],
                    "factor": factor,
                    "calibration_rows": len(calibration),
                    "checkpoint_sha256": file_sha(checkpoint),
                    "train_recordings": sorted(set(values["group_id"][calibration])),
                }
            )
        merged = {k: np.concatenate([v[k] for v in collected]) for k in collected[0]}
        save_npz(args.output / f"seed_{seed}.npz", merged)
        report["seeds"][str(seed)] = {
            "scales": scales,
            "heldout": phase_interval_metrics(merged["target"], merged["lower"], merged["upper"]),
        }
    atomic_receipt(args.output / "REPORT.json", report)


if __name__ == "__main__":
    main()
