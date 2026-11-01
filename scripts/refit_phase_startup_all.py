"""Fixed all-eligible-data refit; no validation claim for fitted labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_campaign.data import read
from operational.phase_startup.runtime import RouterEnsemble
from scripts.train_phase_robust import load_arrays, scored
from scripts.train_phase_startup import fit_router


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    freeze = read(args.training / "FINAL_FREEZE.json")
    a = load_arrays(args.training / "POPULATION.npz")
    atomic_receipt(
        args.output / "CONFIG.json",
        {
            "parent_freeze_sha256": file_sha(args.training / "FINAL_FREEZE.json"),
            "population_sha256": file_sha(args.training / "POPULATION.npz"),
            "scope": (
                "all 2677 eligible labeled rows; descriptive resubstitution, "
                "no remaining independent test; 8 overlap recordings excluded"
            ),
            "selected": freeze["selected"],
            "coordinate": freeze["coordinate"],
            "seeds": [7, 13, 21],
            "recipe": "inherited 300updates AdamW lr.01 wd.01 family-balanced loss; no retuning",
            "fit_groups": sorted(set(a["group"])),
            "sources": {
                str(x): file_sha(x)
                for x in [
                    Path(__file__),
                    Path("scripts/train_phase_startup.py"),
                    Path("operational/phase_startup/model.py"),
                ]
            },
            "uncertainty": (
                "calibration not valid after refitting its labels; no interval delivered"
            ),
        },
    )
    atomic_receipt(
        args.output / "SELECTION.json",
        {"selected": freeze["selected"], "reason": "fixed inherited selection"},
    )
    mask = np.ones(len(a["target"]), bool)
    models = [
        fit_router(
            a, mask, freeze["selected"], s, args.output / "final" / str(s), freeze["coordinate"]
        )
        for s in (7, 13, 21)
    ]
    ensemble = RouterEnsemble(models).eval()
    with torch.inference_mode():
        pred = ensemble(
            *(torch.from_numpy(a[k]) for k in ("native", "geometry", "quality"))
        ).numpy()
    save_npz(args.output / "PREDICTIONS.npz", {**a, "prediction": pred})
    atomic_receipt(
        args.output / "FINAL_FREEZE.json",
        {
            "selected": freeze["selected"],
            "coordinate": freeze["coordinate"],
            "config_sha256": file_sha(args.output / "CONFIG.json"),
            "selection_sha256": file_sha(args.output / "SELECTION.json"),
            "checkpoints": {
                str(s): file_sha(args.output / "final" / str(s) / "checkpoint.pt")
                for s in (7, 13, 21)
            },
        },
    )
    atomic_receipt(
        args.output / "REPORT.json",
        {
            "scope": "resubstitution only",
            "metrics": {
                name: {"F0": scored(a, a["native"][:, 0], m), "candidate": scored(a, pred, m)}
                for name, m in {
                    "road24": a["family"] != "Slider",
                    "slider_all": a["family"] == "Slider",
                    "all": mask,
                }.items()
            },
        },
    )
    print("COMPLETE all-data router refit", flush=True)


if __name__ == "__main__":
    main()
