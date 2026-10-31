"""Fixed all-eligible-data refit after confirmation; scores are resubstitution."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_robust.runtime import FusionEnsemble
from scripts.train_phase_robust import fit, load_arrays, scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("training", "external", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    a = load_arrays(args.training / "POPULATION.npz")
    for speed in (750, 1000):
        path = args.external / f"Slider-{speed}"
        p, y = load_arrays(path / "PREDICTIONS.npz"), load_arrays(path / "TARGETS.npz")
        n = len(y["target"])
        extra = {
            "native": p["native"],
            "target": y["target"],
            "geometry": p["geometry"],
            "quality": p["quality"],
            "group": np.repeat(f"Slider-{speed}", n),
            "family": np.repeat("Slider", n),
            "token": np.array([f"Slider-{speed}:{i}" for i in range(n)]),
        }
        for k in a:
            a[k] = np.concatenate((a[k], extra[k])).astype(a[k].dtype)
    config = {
        "parent_freeze_sha256": file_sha(args.training / "FINAL_FREEZE.json"),
        "scope": "all-data release refit, all metrics descriptive; no held-out "
        "confirmation remains for these weights",
        "unchanged": "quality401 gate; seeds7/13/21;200epochs;AdamW lr.01 wd.01; "
        "family-balanced loss; no search after reserve",
        "fit_groups": sorted(set(a["group"])),
        "rows": len(a["target"]),
        "exclusion": "original eight overlap road recordings remain excluded",
        "inputs": {
            str(p): file_sha(p)
            for p in [
                args.training / "POPULATION.npz",
                *args.external.glob("Slider-*/PREDICTIONS.npz"),
                *args.external.glob("Slider-*/TARGETS.npz"),
            ]
        },
        "sources": {
            str(p): file_sha(p)
            for p in [
                Path(__file__),
                Path("scripts/train_phase_robust.py"),
                Path("operational/phase_robust/model.py"),
            ]
        },
        "default_evaluated_candidate": str(args.training),
        "uncertainty": "previous separate calibration cannot be transferred as a "
        "validated interval after fitting its labels",
    }
    atomic_receipt(args.output / "CONFIG.json", config)
    atomic_receipt(
        args.output / "SELECTION.json",
        {
            "selected": "quality",
            "reason": "fixed selection inherited before external results; no new "
            "architecture or hyperparameter selection",
        },
    )
    mask = np.ones(len(a["target"]), bool)
    models = [fit(a, mask, "quality", s, args.output / "final" / str(s)) for s in (7, 13, 21)]
    ensemble = FusionEnsemble(models).eval()
    with torch.inference_mode():
        pred = ensemble(
            *(torch.from_numpy(a[k]) for k in ("native", "geometry", "quality"))
        ).numpy()
    save_npz(args.output / "PREDICTIONS.npz", {"prediction": pred, **a})
    atomic_receipt(
        args.output / "FINAL_FREEZE.json",
        {
            "config_sha256": file_sha(args.output / "CONFIG.json"),
            "selection_sha256": file_sha(args.output / "SELECTION.json"),
            "selected": "quality",
            "checkpoints": {
                str(s): file_sha(args.output / "final" / str(s) / "checkpoint.pt")
                for s in (7, 13, 21)
            },
        },
    )
    torch.jit.script(ensemble).save(str(args.output / "fusion.ts"))
    atomic_receipt(
        args.output / "REPORT.json",
        {
            "scope": config["scope"],
            "fit_metrics": {
                name: {"F0": scored(a, a["native"][:, 0], m), "candidate": scored(a, pred, m)}
                for name, m in {
                    "road24": a["family"] != "Slider",
                    "slider_all": a["family"] == "Slider",
                    "all": mask,
                }.items()
            },
            "fusion_sha256": file_sha(args.output / "fusion.ts"),
        },
    )
    print(
        f"COMPLETE all-data refit: {len(a['target'])} rows, {len(set(a['group']))} recordings",
        flush=True,
    )


if __name__ == "__main__":
    main()
