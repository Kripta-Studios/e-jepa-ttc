"""Fit one disclosed final candidate without direct benchmark-recording labels."""

from __future__ import annotations

import argparse
import io
import shutil
from pathlib import Path

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_campaign.data import read
from operational.phase_robust.bundle import verified_upstream_paths
from operational.phase_startup.runtime import RouterEnsemble, load_router
from operational.phase_submission.protocol import eligible_population
from scripts.train_phase_robust import load_arrays, scored
from scripts.train_phase_startup import fit_router


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("population", "exclusions", "previous_bundle", "output"):
        p.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    excluded = set(read(args.exclusions)["excluded_recordings"])
    a = eligible_population(load_arrays(args.population), excluded)
    if len(a["target"]) != 2608 or len(set(a["group"])) != 25:
        raise ValueError("Expected24eligible road recordings and Slider500:2608rows")
    out = args.output
    config = {
        "model_name": "PHASE-TTC-Adapt-20261031",
        "scope": (
            "adapted research candidate; no direct benchmark-recording fit; "
            "historical exposure remains"
        ),
        "recipe": (
            "previously selected inverse QA,688parameters,300updates AdamW lr.01 wd.01,"
            "unchanged augmentation/loss"
        ),
        "seeds": [7, 13, 21],
        "coordinate": "inverse",
        "selected": "QA",
        "fit_groups": sorted(set(a["group"])),
        "excluded_road": sorted(excluded),
        "excluded_slider": ["Slider-750", "Slider-1000"],
        "rows": len(a["target"]),
        "historical_exposure": (
            "all32road plus all3Slider inspected previously; earlier recipe "
            "selection used exposed Slider labels; cannot claim virgin benchmark"
        ),
        "inputs": {str(path): file_sha(path) for path in (args.population, args.exclusions)},
        "sources": {
            str(path): file_sha(path)
            for path in [
                Path(__file__),
                Path("scripts/train_phase_startup.py"),
                Path("operational/phase_startup/model.py"),
                Path("operational/phase_submission/protocol.py"),
            ]
        },
        "uncertainty": "no calibrated interval; prior calibration labels included in final fit",
    }
    atomic_receipt(out / "training/CONFIG.json", config)
    atomic_receipt(
        out / "training/SELECTION.json",
        {"selected": "QA", "reason": "inherited fixed recipe; no new benchmark-dependent search"},
    )
    save_npz(out / "training/POPULATION.npz", a)
    mask = np.ones(len(a["target"]), bool)
    models = [
        fit_router(a, mask, "QA", s, out / "training/final" / str(s), "inverse")
        for s in (7, 13, 21)
    ]
    model = RouterEnsemble(models).eval()
    inputs = [torch.from_numpy(a[k]) for k in ("native", "geometry", "quality")]
    with torch.inference_mode():
        prediction = model(*inputs).numpy()
    save_npz(out / "training/PREDICTIONS.npz", {"prediction": prediction})
    atomic_receipt(
        out / "training/FINAL_FREEZE.json",
        {
            "selected": "QA",
            "coordinate": "inverse",
            "config_sha256": file_sha(out / "training/CONFIG.json"),
            "selection_sha256": file_sha(out / "training/SELECTION.json"),
            "checkpoints": {
                str(s): file_sha(out / "training/final" / str(s) / "checkpoint.pt")
                for s in (7, 13, 21)
            },
        },
    )
    torch.jit.script(model).save(str(out / "training/router.ts"))
    loaded = torch.jit.load(io.BytesIO((out / "training/router.ts").read_bytes()))
    with torch.inference_mode():
        np.testing.assert_array_equal(prediction, loaded(*inputs).numpy())
        np.testing.assert_array_equal(prediction, load_router(out / "training")(*inputs).numpy())
    bundle = out / "FINAL_MODEL"
    if bundle.exists():
        raise ValueError("Bundle already exists")
    verified_upstream_paths(args.previous_bundle / "upstream")
    shutil.copytree(args.previous_bundle / "upstream", bundle / "upstream")
    for name in ("CONFIG.json", "SELECTION.json", "FINAL_FREEZE.json", "router.ts"):
        (bundle / "router").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / "training" / name, bundle / "router" / name)
    shutil.copytree(out / "training/final", bundle / "router/final")
    atomic_receipt(
        bundle / "MODEL.json",
        {
            "schema": "phase_evttc_submission_model_v1",
            "name": config["model_name"],
            "upstream_sha256": file_sha(bundle / "upstream/BUNDLE.json"),
            "router_freeze_sha256": file_sha(bundle / "router/FINAL_FREEZE.json"),
            "export_sha256": file_sha(bundle / "router/router.ts"),
            "implementation": {
                str(path): file_sha(path) for path in Path("operational/phase_startup").glob("*.py")
            },
            "fit_rows": 2608,
            "fit_recordings": 25,
            "scope": config["scope"],
        },
    )
    atomic_receipt(
        out / "training/REPORT.json",
        {
            "scope": "resubstitution only, no benchmark evaluation performed during fit",
            "fit_metrics": scored(a, prediction, mask),
            "export_exact_rows": len(prediction),
            "model_sha256": file_sha(bundle / "MODEL.json"),
        },
    )
    print("COMPLETE final candidate:2608rows,25recordings,three seeds", flush=True)


if __name__ == "__main__":
    main()
