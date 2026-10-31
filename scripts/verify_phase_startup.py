"""Check exact feature reconstruction, deterministic refit and cached raw replay."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.data import read
from operational.phase_campaign.evaluation import metrics
from operational.phase_startup.runtime import load_router
from scripts.train_phase_robust import load_arrays
from scripts.train_phase_startup import build_population, fit_router


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "previous", "data", "campaign"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    training = args.root / "training_rate"
    a = load_arrays(training / "POPULATION.npz")
    rebuilt = build_population(args.previous, args.data, args.campaign)
    for name in a:
        np.testing.assert_array_equal(a[name], rebuilt[name])
    mask = ~np.isin(a["family"], ["CCRm", "CPNAO", "CPLA"])
    replica = fit_router(a, mask, "QA", 7, args.root / "verification/reproduced_fold", "inverse")
    expected = torch.load(
        training / "cv/QA/7/CPLA/checkpoint.pt", map_location="cpu", weights_only=True
    )["state_dict"]
    for name, tensor in expected.items():
        if not torch.equal(tensor, replica.state_dict()[name]):
            raise AssertionError(f"Non-deterministic fit: {name}")
    config = read(training / "CONFIG.json")
    for path, digest in config["source_hashes"].items():
        archived = training / "executed_source" / Path(path).resolve().relative_to(Path.cwd())
        if file_sha(archived) != digest:
            raise AssertionError("Executed source snapshot differs")
    raw = load_arrays(args.root / "raw_robustness/PREDICTIONS.npz")
    tokens = {str(t): i for i, t in enumerate(a["token"])}
    indices = [tokens[str(t)] for t in raw["tokens"]]
    router = load_router(training)
    maximum = 0.0
    with torch.inference_mode():
        for key in raw:
            if not key.endswith("_native"):
                continue
            current = router(
                torch.tensor(raw[key], dtype=torch.float32),
                torch.from_numpy(a["geometry"][indices]),
                torch.from_numpy(a["quality"][indices]),
            ).numpy()
            cached = raw[key.replace("_native", "_candidate")]
            np.testing.assert_allclose(current, cached, rtol=1e-6, atol=1e-6)
            maximum = max(maximum, float(np.max(np.abs(current - cached))))
    final = load_arrays(training / "FINAL_PREDICTIONS.npz")["prediction"]
    native = read(args.root / "NATIVE_REPLAY.json")["rows"][0]
    ni = tokens["evttc:" + native["query_id"]]
    slider = read(args.root / "SLIDER_STARTUP_REPLAY.json")["rows"]
    si = np.flatnonzero(a["group"] == "Slider-750")[: len(slider)]
    difference = float(np.max(np.abs(final[si] - [r["ttc_seconds"] for r in slider])))
    startup = {}
    for group in ("Slider-500", "Slider-750", "Slider-1000"):
        selected = np.flatnonzero(a["group"] == group)[:2]
        startup[group] = {
            k: metrics(a["target"][selected], v[selected])
            for k, v in {
                "F0": a["native"][:, 0],
                "previous_quality": a["previous_quality_frozen"],
                "candidate_fit": final,
            }.items()
        }
    atomic_receipt(
        args.root / "verification/REPRODUCIBILITY.json",
        {
            "population_arrays_exact": list(a),
            "rows": len(a["target"]),
            "reproduced_fit": "QA inverse seed7 CPLA held out; all tensors bitwise equal",
            "executed_source_snapshots_valid": True,
            "raw_cache_current_router_max_abs_seconds": maximum,
            "native_fresh_replay_abs_seconds": abs(float(final[ni]) - native["ttc_seconds"]),
            "slider_fresh_replay_max_abs_seconds": difference,
            "sensor_replay_scope": (
                "fresh CLI disables TF32; old Slider cache inherited configure_cuda TF32. "
                "Measured difference, not bitwise sensor parity"
            ),
            "startup_first_two": startup,
            "startup_scope": "historically exposed and fitted Slider rows; diagnostic only",
        },
    )
    print("COMPLETE exact data/refit reproduction and current raw-cache replay", flush=True)


if __name__ == "__main__":
    main()
