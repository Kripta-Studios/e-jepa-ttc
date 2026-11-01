"""Validate the final router on frozen source features and cached sensor corruptions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.final_refit.training import PhaseOnlyFinal
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_campaign.data import load_data
from operational.phase_robust.bundle import verified_upstream_paths
from operational.phase_robust.data import histories
from operational.phase_startup.runtime import predict_history
from operational.phase_submission.runtime import load_submission_model
from scripts.evaluate_phase_submission import metric
from scripts.final_train40_transfer import HEAD


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "data", "raw_cache", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load_submission_model(args.bundle)
    paths = verified_upstream_paths(args.bundle / "upstream")
    values, metadata, _ = load_data(args.data)
    baseline = PhaseOnlyFinal().eval()
    baseline.load_state_dict(
        torch.load(paths[HEAD], map_location="cpu", weights_only=False)["model_state_dict"]
    )
    source = values["domain"] == "eap"
    with torch.inference_mode():
        original = baseline(
            torch.from_numpy(values["features"][source]),
            torch.from_numpy(values["rgb_width_logits"][source]),
        )["ttc"]
        fallback = model(
            original[:, None].expand(-1, 3),
            torch.full_like(original, float("nan")),
            torch.zeros(len(original), 20),
        )
    if not len(original) or not torch.equal(original, fallback):
        raise AssertionError("Source-domain absent-history fallback changed")
    tracks = histories(list(metadata.values()))
    cache = np.load(args.raw_cache)
    predictions = {}
    variants = ("clean", "drop_70", "drop_positive", "drop_negative", "jitter_1000", "background_5")
    invariants = 0
    for variant in variants:
        predicted = []
        for token, native in zip(cache["tokens"], cache[variant + "_native"], strict=True):
            row = metadata[str(token)]
            boxes, clocks = tracks[row["sequence_id"]]
            point, _ = predict_history(model, native, boxes, clocks, row["anchor_us"])
            predicted.append(point)
            if variant == "clean":
                shifted, _ = predict_history(
                    model, native, boxes, clocks + 1000000, row["anchor_us"] + 1000000
                )
                future, _ = predict_history(
                    model,
                    native,
                    np.r_[boxes, [[0, 0, 1, 1]]],
                    np.r_[clocks, row["anchor_us"] + 99999999],
                    row["anchor_us"],
                )
                if shifted != point or future != point:
                    raise AssertionError("Causal clock/future invariance failed")
                invariants += 1
        predictions[variant] = np.array(predicted)
    save_npz(
        args.output / "CACHED_ROBUSTNESS.npz",
        {**predictions, "target": cache["target"], "tokens": cache["tokens"]},
    )
    atomic_receipt(
        args.output / "VALIDATION.json",
        {
            "model_sha256": file_sha(args.bundle / "MODEL.json"),
            "source_fallback_rows": len(original),
            "source_fallback_bitwise_exact": True,
            "clock_shift_and_future_append_exact_queries": invariants,
            "raw_cache_sha256": file_sha(args.raw_cache),
            "scope": (
                "diagnostic on previously exposed records; reuse frozen perturbed native outputs"
            ),
            "sensor_robustness": {
                v: {
                    "adapted": metric(cache["target"], predictions[v]),
                    "original": metric(cache["target"], cache[v + "_F0"]),
                }
                for v in variants
            },
        },
    )
    print(
        f"COMPLETE {len(original)} exact fallback rows,144 cached corruptions,24 causal invariants"
    )


if __name__ == "__main__":
    main()
