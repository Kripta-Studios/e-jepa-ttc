"""Frozen fusion on perturbed raw sensors at one midpoint per eligible recording."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_adapt_runtime import CachedAdaptPredictor
from operational.phase_adapt_training import configure_cuda
from operational.phase_campaign.data import load_data, read
from operational.phase_campaign.evaluation import metrics
from operational.phase_campaign.perturbations import perturb_events
from operational.phase_robust.data import histories
from operational.phase_startup.runtime import load_router, predict_history
from scripts.final_train40_transfer import HEAD, verify_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "candidate", "fusion", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    configure_cuda()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    values, rows, _ = load_data(args.data)
    tracks = histories(list(rows.values()))
    config = read(args.fusion / "CONFIG.json")
    groups = sorted(
        set(
            config["development_groups"]
            + config["calibration_groups"]
            + config["confirmation_groups"]
        )
        - {"Slider-500", "Slider-750", "Slider-1000"}
    )
    indices = []
    for group in groups:
        available = np.flatnonzero(values["group_id"] == group)
        ordered = sorted(available, key=lambda i: rows[str(values["sample_token"][i])]["anchor_us"])
        indices.append(ordered[len(ordered) // 2])
    variants = ("clean", "drop_70", "drop_positive", "drop_negative", "jitter_1000", "background_5")
    atomic_receipt(
        args.output / "FREEZE.json",
        {
            "fusion_freeze_sha256": file_sha(args.fusion / "FINAL_FREEZE.json"),
            "tokens": values["sample_token"][indices].tolist(),
            "variants": variants,
            "selection": (
                "midpoint by sensor clock per eligible recording, no error-based selection"
            ),
            "source": {
                str(p): file_sha(p)
                for p in (Path(__file__), Path("operational/phase_campaign/perturbations.py"))
            },
            "scope": (
                "diagnostic after freeze; includes exposed "
                "fit/calibration/confirmation records; no tuning"
            ),
        },
    )
    fusion = load_router(args.fusion)
    paths, _ = verify_candidate(args.candidate)
    runtime = CachedAdaptPredictor(paths)
    captured = []
    hook = runtime.models[HEAD].register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().cpu())
    )
    output = {f"{v}_{k}": [] for v in variants for k in ("F0", "candidate", "native")}
    for qi, index in enumerate(indices):
        row = {
            k: v for k, v in rows[str(values["sample_token"][index])].items() if k != "ttc_csv_path"
        }
        original = runtime.inputs.iter_window_chunks
        for variant in variants:

            def changed(
                start: int,
                end: int,
                *,
                chunk_events: int = 250000,
                source: Callable = original,
                intervention: str = variant,
            ) -> Iterator[dict[str, np.ndarray]]:
                chunks = list(source(start, end, chunk_events=chunk_events))
                if sum(sum(a.nbytes for a in c.values()) for c in chunks) > 1024**3:
                    raise ValueError("Perturbation window exceeds 1 GiB")
                raw = {
                    k: np.concatenate([c[k] for c in chunks])
                    if chunks
                    else np.empty(0, dtype=np.int64)
                    for k in ("x", "y", "t", "p")
                }
                perturbed = perturb_events(raw, intervention, start, end, 1280, 720)
                for offset in range(0, len(perturbed["t"]), chunk_events):
                    yield {k: v[offset : offset + chunk_events] for k, v in perturbed.items()}

            captured.clear()
            with patch.object(runtime.inputs, "iter_window_chunks", changed):
                result = runtime.predict(row)
            base = result["predictions"]["PHASE_ONLY_TRAIN40_S7"]
            if not captured:
                raise ValueError("Expected full-context query in robustness population")
            features = captured[-1][0]
            native = np.array(
                [base, float(phase_to_ttc(features[161])), float(phase_to_ttc(features[331]))],
                np.float32,
            )
            if variant == "clean":
                np.testing.assert_allclose(
                    features, values["features"][index], rtol=1e-5, atol=1e-5
                )
            boxes, clocks = tracks[row["sequence_id"]]
            pred, _ = predict_history(fusion, native, boxes, clocks, row["anchor_us"])
            output[f"{variant}_native"].append(native)
            output[f"{variant}_F0"].append(base)
            output[f"{variant}_candidate"].append(pred)
        print(f"raw robustness {qi + 1}/{len(indices)}", flush=True)
    hook.remove()
    runtime.close()
    arrays = {k: np.array(v) for k, v in output.items()}
    arrays.update(
        target=phase_to_ttc(torch.from_numpy(values["target_phase"][indices])).numpy(),
        tokens=values["sample_token"][indices],
    )
    save_npz(args.output / "PREDICTIONS.npz", arrays)
    atomic_receipt(
        args.output / "REPORT.json",
        {
            "scores": {
                k: metrics(arrays["target"], arrays[k]) for k in output if not k.endswith("_native")
            },
            "clean_feature_parity": True,
        },
    )
    print("COMPLETE raw-sensor robustness", flush=True)


if __name__ == "__main__":
    main()
