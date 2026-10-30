"""Predeclared raw-event robustness on two temporal quartiles per eligible recording."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import yaml

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_adapt_runtime import CachedAdaptPredictor
from operational.phase_adapt_training import configure_cuda, load_head
from operational.phase_campaign.data import load_data
from operational.phase_campaign.evaluation import metrics
from operational.phase_campaign.perturbations import perturb_events
from scripts.final_train40_transfer import HEAD, verify_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "candidate", "adapted", "protocol", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    configure_cuda()
    torch.set_num_threads(2)
    values, metadata, _ = load_data(args.data)
    records = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))["records"]
    groups = sorted(r["sequence_id"] for r in records if r["role"] == "development_exposed")
    indices = []
    for group in groups:
        available = np.flatnonzero((values["domain"] == "evttc") & (values["group_id"] == group))
        ordered = sorted(
            available, key=lambda i: metadata[str(values["sample_token"][i])]["anchor_us"]
        )
        indices.extend([ordered[len(ordered) // 4], ordered[3 * len(ordered) // 4]])
    variants = [
        "clean",
        "drop_30",
        "drop_70",
        "drop_positive",
        "drop_negative",
        "jitter_200",
        "jitter_1000",
        "background_5",
        "event_roi_75",
    ]
    paths, parent = verify_candidate(args.candidate)
    atomic_receipt(
        args.output / "FREEZE.json",
        {
            "parent": parent,
            "adapted_sha256": file_sha(args.adapted),
            "data_sha256": file_sha(args.data / "COMPLETE.json"),
            "tokens": values["sample_token"][indices].tolist(),
            "variants": variants,
            "population": "two quartiles among target-supported rows in each eligible recording",
            "selection_uses_prediction_errors": False,
            "seed": 7,
            "timing_jitter": "bounded uniform radius; discard events leaving original window",
            "scope": "perturbed exposed development; no independent-generalization claim",
            "source_sha256": {
                str(p): file_sha(p)
                for p in (
                    Path(__file__).relative_to(Path.cwd()),
                    Path("operational/phase_campaign/perturbations.py"),
                )
            },
        },
    )
    runtime = CachedAdaptPredictor(paths)
    adapted = load_head(args.adapted, file_sha(args.adapted)).to("cuda")
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture(module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
        del module
        captured.append((inputs[0].detach().clone(), inputs[1].detach().clone()))

    hook = runtime.models[HEAD].register_forward_pre_hook(capture)
    target_phase = values["target_phase"][indices]
    target = phase_to_ttc(torch.from_numpy(target_phase)).numpy()
    output = {
        v: {k: np.full(len(indices), np.nan) for k in ("F0", "F1", "q10", "q90")} for v in variants
    }
    details = []
    for qi, index in enumerate(indices):
        original = {
            k: v
            for k, v in metadata[str(values["sample_token"][index])].items()
            if k != "ttc_csv_path"
        }
        for variant in variants:
            row = dict(original)
            if variant == "event_roi_75":
                box = np.array(row["square_xyxy"], dtype=float)
                center = (box[:2] + box[2:]) / 2
                half = 0.75 * (box[2:] - box[:2]) / 2
                row["square_xyxy"] = np.r_[center - half, center + half].tolist()
            raw_iterator = runtime.inputs.iter_window_chunks

            def transformed(
                start: int,
                end: int,
                *,
                chunk_events: int = 250000,
                source: Callable = raw_iterator,
                intervention: str = variant,
            ) -> Iterator[dict[str, np.ndarray]]:
                chunks = list(source(start, end, chunk_events=chunk_events))
                if sum(sum(a.nbytes for a in c.values()) for c in chunks) > 1024**3:
                    raise ValueError("Raw perturbation window exceeds 1 GiB bound")
                raw = {
                    k: np.concatenate([c[k] for c in chunks])
                    if chunks
                    else np.empty(0, dtype=np.int64)
                    for k in ("x", "y", "t", "p")
                }
                changed = perturb_events(raw, intervention, start, end, 1280, 720)
                for offset in range(0, len(changed["t"]), chunk_events):
                    yield {k: v[offset : offset + chunk_events] for k, v in changed.items()}

            captured.clear()
            start = time.perf_counter()
            failure = None
            result: dict[str, Any] = {}
            try:
                if variant in {"clean", "event_roi_75"}:
                    result = runtime.predict(row)
                else:
                    with patch.object(runtime.inputs, "iter_window_chunks", transformed):
                        result = runtime.predict(row)
                before = result["predictions"]["PHASE_ONLY_TRAIN40_S7"]
                output[variant]["F0"][qi] = before
                if captured:
                    with torch.inference_mode():
                        after = adapted(*captured[-1])
                    output[variant]["F1"][qi] = after["ttc"].item()
                    for key in ("q10", "q90"):
                        output[variant][key][qi] = after[key].item()
                    if variant == "clean":
                        np.testing.assert_allclose(
                            captured[-1][0].cpu().numpy()[0],
                            values["features"][index],
                            rtol=1e-5,
                            atol=1e-5,
                        )
                else:
                    output[variant]["F1"][qi] = before
            except ValueError as exc:
                failure = str(exc)
            details.append(
                {
                    "query": qi,
                    "variant": variant,
                    "failure": failure,
                    "seconds": time.perf_counter() - start,
                    "profile": dict(runtime.inputs.last_profile),
                    "history": result.get("history_lengths"),
                }
            )
        print(f"robustness {qi + 1}/{len(indices)}", flush=True)
    hook.remove()
    runtime.close()
    arrays = {
        f"{variant}_{key}": v for variant, result in output.items() for key, v in result.items()
    }
    arrays.update(target=target, target_phase=target_phase, tokens=values["sample_token"][indices])
    save_npz(args.output / "predictions.npz", arrays)
    report: dict[str, Any] = {
        "variants": {},
        "details": details,
        "source": "raw sensors before unchanged encoders; TRAIN40+EvTTC exposed F1",
    }
    for variant, predictions in output.items():
        valid = np.isfinite(predictions["q10"]) & np.isfinite(predictions["q90"])
        report["variants"][variant] = {
            "F0": metrics(target, predictions["F0"]),
            "F1": metrics(target, predictions["F1"]),
            "native_F1_interval_coverage_80": float(
                np.mean(
                    (predictions["q10"][valid] <= target_phase[valid])
                    & (predictions["q90"][valid] >= target_phase[valid])
                )
            )
            if valid.any()
            else None,
            "native_F1_phase_interval_width": float(
                np.mean(predictions["q90"][valid] - predictions["q10"][valid])
            )
            if valid.any()
            else None,
            "interval_rows": int(valid.sum()),
        }
    atomic_receipt(args.output / "REPORT.json", report)


if __name__ == "__main__":
    main()
