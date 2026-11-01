"""Verify reload/export, source fallback and bbox perturbations of the frozen gate."""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.final_refit.training import PhaseOnlyFinal
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_campaign.data import load_data, read
from operational.phase_campaign.evaluation import metrics
from operational.phase_robust.data import histories, query_geometry
from operational.phase_robust.runtime import load_fusion, predict_history
from scripts.final_train40_transfer import HEAD, verify_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "candidate", "fusion", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load_fusion(args.fusion)
    values, metadata, _ = load_data(args.data)
    tracks = histories(list(metadata.values()))
    paths, parent = verify_candidate(args.candidate)
    baseline = PhaseOnlyFinal().eval()
    baseline.load_state_dict(
        torch.load(paths[HEAD], map_location="cpu", weights_only=False)["model_state_dict"]
    )
    source = values["domain"] == "eap"
    if not source.any():
        raise ValueError("No source-domain rows available for fallback validation")
    with torch.inference_mode():
        native_base = baseline(
            torch.from_numpy(values["features"][source]),
            torch.from_numpy(values["rgb_width_logits"][source]),
        )["ttc"]
        native = native_base[:, None].expand(-1, 3)
        output = model(
            native, torch.full_like(native_base, float("nan")), torch.zeros(len(native_base), 20)
        )
    if not torch.equal(output, native_base):
        raise AssertionError("Absent-history source fallback differs bitwise")
    args.output.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model)
    scripted.save(str(args.output / "fusion.ts"))
    reloaded = torch.jit.load(io.BytesIO((args.output / "fusion.ts").read_bytes()))
    with np.load(args.fusion / "POPULATION.npz") as p:
        a = {k: p[k] for k in p.files}
    tensors = [torch.from_numpy(a[k]) for k in ("native", "geometry", "quality")]
    with torch.inference_mode():
        expected = model(*tensors)
        exported = reloaded(*tensors)
    torch.testing.assert_close(expected, exported, rtol=0, atol=0, equal_nan=True)
    # Benchmark batch1 after warmup; CPU gate and geometry time are separate.
    times = []
    with torch.inference_mode():
        for i in range(220):
            start = time.perf_counter()
            reloaded(*(x[:1] for x in tensors))
            if i >= 20:
                times.append((time.perf_counter() - start) * 1000)
    selected = []
    for group in np.unique(a["group"]):
        if group == "Slider-500":
            continue
        indices = np.flatnonzero(a["group"] == group)
        selected.append(int(indices[len(indices) // 2]))
    variants = (
        "clean",
        "pixel_jitter_1",
        "pixel_jitter_3",
        "box_dropout_50",
        "box_quantize_4",
        "bbox_missing",
    )
    predictions = {variant: [] for variant in variants}
    geometry_ms = []
    for i in selected:
        row = metadata[str(a["token"][i])]
        boxes, clocks = tracks[row["sequence_id"]]
        mask = (clocks <= row["anchor_us"]) & (clocks >= row["anchor_us"] - 650000)
        boxes, clocks = boxes[mask], clocks[mask]
        for variant in variants:
            changed, ts = boxes.copy(), clocks.copy()
            rng = np.random.default_rng(7)
            if variant.startswith("pixel_jitter"):
                changed += rng.uniform(-int(variant[-1]), int(variant[-1]), changed.shape)
            elif variant == "box_dropout_50":
                keep = rng.random(len(ts)) >= 0.5
                changed, ts = changed[keep], ts[keep]
            elif variant == "box_quantize_4":
                changed = np.round(changed / 4) * 4
            elif variant == "bbox_missing":
                changed, ts = changed[:0], ts[:0]
            start = time.perf_counter()
            pred, _ = predict_history(model, a["native"][i], changed, ts, row["anchor_us"])
            geometry_ms.append((time.perf_counter() - start) * 1000)
            predictions[variant].append(pred)
            if variant == "clean":
                np.testing.assert_allclose(pred, expected[i], rtol=1e-6, atol=1e-6)
        # The same query with shifted clocks and appended arbitrary future is identical.
        shifted, _ = predict_history(
            model, a["native"][i], boxes, clocks + 1234567, row["anchor_us"] + 1234567
        )
        future, _ = predict_history(
            model,
            a["native"][i],
            np.concatenate((boxes, [[0, 0, 1, 1]])),
            np.r_[clocks, row["anchor_us"] + 1000000],
            row["anchor_us"],
        )
        assert shifted == predictions["clean"][-1] == future
        physical = query_geometry(tracks[row["sequence_id"]], row["anchor_us"])
        assert physical["count"] <= 32
    arrays = {k: np.array(v) for k, v in predictions.items()}
    arrays.update(
        target=a["target"][selected], native=a["native"][selected, 0], token=a["token"][selected]
    )
    save_npz(args.output / "BBOX_ROBUSTNESS.npz", arrays)
    save_npz(
        args.output / "example_input.npz", {k: a[k][:8] for k in ("native", "geometry", "quality")}
    )
    save_npz(args.output / "example_output.npz", {"ttc_seconds": exported[:8].numpy()})
    radius = read(args.fusion / "CONFIRMATION.json")["calibration"]["radius_seconds_80"]
    atomic_receipt(
        args.output / "VALIDATION.json",
        {
            "source_fallback_rows": int(source.sum()),
            "source_fallback_bitwise_equal": True,
            "source_scope": (
                "F0 native predictions preserved when bbox history absent; eAP with "
                "geometry not evaluated"
            ),
            "original_parent": parent,
            "fusion_sha256": file_sha(args.output / "fusion.ts"),
            "export_parity_rows": len(expected),
            "export_exact": True,
            "gate_cpu_batch1_ms": {
                "median": float(np.median(times)),
                "p95": float(np.quantile(times, 0.95)),
            },
            "geometry_and_gate_cpu_ms": {
                "median": float(np.median(geometry_ms)),
                "p95": float(np.quantile(geometry_ms, 0.95)),
            },
            "bbox_robustness": {k: metrics(arrays["target"], arrays[k]) for k in variants},
            "native_reference": metrics(arrays["target"], arrays["native"]),
            "clock_shift_and_future_append_exact_queries": len(selected),
            "interval_radius_seconds_80": radius,
            "interval_claim": (
                "descriptive calibration on separate CPNAO; not certified safety coverage"
            ),
            "feature_contract": (
                "native[B,3]=F0,RGBcontext,eventcontext in seconds; geometry[B] "
                "signed TTC or NaN; quality[B,20] per geometry.py; bbox history "
                "<=650ms"
            ),
        },
    )
    print(
        "COMPLETE strict source fallback, export, physical invariance and bbox robustness",
        flush=True,
    )


if __name__ == "__main__":
    main()
