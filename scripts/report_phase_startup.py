"""Regenerate candidate evidence and a portable, identity-bound model bundle."""

from __future__ import annotations

import argparse
import csv
import io
import shutil
from pathlib import Path

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.data import read
from operational.phase_campaign.evaluation import metrics
from operational.phase_startup.runtime import load_bundle_router, load_router
from scripts.train_phase_robust import load_arrays


def grouped_delta(a: dict, old: np.ndarray, new: np.ndarray, mask: np.ndarray) -> dict:
    """Paired resampling of whole development families; never bootstrap windows."""
    groups = np.unique(a["family"][mask])
    pieces = [np.flatnonzero(mask & (a["family"] == g)) for g in groups]
    delta = (np.abs(new - a["target"]) - np.abs(old - a["target"])) / np.abs(a["target"])
    rng = np.random.default_rng(7)
    samples = []
    for _ in range(5000):
        indices = np.concatenate([pieces[i] for i in rng.integers(0, len(pieces), len(pieces))])
        samples.append(float(np.mean(delta[indices]) * 100))
    return {
        "families": groups.tolist(),
        "units": len(groups),
        "replicates": 5000,
        "mean_delta_pp": float(np.mean(delta[mask]) * 100),
        "ci95_delta_pp": np.quantile(samples, [0.025, 0.975]).tolist(),
        "scope": "descriptive paired family bootstrap on exposed development, only five units",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    args = p.parse_args()
    root, previous, bundle = args.root, args.previous, args.bundle
    if bundle.exists():
        raise ValueError("Use a fresh bundle destination")
    torch.set_num_threads(2)
    training = root / "training_rate"
    a = load_arrays(training / "POPULATION.npz")
    cv = load_arrays(training / "CV_PREDICTIONS.npz")
    final = load_arrays(training / "FINAL_PREDICTIONS.npz")["prediction"]
    refit = load_arrays(root / "all_data_refit/PREDICTIONS.npz")["prediction"]
    selected = read(training / "FINAL_FREEZE.json")["selected"]
    road = ~np.isin(a["family"], ["CCRm", "CPNAO", "Slider"])
    delta = grouped_delta(a, a["previous_quality_oof"], cv[selected], road)
    report = root / "report"
    report.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(population: str, scope: str, model: str, target: np.ndarray, pred: np.ndarray) -> None:
        rows.append({"cohort": population, "scope": scope, "model": model, **metrics(target, pred)})

    for name, pred in {
        "F0": a["native"][:, 0],
        "previous_quality_oof": a["previous_quality_oof"],
        "router_inverse_oof": cv[selected],
    }.items():
        add(
            "road_development_1747",
            "family-held-out OOF, historically exposed",
            name,
            a["target"][road],
            pred[road],
        )
    confirm = a["family"] == "CCRm"
    for name, pred in {
        "F0": a["native"][:, 0],
        "previous_quality": a["previous_quality_frozen"],
        "router_inverse": final,
    }.items():
        add(
            "CCRm_580",
            "excluded from current fit/selection, historically exposed",
            name,
            a["target"][confirm],
            pred[confirm],
        )
    for group in ("Slider-500", "Slider-750", "Slider-1000"):
        mask = a["group"] == group
        for name, pred in {
            "F0": a["native"][:, 0],
            "previous_quality": a["previous_quality_frozen"],
            "router_all_slider_held_out": cv[selected],
            "router_fit": final,
            "router_all_data_refit": refit,
            "geometry_only": a["geometry"],
        }.items():
            scope = (
                "all Slider held out together"
                if "held_out" in name
                else "exposed; fit rows for router_fit/refit"
            )
            add(group, scope, name, a["target"][mask], pred[mask])
    allfit = read(root / "all_data_refit/REPORT.json")
    seconds = read(root / "training/SELECTION.json")
    rate = read(training / "SELECTION.json")
    confirmation = read(training / "CONFIRMATION.json")
    validation = read(root / "validation_v2/VALIDATION.json")
    raw = read(root / "raw_robustness/REPORT.json")
    oldraw = read(previous / "raw_robustness/REPORT.json")
    atomic_receipt(
        report / "RESULTS.json",
        {
            "rows": rows,
            "paired_family_bootstrap": delta,
            "all_data_fit": allfit,
            "seconds_controls": seconds,
            "inverse_controls": rate,
            "confirmation": confirmation,
            "validation": validation,
            "raw_robustness": raw,
            "previous_raw_robustness": oldraw,
            "decision": (
                "retain previous release as default; inverse router is an additional "
                "trained candidate with mixed robustness/confirmation outcomes"
            ),
            "stale_policy": (
                "after failed dropout diagnostic, reuse existing 150ms freshness contract "
                "for whole router; no stale training/evaluation rows, weights unchanged; "
                "stress test now exposed"
            ),
        },
    )
    with (report / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Preserve original component manifest byte-for-byte; fusion variant binding is separate.
    upstream = previous / "FINAL_MODEL"
    (bundle / "upstream").mkdir(parents=True)
    shutil.copyfile(upstream / "BUNDLE.json", bundle / "upstream/BUNDLE.json")
    shutil.copytree(upstream / "components", bundle / "upstream/components")
    export_proofs = {}
    for variant, source, prediction in (
        ("evaluated", training, final),
        ("all_data_refit", root / "all_data_refit", refit),
    ):
        dest = bundle / variant
        dest.mkdir()
        for name in ("CONFIG.json", "SELECTION.json", "FINAL_FREEZE.json"):
            shutil.copyfile(source / name, dest / name)
        shutil.copytree(source / "final", dest / "final")
        if variant == "evaluated":
            shutil.copyfile(source / "CONFIRMATION.json", dest / "CONFIRMATION.json")
        model = load_router(dest)
        scripted = torch.jit.script(model)
        scripted.save(str(dest / "router.ts"))
        reloaded = torch.jit.load(io.BytesIO((dest / "router.ts").read_bytes()))
        inputs = [torch.from_numpy(a[k]) for k in ("native", "geometry", "quality")]
        with torch.inference_mode():
            eager, exported = model(*inputs), reloaded(*inputs)
        torch.testing.assert_close(eager, exported, rtol=0, atol=0, equal_nan=True)
        np.testing.assert_array_equal(eager.numpy(), prediction)
        export_proofs[variant] = {
            "rows": len(prediction),
            "bitwise_equal": True,
            "sha256": file_sha(dest / "router.ts"),
        }
    atomic_receipt(
        bundle / "ROUTER_BINDING.json",
        {
            "schema": "phase_startup_bundle_v1",
            "upstream_sha256": file_sha(bundle / "upstream/BUNDLE.json"),
            "freezes": {
                v: file_sha(bundle / v / "FINAL_FREEZE.json")
                for v in ("evaluated", "all_data_refit")
            },
            "artifacts": {
                v: file_sha(bundle / v)
                for v in (
                    "evaluated/router.ts",
                    "all_data_refit/router.ts",
                    "evaluated/CONFIRMATION.json",
                )
            },
            "implementation": {
                n: file_sha(Path("operational/phase_startup") / n)
                for n in ("model.py", "geometry.py", "runtime.py")
            },
            "status": "research candidate; previous release default retained",
        },
    )
    for variant in ("evaluated", "all_data_refit"):
        load_bundle_router(bundle, variant)
    atomic_receipt(report / "EXPORT.json", export_proofs)
    table = ["| Population | Model | RAE % | MAE s | Coverage |", "|---|---|---:|---:|---:|"]
    for row in rows:
        table.append(
            f"| {row['cohort']} | {row['model']} | "
            f"{row['relative_absolute_error_pct']:.4f} | {row['mae_seconds']:.6f} | "
            f"{row['finite']}/{row['population']} |"
        )
    road_fit = allfit["metrics"]["road24"]["candidate"]["relative_absolute_error_pct"]
    slider_fit = allfit["metrics"]["slider_all"]["candidate"]["relative_absolute_error_pct"]
    rejected = seconds["controls"]["QA"]["road"]
    prior = seconds["prior_quality"]["road_oof"]
    failed_stress = read(root / "validation/VALIDATION.json")["bbox_robustness"]["box_dropout_50"]
    fixed_stress = validation["bbox_robustness"]["box_dropout_50"]
    audit = read(root / "AUDIT.json")["scores"]
    reproduction = read(root / "verification/REPRODUCIBILITY.json")
    n_road = int(road.sum())
    n_slider = int(np.sum(a["family"] == "Slider"))
    n_confirmation = int(confirm.sum())
    n_calibration = int(np.sum(a["family"] == "CPNAO"))
    n_total = len(a["target"])
    n_records = len(np.unique(a["group"]))
    n_fits = sum(
        len(list((root / name).rglob("COMPLETE.json")))
        for name in ("training", "training_rate", "all_data_refit")
    )
    text = "\n".join(
        [
            "# PHASE-TTC startup router — 2026-10-31",
            "",
            "Additional trained candidate; previous robust release remains the default. "
            "Inverse-rate mixing improves aggregate road OOF slightly and substantially "
            "improves Slider fitting, but does not dominate confirmation MAE or raw perturbations.",
            "",
            "## Protocol",
            "",
            "Frozen nine-component TRAIN40 upstream; new 688-parameter four-expert router, "
            "three seeds 7/13/21 (2064 learned parameters in ensemble). Causal boxes <=650ms; "
            "two frames >=10ms; stale endpoint >150ms or no history returns exact F0. "
            "Signed inverse rates are bounded to TTC magnitude [.1,60]s; exact rate cancellation "
            "falls back to F0. Opposite-sign experts can recover a failed component; no target or "
            "sequence identity enters the gate. Native road uses left RGB/history; Slider uses "
            "the competition adapter. These contracts are not interchangeable.",
            "",
            f"Development: {n_road} road + {n_slider} Slider = {n_road + n_slider} rows; "
            "five road families plus one Slider family. Each CV fold excludes the complete "
            f"family. Current fit/selection excludes CCRm {n_confirmation} "
            f"and CPNAO {n_calibration}. Every dataset is historically exposed. "
            "No clean external test. "
            "Compared with the previous gate, development now includes Slider750/1000; "
            "the old/new difference cannot be attributed solely to architecture. "
            "Both coordinates run Q/QA/NAcap with equal 688-parameter capacity and seeds 7/13/21, "
            "300 updates, fixed AdamW .01/.01. Train-only normalization and family-balanced loss. "
            f"{n_fits} experiment fits total including both trials and all-data refit; "
            "none discarded.",
            "",
            "The seconds-coordinate trial is rejected: road QA "
            f"{rejected['relative_absolute_error_pct']:.4f}%/{rejected['mae_seconds']:.6f}s versus "
            f"previous quality {prior['relative_absolute_error_pct']:.4f}%/"
            f"{prior['mae_seconds']:.6f}s. Inverse QA is retained as a candidate only. "
            "Its raw-event midpoint stress test is worse than the previous gate; see JSON. "
            f"A failed bbox-dropout test ({failed_stress['relative_absolute_error_pct']:.4f}%) "
            "exposed stale-history routing. The explicit 150ms fallback fixes that contract "
            f"({fixed_stress['relative_absolute_error_pct']:.4f}%, identical F0). This stress test "
            "has now informed implementation and is not independent validation.",
            "",
            "Table metrics pool all queries; model selection uses macro-record metrics in JSON.",
            "",
            *table,
            "",
            f"Paired family bootstrap of road OOF change (percentage points): {delta}.",
            "",
            "## All-data weights and uncertainty",
            "",
            f"All-data refit uses all {n_total} eligible rows ({n_records} recordings), retains "
            f"eight-overlap exclusion. Fit metrics: {road_fit:.4f}% road and "
            f"{slider_fit:.4f}% Slider. "
            "These are resubstitution, not generalization. Encoders/producers were not retrained. "
            "The all-data model returns no calibrated interval because calibration "
            "labels were fitted.",
            "",
            f"Evaluated-candidate interval: {confirmation['calibration']}. "
            "Calibration uses separate CPNAO, historical exposure and correlated sequences remain; "
            "coverage is descriptive, not a safety guarantee.",
            "",
            "## Verification and runtime",
            "",
            f"Source fallback is bitwise equal on {validation['source_fallback_rows']} "
            "eAP examples "
            "without bbox history; no source-domain geometry transfer claim. TorchScript eager/"
            f"reload parity is exact for both {n_total}-row variants. "
            "Perturbations include raw event "
            "loss/polarity loss/jitter/background and bbox jitter/dropout/quantization/missing. "
            "Raw robustness caches all three native experts for future reuse; avoids repeating "
            "sensor decoding when testing another fusion. Training uses cached features and "
            "two CPU threads; native raw replay retains cached input/producer acceleration. "
            f"Gate CPU batch1 timing: {validation['gate_cpu_batch1_ms']} "
            "(not end-to-end sensor latency).",
            "",
            "Fresh sensor replay is distinct from tensor-export parity: native query difference "
            f"{reproduction['native_fresh_replay_abs_seconds']:.9f}s; eight-query Slider replay "
            f"max difference {reproduction['slider_fresh_replay_max_abs_seconds']:.9f}s. "
            "The new CLI disables TF32 while the earlier Slider cache inherited TF32 settings; "
            "this is a measured precision-contract difference, not bitwise sensor parity. "
            "The complete QA inverse seed7/CPLA-held-out refit reproduced every weight exactly.",
            "",
            "## Deployment",
            "",
            "Install repository dependencies; set PYTHONPATH=src;. on Windows. Run from repo root:",
            "",
            "```powershell",
            "python -m scripts.predict_phase_startup --bundle PATH_TO_BUNDLE "
            "--contract native_road --manifest SENSOR_ONLY.json --device cuda --output NEW.json",
            "python -m scripts.predict_phase_startup --bundle PATH_TO_BUNDLE "
            "--contract slider --sensors SENSOR_ROOT --sequence Slider-750 "
            "--device cuda --output NEW.json",
            "# Add --variant all_data_refit for fully fitted weights (no validated uncertainty).",
            "```",
            "",
            "Bundle binds every upstream checkpoint/normalizer, named variant, calibration, "
            "export and implementation hashes. Raw data are not included. Model inputs are "
            "native[B,3]=(F0,RGBcontext,eventcontext) seconds, geometry[B] seconds/NaN and "
            "quality[B,20] documented in geometry.py. Example tensor inputs/outputs "
            "in validation_v2.",
            "",
            "## Unresolved evidence",
            "",
            f"Official PDF750 is {audit['750']['official_pdf_relative_error_pct']:.4f}%, "
            "while original submitted predictions on public targets yield "
            f"{audit['750']['submitted_on_public_targets']['relative_absolute_error_pct']:.4f}%; "
            "1000 reproduces "
            f"{audit['1000']['submitted_on_public_targets']['relative_absolute_error_pct']:.4f}%. "
            "PDF contains no per-query aligned targets or "
            "evaluator. No offset, clipping or denominator was fitted to reconcile it. "
            "Audit JSON retains original file hashes and synthetic parallax at seven depths. "
            "Slider extrinsic remains a candidate, not authoritative; no ground-truth depth "
            "was injected into inference. New independent acquisitions/official evaluator are "
            "still needed; this does not establish SOTA. JEPA/encoder retraining is not justified "
            "by these fusion experiments alone and has not been claimed complete.",
            "",
        ]
    )
    (report / "REPORT.md").write_text(text, encoding="utf-8")
    (bundle / "MODEL_CARD.md").write_text(text, encoding="utf-8")
    print(f"COMPLETE report and verified bundle {bundle}", flush=True)


if __name__ == "__main__":
    main()
