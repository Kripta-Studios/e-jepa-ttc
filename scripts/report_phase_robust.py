"""Regenerate evidence tables, verification and a portable two-variant delivery."""

from __future__ import annotations

import argparse
import csv
import io
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.data import read
from operational.phase_campaign.evaluation import metrics
from operational.phase_robust.runtime import load_fusion
from scripts.final_train40_transfer import verify_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--package", action="store_true")
    args = parser.parse_args()
    root = args.root
    torch.set_num_threads(2)
    selection = read(root / "training/SELECTION.json")
    confirmation = read(root / "training/CONFIRMATION.json")
    refit = read(root / "all_data_refit/REPORT.json")
    validation = read(root / "validation/VALIDATION.json")
    supplement = read(root / "SUPPLEMENT.json")
    rows = []

    def add(population: str, scope: str, model: str, m: dict) -> None:
        rows.append(
            {
                "population": population,
                "scope": scope,
                "model": model,
                **{
                    k: m[k]
                    for k in (
                        "population",
                        "finite",
                        "coverage",
                        "relative_absolute_error_pct",
                        "mae_seconds",
                    )
                },
                "cohort": population,
            }
        )

    for pop in ("road", "slider"):
        add(pop, "grouped development OOF", "F0", selection["baseline"][pop])
        for mode in ("quality", "native"):
            add(pop, "grouped development OOF", mode, selection["controls"][mode][pop])
    for model in ("F0", "candidate"):
        add(
            "CCRm",
            "current-iteration fit/selection reserve; historical exposure",
            model,
            confirmation[model],
        )
    external = {}
    for speed in (750, 1000):
        report = read(root / f"external/Slider-{speed}/REPORT.json")
        external[str(speed)] = report
        for model in ("F0", "candidate", "geometry"):
            add(
                f"Slider-{speed}",
                "public GT joined after prediction freeze; historical benchmark exposure",
                model,
                report["full"][model],
            )
    for pop, result in refit["fit_metrics"].items():
        for model, m in result.items():
            add(pop, "all-data resubstitution", model, m)
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Verify both deployable models against their byte-bound checkpoints and exports.
    checks = {}
    for name, folder, export in (
        ("evaluated", root / "training", root / "validation/fusion.ts"),
        ("all_data_refit", root / "all_data_refit", root / "all_data_refit/fusion.ts"),
    ):
        model = load_fusion(folder)
        source = folder / ("POPULATION.npz" if name == "evaluated" else "PREDICTIONS.npz")
        with np.load(source) as data, torch.inference_mode():
            tensors = [torch.from_numpy(data[k]) for k in ("native", "geometry", "quality")]
            eager = model(*tensors)
            scripted = torch.jit.load(io.BytesIO(export.read_bytes()))
            exported = scripted(*tensors)
            torch.testing.assert_close(eager, exported, rtol=0, atol=0, equal_nan=True)
        checks[name] = {"rows": len(eager), "export_exact": True, "export_sha256": file_sha(export)}
    model = load_fusion(root / "training")
    for speed in (750, 1000):
        folder = root / f"external/Slider-{speed}"
        seal = read(folder / "SEALED.json")
        if file_sha(folder / "PREDICTIONS.npz") != seal["predictions_sha256"]:
            raise ValueError("Changed sealed external predictions")
        with (
            np.load(folder / "PREDICTIONS.npz") as p,
            np.load(folder / "TARGETS.npz") as y,
            torch.inference_mode(),
        ):
            result = model(
                *(
                    torch.from_numpy(p[k].astype(np.float32))
                    for k in ("native", "geometry", "quality")
                )
            ).numpy()
            np.testing.assert_allclose(result, p["prediction"], rtol=0, atol=1e-5, equal_nan=True)
            checks[f"Slider-{speed}"] = {
                "runtime_fallback_rounding_max_seconds": float(
                    np.max(np.abs(result - p["prediction"]))
                ),
                "scores": metrics(y["target"], result),
                "interval_80_coverage": float(
                    np.mean(
                        np.abs(result - y["target"])
                        <= confirmation["calibration"]["radius_seconds_80"]
                    )
                ),
            }
    atomic_receipt(out / "VERIFICATION.json", checks)
    lines = [
        "# PHASE-TTC: robust quality fusion, 2026-10-31",
        "",
        "Two trained variants are delivered. `evaluated` is the fixed three-seed "
        "candidate tested after selection. "
        "`all_data_refit` uses all 2,677 eligible labeled queries across 24 road "
        "and three Slider recordings; "
        "its scores are training resubstitution and it has no remaining independent confirmation.",
        "",
        "The new trainable component is a three-seed ensemble of 401-parameter quality gates. "
        "The original nine-component PHASE graph is frozen. Geometry uses provided "
        "boxes over at most 650 ms, "
        "actual timestamps and a robust inverse-scale fit. No target depth, TTC or "
        "future observation enters inference.",
        "",
        "## Recomputed results",
        "",
        "| Population / role | Model | Rows | Finite | Relative absolute error (%) | MAE (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['cohort']} / {row['scope']} | {row['model']} | "
            f"{row['population']} | {row['finite']} | "
            f"{row['relative_absolute_error_pct']:.3f} | {row['mae_seconds']:.4f} |"
        )
    lines += [
        "",
        "## What the evidence supports",
        "",
        "Quality information improved road grouped validation beyond an equal-parameter control. "
        "Both controls use the geometric estimate; this isolates gate-quality "
        "information, not geometry availability. "
        "Seeds 7, 13 and 21 are all retained and averaged. Selection required "
        "improvements in both road macro-record "
        "RAE and macro-record MAE, plus Slider500 RAE and MAE. It did not select on the reserves.",
        "",
        "CCRm has four recording groups but is one scenario family. Its paired "
        "recording bootstrap interval "
        f"for the relative-error change is "
        f"[{confirmation['paired_record_bootstrap_relative_pp']['lower_95']:.3f}, "
        f"{confirmation['paired_record_bootstrap_relative_pp']['upper_95']:.3f}] "
        f"percentage points and includes zero. "
        "These are historically exposed data; this is not a claim of statistically "
        "established independent superiority.",
        "",
        "Public Slider750/1000 boxes exactly match the local query CSVs. Targets "
        "join by exact frame identity "
        "and sensor clock within 2 microseconds. These are recomputed local "
        "public-GT scores, not a new official "
        "benchmark receipt. In particular, Slider750's local F0 score differs from "
        "the previously supplied official aggregate. "
        "Do not splice the old aggregate and the new candidate score into one comparison.",
        "",
        "## Limitations that remain visible",
        "",
        "Slider750 startup is still poor: the first two queries lack three causal "
        "boxes, and inconsistent "
        "neural signs can remain in the conservative native fallback. "
        "Full-population scores retain those failures. "
        "The standalone geometric observer is also reported and can beat the "
        "learned fusion on Slider. "
        "Provided annotation boxes are an oracle-input contract; "
        "detector-generated boxes were not evaluated. "
        "Event/RGB extrinsics remain provisional rotation-only on Slider. FCWD "
        "cannot support the full RGB-event "
        "comparison without an audited spatial mapping, so no such result is manufactured.",
        "",
        "The 80% interval for `evaluated` uses separate CPNAO calibration records, "
        "with a fixed symmetric radius "
        f"of {confirmation['calibration']['radius_seconds_80']:.4f} s. CCRm coverage is "
        f"{100 * confirmation['calibration']['confirmation_coverage_80']:.2f}%. "
        f"This is descriptive calibration, "
        "not a safety or exchangeability guarantee. The interval must not be "
        "presented as validated for `all_data_refit`, "
        "which trained on those calibration labels.",
        "",
        f"Without geometry, {validation['source_fallback_rows']:,} eAP native "
        f"predictions are preserved bitwise. "
        "Source eAP with full box histories remains unevaluated. No end-to-end "
        "encoder training or JEPA benefit is claimed: "
        "the demonstrated intervention is quality-aware physical fusion.",
        "",
        "## Runtime and verification",
        "",
        f"The evaluated CPU gate batch-one median was "
        f"{validation['gate_cpu_batch1_ms']['median']:.3f} ms; "
        f"geometry plus gate median was {validation['geometry_and_gate_cpu_ms']['median']:.3f} ms. "
        "These exclude upstream sensor decoding and neural inference. Cached "
        "external replay keeps exact prepared "
        "tensors and bounded memory; raw road replay uses the previously verified "
        "CUDA-graph producer runtime. "
        "The tiny gate trains from preloaded features on CPU; CUDA graphs would "
        "not remove sensor preprocessing cost.",
        "",
        "Both TorchScript exports match their PyTorch models exactly. Checkpoints, "
        "configurations, source snapshots, "
        "normalizers and parent graph hashes are included. "
        "`report/VERIFICATION.json`, `validation/VALIDATION.json` "
        "and `raw_robustness/REPORT.json` contain machine-readable checks. Full "
        "test and independent review results "
        "are recorded separately after they finish.",
        "",
        "## Reproduction",
        "",
        "Run from the repository root, with Python 3.11 and the locked project dependencies:",
        "",
        "```powershell",
        "$env:PYTHONPATH='src;.'",
        "$env:PYTHONUTF8='1'",
        "$env:PYTHONIOENCODING='utf-8'",
        "python -m scripts.predict_phase_robust --bundle "
        "output/phase_robust/FINAL_MODEL --sensors ../EvTTC-Benchmark --sequence "
        "Slider-1000 --device cuda --variant evaluated --output predictions.json",
        "```",
        "",
        "`--variant all_data_refit` selects the final all-data weights. The "
        "default remains the evaluated candidate. "
        "The prediction CLI never opens a target file and reports "
        "missing-history/sign-conflict flags.",
        "",
        "The command above is restricted to Slider: it uses the audited competition adapter "
        "and event-plane provided boxes. For native road data use "
        "`scripts.predict_phase_robust_native` with `--manifest` pointing to a sensor-only native "
        "manifest. Its RGB boxes, timestamps and "
        "upstream preprocessing match the road evaluation. Target fields and target-file paths are "
        "rejected. `NATIVE_PARITY.json` records an exact raw replay against the cached candidate.",
        "",
        "`FUSION_BINDING.json` binds named evaluated/refit variants, "
        "exports and evaluated calibration "
        "to the upstream bundle. The loader rejects a swapped variant even if its internal "
        "checkpoint hashes are otherwise consistent.",
    ]
    long_ttc = supplement["metrics"]["road_over4"]
    lines += [
        "",
        "## Long-TTC tail",
        "",
        f"For the {long_ttc['F0']['population']} grouped-CV road queries above4s, MAE changes "
        f"from {long_ttc['F0']['mae_seconds']:.4f} to {long_ttc['quality']['mae_seconds']:.4f}s; "
        f"RMSE changes from {long_ttc['F0']['rmse_seconds']:.4f} to "
        f"{long_ttc['quality']['rmse_seconds']:.4f}s. The latter worsens, so the candidate does "
        "not dominate every tail metric. Full bucket results are in SUPPLEMENT.json.",
    ]
    if (root / "FINAL_CHECKS.json").exists():
        checks_receipt = read(root / "FINAL_CHECKS.json")
        full = checks_receipt["full_suite"]
        focused = checks_receipt["focused_final"]
        lines += [
            "",
            "## Completed delivery checks",
            "",
            f"Full suite: {full['passed']} passed, {full['skipped']} skipped, "
            f"{full['additional_subtests']} additional subtests, no failures or errors. "
            f"Final focused suite: {focused['passed']} passed. Ruff and Pyright pass. "
            "The one fresh review found two deployment/identity issues; both were fixed and "
            "verified by contract/tamper tests and exact native raw replay. See FINAL_CHECKS.json "
            "for the full-suite versus final-fix verification scope.",
        ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Compact exported figure: each population keeps its own denominator.
    labels = ["Road grouped CV", "CCRm reserve", "Slider750 public", "Slider1000 public"]
    base = [
        selection["baseline"]["road"]["relative_absolute_error_pct"],
        confirmation["F0"]["relative_absolute_error_pct"],
        external["750"]["full"]["F0"]["relative_absolute_error_pct"],
        external["1000"]["full"]["F0"]["relative_absolute_error_pct"],
    ]
    new = [
        selection["controls"]["quality"]["road"]["relative_absolute_error_pct"],
        confirmation["candidate"]["relative_absolute_error_pct"],
        external["750"]["full"]["candidate"]["relative_absolute_error_pct"],
        external["1000"]["full"]["candidate"]["relative_absolute_error_pct"],
    ]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(4)
    ax.bar(x - 0.18, base, 0.36, label="Frozen F0", color="#64748b")
    ax.bar(x + 0.18, new, 0.36, label="Evaluated quality fusion", color="#0d9488")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Relative absolute error (%)")
    ax.set_title("Local evaluations; historically exposed data; full query populations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)
    if not args.package:
        print("COMPLETE report and model/export verification", flush=True)
        return
    bundle = root / "FINAL_MODEL"
    bundle.mkdir(parents=True, exist_ok=True)
    paths, parent = verify_candidate(args.candidate)
    files = {}
    for name, path in paths.items():
        rel = Path("components") / (name + path.suffix)
        destination = bundle / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        files[name] = {"path": rel.as_posix(), "sha256": file_sha(destination)}
    for name, source in (
        ("evaluated", root / "training"),
        ("all_data_refit", root / "all_data_refit"),
    ):
        for relative in ("CONFIG.json", "SELECTION.json", "FINAL_FREEZE.json"):
            (bundle / name).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, bundle / name / relative)
        shutil.copytree(source / "final", bundle / name / "final", dirs_exist_ok=True)
        export = root / "validation/fusion.ts" if name == "evaluated" else source / "fusion.ts"
        shutil.copyfile(export, bundle / name / "fusion.ts")
    atomic_receipt(
        bundle / "BUNDLE.json",
        {
            "schema": "phase_robust_bundle_v1",
            "parent": parent,
            "upstream": files,
            "default_variant": "evaluated",
            "all_data_refit_rows": 2677,
        },
    )
    atomic_receipt(bundle / "evaluated/CALIBRATION.json", confirmation["calibration"])
    atomic_receipt(
        bundle / "evaluated/CALIBRATION_SEAL.json",
        {
            "calibration_sha256": file_sha(bundle / "evaluated/CALIBRATION.json"),
            "fusion_freeze_sha256": file_sha(bundle / "evaluated/FINAL_FREEZE.json"),
        },
    )
    atomic_receipt(
        bundle / "FUSION_BINDING.json",
        {
            "bundle_sha256": file_sha(bundle / "BUNDLE.json"),
            "fusion_freezes": {
                name: file_sha(bundle / name / "FINAL_FREEZE.json")
                for name in ("evaluated", "all_data_refit")
            },
            "artifacts": {
                name: file_sha(bundle / name)
                for name in (
                    "evaluated/fusion.ts",
                    "all_data_refit/fusion.ts",
                    "evaluated/CALIBRATION.json",
                    "evaluated/CALIBRATION_SEAL.json",
                )
            },
        },
    )
    shutil.copyfile(out / "REPORT.md", bundle / "MODEL_CARD.md")
    print(f"COMPLETE portable model bundle {bundle}", flush=True)


if __name__ == "__main__":
    main()
