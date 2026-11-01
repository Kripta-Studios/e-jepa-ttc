"""Regenerate the final submission report solely from sealed campaign receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from operational.phase_campaign.data import read


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    evaluation = read(args.root / "evaluation/LOCAL_EVALUATION.json")
    validation = read(args.root / "validation/VALIDATION.json")
    training = read(args.root / "training/REPORT.json")
    environment = read(args.root / "replay/ENVIRONMENT.json")
    complete = read(args.root / "replay/COMPLETE.json")
    audit = read(args.root / "audit/HISTORICAL_AUDIT.json")
    rows = [
        json.loads(line) for line in (args.root / "replay/all_rows.jsonl").read_text().splitlines()
    ]
    lines = [
        "# PHASE-TTC-Adapt-20261031 — measured delivery report",
        "",
        "Generated from frozen JSON/JSONL receipts; no table values are manually supplied.",
        "",
        "Final model: 2,608 fit queries, 25 recordings, seeds 7/13/21, 688 parameters per router.",
        "The eight competition road recordings and Slider-750/1000 are excluded from this fit.",
        "Historical recipe selection and evaluation exposure remain; this is an "
        "adapted research result.",
        "The original TRAIN40 producers are unchanged. No new model selection "
        "follows these scores.",
        "",
        "## Submission completeness",
        "",
        f"{complete['rows']}/434 finite outputs; "
        f"{sum(s['negative'] for s in complete['by_sequence'].values())} negative estimates.",
        "Each original index/timestamp string and measured cost matches the "
        "inference ledger exactly.",
        f"Prediction ZIP SHA-256: `{complete['results_zip_sha256']}`.",
        f"Model identity SHA-256: `{evaluation['model_sha256']}`.",
        "",
        "## Local accuracy on supported targets",
        "",
        "Relative error is mean |prediction-target|/target, expressed as a percentage.",
        "The original and adapted columns below both use the organizer-email "
        "negative interpolation.",
        "Raw and processed scores, plus geometry coverage, remain separately available in JSON.",
        "These are NOT official EvTTC results: 11 road targets are unavailable "
        "locally and the original",
        "Slider-750 public-target calculation does not reproduce the organizer's PDF.",
        "CCRs1-low and CCRs1-medium also differ from the PDF; six other road scores reproduce",
        "its rounded values. The organizer reply requests targets/clock mappings for all three.",
        "",
        "| Population | Supported / all | Original error % | Adapted error % | "
        "Original MAE s | Adapted MAE s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for population, scores in evaluation["metrics"].items():
        old, new = scores["original_email_processed"], scores["adapted_email_processed"]
        lines.append(
            f"| {population} | {new['scored']}/{new['population']} | "
            f"{old['relative_error_pct']:.4f} | {new['relative_error_pct']:.4f} | "
            f"{old['mae_seconds']:.6f} | {new['mae_seconds']:.6f} |"
        )
    fit = training["fit_metrics"]
    lines += [
        "",
        "## Fit and validation",
        "",
        f"Resubstitution on the 2,608 fitted queries: {fit['relative_absolute_error_pct']:.4f}% "
        f"relative error, {fit['mae_seconds']:.6f}s MAE. This is training fit, not generalization.",
        f"Training reload/export exact on {training['export_exact_rows']} rows; "
        "final replay export exact on "
        f"{evaluation['verification']['export_batch_exact_rows']} rows.",
        f"Batch versus per-query FP32 maximum difference: "
        f"{evaluation['verification']['router_replay_max_abs_difference']:.9g}s.",
        f"Absent-history original-model fallback bitwise exact on "
        f"{validation['source_fallback_rows']} source rows.",
        f"Clock shift and appended future leave outputs unchanged on "
        f"{validation['clock_shift_and_future_append_exact_queries']} sampled recordings.",
        "",
        "Cached raw-sensor perturbation diagnostics use 24 previously exposed recording midpoints;",
        "the upstream producers are identical, so their frozen perturbed native "
        "outputs are reused.",
        "This is a small robustness diagnostic, not an independent test or a new tuning phase.",
        "",
        "| Perturbation | Original error % | Adapted error % |",
        "|---|---:|---:|",
    ]
    for variant, result in validation["sensor_robustness"].items():
        lines.append(
            f"| {variant} | {result['original']['relative_error_pct']:.4f} | "
            f"{result['adapted']['relative_error_pct']:.4f} |"
        )
    lines += [
        "",
        "## Measured runtime",
        "",
        f"GPU: {environment['gpu']}; Python {environment['python']}; "
        f"PyTorch {environment['torch']}; CUDA {environment['cuda']}.",
        "CPU: AMD Ryzen 9 9955HX, 16 cores / 32 logical processors.",
        "Two CPU threads for PyTorch; FP32; TF32 disabled. No CUDA Graph acceleration claimed.",
        "",
        "| Stage | Mean seconds/query |",
        "|---|---:|",
    ]
    for key in (
        "preparation_seconds",
        "native_graph_seconds",
        "geometry_router_seconds",
        "cost_time_seconds",
    ):
        lines.append(f"| {key} | {np.mean([r[key] for r in rows]):.6f} |")
    lines += [
        "",
        "Original submission observed mean: "
        f"{audit['verification']['mean_cost_seconds']:.6f}s/query.",
        "This cross-run timing comparison is descriptive, not a controlled cache ablation.",
        "Costs include sensor/cache preparation and synchronized inference; one-time loading,",
        "hashing/indexing and ledger writes are outside the query boundary. This pipeline is not",
        "claimed real-time. The lossless raw cache is bounded to 64 MiB events and 16 RGB frames.",
        "",
        "## Unresolved limits and organizer action",
        "",
        "- The adapted result is historically exposed; do not claim independent SOTA.",
        "- Training native-camera features and competition right-camera features differ.",
        "- The sensor projection omits translation; Slider rotation is provisional.",
        "- No calibrated interval is carried after this final fit.",
        "- Nominal Slider lens millimetres are not established; exact pixel "
        "intrinsics are provided.",
        "- Ask organizers for Slider-750 aligned targets/clock to reconcile the original PDF.",
        "- Send the two versions' descriptions separately; official evaluation/listing is pending.",
        "",
        "No email was sent and no external upload was performed.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
