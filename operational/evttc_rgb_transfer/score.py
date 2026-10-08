"""Compare sealed full-Garl predictions against unchanged inherited predictions."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from operational.efficient_context.common import atomic_json, digest
from operational.evttc_transfer.score import metrics

from .run import read, verify_baseline

MODELS = (
    "H8_seed7",
    "H8_seed13",
    "H8_seed23",
    "public_Garl_event_lhr",
    "public_Garl_rgb_event_full",
)


def run(output: Path, baseline: Path) -> None:
    """Validate all seals before accessing the already published baseline labels."""
    manifest, old = verify_baseline(baseline, read(output / "BASELINE_PRESERVATION.json"))
    seal = read(output / "PREDICTIONS_SEALED.json")
    expected_names = {f"predictions/query_{i:05d}.json" for i in range(len(old))}
    if (
        seal["status"] != "COMPLETE"
        or seal["queries"] != len(old)
        or set(seal["fragments"]) != expected_names
        or seal["binding_sha256"] != digest(output / "INFERENCE_FREEZE.json")
        or seal["manifest_sha256"] != digest(baseline / "QUERY_MANIFEST.json")
    ):
        raise ValueError("Complete consistent full-Garl seal required")
    new = []
    for i, (row, previous) in enumerate(zip(manifest["rows"], old, strict=True)):
        name = f"predictions/query_{i:05d}.json"
        if digest(output / name) != seal["fragments"][name]:
            raise ValueError("Full-Garl prediction changed")
        p = read(output / name)
        if (
            p["query_id"] != row["query_id"]
            or p["binding_sha256"] != seal["binding_sha256"]
            or p["baseline_fragment_sha256"] != digest(baseline / name)
            or any(p["ttc"][m] != previous["ttc"][m] for m in MODELS[:-1])
        ):
            raise ValueError("Full-Garl identity or inherited output mismatch")
        new.append(p)
    with (baseline / "SCORED_PREDICTIONS.csv").open(encoding="utf-8") as h:
        labeled_rows = list(csv.DictReader(h))
    if [r["query_id"] for r in labeled_rows] != [r["query_id"] for r in manifest["rows"]]:
        raise ValueError("Inherited labels population/order mismatch")
    truth = np.array(
        [float(r["truth_ttc_seconds"]) if r["truth_ttc_seconds"] else np.nan for r in labeled_rows]
    )
    predictions = {
        m: np.array([p["ttc"][m] if p["ttc"][m] is not None else np.nan for p in new])
        for m in MODELS
    }
    seq = np.array([r["sequence_id"] for r in labeled_rows])
    summaries = {m: metrics(p, truth) for m, p in predictions.items()}
    full = predictions[MODELS[-1]]
    common = np.isfinite(truth) & np.isfinite(full)
    for p in predictions.values():
        common &= np.isfinite(p)
    common_summaries = {m: metrics(p[common], truth[common]) for m, p in predictions.items()}
    paired = {}
    for m in MODELS[:-1]:
        pair_common = np.isfinite(truth) & np.isfinite(full) & np.isfinite(predictions[m])
        diffs = []
        for sequence in np.unique(seq):
            mask = pair_common & (seq == sequence)
            if mask.any():
                diffs.append(
                    float(
                        (
                            np.abs(predictions[m][mask] - truth[mask])
                            - np.abs(full[mask] - truth[mask])
                        ).mean()
                    )
                )
        if diffs:
            d = np.array(diffs)
            rng = np.random.default_rng(20261008)
            boot = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(1)
            paired[m] = dict(
                metric="macro_sequence_MAE_difference_model_minus_Garl_full",
                sequences=len(d),
                common_queries=int(pair_common.sum()),
                difference_seconds=float(d.mean()),
                sequence_bootstrap_95pct_seconds=np.quantile(boot, [0.025, 0.975]).tolist(),
            )
    per_sequence = []
    for sequence in np.unique(seq):
        for m, p in predictions.items():
            mask = seq == sequence
            per_sequence.append(
                dict(sequence_id=str(sequence), model=m, **metrics(p[mask], truth[mask]))
            )
    output_rows = []
    for r, p in zip(labeled_rows, new, strict=True):
        output_rows.append(
            {
                **r,
                MODELS[-1]: p["ttc"][MODELS[-1]],
                "full_unavailable_reason": p["unavailable_reason"],
            }
        )
    for name, rows in (("SCORED_PREDICTIONS.csv", output_rows), ("PER_SEQUENCE.csv", per_sequence)):
        with (output / name).open("w", newline="", encoding="utf-8") as h:
            writer = csv.DictWriter(h, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result: dict = dict(
        status="COMPLETE",
        role="FROZEN_EVTTC_FULL_GARL_FOLLOWUP",
        optimizer_updates=0,
        own_predictions_reused_exactly=True,
        no_checkpoint_selection=True,
        manifest_sha256=seal["manifest_sha256"],
        prediction_seal_sha256=digest(output / "PREDICTIONS_SEALED.json"),
        inherited_scored_predictions_sha256=digest(baseline / "SCORED_PREDICTIONS.csv"),
        metrics=summaries,
        common_support_metrics=common_summaries,
        paired_sequence_comparison=paired,
        unavailability=dict(
            Counter(p["unavailable_reason"] for p in new if p["unavailable_reason"])
        ),
        limitations=[
            "Historical development cohort and follow-up after seeing event-only results; "
            "not blind model selection.",
            "H8 uses events plus oracle ROI metadata; full Garl additionally receives RGB pixels.",
            "Native frozen temporal contexts differ across models.",
            "RGB/event timing and geometry approximations are declared "
            "in the frozen input contract.",
            "H8 native TTC support +/-60 seconds; Garl native conversion unbounded, no clipping.",
            "Public Garl training/selection ancestry not independently certified.",
            "Sampled cross-dataset transfer comparison, "
            "not a reproduction of the official paper benchmark.",
        ],
    )
    atomic_json(output / "RESULT.json", result)
    lines = [
        "# Frozen EvTTC comparison with full RGB+event Garl",
        "",
        "Previous event-only results are preserved; all three H8 heads are reused exactly. "
        "No training or tuning.",
        "",
        "## Accuracy on identical common finite support",
        "",
        "| Model | Labeled | Finite | MAE s | Median AE s | RMSE s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m, v in common_summaries.items():
        lines.append(
            f"| {m} | {v['labeled_queries']} | {v['finite_predictions_on_labels']} | "
            f"{v['mae_seconds']} | {v['median_absolute_error_seconds']} | {v['rmse_seconds']} |"
        )
    lines += [
        "",
        f"Common finite support across all models: {int(common.sum())} queries.",
        "",
        "## Paired macro-sequence differences versus full Garl",
        "",
    ]
    for m, v in paired.items():
        lines.append(
            f"- {m}: {v['difference_seconds']:.6f} s, "
            f"95% sequence-bootstrap CI {v['sequence_bootstrap_95pct_seconds']}."
        )
    lines += ["", "## Coverage on the original population", ""]
    for m, v in summaries.items():
        lines.append(
            f"- {m}: {v['finite_predictions_on_labels']}/{v['labeled_queries']} "
            f"labeled queries; coverage {v['coverage']}."
        )
    lines += ["", "## Limitations", ""] + [f"- {s}" for s in result["limitations"]]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.baseline)
