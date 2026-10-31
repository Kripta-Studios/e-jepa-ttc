"""Summarize the closed JEPA pilot using existing evidence only (stdlib)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

SEEDS = (7, 13, 21)
ARMS = ("B_static", "C_one_step", "C_shuffle", "random_init")
PRIMARY = "encoder_dense/temporal_difference|measured_log_height_ratio"


def load(path: Path) -> Any:
    """Read recorded JSON without modifying its source."""
    return json.loads(path.read_text(encoding="utf-8"))


def correlation(x: list[float], y: list[float]) -> float | None:
    """Descriptive Pearson correlation; no independent-window inference."""
    xc, yc = [v - mean(x) for v in x], [v - mean(y) for v in y]
    scale = math.sqrt(sum(v * v for v in xc) * sum(v * v for v in yc))
    return sum(a * b for a, b in zip(xc, yc, strict=True)) / scale if scale else None


def main() -> None:
    """Generate closure tables from all seeds and all held-out sequences."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.evidence, args.output
    result = load(root / "results.json")
    status = load(root / "CAMPAIGN_STATUS.json")
    summaries = {s: load(root / f"seed{s}/summary.json") for s in SEEDS}
    if status["status"] != "completed" or status["evaluation_returncode"] != 0:
        raise ValueError("Incomplete campaign")
    if result["primary_benefit_gate_passed"]:
        raise ValueError("This closure is for the failed primary gate")

    def metric(seed: int, arm: str) -> Any:
        return result["metrics"][f"{seed}|{arm}|{PRIMARY}"]

    groups = sorted(metric(7, "C_one_step")["per_sequence"])
    if len(groups) != 8:
        raise ValueError("Unexpected held-out cohort")
    rows = []
    for group in groups:
        for control in ("B_static", "C_shuffle"):
            expansion, sign, latent, increment = [], [], [], []
            for seed in SEEDS:
                c = metric(seed, "C_one_step")["per_sequence"][group]
                b = metric(seed, control)["per_sequence"][group]
                expansion.append(c["mae"] - b["mae"])
                sign.append(c["sign_accuracy"] - b["sign_accuracy"])
                lc = summaries[seed]["exports"]["C_one_step"]["H_teacher_metrics"]["per_sequence"][
                    group
                ]
                lb = summaries[seed]["exports"][control]["H_teacher_metrics"]["per_sequence"][group]
                latent.append(lc["future_true"]["mse"] - lb["future_true"]["mse"])
                increment.append(lc["delta_true"]["mse"] - lb["delta_true"]["mse"])
            rows.append(
                {
                    "sequence": group,
                    "control": control,
                    "delta_expansion_mae": mean(expansion),
                    "delta_sign_accuracy": mean(sign),
                    "delta_future_mse": mean(latent),
                    "delta_increment_mse": mean(increment),
                    "expansion_improves_all_seeds": all(v < 0 for v in expansion),
                    "joint_future_expansion_sign_improvement": mean(latent) < 0
                    and mean(expansion) < 0
                    and mean(sign) > 0,
                }
            )
    both_mean, both_all_seeds = [], []
    for group in groups:
        selected = [r for r in rows if r["sequence"] == group]
        if all(r["delta_expansion_mae"] < 0 for r in selected):
            both_mean.append(group)
        if all(r["expansion_improves_all_seeds"] for r in selected):
            both_all_seeds.append(group)
    comparisons = {}
    for control in ("B_static", "C_shuffle"):
        selected = [r for r in rows if r["control"] == control]
        comparisons[control] = {
            "future_vs_expansion_pearson_n8": correlation(
                [r["delta_future_mse"] for r in selected],
                [r["delta_expansion_mae"] for r in selected],
            ),
            "joint_improvement_sequences": [
                r["sequence"] for r in selected if r["joint_future_expansion_sign_improvement"]
            ],
        }
    sources = [root / "results.json", root / "CAMPAIGN_STATUS.json"]
    sources += [root / f"seed{s}/summary.json" for s in SEEDS]
    summary = {
        "decision": "closed_no_additional_training",
        "scope": "post_hoc_descriptive_analysis_of_historically_exposed_sequences",
        "primary_benefit_gate_passed": False,
        "elapsed_seconds": status["elapsed_seconds"],
        "primary_mae": {a: mean(metric(s, a)["sequence_mean_mae"] for s in SEEDS) for a in ARMS},
        "primary_comparisons": {
            c: result["comparisons"][PRIMARY.replace("|", "/") + "/C_minus_" + c]
            for c in ("B_static", "C_shuffle")
        },
        "latent_comparisons": result["latent_comparisons"],
        "both_controls_improved_mean_sequences": both_mean,
        "both_controls_improved_every_seed_sequences": both_all_seeds,
        "descriptive_associations": comparisons,
        "source_sha256": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "limitations": [
            "Eight sequences; three seeds kept together.",
            "Post hoc subgroup patterns cannot rescue the preregistered gate.",
            "Expansion is annotated log-height change, not a rate or TTC seconds.",
            "No physical scenario labels assigned from opaque sequence IDs.",
            "Predictor-retained TTC and autoregressive rollouts remain untested.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "closure_analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (out / "sequence_deltas.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Generated pilot closure tables",
        "",
        "MAE is dimensionless annotated log-height change; lower is better.",
        "",
        "| Arm | Mean primary MAE |",
        "|---|---:|",
    ]
    lines += [f"| {a} | {v:.8f} |" for a, v in summary["primary_mae"].items()]
    lines += [
        "",
        "| Sequence | Control | Delta expansion MAE | Delta sign accuracy | "
        "Delta future MSE | Delta increment MSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['sequence']} | {r['control']} | {r['delta_expansion_mae']:+.8f} | "
        f"{r['delta_sign_accuracy']:+.5f} | {r['delta_future_mse']:+.6f} | "
        f"{r['delta_increment_mse']:+.6f} |"
        for r in rows
    ]
    lines += [
        "",
        "Descriptive only: all eight sequences retained; no subgroup selection or new training.",
    ]
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "both_controls_every_seed": both_all_seeds,
                "associations": comparisons,
                "output": str(out),
            }
        )
    )


if __name__ == "__main__":
    main()
