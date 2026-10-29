"""Regenerate the public EvTTC competition tables and figures.

This generator is deliberately self-contained: it reads only the sanitized CSV
and JSON bundled under ``data/``. It never opens the private inference ledgers.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
CSV_PATH = DATA / "evttc_competition_predictions.csv"
SUMMARY_PATH = DATA / "evttc_competition_summary.json"

SEQUENCES = [
    "CCRs1-low",
    "CCRs1-medium",
    "CCRs1-high",
    "CCRs2-low",
    "CCRs2-medium",
    "CCRs2-high",
    "CCRm-low",
    "CCRm-medium",
    "Slider-750",
    "Slider-1000",
]
MODELS = ["phase", "garl_full", "garl_event", "garl_visual"]
VARIANTS = MODELS[1:]
LABELS = {
    "phase": "PHASE-TTC",
    "garl_full": "Garl full",
    "garl_event": "Garl event-only",
    "garl_visual": "Garl visual-only",
}
COLORS = {
    "phase": "#3769DB",
    "garl_full": "#268B86",
    "garl_event": "#AD534B",
    "garl_visual": "#B57732",
}
INK, MUTED, GRID, LIGHT = "#172335", "#667386", "#D5DCE7", "#F5F7FA"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.7,
        "axes.facecolor": LIGHT,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 11.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.fontsize": 10.5,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.72,
    }
)


def finite(value: str) -> float | None:
    if value == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite numeric value: {value!r}")
    return number


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q, method="linear"))


def assert_close(actual: float, expected: float, label: str, tolerance: float = 1e-10) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{label}: {actual} != {expected}")


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def load_public_data() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_text = CSV_PATH.read_text(encoding="utf-8") + SUMMARY_PATH.read_text(encoding="utf-8")
    forbidden = [r"[A-Za-z]:\\", r"/Users/", r"\\Users\\", "path_resolved", "hostname"]
    for pattern in forbidden:
        if re.search(pattern, raw_text, flags=re.IGNORECASE):
            raise ValueError(f"Public evidence contains forbidden path or host token: {pattern}")

    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        rows: list[dict[str, Any]] = list(csv.DictReader(handle))
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary["schema"] != "phase_ttc_evttc_public_evidence_v1":
        raise ValueError("Unexpected public evidence schema")
    return rows, summary


def validate(rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    if len(rows) != 1736:
        raise ValueError(f"Expected 1,736 prediction rows, found {len(rows)}")
    counts = Counter(row["model"] for row in rows)
    if counts != Counter({model: 434 for model in MODELS}):
        raise ValueError(f"Unexpected model counts: {counts}")

    identities: dict[str, set[tuple[str, int, int, str]]] = defaultdict(set)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_identity: dict[tuple[str, int, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        model = row["model"]
        if model not in MODELS or row["sequence_id"] not in SEQUENCES:
            raise ValueError(f"Unknown model or sequence: {row}")
        identity = (
            row["sequence_id"],
            int(row["row_index"]),
            int(row["index"]),
            row["timestamp_text"],
        )
        if model in by_identity[identity]:
            raise ValueError(f"Duplicate model/identity: {model}, {identity}")
        identities[model].add(identity)
        by_identity[identity][model] = row
        by_model[model].append(row)
        prediction = finite(row["prediction_ttc_s"])
        status = row["status"]
        if status == "ok" and prediction is None:
            raise ValueError(f"Available row lacks prediction: {identity}, {model}")
        if status == "unavailable" and prediction is not None:
            raise ValueError(f"Unavailable row has prediction: {identity}, {model}")
        if status not in {"ok", "unavailable"}:
            raise ValueError(f"Unknown status: {status}")
        model_wall = finite(row["model_wall_s"])
        cost = finite(row["cost_s"])
        if model_wall is None or cost is None or model_wall < 0 or cost < model_wall:
            raise ValueError(f"Invalid timing: {identity}, {model}")

    baseline = identities["phase"]
    if any(identities[model] != baseline for model in MODELS):
        raise ValueError("The four models do not share the same 434 identities")
    if any(set(group) != set(MODELS) for group in by_identity.values()):
        raise ValueError("At least one identity lacks one of the four model rows")

    sequence_counts = Counter(identity[0] for identity in baseline)
    if dict(sequence_counts) != summary["query_population"]["sequence_queries"]:
        raise ValueError("Sequence population does not match the public summary")

    stats: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        predictions = [finite(row["prediction_ttc_s"]) for row in by_model[model]]
        available = [value for value in predictions if value is not None]
        model_walls = [float(row["model_wall_s"]) for row in by_model[model]]
        costs = [float(row["cost_s"]) for row in by_model[model]]
        stats[model] = {
            "records": len(predictions),
            "finite_predictions": len(available),
            "null_predictions": len(predictions) - len(available),
            "negative_predictions": sum(value < 0 for value in available),
            "prediction_min_s": min(available),
            "prediction_max_s": max(available),
            "model_wall_median_s": statistics.median(model_walls),
            "model_wall_p95_s": percentile(model_walls, 95),
            "full_cost_median_s": statistics.median(costs),
            "full_cost_p95_s": percentile(costs, 95),
        }
        reference = summary["phase_stats"] if model == "phase" else summary["variant_stats"][model]
        for key, value in stats[model].items():
            if isinstance(value, float):
                assert_close(value, float(reference[key]), f"{model}/{key}")
            elif value != reference[key]:
                raise ValueError(f"{model}/{key}: {value} != {reference[key]}")

    disagreements: dict[str, dict[str, float | int]] = {}
    per_sequence: list[dict[str, Any]] = []
    for model in VARIANTS:
        values: list[float] = []
        sequence_values: dict[str, list[float]] = defaultdict(list)
        for identity, group in by_identity.items():
            phase = finite(group["phase"]["prediction_ttc_s"])
            other = finite(group[model]["prediction_ttc_s"])
            if phase is not None and other is not None:
                difference = abs(other - phase)
                values.append(difference)
                sequence_values[identity[0]].append(difference)
        disagreements[model] = {
            "shared_valid_queries": len(values),
            "null_gap_queries": 434 - len(values),
            "absolute_disagreement_median_s": statistics.median(values),
            "absolute_disagreement_p95_s": percentile(values, 95),
        }
        reference = summary["prediction_disagreement"][model]
        for key, value in disagreements[model].items():
            if isinstance(value, float):
                assert_close(value, float(reference[key]), f"{model}/{key}")
            elif value != reference[key]:
                raise ValueError(f"{model}/{key}: {value} != {reference[key]}")
        for sequence in SEQUENCES:
            values_sequence = sequence_values[sequence]
            queries = sequence_counts[sequence]
            per_sequence.append(
                {
                    "sequence_id": sequence,
                    "variant": model,
                    "queries": queries,
                    "shared_valid_queries": len(values_sequence),
                    "null_gap_queries": queries - len(values_sequence),
                    "coverage_fraction": len(values_sequence) / queries,
                    "absolute_disagreement_median_s": statistics.median(values_sequence),
                    "absolute_disagreement_p95_s": percentile(values_sequence, 95),
                }
            )

    expected_sequence = {
        (row["sequence_id"], row["variant"]): row
        for row in summary["per_sequence_prediction_disagreement"]
    }
    if len(expected_sequence) != 30:
        raise ValueError("Expected 30 sequence-variant summary cells")
    for row in per_sequence:
        expected = expected_sequence[(row["sequence_id"], row["variant"])]
        for key, value in row.items():
            if isinstance(value, float):
                assert_close(value, float(expected[key]), f"per-sequence/{key}")
            elif value != expected[key]:
                raise ValueError(f"per-sequence/{key}: {value} != {expected[key]}")

    return {
        "by_model": by_model,
        "by_identity": by_identity,
        "sequence_counts": sequence_counts,
        "stats": stats,
        "disagreements": disagreements,
        "per_sequence": per_sequence,
    }


def write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_macros(derived: dict[str, Any], summary: dict[str, Any]) -> None:
    stats = derived["stats"]
    disagreement = derived["disagreements"]
    lines = [r"% Generated by scripts/make_evttc_results.py. Do not edit manually."]
    lines.extend(
        [
            r"\newcommand{\EvTTCQueryCount}{434}",
            r"\newcommand{\EvTTCSequenceCount}{10}",
            r"\newcommand{\EvTTCPredictionRowCount}{1,736}",
        ]
    )
    prefixes = {
        "phase": "EvTTCPhase",
        "garl_full": "EvTTCGarlFull",
        "garl_event": "EvTTCGarlEvent",
        "garl_visual": "EvTTCGarlVisual",
    }
    for model in MODELS:
        prefix = prefixes[model]
        model_stats = stats[model]
        lines.extend(
            [
                rf"\newcommand{{\{prefix}Finite}}{{{model_stats['finite_predictions']}}}",
                rf"\newcommand{{\{prefix}Coverage}}{{{model_stats['finite_predictions']}/434}}",
                rf"\newcommand{{\{prefix}NegativeCount}}{{{model_stats['negative_predictions']}}}",
                rf"\newcommand{{\{prefix}PredictionMin}}{{{model_stats['prediction_min_s']:.3f}}}",
                rf"\newcommand{{\{prefix}PredictionMax}}{{{model_stats['prediction_max_s']:.3f}}}",
                rf"\newcommand{{\{prefix}ModelMedian}}{{{model_stats['model_wall_median_s']:.3f}\,s}}",
                rf"\newcommand{{\{prefix}ModelPninetyfive}}{{{model_stats['model_wall_p95_s']:.3f}\,s}}",
                rf"\newcommand{{\{prefix}CostMedian}}{{{model_stats['full_cost_median_s']:.3f}\,s}}",
                rf"\newcommand{{\{prefix}CostPninetyfive}}{{{model_stats['full_cost_p95_s']:.3f}\,s}}",
            ]
        )
        if model != "phase":
            values = disagreement[model]
            lines.extend(
                [
                    rf"\newcommand{{\{prefix}MedianDisagreement}}{{{values['absolute_disagreement_median_s']:.3f}\,s}}",
                    rf"\newcommand{{\{prefix}PninetyfiveDisagreement}}{{{values['absolute_disagreement_p95_s']:.3f}\,s}}",
                ]
            )
    average = next(
        row["rte_percent"]
        for row in summary["published_garl_rte"]
        if row["sequence_id"] == "Unweighted average"
    )
    lines.append(rf"\newcommand{{\EvTTCPublishedGarlAverageRTE}}{{{average:.2f}\%}}")
    write(TABLES / "evttc_macros.tex", lines)


def make_model_summary(derived: dict[str, Any]) -> None:
    stats = derived["stats"]
    disagreement = derived["disagreements"]
    lines = [
        r"\begin{table}[tbp]",
        r"\centering\small",
        r"\caption{Predictions on 434 EvTTC competition queries. The table reports "
        r"finite-output coverage, signed TTC range, and negative counts. Median and "
        r"P95 summarize absolute differences from PHASE-TTC where both models "
        r"produce finite outputs. \evidence{T01}}",
        r"\label{tab:evttc-model-summary}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Finite} & \textbf{Range (s)} & "
        r"\textbf{$<0$} & \textbf{Median (s)} & \textbf{P95 (s)} \\",
        r"\midrule",
    ]
    for model in MODELS:
        model_stats = stats[model]
        if model == "phase":
            median, p95 = "--", "--"
        else:
            median = f"{disagreement[model]['absolute_disagreement_median_s']:.3f}"
            p95 = f"{disagreement[model]['absolute_disagreement_p95_s']:.3f}"
        lines.append(
            f"{LABELS[model]} & {model_stats['finite_predictions']}/434 & "
            f"[{model_stats['prediction_min_s']:.3f}, {model_stats['prediction_max_s']:.3f}] & "
            f"{model_stats['negative_predictions']} & {median} & {p95} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write(TABLES / "evttc_model_summary.tex", lines)


def make_sequence_table(derived: dict[str, Any]) -> None:
    indexed = {(row["sequence_id"], row["variant"]): row for row in derived["per_sequence"]}
    lines = [
        r"\begin{table}[H]",
        r"\centering\scriptsize",
        r"\caption{Garl--PHASE prediction comparisons for all ten sequences. "
        r"Each entry gives shared-finite coverage and median absolute "
        r"prediction difference in seconds. \evidence{T01}}",
        r"\label{tab:evttc-sequence-disagreement}",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"\textbf{Sequence} & \textbf{Full $n/N$; med.} & "
        r"\textbf{Event $n/N$; med.} & \textbf{Visual $n/N$; med.} \\",
        r"\midrule",
    ]
    for sequence in SEQUENCES:
        cells = []
        for model in VARIANTS:
            row = indexed[(sequence, model)]
            cells.append(
                f"{row['shared_valid_queries']}/{row['queries']}; "
                f"{row['absolute_disagreement_median_s']:.3f}"
            )
        lines.append(f"{tex_escape(sequence)} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write(TABLES / "evttc_sequence_disagreement.tex", lines)


def make_runtime_table(derived: dict[str, Any]) -> None:
    stats = derived["stats"]
    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\caption{Offline processing time in seconds per query. Model time covers "
        r"the synchronized forward pass and host output; full time includes "
        r"tensor preparation. The Garl variants ran in full/event/visual order; "
        r"PHASE-TTC used its own adapter and history length. \evidence{T01}}",
        r"\label{tab:evttc-runtime}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Model med.} & \textbf{Model P95} & "
        r"\textbf{Full med.} & \textbf{Full P95} \\",
        r"\midrule",
    ]
    for model in MODELS:
        row = stats[model]
        lines.append(
            f"{LABELS[model]} & {row['model_wall_median_s']:.3f} & "
            f"{row['model_wall_p95_s']:.3f} & {row['full_cost_median_s']:.3f} & "
            f"{row['full_cost_p95_s']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write(TABLES / "evttc_runtime.tex", lines)


def make_published_table(summary: dict[str, Any]) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\caption{Garl-TTC's published zero-shot relative TTC error on three "
        r"EvTTC sequences, as reported in the eAP paper, Table VI. \evidence{T02}}",
        r"\label{tab:evttc-published-rte}",
        r"\begin{tabular}{@{}lr@{}}",
        r"\toprule",
        r"\textbf{Published sequence} & \textbf{RTE (\%)} \\",
        r"\midrule",
    ]
    for row in summary["published_garl_rte"]:
        prefix = r"\textbf{" if row["sequence_id"] == "Unweighted average" else ""
        suffix = "}" if prefix else ""
        lines.append(
            f"{prefix}{tex_escape(row['sequence_id'])}{suffix} & "
            f"{prefix}{row['rte_percent']:.2f}{suffix} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write(TABLES / "evttc_published_rte.tex", lines)


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def grouped_series(derived: dict[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    output: dict[str, dict[str, list[dict[str, Any]]]] = {
        sequence: {model: [] for model in MODELS} for sequence in SEQUENCES
    }
    for model, rows in derived["by_model"].items():
        for row in rows:
            output[row["sequence_id"]][model].append(row)
    for sequence in SEQUENCES:
        for model in MODELS:
            output[sequence][model].sort(key=lambda row: int(row["row_index"]))
    return output


def make_prediction_figure(derived: dict[str, Any], symlog: bool) -> None:
    series = grouped_series(derived)
    available = [
        float(row["prediction_ttc_s"])
        for rows in derived["by_model"].values()
        for row in rows
        if row["prediction_ttc_s"] != ""
    ]
    global_min, global_max = min(available), max(available)
    padding = 0.035 * (global_max - global_min)
    fig, axes = plt.subplots(5, 2, figsize=(11.2, 13.2), sharey=True)
    handles = []
    for axis_index, (ax, sequence) in enumerate(zip(axes.ravel(), SEQUENCES, strict=True)):
        for model in MODELS:
            rows = series[sequence][model]
            x = np.asarray([float(row["timestamp_text"]) for row in rows])
            y = np.asarray(
                [
                    np.nan if row["prediction_ttc_s"] == "" else float(row["prediction_ttc_s"])
                    for row in rows
                ]
            )
            (line,) = ax.plot(
                x,
                y,
                color=COLORS[model],
                lw=1.35 if model == "phase" else 1.05,
                alpha=0.96,
                label=LABELS[model],
                zorder=4 if model == "phase" else 3,
            )
            if axis_index == 0:
                handles.append(line)
            missing = np.isnan(y)
            if missing.any():
                ax.plot(
                    x[missing],
                    np.full(int(missing.sum()), 0.035),
                    marker="x",
                    ls="none",
                    ms=5,
                    mew=1.1,
                    color=COLORS[model],
                    transform=ax.get_xaxis_transform(),
                    clip_on=False,
                    zorder=7,
                )
        ax.axhline(0, color=INK, lw=0.65, alpha=0.7)
        ax.set_title(f"{sequence}  /  n={len(series[sequence]['phase'])}", loc="left", pad=6)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        if symlog:
            ax.set_yscale("symlog", linthresh=1.0, linscale=1.0)
        else:
            ax.set_ylim(global_min - padding, global_max + padding)
        if axis_index // 2 == 4:
            ax.set_xlabel("Query timestamp (s)")
        if axis_index % 2 == 0:
            ax.set_ylabel("Predicted TTC (s)")
    title = (
        "All signed predictions on a common symmetric-log scale"
        if symlog
        else "All signed predictions on one common linear scale"
    )
    subtitle = (
        "Symmetric log (linear within +/-1 s) retains the 406 s outlier; "
        "lower-edge x markers are null gaps"
        if symlog
        else "Full range retained; x markers at the lower axis edge denote explicit null gaps"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.997)
    fig.text(0.5, 0.977, subtitle, ha="center", va="top", fontsize=10.5, color=MUTED)
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.957))
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.2, w_pad=1.25)
    save_figure(fig, "evttc_predictions_symlog" if symlog else "evttc_predictions_linear")


def make_coverage_figure(derived: dict[str, Any]) -> None:
    counts = derived["sequence_counts"]
    indexed = {(row["sequence_id"], row["variant"]): row for row in derived["per_sequence"]}
    fig, ax = plt.subplots(figsize=(10.8, 5.3))
    x = np.arange(len(SEQUENCES))
    width = 0.24
    for offset, model in zip([-1, 0, 1], VARIANTS, strict=True):
        values = [100 * indexed[(sequence, model)]["coverage_fraction"] for sequence in SEQUENCES]
        bars = ax.bar(
            x + offset * width,
            values,
            width,
            color=COLORS[model],
            label=LABELS[model],
        )
        for bar, sequence, value in zip(bars, SEQUENCES, values, strict=True):
            available = indexed[(sequence, model)]["shared_valid_queries"]
            if available != counts[sequence]:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value - 0.4,
                    f"{available}/{counts[sequence]}",
                    ha="center",
                    va="top",
                    rotation=90,
                    fontsize=8.5,
                    color="white",
                    fontweight="bold",
                )
    ax.set_xticks(x, [sequence.replace("-", "\n", 1) for sequence in SEQUENCES])
    ax.set_ylim(90, 101.6)
    ax.set_ylabel("Finite prediction coverage (%)")
    ax.set_title("Coverage is reported without filling unavailable predictions", pad=42)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.035))
    fig.text(
        0.995,
        0.012,
        "PHASE-TTC: 434/434 finite (not shown; 100% in every sequence).",
        ha="right",
        fontsize=9.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, "evttc_coverage")


def make_published_rte_figure(summary: dict[str, Any]) -> None:
    rows = summary["published_garl_rte"]
    sequence_rows = [row for row in rows if row["sequence_id"] != "Unweighted average"]
    average = next(row for row in rows if row["sequence_id"] == "Unweighted average")
    fig, ax = plt.subplots(figsize=(8.7, 4.4))
    bars = ax.bar(
        [row["sequence_id"].replace("-", "\n", 1) for row in sequence_rows],
        [row["rte_percent"] for row in sequence_rows],
        color=["#3769DB", "#268B86", "#B57732"],
        width=0.62,
    )
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=11.5, color=INK)
    ax.axhline(average["rte_percent"], color="#AD534B", lw=1.4, ls=(0, (5, 3)))
    ax.text(
        -0.43,
        average["rte_percent"] + 0.18,
        f"Published average {average['rte_percent']:.2f}%",
        ha="left",
        va="bottom",
        color="#AD534B",
        fontsize=10.5,
    )
    ax.set_ylim(0, 15.5)
    ax.set_ylabel("Published relative TTC error (%)")
    ax.set_title("Published Garl-TTC zero-shot EvTTC result", pad=12)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.text(
        0.5,
        0.015,
        "Separate three-sequence literature evaluation; not an accuracy comparison with T01.",
        ha="center",
        fontsize=10,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, "evttc_published_rte")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    rows, summary = load_public_data()
    derived = validate(rows, summary)
    make_macros(derived, summary)
    make_model_summary(derived)
    make_sequence_table(derived)
    make_runtime_table(derived)
    make_published_table(summary)
    make_prediction_figure(derived, symlog=False)
    make_prediction_figure(derived, symlog=True)
    make_coverage_figure(derived)
    make_published_rte_figure(summary)
    print(
        json.dumps(
            {
                "prediction_rows": len(rows),
                "query_identities": 434,
                "sequence_variant_cells": 30,
                "stats": derived["stats"],
                "disagreements": derived["disagreements"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
