"""Render the verified official benchmark comparison from its public ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "official_benchmark.json"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"

PHASE = "#268B86"
GARL = "#AAB5C4"
INK = "#172335"
MUTED = "#667386"
GRID = "#D5DCE7"
LIGHT = "#F5F7FA"

METRICS = (
    ("overall_MiD", "Overall"),
    ("MiDc", "Crucial"),
    ("MiDs", "Small"),
    ("MiDl", "Large"),
    ("MiDn", "Negative"),
)
BAND_METRICS = METRICS[1:]
EXPECTED_SOURCE_HASHES = {
    "B01": "0c7114c5acfded5e661446cd483735da99c0156fe553e81cc3ca1fe14c558067",
    "B02": "e06e99ba1e5d6c4eccc0a475384d797961de771f61d7dbb0fc56a00dc9e75092",
}

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
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.fontsize": 11.5,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
    }
)


def load_ledger(path: Path) -> dict[str, Any]:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    required_top = {"as_of", "benchmark", "phase_submission", "competitor"}
    if not required_top.issubset(ledger):
        raise ValueError(f"Incomplete official benchmark ledger: {path}")

    benchmark = ledger["benchmark"]
    if benchmark.get("task_id") != 34531 or benchmark.get("lower_is_better") is not True:
        raise ValueError("Unexpected official benchmark identity or metric direction")
    if benchmark.get("name") != "GarlTTC Test":
        raise ValueError("Unexpected official benchmark display name")

    phase = ledger["phase_submission"]
    garl = ledger["competitor"]
    if phase.get("id") != 1016512 or phase.get("rank") != 1:
        raise ValueError("The ledger does not identify the verified first-place PHASE submission")
    if garl.get("id") != 817519:
        raise ValueError("The ledger does not identify the official Garl-TTC comparison")

    required_metrics = {key for key, _ in METRICS} | {"failed_rate", "num_samples"}
    for entry in (phase, garl):
        metrics = entry.get("metrics", {})
        if not required_metrics.issubset(metrics):
            raise ValueError(f"Missing benchmark metrics for submission {entry.get('id')}")
        if any(not isinstance(metrics[key], (int, float)) for key in required_metrics):
            raise TypeError(f"Non-numeric benchmark metric for submission {entry.get('id')}")
        if metrics["num_samples"] != 6762 or metrics["failed_rate"] != 0:
            raise ValueError("Official comparison requires 6,762 samples and zero failures")

    if phase["metrics"]["overall_MiD"] >= garl["metrics"]["overall_MiD"]:
        raise ValueError("Verified PHASE submission must have the lower official Overall MiD")
    weights = benchmark.get("band_weights", {})
    if weights != {"MiDc": 0.5, "MiDs": 0.3, "MiDl": 0.1, "MiDn": 0.1}:
        raise ValueError("Unexpected official band weights")
    provenance = ledger.get("provenance", {})
    for source_id, expected_hash in EXPECTED_SOURCE_HASHES.items():
        source = provenance.get(source_id, {})
        if source.get("original_sha256") != expected_hash:
            raise ValueError(f"Official source hash changed for {source_id}")
    return ledger


def reduction(phase_value: float, competitor_value: float) -> float:
    return 100.0 * (competitor_value - phase_value) / competitor_value


def macro_name(label: str) -> str:
    return {
        "Overall": "Overall",
        "Crucial": "Crucial",
        "Small": "Small",
        "Large": "Large",
        "Negative": "Negative",
    }[label]


def weighted_contributions(ledger: dict[str, Any]) -> list[tuple[str, float, float, float]]:
    phase = ledger["phase_submission"]["metrics"]
    garl = ledger["competitor"]["metrics"]
    weights = ledger["benchmark"]["band_weights"]
    contributions = []
    for key, label in BAND_METRICS:
        weight = float(weights[key])
        difference = float(garl[key]) - float(phase[key])
        contributions.append((label, weight, difference, weight * difference))
    return contributions


def write_macros(ledger: dict[str, Any]) -> None:
    phase = ledger["phase_submission"]
    garl = ledger["competitor"]
    contributions = weighted_contributions(ledger)
    band_subtotal = sum(item[3] for item in contributions)
    overall_gap = float(garl["metrics"]["overall_MiD"]) - float(phase["metrics"]["overall_MiD"])
    rounding_residual = overall_gap - band_subtotal
    crucial_small_share = 100.0 * sum(item[3] for item in contributions[:2]) / overall_gap
    lines = [
        "% Generated by scripts/make_official_results.py; do not edit by hand.",
        rf"\providecommand{{\OfficialPhasePublishedOverallMiD}}{{{phase['metrics']['overall_MiD']:.2f}}}",
        rf"\providecommand{{\OfficialOverallAbsoluteGain}}{{{overall_gap:.9f}}}",
        rf"\providecommand{{\OfficialBandContributionSubtotal}}{{{band_subtotal:.9f}}}",
        rf"\providecommand{{\OfficialDisplayRoundingResidual}}{{{rounding_residual:.4f}}}",
        rf"\providecommand{{\OfficialCrucialSmallContributionShare}}{{{crucial_small_share:.1f}\%}}",
    ]
    for key, label in METRICS:
        stem = macro_name(label)
        phase_value = float(phase["metrics"][key])
        garl_value = float(garl["metrics"][key])
        lines.extend(
            (
                rf"\providecommand{{\OfficialPhase{stem}MiD}}{{{phase_value:.4f}}}",
                rf"\providecommand{{\OfficialGarl{stem}MiD}}{{{garl_value:.2f}}}",
                (
                    rf"\providecommand{{\Official{stem}RelativeGain}}"
                    rf"{{{reduction(phase_value, garl_value):.2f}\%}}"
                ),
            )
        )
    lines.extend(
        (
            rf"\providecommand{{\OfficialSampleCount}}{{{int(phase['metrics']['num_samples']):,}}}",
            rf"\providecommand{{\OfficialPhaseFailureRate}}{{{float(phase['metrics']['failed_rate']):.1f}\%}}",
            rf"\providecommand{{\OfficialGarlFailureRate}}{{{float(garl['metrics']['failed_rate']):.1f}\%}}",
            rf"\providecommand{{\OfficialPhaseSubmissionID}}{{{int(phase['id'])}}}",
            rf"\providecommand{{\OfficialGarlSubmissionID}}{{{int(garl['id'])}}}",
        )
    )
    (TABLES / "official_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table(ledger: dict[str, Any]) -> None:
    phase = ledger["phase_submission"]
    garl = ledger["competitor"]
    rows = []
    for key, label in METRICS:
        phase_value = float(phase["metrics"][key])
        garl_value = float(garl["metrics"][key])
        rows.append(
            f"{label} & {phase_value:.4f} & {garl_value:.2f} & "
            f"{reduction(phase_value, garl_value):.2f}\\% \\\\"
        )
    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\caption{Official GarlTTC Test scores on 28 October 2026 (lower is better). "
        r"PHASE-TTC scores are shown to four decimals and Garl-TTC at published "
        r"precision. Relative reductions use unrounded PHASE-TTC scores. "
        r"Both entries have 6,762 scored tokens and 0\% scorer-defined failures. "
        r"\evidence{B01,B02}}",
        r"\label{tab:official-benchmark}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{PHASE-TTC} & \textbf{Garl-TTC} & \textbf{Decrease} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (TABLES / "official_benchmark.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_contribution_table(ledger: dict[str, Any]) -> None:
    contributions = weighted_contributions(ledger)
    phase = ledger["phase_submission"]["metrics"]
    garl = ledger["competitor"]["metrics"]
    band_subtotal = sum(item[3] for item in contributions)
    overall_gap = float(garl["overall_MiD"]) - float(phase["overall_MiD"])
    residual = overall_gap - band_subtotal
    rows = [
        f"{label} & {weight:.1f} & {difference:.4f} & {contribution:.4f} \\\\"
        for label, weight, difference, contribution in contributions
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\caption{Additive decomposition of the official Overall MiD difference. "
        r"Each band contributes $w_b(\mathrm{MiD}^{\mathrm{Garl}}_b-"
        r"\mathrm{MiD}^{\mathrm{PHASE}}_b)$. The 0.0010-point reconciliation residual "
        r"comes from Garl-TTC's published two-decimal band scores and appears "
        r"separately below. \evidence{B01,B02}}",
        r"\label{tab:official-weighted-contributions}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Band} & \textbf{Weight} & \textbf{MiD difference} & "
        r"\textbf{Weighted contribution} \\",
        r"\midrule",
        *rows,
        r"\midrule",
        f"Band subtotal & 1.0 & -- & {band_subtotal:.9f} \\\\ ",
        f"Published-score rounding residual & -- & -- & {residual:.4f} \\\\ ",
        f"Official Overall difference & -- & -- & {overall_gap:.9f} \\\\ ",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (TABLES / "official_weighted_contributions.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_figure(ledger: dict[str, Any]) -> None:
    phase = ledger["phase_submission"]
    garl = ledger["competitor"]
    phase_values = [float(phase["metrics"][key]) for key, _ in METRICS]
    garl_values = [float(garl["metrics"][key]) for key, _ in METRICS]
    labels = [label for _, label in METRICS]

    x = np.arange(len(METRICS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.15, 3.05))
    phase_bars = ax.bar(x - width / 2, phase_values, width, color=PHASE, label="PHASE-TTC")
    garl_bars = ax.bar(x + width / 2, garl_values, width, color=GARL, label="Garl-TTC")
    ax.bar_label(phase_bars, fmt="%.2f", padding=-18, fontsize=11.5, color="white")
    ax.bar_label(garl_bars, fmt="%.2f", padding=3, fontsize=11.5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("MiD (lower is better)")
    ax.set_ylim(0, max(garl_values) * 1.22)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=2)
    overall_gain = reduction(phase_values[0], garl_values[0])
    ax.set_title(f"Official benchmark: {overall_gain:.2f}% lower Overall MiD", pad=13)
    fig.tight_layout(pad=0.55)
    fig.savefig(FIGURES / "official_benchmark.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(
        FIGURES / "official_benchmark.png",
        bbox_inches="tight",
        pad_inches=0.08,
        dpi=220,
    )
    plt.close(fig)


def write_contribution_figure(ledger: dict[str, Any]) -> None:
    contributions = weighted_contributions(ledger)
    labels = [item[0] for item in contributions]
    values = [item[3] for item in contributions]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.15, 2.8))
    bars = ax.barh(y, values, color=(PHASE, PHASE, "#58AAA5", "#8BC5C1"))
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=11.5)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Contribution to lower Overall MiD (points)")
    ax.set_xlim(0, max(values) * 1.18)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_title("Crucial and small bands dominate the weighted difference", pad=11)
    fig.tight_layout(pad=0.55)
    fig.savefig(
        FIGURES / "official_weighted_contributions.pdf",
        bbox_inches="tight",
        pad_inches=0.08,
    )
    fig.savefig(
        FIGURES / "official_weighted_contributions.png",
        bbox_inches="tight",
        pad_inches=0.08,
        dpi=220,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    ledger = load_ledger(args.data)
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    write_macros(ledger)
    write_table(ledger)
    write_contribution_table(ledger)
    write_figure(ledger)
    write_contribution_figure(ledger)


if __name__ == "__main__":
    main()
