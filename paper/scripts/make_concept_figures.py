"""Regenerate the paper's explanatory vector schematics."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
INK, BLUE, TEAL, RED = "#172335", "#3769DB", "#268B86", "#AD534B"
MUTED, GRID, LIGHT, GRAY = "#667386", "#D5DCE7", "#F5F7FA", "#B5BDCA"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.7,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def box(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    color: str = BLUE,
    size: float = 11.5,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.065",
            facecolor=color + "0E",
            edgecolor=color + "78",
            linewidth=0.8,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.4)


def arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    dashed: bool = False,
    color: str = MUTED,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color=color,
            linestyle="--" if dashed else "-",
        )
    )


def save(fig: Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def overview() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 3.7))
    ax.set(xlim=(0, 10), ylim=(0, 3.7))
    ax.axis("off")
    headings = [
        "Future predictive\nlearning",
        "Geometric scale\nand transport",
        "Causal temporal\nfusion",
        "Streaming\nTTC",
    ]
    texts = [
        "Event windows\nLatent future targets\nEMA teacher",
        "Paired event windows\nApparent expansion\nSigned phase",
        "Event + RGB features\nTemporal context\nPhase correction",
        "Persistent state\nPhase-to-TTC readout\nDeadline response",
    ]
    colors = [BLUE, TEAL, BLUE, TEAL]
    for i, (h, t, c) in enumerate(zip(headings, texts, colors, strict=False)):
        x = 0.08 + i * 2.53
        box(ax, x, 0.88, 2.22, 2.22, "", c)
        ax.text(
            x + 1.11,
            2.59,
            h,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=c,
            linespacing=1.35,
        )
        ax.plot([x + 0.2, x + 2.02], [2.05, 2.05], color=c + "45", lw=0.7)
        ax.text(x + 1.11, 1.48, t, ha="center", va="center", fontsize=11.2, linespacing=1.5)
        if i < 3:
            arrow(ax, (x + 2.25, 1.98), (x + 2.49, 1.98))
    ax.text(
        5,
        0.30,
        "Four research stages culminating in the deployed causal fusion design.",
        ha="center",
        fontsize=10.8,
        color=MUTED,
    )
    save(fig, "overview")


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 9.1))
    ax.set(xlim=(0, 10), ylim=(0, 9))
    ax.axis("off")
    ax.text(
        0.1,
        8.75,
        "A   Predictive representation learning",
        fontweight="bold",
        fontsize=14,
        color=BLUE,
    )
    box(ax, 0.1, 7.5, 1.55, 0.72, "Context events\ncausal windows", size=10.8)
    box(ax, 2.05, 7.5, 1.6, 0.72, "Online encoder\n" + r"$ f_\theta $", size=10.8)
    box(ax, 4.1, 7.5, 1.85, 0.72, "Horizon-conditioned\npredictor " + r"$g_\psi$", size=9.8)
    box(ax, 6.4, 7.5, 1.5, 0.72, "Predicted\nlatent future", size=10.8)
    box(ax, 8.35, 7.5, 1.45, 0.72, "Predictive\nloss", TEAL, size=10.8)
    for s, e in [(1.67, 2.02), (3.68, 4.07), (5.98, 6.37), (7.92, 8.32)]:
        arrow(ax, (s, 7.86), (e, 7.86))
    box(ax, 0.1, 6.25, 1.55, 0.72, "Future events\ntraining target", TEAL, size=10.8)
    box(ax, 2.05, 6.25, 1.6, 0.72, "Target encoder\n" + r"$ f_\xi $", TEAL, size=10.8)
    box(ax, 6.4, 6.25, 1.5, 0.72, "Stop-gradient\nlatent target", TEAL, size=10.8)
    arrow(ax, (1.67, 6.61), (2.02, 6.61))
    arrow(ax, (3.68, 6.61), (6.37, 6.61))
    arrow(ax, (7.93, 6.61), (9.08, 7.46))
    arrow(ax, (2.85, 7.47), (2.85, 7.0), True, TEAL)
    ax.text(3.05, 7.19, "EMA", fontsize=10.5, color=TEAL)
    ax.text(
        0.1,
        5.91,
        "Configured predictive losses use future-event targets during training.",
        fontsize=10.6,
        color=MUTED,
    )
    ax.axhline(5.65, color=GRID, lw=0.8)
    ax.text(
        0.1,
        5.3,
        "B   Causal event geometry and transport",
        fontweight="bold",
        fontsize=14,
        color=TEAL,
    )
    labels = [
        "Endpoint pair\ncommon ROI\n(annotated)",
        "Shared dense\nevent encoder",
        "Foreground scale /\nlocal transport",
        "Signed log-ratio\nand validity",
        "Signed TTC /\ninverse TTC",
    ]
    for i, t in enumerate(labels):
        x = 0.1 + 2.02 * i
        box(ax, x, 3.95, 1.7, 0.93, t, TEAL, 10.2)
        if i < 4:
            arrow(ax, (x + 1.73, 4.42), (x + 1.98, 4.42))
    ax.text(
        0.1,
        3.45,
        "Geometry-supervised training; prefix-causal inputs at inference.",
        fontsize=10.6,
        color=MUTED,
    )
    ax.axhline(3.12, color=GRID, lw=0.8)
    ax.text(
        0.1,
        2.8,
        "C   Multimodal causal phase correction",
        fontweight="bold",
        fontsize=14,
        color=BLUE,
    )
    box(ax, 0.1, 1.65, 2.1, 0.65, "Event expert\nfeatures", BLUE)
    box(ax, 0.1, 0.55, 2.1, 0.65, "RGB context\n+ phase", TEAL)
    box(ax, 3.05, 1.0, 2.6, 1.0, "Causal fusion / residual\nphase correction", BLUE)
    box(ax, 6.4, 1.0, 1.55, 1.0, "Phase-to-TTC\nreadout", TEAL)
    box(ax, 8.5, 1.0, 1.3, 1.0, "Validity +\noutputs", TEAL)
    arrow(ax, (2.23, 1.97), (3.02, 1.6))
    arrow(ax, (2.23, 0.87), (3.02, 1.35))
    arrow(ax, (5.68, 1.5), (6.37, 1.5))
    arrow(ax, (7.98, 1.5), (8.47, 1.5))
    ax.text(
        0.1,
        0.06,
        "Family schematic. Controls: TRUE / ZERO; SHUFFLE / REPEAT where executed.",
        fontsize=10.6,
        color=MUTED,
    )
    save(fig, "architecture")


def history() -> None:
    d = json.loads((ROOT / "data/git_summary.json").read_text())
    fig, ax = plt.subplots(figsize=(9.2, 2.8))
    vals = d["commits_by_author_month"]
    bars = ax.bar(
        [k[5:] for k in vals], list(vals.values()), color=[BLUE, BLUE, TEAL, TEAL, RED], width=0.58
    )
    ax.bar_label(bars, padding=4, fontsize=12)
    ax.set_ylim(0, max(vals.values()) * 1.20)
    ax.set_ylabel("Reachable commits")
    ax.set_xlabel("2026 author month (June is partial)")
    ax.set_facecolor(LIGHT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, alpha=0.65, linewidth=0.6)
    ax.set_axisbelow(True)
    save(fig, "history_activity")


def evidence_flow() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 3.1))
    ax.set(xlim=(0, 10), ylim=(0, 3.1))
    ax.axis("off")
    for x, t in [
        (0.1, "Prediction records\nJSON / CSV"),
        (2.65, "Protocol identity\nrows, split, seed"),
        (5.2, "Evaluation definition\nmetric + population"),
        (7.75, "Reported estimate\nwith uncertainty"),
    ]:
        box(ax, x, 1.6, 2.08, 1.05, t, TEAL, 11.5)
    for x in [2.2, 4.75, 7.3]:
        arrow(ax, (x, 2.13), (x + 0.4, 2.13))
    box(
        ax,
        1.0,
        0.25,
        8.0,
        0.72,
        "Invalid predictions and failed runs retain their original evaluation populations.",
        BLUE,
        11.5,
    )
    save(fig, "evidence_flow")


def deployed_architecture() -> None:
    """Exact deployed graph: four producers and five heads in one inference chain."""
    fig, ax = plt.subplots(figsize=(13.4, 5.8))
    ax.set(xlim=(0, 20), ylim=(0, 6.7))
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.05, 3.45),
            13.50,
            2.75,
            boxstyle="round,pad=.02,rounding_size=.12",
            facecolor=TEAL + "08",
            edgecolor=TEAL + "30",
            linewidth=0.8,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.05, 0.35),
            13.50,
            2.75,
            boxstyle="round,pad=.02,rounding_size=.12",
            facecolor=BLUE + "08",
            edgecolor=BLUE + "30",
            linewidth=0.8,
        )
    )
    ax.text(0.25, 5.92, "EVENT MEASUREMENT STREAM", fontsize=11.2, fontweight="bold", color=TEAL)
    ax.text(0.25, 2.82, "RGB MEASUREMENT STREAM", fontsize=11.2, fontweight="bold", color=BLUE)

    def lane(y: float, color: str, prefix: str, input_text: str) -> None:
        numbers = ("1", "2", "3", "4") if prefix == "E" else ("5", "6", "7", "8")
        box(ax, 0.25, y + 0.30, 1.55, 0.98, input_text, color, 11.0)
        box(ax, 2.30, y + 0.86, 1.40, 0.68, f"{numbers[0]}  {prefix}-A5", color, 11.0)
        box(ax, 2.30, y + 0.06, 1.40, 0.68, f"{numbers[1]}  {prefix}-C2F", color, 11.0)
        box(ax, 4.70, y + 0.86, 1.55, 0.68, f"{numbers[2]}  PAIR-{prefix}", color, 11.0)
        box(ax, 7.00, y + 0.18, 2.70, 1.32, "PHASE17\n+ 4 timing values\n+ valid mask", color, 11.0)
        box(
            ax,
            10.60,
            y + 0.14,
            2.70,
            1.40,
            f"{numbers[3]}  {prefix}-CTX\n2-layer GRU\nwidth 160\nhistory $\\leq 8$",
            color,
            11.0,
        )
        arrow(ax, (1.83, y + 0.79), (2.26, y + 1.20), color=color)
        arrow(ax, (1.83, y + 0.79), (2.26, y + 0.40), color=color)
        arrow(ax, (3.73, y + 1.20), (4.66, y + 1.20), color=color)
        # A5 diagnostics route bends above PAIR, keeping the dependency visible
        # without drawing through the PAIR box or its label.
        bend_y = y + 1.70
        ax.plot([3.73, 4.20, 6.55], [y + 1.20, bend_y, bend_y], color=color, lw=1.35)
        arrow(ax, (6.55, bend_y), (6.96, y + 1.30), color=color)
        arrow(ax, (3.73, y + 0.40), (6.96, y + 0.61), color=color)
        arrow(ax, (6.28, y + 1.20), (6.96, y + 0.96), color=color)
        arrow(ax, (9.73, y + 0.83), (10.56, y + 0.83), color=color)

    lane(3.67, TEAL, "E", "Causal\nevents")
    lane(0.57, BLUE, "R", "Causal\nRGB")
    box(
        ax,
        14.25,
        2.18,
        3.10,
        1.72,
        "9  PHASE-TTC\nRGB anchor\n20-value fusion\n$20\\rightarrow32\\rightarrow32\\rightarrow3$",
        BLUE,
        11.0,
    )
    box(ax, 18.00, 2.45, 1.90, 1.20, "Signed TTC\nphase\nbounds", TEAL, 11.0)
    arrow(ax, (13.33, 4.74), (14.21, 3.52), color=TEAL)
    arrow(ax, (13.33, 1.64), (14.21, 2.58), color=BLUE)
    arrow(ax, (17.38, 3.05), (17.96, 3.05), color=MUTED)
    ax.text(
        10.0,
        0.08,
        "Five out-of-fold rotations generate training features; deployment executes the "
        "single nine-component graph above.",
        ha="center",
        fontsize=11.0,
        color=MUTED,
    )
    save(fig, "deployed_architecture")


def fusion_interface() -> None:
    """The deployed final head selects 20 phase, timing, and support values."""
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    ax.set(xlim=(0, 18), ylim=(0, 6.2))
    ax.axis("off")
    ax.text(
        0.20, 5.88, "FROZEN 340-VALUE CONTEXT CACHE", fontsize=11.5, fontweight="bold", color=MUTED
    )
    ax.text(3.25, 5.38, "RGB block: 170", ha="center", fontsize=11.0, fontweight="bold", color=BLUE)
    ax.text(
        9.75, 5.38, "Event block: 170", ha="center", fontsize=11.0, fontweight="bold", color=TEAL
    )
    box(
        ax,
        0.20,
        3.95,
        3.70,
        1.10,
        "RGB hidden: 160\nupstream context\nends before head",
        GRAY,
        11.0,
    )
    box(ax, 3.95, 3.95, 2.60, 1.10, "10 selected RGB\nphase / timing\n/ support", BLUE, 11.0)
    box(
        ax,
        6.70,
        3.95,
        3.70,
        1.10,
        "Event hidden: 160\nupstream context\nends before head",
        GRAY,
        11.0,
    )
    box(ax, 10.45, 3.95, 2.60, 1.10, "10 selected event\nphase / timing\n/ support", TEAL, 11.0)
    box(
        ax,
        5.00,
        2.52,
        4.20,
        1.02,
        "20 selected values\nscale 5 phase fields / modality\nvalid-history count / 8 included",
        BLUE,
        11.0,
    )
    box(
        ax,
        13.85,
        3.90,
        3.75,
        1.20,
        "RGB raw-location anchor\n+ separately supplied\nRGB width logits",
        BLUE,
        11.0,
    )
    arrow(ax, (5.25, 3.92), (6.20, 3.58), color=BLUE)
    arrow(ax, (11.75, 3.92), (8.00, 3.58), color=TEAL)
    box(ax, 0.20, 0.62, 2.65, 1.15, "Linear $20\\rightarrow32$\nLayerNorm\nSiLU", BLUE, 11.0)
    box(ax, 3.25, 0.62, 2.65, 1.15, "Linear $32\\rightarrow32$\nLayerNorm\nSiLU", BLUE, 11.0)
    box(ax, 6.30, 0.62, 2.45, 1.15, "Linear $32\\rightarrow3$\nzero-initialized", BLUE, 11.0)
    box(ax, 9.30, 0.62, 3.85, 1.15, "Correct RGB raw location\nand 2 width logits", BLUE, 11.0)
    box(ax, 14.10, 0.62, 3.50, 1.15, "Point phase\nsigned TTC\n2 phase bounds", TEAL, 11.0)
    arrow(ax, (7.05, 2.49), (1.53, 1.71), color=MUTED)
    arrow(ax, (2.89, 1.20), (3.21, 1.20))
    arrow(ax, (5.94, 1.20), (6.26, 1.20))
    arrow(ax, (8.79, 1.20), (9.26, 1.20))
    arrow(ax, (13.19, 1.20), (14.06, 1.20))
    arrow(ax, (15.73, 3.87), (11.65, 1.81), color=BLUE)
    ax.text(
        9.0,
        0.12,
        "Final fusion head: 1,955 trainable parameters.",
        ha="center",
        fontsize=11.0,
        color=MUTED,
    )
    save(fig, "fusion_interface")


def crop_geometry() -> None:
    """Conceptual demonstration of why both endpoints must share a crop."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    for ax in axes:
        ax.set(xlim=(0, 10), ylim=(0, 6))
        ax.set_aspect("equal")
        ax.axis("off")
    axes[0].set_title(
        "Shared coordinates preserve apparent growth", loc="left", fontsize=13, pad=12
    )
    axes[0].add_patch(Rectangle((0.6, 0.6), 8.8, 4.8, facecolor=LIGHT, edgecolor=GRID, lw=1))
    axes[0].add_patch(
        Rectangle((1.75, 1.78), 1.35, 2.45, facecolor=TEAL + "25", edgecolor=TEAL, lw=1.5)
    )
    axes[0].add_patch(
        Rectangle((6.20, 1.25), 2.15, 3.50, facecolor=BLUE + "20", edgecolor=BLUE, lw=1.5)
    )
    axes[0].text(2.43, 0.88, "earlier", ha="center", color=TEAL, fontweight="bold")
    axes[0].text(7.28, 0.88, "current", ha="center", color=BLUE, fontweight="bold")
    arrow(axes[0], (3.45, 3.0), (5.85, 3.0), color=MUTED)
    axes[0].text(4.65, 3.25, "same raster", ha="center", fontsize=10.2, color=MUTED)
    axes[1].set_title(
        "Independent resizing removes that scale cue", loc="left", fontsize=13, pad=12
    )
    axes[1].add_patch(Rectangle((0.6, 0.6), 3.55, 4.8, facecolor=LIGHT, edgecolor=GRID, lw=1))
    axes[1].add_patch(Rectangle((5.85, 0.6), 3.55, 4.8, facecolor=LIGHT, edgecolor=GRID, lw=1))
    for x, c, label in [(1.42, TEAL, "earlier"), (6.67, BLUE, "current")]:
        axes[1].add_patch(Rectangle((x, 1.25), 1.9, 3.5, facecolor=c + "20", edgecolor=c, lw=1.5))
        axes[1].text(x + 0.95, 0.88, label, ha="center", color=c, fontweight="bold")
    axes[1].text(5.0, 3.0, "→", ha="center", va="center", fontsize=24, color=MUTED)
    fig.text(
        0.5,
        0.015,
        "Conceptual common-ROI construction; boxes denote object regions in a shared raster.",
        ha="center",
        fontsize=11.0,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=2.4)
    save(fig, "crop_geometry")


def phase_geometry() -> None:
    """Analytic benchmark phase coordinate, with both valid signed branches."""
    import numpy as np

    fig = plt.figure(figsize=(10.8, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[4, 1.25, 4], wspace=0.12)
    left = fig.add_subplot(gs[0, 0])
    mid = fig.add_subplot(gs[0, 1])
    right = fig.add_subplot(gs[0, 2])
    tn = np.linspace(-10, -0.012, 900)
    tp = np.geomspace(0.1005, 10, 900)
    left.plot(tn, -np.log(1 - 0.1 / tn), color=TEAL, lw=2.2)
    right.plot(tp, -np.log(1 - 0.1 / tp), color=BLUE, lw=2.2)
    left.set(
        xlim=(-10, 0),
        ylim=(-2.5, 0.15),
        xlabel="Receding TTC $T$ (s)",
        ylabel=r"Benchmark phase $\phi(T)$",
    )
    right.set(xlim=(0, 10), ylim=(-0.1, 5.5), xlabel="Approaching TTC $T$ (s)")
    for ax in (left, right):
        ax.grid(color=GRID, lw=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
    mid.set(xlim=(0, 0.1), ylim=(0, 1))
    mid.axis("off")
    mid.add_patch(
        Rectangle(
            (0, 0.08), 0.1, 0.84, facecolor=RED + "12", edgecolor=RED + "65", lw=0.9, hatch="///"
        )
    )
    mid.text(
        0.05,
        0.60,
        "invalid\ninterval",
        ha="center",
        va="center",
        fontsize=10.6,
        color=RED,
        fontweight="bold",
    )
    mid.text(0, 0.01, "0", ha="center", va="bottom", fontsize=9.8, color=MUTED)
    mid.text(0.1, 0.01, "0.1 s", ha="center", va="bottom", fontsize=9.8, color=MUTED)
    fig.suptitle(
        r"Analytic benchmark coordinate: $\phi(T)=-\log(1-0.1/T)$",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    left.set_title("Valid branch: $T<0$", fontsize=11.0, pad=8)
    right.set_title("Valid branch: $T>0.1$ s", fontsize=11.0, pad=8)
    fig.text(
        0.5,
        0.01,
        "Analytic phase coordinate: distant positive TTC is compressed, with singularities "
        "at the domain boundaries.",
        ha="center",
        fontsize=11.0,
        color=MUTED,
    )
    fig.subplots_adjust(bottom=0.18, top=0.84)
    save(fig, "phase_geometry")


if __name__ == "__main__":
    overview()
    architecture()
    history()
    evidence_flow()
    deployed_architecture()
    fusion_interface()
    crop_geometry()
    phase_geometry()
