"""Build manuscript figures and tables from the bundled measured-result ledgers."""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "figures"
TABLES = ROOT / "tables"
TABLES.mkdir(exist_ok=True)
BLUE, TEAL, RED, GRAY = "#3769DB", "#268B86", "#AD534B", "#B5BDCA"
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
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
    }
)
late = json.loads((DATA / "late_results.json").read_text(encoding="utf-8"))["rows"]
middle = json.loads((DATA / "middle_results.json").read_text(encoding="utf-8"))


def esc(x: object) -> str:
    return str(x).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def table(
    name: str,
    rows: list[dict[str, object]],
    columns: list[tuple[str, str, str]],
    caption: str,
    label: str,
) -> None:
    placement = "H" if name == "late_transfer" else "tbp"
    lines = [
        r"\begin{table}[" + placement + "]",
        r"\centering\small",
        r"\caption{" + caption + "}",
        r"\label{tab:" + label + "}",
        r"\begin{tabular}{l" + "r" * (len(columns) - 1) + "}",
        r"\toprule",
        " & ".join(r"\textbf{" + c[1] + "}" for c in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        items = []
        for key, _, fmt in columns:
            v = row.get(key, "--")
            items.append(format(v, fmt) if isinstance(v, (int, float)) else esc(v))
        lines.append(" & ".join(items) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def save(fig: Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def late_tables() -> None:
    r = [x for x in late if x["version"] == "TRAIN40" and x["cohort"] == "extended_new"]
    names = {
        "E_CTX_TRAIN40": "Event context",
        "Garl_full": "Garl reproduced",
        "PHASE_ONLY_TRAIN40_S7": "Phase correction",
        "R_CTX_TRAIN40": "RGB context",
    }
    r = [dict(x, model=names[x["model"]]) for x in r]
    table(
        "late_transfer",
        r,
        [
            ("model", "Model", ""),
            ("MAE_s", "MAE (s)", ".3f"),
            ("RTE_pct", r"RTE (\%)", ".2f"),
            ("median_AE_s", "Median AE (s)", ".3f"),
            ("p95_AE_s", "P95 AE (s)", ".3f"),
        ],
        r"Frozen TRAIN40 transfer on \texttt{extended\_new}: 468 GT-valid queries, 32 "
        r"sequences, positive TTC only, annotated ROI. Pooled metrics; lower is better. "
        r"Source \evidence{L01}.",
        "late-transfer",
    )
    r = [
        x
        for x in late
        if x["version"] == "V18"
        and x["cohort"] == "V"
        and "S" in x["model"]
        and x["model"].startswith("FUSION")
    ]
    r.sort(key=lambda x: (x["seed"], x["model"]))
    table(
        "late_fusion",
        r,
        [
            ("model", "Head / seed", ""),
            ("overall_MiD", "Overall MiD", ".3f"),
            ("MiDc", "Crucial MiD", ".3f"),
            ("MAE_s", "MAE (s)", ".3f"),
        ],
        r"V18 C1 equal-capacity TRUE/ZERO event controls on exposed V (16,432 rows), "
        r"using adaptation/head seeds 7, 13, and 23. Source \evidence{L03}.",
        "late-fusion",
    )
    for name, version, cohort, cap in [
        (
            "late_v14",
            "V14",
            "V",
            r"V14 factorial study on exposed V: apparent crucial-range gains can coexist "
            r"with non-crucial regressions. Source \evidence{L02}.",
        ),
        (
            "late_v18",
            "V18",
            "H",
            r"V18 primary and adaptive experiments on development subset H (18,422 rows). "
            r"Sources \evidence{L10--L14,L20--L25}.",
        ),
        (
            "late_v19",
            "V19",
            "H",
            r"V19 decision recovery on exposed H. LW trades a small overall MiD increase "
            r"against improved MAE and negative-range behavior. Source \evidence{L30}.",
        ),
    ]:
        r = [x for x in late if x["version"] == version and x["cohort"] == cohort]
        table(
            name,
            r,
            [
                ("model", "Model", ""),
                ("overall_MiD", "MiD", ".2f"),
                ("MiDc", "Crucial", ".2f"),
                ("MiDn", "Negative", ".2f"),
                ("MAE_s", "MAE (s)", ".2f"),
            ],
            cap,
            name,
        )


def transfer() -> None:
    names = ["Garl_full", "E_CTX_TRAIN40", "PHASE_ONLY_TRAIN40_S7", "R_CTX_TRAIN40"]
    labels = ["Garl\nreproduced", "Event\ncontext", "Phase\ncorrection", "RGB\ncontext"]
    rows = [
        next(x for x in late if x["cohort"] == "extended_new" and x["model"] == n) for n in names
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, key, unit, title in zip(
        axes,
        ["RTE_pct", "MAE_s"],
        ["RTE (%)", "MAE (s)"],
        ["Relative TTC error", "Absolute TTC error"],
        strict=False,
    ):
        bars = ax.bar(labels, [x[key] for x in rows], color=[GRAY, BLUE, TEAL, RED], width=0.62)
        ax.bar_label(
            bars, fmt="%.3f" if key == "MAE_s" else "%.2f", padding=4, fontsize=11.5, color=INK
        )
        ax.set_ylim(0, max(x[key] for x in rows) * 1.22)
        ax.set_ylabel(unit)
        ax.set_title(title, pad=12)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    fig.suptitle(
        "Exploratory EvTTC transfer / 468 valid queries / 32 sequences", fontsize=11.5, color=MUTED
    )
    fig.tight_layout(w_pad=2.4, pad=0.6)
    save(fig, "transfer_results")


def transfer_query_rows() -> list[dict[str, str]]:
    """Load the path-free public projection, with the private snapshot as fallback."""
    public = DATA / "transfer_queries.csv"
    private = DATA / "late_sources" / "L46.csv"
    path = public if public.is_file() else private
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "query_id",
        "group_id",
        "cohort",
        "target_ttc",
        "target_exclusion",
        "PHASE_ONLY_TRAIN40_S7",
        "R_CTX_TRAIN40",
        "Garl_full",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Transfer query projection is incomplete: {path}")
    valid = [row for row in rows if row["cohort"] == "extended_new" and not row["target_exclusion"]]
    if len(valid) != 468 or len({row["group_id"] for row in valid}) != 32:
        raise ValueError("Expected 468 valid extended_new queries from 32 sequence groups")
    if not all(float(row["target_ttc"]) > 0 for row in valid):
        raise ValueError("Extended-new transfer plot requires positive TTC targets")
    return valid


def sequence_label(group_id: str) -> str:
    """Shorten repeated overlap wording without changing sequence identity or order."""
    return group_id.replace("-overlap-", " / overlap ")


def transfer_sequence() -> None:
    rows = transfer_query_rows()
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    models = ["PHASE_ONLY_TRAIN40_S7", "Garl_full", "R_CTX_TRAIN40"]
    records = []
    for group_id, items in grouped.items():
        rte = {
            model: statistics.mean(
                100
                * abs(float(row[model]) - float(row["target_ttc"]))
                / abs(float(row["target_ttc"]))
                for row in items
            )
            for model in models
        }
        records.append(
            {
                "group_id": group_id,
                "n": len(items),
                "phase_minus_garl": rte[models[0]] - rte[models[1]],
                "phase_minus_rgb": rte[models[0]] - rte[models[2]],
            }
        )
    records.sort(key=lambda row: (row["phase_minus_garl"], row["group_id"]))
    wins_garl = sum(row["phase_minus_garl"] < 0 for row in records)
    wins_rgb = sum(row["phase_minus_rgb"] < 0 for row in records)
    if (wins_garl, wins_rgb) != (26, 14):
        raise ValueError(f"Unexpected sequence win counts: {(wins_garl, wins_rgb)}")

    y = np.arange(len(records))
    labels = [sequence_label(row["group_id"]) for row in records]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 12.0), sharey=True)
    panels = [
        ("phase_minus_garl", "PHASE-TTC minus Garl full", wins_garl),
        ("phase_minus_rgb", "PHASE-TTC minus RGB context", wins_rgb),
    ]
    for ax, (key, title, wins) in zip(axes, panels, strict=False):
        values = [row[key] for row in records]
        colors = [TEAL if value < 0 else RED for value in values]
        ax.barh(y, values, color=colors, height=0.68)
        ax.axvline(0, color=INK, lw=1)
        ax.set_xscale("symlog", linthresh=5, linscale=1)
        ticks = [-1000, -100, -10, 0, 10] if key == "phase_minus_garl" else [-10, 0, 10, 100]
        ax.set_xticks(ticks, [str(value).replace("-", "−") for value in ticks])
        ax.set_xlabel("RTE difference\n(percentage points)", fontsize=13)
        ax.tick_params(axis="x", labelsize=13)
        ax.set_title(f"{title}\nPHASE-TTC lower on {wins}/32 sequences", pad=12, fontsize=12.5)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, labels)
    axes[0].tick_params(axis="y", labelsize=13)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Every extended-new sequence / negative values favor PHASE-TTC",
        fontsize=12,
        color=MUTED,
        y=0.995,
    )
    fig.tight_layout(w_pad=2.6, rect=(0, 0, 1, 0.985))
    save(fig, "transfer_sequence")
    return {
        "queries": len(rows),
        "sequence_groups": len(records),
        "phase_lower_rte_than_garl_groups": wins_garl,
        "phase_lower_rte_than_rgb_groups": wins_rgb,
        "records": records,
    }


def transfer_tradeoff() -> None:
    models = ["PHASE_ONLY_TRAIN40_S7", "R_CTX_TRAIN40"]
    rows = [
        next(row for row in late if row["cohort"] == "extended_new" and row["model"] == model)
        for model in models
    ]
    metrics = [("MAE_s", "Mean AE"), ("median_AE_s", "Median AE"), ("p95_AE_s", "P95 AE")]
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    for index, (row, label, color) in enumerate(
        zip(rows, ["PHASE-TTC", "RGB context"], [TEAL, BLUE], strict=False)
    ):
        values = [row[key] for key, _ in metrics]
        bars = ax.bar(x + (index - 0.5) * width, values, width, color=color, label=label)
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10.5)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("Absolute TTC error (s)")
    ax.set_title("Mean and tail error favor different models", pad=12)
    ax.set_ylim(0, max(row["p95_AE_s"] for row in rows) * 1.2)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    fig.suptitle(
        "Extended-new / same 468 valid queries / lower is better",
        fontsize=11.5,
        color=MUTED,
        y=0.96,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save(fig, "transfer_tradeoff")


def fusion() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    seeds = [7, 13, 23]
    x = np.arange(3)
    for ax, key, title in zip(
        axes, ["overall_MiD", "MiDc"], ["Overall MiD", "Crucial-range MiD"], strict=False
    ):
        for j, (arm, c) in enumerate([("TRUE", TEAL), ("ZERO", GRAY)]):
            vals = [next(r[key] for r in late if r["model"] == f"FUSION-{arm}-S{s}") for s in seeds]
            bars = ax.bar(x + (j - 0.5) * 0.34, vals, width=0.32, color=c, label=arm)
            ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=10)
        ax.set_xticks(x, [f"Seed {s}" for s in seeds])
        ax.set_ylabel("MiD (lower is better)")
        ax.set_title(title, pad=10)
        ax.set_ylim(0, 160 if key == "MiDc" else 110)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(w_pad=2.6, rect=(0, 0, 1, 0.93))
    save(fig, "fusion_ablation")


def range_tradeoffs() -> None:
    rows = [r for r in late if r["version"] == "V14"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for ax, k, t in zip(
        axes, ["MiDc", "MiDn"], ["Crucial-range MiD", "Negative-range MiD"], strict=False
    ):
        labels = [r["model"].replace("RT-", "").replace("-S7", "") for r in rows]
        bars = ax.barh(labels, [r[k] for r in rows], color=[GRAY, BLUE, TEAL, TEAL, RED, RED])
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=11)
        ax.set_xlim(0, max(r[k] for r in rows) * 1.24)
        ax.invert_yaxis()
        ax.set_xlabel("MiD (lower is better)")
        ax.set_title(t, pad=12)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=3)
    save(fig, "range_tradeoffs")


def runtime() -> None:
    rows = [r for r in late if r["version"] == "TRAIN40_runtime"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    keys = ["mean_ms", "p50_ms", "p95_ms", "p99_ms"]
    x = np.arange(4)
    for j, r in enumerate(rows):
        bars = axes[0].bar(
            x + (j - 0.5) * 0.35,
            [r[k] for k in keys],
            0.33,
            color=[RED, TEAL][j],
            label=["Deadline response", "Raw-availability response"][j],
        )
        axes[0].bar_label(bars, fmt="%.1f", padding=3, fontsize=9.5)
    axes[0].axhline(20, ls=(0, (4, 3)), lw=1.1, color=INK)
    axes[0].set_xticks(x, ["Mean", "P50", "P95", "P99"])
    axes[0].set_ylabel("Response time (ms)")
    axes[0].set_ylim(0, 39)
    axes[0].set_title("Response-time distribution", pad=12)
    deadline = next(r for r in rows if r["model"] == "deadline_response_ms")
    bars = axes[1].bar(
        ["Below 20 ms", "At least 20 ms"],
        [deadline["strictly_under_20ms_count"], deadline["at_least_20ms_count"]],
        color=[TEAL, RED],
        width=0.6,
    )
    axes[1].bar_label(bars, padding=4, fontsize=12)
    axes[1].set_ylim(0, 250)
    axes[1].set_ylabel("Updates (250 total)")
    axes[1].set_title("20 ms deadline", pad=12)
    for ax in axes:
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Paced replay with preloaded sensor observations", fontsize=11.5, color=MUTED)
    fig.tight_layout(w_pad=2.8, rect=(0, 0, 1, 0.85))
    save(fig, "runtime_deadlines")


def h16() -> None:
    d = middle["h16_replication"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    x = np.arange(3)
    for j, (k, c) in enumerate([("h8_mid", GRAY), ("h16_mid", TEAL)]):
        axes[0].bar(
            x + (j - 0.5) * 0.35,
            [d[str(s)][k] for s in [7, 13, 23]],
            0.33,
            color=c,
            label=k.split("_")[0].upper(),
        )
    axes[0].set_xticks(x, ["7\n(exploratory)", "13", "23"])
    axes[0].set_xlabel("Head seed")
    axes[0].set_ylabel("Sequence-macro MiD")
    axes[0].legend(loc="upper left", ncol=2, bbox_to_anchor=(-0.02, 1.17))
    axes[0].grid(axis="y")
    axes[0].set_axisbelow(True)
    keys = ["7", "13", "23", "new13_23", "all7_13_23"]
    for i, k in enumerate(keys):
        v = d[k]
        lo, hi = v["hierarchical_ci95"]
        val = v["paired_loss_delta"]
        axes[1].plot([lo, hi], [i, i], color=TEAL, lw=2.5, solid_capstyle="round")
        axes[1].plot(val, i, "o", color=TEAL, ms=6)
    axes[1].axvline(0, color=RED, ls=(0, (4, 3)), lw=1)
    axes[1].set_yticks(range(5), ["Seed 7", "Seed 13", "Seed 23", "New seeds only", "All three"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Paired MiD delta: H16 - H8")
    axes[1].set_title("Hierarchical 95% intervals", pad=15)
    axes[1].grid(axis="x")
    axes[1].set_axisbelow(True)
    axes[1].margins(y=0.16)
    fig.tight_layout(w_pad=3.4)
    save(fig, "h16_replication")


def early() -> None:
    rows = json.loads((DATA / "early_results.json").read_text(encoding="utf-8"))
    groups = []
    for r in rows:
        if "MiD" in r["metric"] or "mid" in r["metric"].lower():
            if r["group"] not in groups:
                groups.append(r["group"])
    # Separate populations; select first three recovery groups with >= 3 arm scores.
    chosen = []
    for g in groups:
        rs = [
            r
            for r in rows
            if r["group"] == g
            and ("mid" in r["metric"].lower())
            and "delta" not in r["metric"].lower()
        ]
        if len(rs) >= 3 and len(rs) <= 8 and "V" in g:
            chosen.append((g, rs))
        if len(chosen) == 3:
            break
    fig, axes = plt.subplots(len(chosen), 1, figsize=(9.2, 7.8), squeeze=False)
    for ax, (g, rs) in zip(axes[:, 0], chosen, strict=False):
        names = {
            "a4": "A4",
            "a6": "A6",
            "a8_0": "A8.0",
            "garl": "Garl",
            "a5": "A5",
            "soft": "SOFT",
            "c2f": "C2F",
            "t20": "T20",
            "cap_s": "CAP-S",
            "soft_partial_freeze": "SOFT partial freeze",
            "router": "Router",
            "timevol20_3": "TimeVol20-3",
            "exp6_3": "Exp6-3",
            "gated_exp6_3": "Gated Exp6-3",
        }
        labels = [names.get(r["model"], r["model"].replace("_", "-")) for r in rs]
        bars = ax.barh(
            labels,
            [r["value"] for r in rs],
            color=[TEAL if i == 0 else BLUE for i in range(len(rs))],
        )
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=11.5)
        ax.set_xlim(0, max(r["value"] for r in rs) * 1.15)
        ax.invert_yaxis()
        ax.set_title(g, fontsize=12.5, loc="left", pad=12)
        ax.set_xlabel("MiD (lower is better)")
        ax.tick_params(axis="y", labelsize=11.5)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
    fig.tight_layout(h_pad=2)
    save(fig, "early_ablation")


if __name__ == "__main__":
    late_tables()
    transfer()
    sequence_summary = transfer_sequence()
    transfer_tradeoff()
    fusion()
    range_tradeoffs()
    runtime()
    h16()
    early()
    repo = ROOT.parents[2]
    if (repo / ".git").exists():
        verification = repo / "tmp" / "pdfs" / "motion_to_time_review" / "transfer.json"
        verification.parent.mkdir(parents=True, exist_ok=True)
        verification.write_text(json.dumps(sequence_summary, indent=2), encoding="utf-8")
