"""Plot every reported band and all three adaptation seeds from frozen fusion evidence."""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from operational.range_transfer.common import read, sha, write

from .common import ROOT, RUN

FIELDS = (
    ("Agregado interno", "overall_MiD", 1),
    ("MiDc · 0–3 s", "bins.crucial.group_macro_mid", 1),
    ("MiDs · 3–6 s", "bins.small.group_macro_mid", 1),
    ("MiDl · 6–10 s", "bins.large.group_macro_mid", 1),
    ("MiDn · −10–0 s", "bins.negative.group_macro_mid", 1),
    ("MAE · segundos", "diagnostics.mae_s", 1),
    ("Error de signo · %", "diagnostics.sign_error_rate", 100),
    ("P95 del error absoluto · segundos", "diagnostics.p95_ae_s", 1),
)


def scalar(metrics: dict, path: str) -> float:
    """Read the same scalar path from every seed and historical comparator."""
    value: object = metrics
    for key in path.split("."):
        if not isinstance(value, dict):
            raise ValueError(f"Invalid metric path: {path}")
        value = value[key]
    if not isinstance(value, (int, float)):
        raise ValueError(f"Metric is not a scalar: {path}")
    return float(value)


def build() -> dict:
    """Export plotted data, a readable PNG/PDF, and exact source/output bindings."""
    source = RUN / "replication/fusion_C1/evaluation/RESULT.json"
    result = read(source)
    if result["status"] != "REPLICA_METRICS_READY_FOR_INTERPRETATION":
        raise ValueError("Complete three-seed evidence required")
    output = RUN / "report/fusion_replication_figure"
    output.mkdir(parents=True, exist_ok=True)
    png = ROOT / "docs/joint_measurement_20261012/figures/fusion_replicas.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    colors = {"TRUE": "#087F8C", "ZERO": "#66717E", "R_CTX": "#BD733C", "F_TRUE": "#8B5CA0"}
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, (title, field, scale) in zip(axes.flat, FIELDS, strict=True):
        for x, mode in enumerate(("TRUE", "ZERO")):
            values = []
            for seed, jitter in zip((7, 13, 23), (-0.08, 0, 0.08), strict=True):
                name = f"FUSION-{mode}-S{seed}"
                value = scale * scalar(result["metrics"][name], field)
                values.append(value)
                rows.append({"model": name, "field": field, "plot_value": value, "scale": scale})
                axis.scatter(x + jitter, value, color=colors[mode], s=28, zorder=3)
            mean = float(np.mean(values))
            if field != "diagnostics.p95_ae_s" and not np.isclose(
                mean, scale * scalar(result["means"][mode], field), rtol=1e-12, atol=1e-12
            ):
                raise ValueError("Reported scalar mean differs from plotted seeds")
            axis.errorbar(
                x,
                mean,
                yerr=np.std(values, ddof=1),
                color=colors[mode],
                fmt="D",
                markersize=5,
                capsize=5,
                linewidth=1.5,
                zorder=4,
            )
        for name in ("R_CTX", "F_TRUE"):
            value = scale * scalar(result["metrics"][name], field)
            rows.append({"model": name, "field": field, "plot_value": value, "scale": scale})
            axis.axhline(value, color=colors[name], linestyle="--", linewidth=1.2, label=name)
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=12)
        axis.set_xticks([0, 1], ["Fusión TRUE", "Control ZERO"])
        axis.set_xlim(-0.45, 1.45)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0, 1].axhline(
        0.9 * scalar(result["metrics"]["F_TRUE"], "bins.crucial.group_macro_mid"),
        color="#C44135",
        linestyle=":",
        linewidth=1.5,
        label="MiDc: −10% frente a F_TRUE",
    )
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.9), ncol=3, frameon=False
    )
    fig.suptitle(
        "V18 · Fusión C1 + RGB: señal controlada, sin promoción del sistema",
        x=0.07,
        y=0.98,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.925,
        "V interno expuesto · 16.432 consultas · menor es mejor en los ocho paneles",
        fontsize=11,
        color="#444444",
    )
    fig.text(
        0.07,
        0.06,
        "Puntos: seeds 7/13/23. Rombo y barra: media ± desviación entre seeds (n=3), no IC.\n"
        "Comparten A5 inicial y anchor R_CTX; son réplicas de adaptación. "
        "No se elige una semilla.\n"
        "La mejora agregada no satisface el gate crucial original. "
        "Los IC por grupos y el contraste V16 están en el informe.",
        fontsize=9,
        linespacing=1.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.20, top=0.80, hspace=0.48, wspace=0.40)
    fig.savefig(png, dpi=160)
    fig.savefig(output / "fusion_replicas.pdf")
    plt.close(fig)
    data = output / "plot_data.csv"
    with data.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt = {
        "status": "GENERATED_PENDING_VISUAL_REVIEW",
        "source_sha256": sha(source),
        "generator_sha256": sha(
            ROOT / "operational/joint_measurement/fusion_replication_figure.py"
        ),
        "png_sha256": sha(png),
        "pdf_sha256": sha(output / "fusion_replicas.pdf"),
        "plot_data_sha256": sha(data),
        "rows": len(rows),
        "system_promotion": False,
    }
    write(output / "RESULT.json", receipt)
    return receipt


if __name__ == "__main__":
    print(build())
