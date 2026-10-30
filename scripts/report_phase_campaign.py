"""Regenerate the bounded campaign report from immutable experiment receipts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.data import read


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "grouped": "F1_grouped/REPORT.json",
        "temporal": "temporal_probes/TEMPORAL.json",
        "probes": "temporal_probes/PROBES.json",
        "calibration": "calibration/REPORT.json",
        "external": "Slider500/REPORT.json",
        "robustness": "robustness/REPORT.json",
        "geometry": "geometry/REPORT.json",
        "restarts": "restarts/REPORT.json",
    }
    reports = {k: read(args.campaign / v) for k, v in paths.items()}
    grouped = reports["grouped"]
    if grouped["status"] != "COMPLETE":
        raise ValueError("All seeds and families must finish before reporting")
    summary = {
        "inputs": {k: {"path": v, "sha256": file_sha(args.campaign / v)} for k, v in paths.items()},
        "seeds": {},
        "claim": "exploratory head adaptation; no SOTA or certified independent transfer",
    }
    baseline = grouped["seeds"]["7"]["models"]["F0"]
    for seed in (7, 13, 21):
        current = grouped["seeds"][str(seed)]
        summary["seeds"][str(seed)] = {
            **current["models"]["F1"],
            "paired_family_bootstrap": current["paired_family_bootstrap"],
            "native_80_coverage": current["intervals"]["F1"]["coverage_80"],
            "train_scaled_80_coverage": reports["calibration"]["seeds"][str(seed)]["heldout"][
                "coverage_80"
            ],
        }
    summary["mean_std_across_training_seeds"] = {
        key: {
            "mean": float(np.mean([r[key] for r in summary["seeds"].values()])),
            "std_ddof1": float(np.std([r[key] for r in summary["seeds"].values()], ddof=1)),
        }
        for key in ("mae_seconds", "relative_absolute_error_pct", "macro_recording_relative_error")
    }
    summary["baseline"] = baseline
    atomic_receipt(args.output / "SUMMARY.json", summary)
    table = [{"model": "F0", **baseline}] + [
        {"model": f"F1 seed {s}", **grouped["seeds"][str(s)]["models"]["F1"]} for s in (7, 13, 21)
    ]
    with (args.output / "grouped_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, key, title in zip(
        axes,
        ("macro_recording_relative_error", "mae_seconds"),
        ("Macro recording relative error (%)", "Query MAE (seconds)"),
        strict=True,
    ):
        axis.bar(
            [r["model"] for r in table],
            [r[key] for r in table],
            color=["#64748b", "#0891b2", "#0891b2", "#0891b2"],
        )
        axis.set_ylabel(title)
        axis.tick_params(axis="x", labelrotation=20)
    fig.suptitle("Current-head family holdouts; upstream historical exposure unresolved")
    fig.savefig(args.output / "grouped_tradeoff.png", dpi=180)
    plt.close(fig)
    lines = [
        "# PHASE-TTC: cierre de campaña local",
        "",
        "La cabeza F1 está entrenada y empaquetada. La validación adicional encuentra una "
        "mejora del error relativo y regresiones en MAE e incertidumbre. Se conserva como "
        "candidato exploratorio; no se promueve como ganador universal ni SOTA.",
        "",
        "## Población y alcance",
        "",
        "21 ajustes: siete familias × semillas 7/13/21. Cada fold reinicia la cabeza original "
        "y conserva TRAIN40 en entrenamiento. Las ocho grabaciones solapadas están excluidas "
        "de esta validación; quedan 24 grabaciones y 2.531 consultas. Los siete grupos son "
        "familias de escenario, no adquisiciones certificadas. La exposición histórica de "
        "los productores impide denominarlo test independiente.",
        "",
        "El modelo final ya entregado usa 88.744 estados TRAIN40 + 3.571 estados EvTTC de "
        "32 grabaciones: 92.315 filas, dos épocas completas y 1.444 actualizaciones por "
        "semilla. Sus métricas sobre esas filas son ajuste. Los ocho componentes anteriores "
        "y normalizadores permanecen congelados. No se declara convergencia ni fine-tuning "
        "de todos los productores.",
        "",
        "## Validación agrupada",
        "",
        "| Modelo | Macro RAE % | Micro RAE % | MAE s | Cobertura |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['model']} | {row['macro_recording_relative_error']:.3f} | "
            f"{row['relative_absolute_error_pct']:.3f} | {row['mae_seconds']:.4f} | "
            f"{100 * row['coverage']:.1f}% |"
        )
    lines += [
        "",
        "Los intervalos siguientes remuestrean familias enteras y mantienen emparejadas "
        "las diferencias por grabación. Son exploratorios y condicionales a esta población; "
        "tres semillas no aportan tres adquisiciones nuevas.",
        "",
        "| Semilla | Diferencia macro, pp | Intervalo 95%, pp | Cobertura nativa 80% | "
        "Reescala solo train |",
        "|---|---:|---:|---:|---:|",
    ]
    for seed, row in summary["seeds"].items():
        ci = row["paired_family_bootstrap"]
        lines.append(
            f"| {seed} | {ci['mean_difference']:.3f} | "
            f"[{ci['lower_95']:.3f}, {ci['upper_95']:.3f}] | "
            f"{100 * row['native_80_coverage']:.2f}% | "
            f"{100 * row['train_scaled_80_coverage']:.2f}% |"
        )
    lines += [
        "",
        "La reescala usa residuos del entrenamiento de cada fold, nunca su familia reservada. "
        "No es split conformal y no ofrece una garantía de cobertura. No se incorpora al bundle.",
        "",
        "## Regímenes de TTC, semilla predefinida 7",
        "",
        "| Régimen | Consultas | RAE F0 % | RAE F1 % | MAE F0 s | MAE F1 s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    regimes = grouped["seeds"]["7"]["regimes"]
    for regime in ("positive_under_1s", "1_to_4s", "over_4s"):
        a, b = regimes["F0"][regime], regimes["F1"][regime]
        lines.append(
            f"| {regime} | {a['population']} | {a['relative_absolute_error_pct']:.3f} | "
            f"{b['relative_absolute_error_pct']:.3f} | {a['mae_seconds']:.4f} | "
            f"{b['mae_seconds']:.4f} |"
        )
    lines += [
        "",
        "No hay targets negativos en esta población: no permite validar generalización "
        "del signo negativo. La regresión en TTC largo queda visible en la tabla.",
        "",
        "## Slider-500 externo al ajuste actual",
        "",
        "Pesos y 77 consultas congelados antes de leer valores TTC; predicciones selladas "
        "antes del join por identidad de frame y reloj (tolerancia 2 μs). Extrínsecas de "
        "Slider provisionales, proyección solo rotacional, parentesco de adquisición sin "
        "resolver. No equivale al score oficial de las diez secuencias.",
        "",
        "| Método | Targets válidos | Salidas evaluables | RAE % | MAE s |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in reports["external"]["metrics"].items():
        relative = row["relative_absolute_error_pct"]
        mae = row["mae_seconds"]
        lines.append(
            f"| {name} | {row['valid_targets']} | {row['finite']} | {relative:.3f} | {mae:.4f} |"
            if relative is not None and mae is not None
            else f"| {name} | {row['valid_targets']} | {row['finite']} | NA | NA |"
        )
    lines += [
        "",
        "Las comparaciones sobre consultas compartidas están en Slider500/REPORT.json. "
        "No se ocultan los huecos de cada método. Los controles CMax y STRTTC lineal son "
        "implementaciones locales acotadas en doce consultas prefijadas, no reproducciones "
        "oficiales; sus resultados y motivos de ausencia están en geometry/REPORT.json.",
        "",
        "## Robustez sobre sensores",
        "",
        "48 consultas: dos cuartiles temporales por grabación elegible, seleccionados sin "
        "mirar errores. Nueve condiciones sobre eventos crudos o ROI, mismo target y pesos; "
        "la condición limpia comprueba paridad con los estados guardados.",
        "",
        "| Condición | F0 RAE % | F1 RAE % | F1 MAE s | Salidas F1 | Cobertura intervalo 80% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in reports["robustness"]["variants"].items():
        a, b = row["F0"], row["F1"]
        lines.append(
            f"| {name} | {a['relative_absolute_error_pct']:.3f} | "
            f"{b['relative_absolute_error_pct']:.3f} | {b['mae_seconds']:.4f} | "
            f"{b['finite']}/{b['population']} | "
            f"{100 * row['native_F1_interval_coverage_80']:.2f}% |"
        )
    temporal = reports["temporal"]
    lines += [
        "",
        "## Decisiones experimentales",
        "",
        f"T0 fijo: RAE {temporal['F0']['relative_absolute_error_pct']:.3f}% → "
        f"{temporal['T0']['relative_absolute_error_pct']:.3f}%, pero el fixture de aceleración "
        "retrasa 100 ms el cruce de alerta de 0,5 s. No se incorpora automáticamente.",
        "",
        "F2/Fcap: no abiertos. La anchura nativa no muestra una asociación consistente con "
        "error dentro de grabación; no se afirma que toda variable de calidad sea inútil. "
        "Densidad, soporte y calidad adicional requieren una hipótesis validada y control "
        "de capacidad antes de ampliar la cabeza.",
        "",
        "P0/J0/J1/J2: no abiertos. Los probes sobre contextos congelados recuperan inverse-TTC "
        "por encima de la media de train. Eso no prueba que el encoder sea óptimo, ni que JEPA "
        "aporte movimiento; no hay evidencia suficiente para atribuir el fallo a representación.",
        "",
        "## Límites que siguen abiertos",
        "",
        "- Genealogía completa de ancestros/teachers y certificación de adquisiciones "
        "independientes.",
        "- Equivalencia exacta con evaluador oficial y cobertura de las 434 consultas: "
        "las 11 fuera de cobertura y los 69 targets Slider originales no verificados "
        "no desaparecen.",
        "- Confirmación Slider con extrínsecas intersensor autoritativas y una adquisición nueva.",
        "- Reproducción oficial de comparadores geométricos y márgenes prácticos de promoción.",
        "- Incertidumbre con cobertura nominal verificada y robustez en TTC negativo.",
        "",
        "No se reutiliza esta evaluación externa para elegir otra variante. Una siguiente "
        "iteración que use sus errores deberá declararla desarrollo expuesto.",
        "",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Regenerated {args.output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
