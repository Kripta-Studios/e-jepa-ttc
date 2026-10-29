"""Regenerate V15 Markdown, plots and a hash-verified local bundle from evidence."""
# ruff: noqa: E501 -- Markdown paragraphs preserve readable prose in generated reports.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/range_readout_20261012"
OUT = ROOT / "docs/range_readout_results_20261012"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def table(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "ausente"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|")

    values = [
        list(frame.columns),
        ["---"] * len(frame.columns),
        *frame.itertuples(index=False, name=None),
    ]
    return "\n" + "\n".join("| " + " | ".join(map(cell, row)) + " |" for row in values) + "\n\n"


def plots() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = RUN / "figures"
    destination.mkdir(exist_ok=True)
    historical = RUN / "historical_V14"
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for mass in (0, 1):
        for directory, label in (
            (historical / "fits" / f"RT-M{mass}-L0-S7", "Uniforme V14"),
            (RUN / "fits" / f"S-M{mass}-L0-S7", "Importancia V15"),
        ):
            frame = pd.read_csv(directory / "CURVE.csv")
            axes[0, mass].plot(
                frame["update"], frame.loss.rolling(100, min_periods=1).mean(), label=label
            )
            axes[1, mass].plot(
                frame["update"],
                (frame.gradient_l2_before_clip > 1).rolling(100, min_periods=1).mean(),
                label=label,
            )
        axes[0, mass].set_title(f"M{mass}: mismo riesgo esperado, media móvil 100")
        axes[0, mass].set_ylabel("Estimador de loss")
        axes[1, mass].set_ylabel("Fracción con clipping")
        axes[1, mass].set_xlabel("Updates")
        axes[0, mass].legend()
    fig.savefig(destination / "S_SAMPLING_CURVES.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    found = False
    for name in ("RT-M0-L0-S7", "R-C05-CTX-S7", "R-C20-CTX-S7"):
        source = historical if name.startswith("RT-") else RUN
        path = source / "fits" / name / "CURVE.csv"
        if not path.exists():
            continue
        found = True
        frame = pd.read_csv(path)
        axes[0].plot(frame["update"], frame.loss.rolling(100, min_periods=1).mean(), label=name)
        axes[1].plot(
            frame["update"],
            (frame.gradient_l2_before_clip > 1).rolling(100, min_periods=1).mean(),
            label=name,
        )
    if found:
        axes[0].set_title("Refinadores M0/L0: objetivo H")
        axes[1].set_title("Refinadores: clipping")
        axes[0].set_ylabel("Loss, media móvil 100")
        axes[1].set_ylabel("Fracción con clipping, media móvil 100")
        axes[1].set_ylim(0, 1.05)
        for ax in axes:
            ax.legend()
            ax.set_xlabel("Updates")
        fig.savefig(destination / "R_CONTEXT_CURVES.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    found = False
    for name in ("R-C05-S7", "R-C20-S7"):
        path = RUN / "fits" / name / "CURVE.csv"
        if not path.exists():
            continue
        found = True
        frame = pd.read_csv(path)
        axes[0].plot(frame["update"], frame.loss.rolling(100, min_periods=1).mean(), label=name)
        axes[1].plot(
            frame["update"],
            (frame.gradient_l2_before_clip > 1).rolling(100, min_periods=1).mean(),
            label=name,
        )
    if found:
        axes[0].set_title("Readout: objetivo nativo P")
        axes[1].set_title("Readout: clipping")
        axes[0].set_ylabel("Loss nativa, media móvil 100")
        axes[1].set_ylabel("Fracción con clipping, media móvil 100")
        axes[1].set_ylim(0, 1.05)
        for ax in axes:
            ax.legend()
            ax.set_xlabel("Updates")
        fig.savefig(destination / "R_READOUT_CURVES.png", dpi=150)
    plt.close(fig)


def generate() -> None:
    snapshot_historical()
    short_producer_table()
    OUT.mkdir(parents=True, exist_ok=True)
    decision = read(RUN / "NEXT_DECISION.json")
    ledger = read(RUN / "ACCOUNTING.json")
    rows = []
    checks = []
    costs = []
    for path in sorted((RUN / "evaluation").glob("*/*/METRICS.json")):
        value = read(path)
        name = path.parent.name
        rows.append(
            {
                "fit": name,
                "stage": path.parent.parent.name,
                **{
                    key: value["bins"][band]["group_macro_mid"]
                    for key, band in (
                        ("MiDc", "crucial"),
                        ("MiDs", "small"),
                        ("MiDl", "large"),
                        ("MiDn", "negative"),
                    )
                },
                "overall_MiD": value["overall_MiD"],
                "mean_MiD": value["mean_MiD"],
                "MAE_s": value["diagnostics"]["mae_s"],
                "late_warning": value["late_warning"]["rate"],
            }
        )
        receipt = read(path.parent / "RECEIPT.json")
        if sha(path.parent / "PREDICTIONS.csv") != receipt["predictions_sha256"]:
            raise ValueError("Prediction hash mismatch")
    for mass in (0, 1):
        name = f"RT-M{mass}-L0-S7"
        value = read(RUN / "historical_V14/evaluation/screen" / name / "METRICS.json")
        rows.append(
            {
                "fit": name,
                "stage": "historical_V14",
                **{
                    key: value["bins"][band]["group_macro_mid"]
                    for key, band in (
                        ("MiDc", "crucial"),
                        ("MiDs", "small"),
                        ("MiDl", "large"),
                        ("MiDn", "negative"),
                    )
                },
                "overall_MiD": value["overall_MiD"],
                "mean_MiD": value["mean_MiD"],
                "MAE_s": value["diagnostics"]["mae_s"],
                "late_warning": value["late_warning"]["rate"],
            }
        )
    for path in sorted((RUN / "fits").glob("*/CHECKPOINT_RECEIPT.json")):
        value = read(path)
        if sha(path.parent / "checkpoint_last.pt") != value["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        checks.append(
            {
                "fit": path.parent.name,
                "status": value["status"],
                "updates": value["completed_updates"],
                "sha256": value["checkpoint_sha256"],
            }
        )
        provenance = read(path.parent / "RUN_PROVENANCE.json")
        costs.append(
            {
                "fit": path.parent.name,
                "seconds": provenance.get("elapsed_fit_s", provenance.get("elapsed_s")),
                "scope": provenance["scope"],
            }
        )
    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(RUN / "RESULTS_TABLE.csv", index=False)
    result_table = table(result_frame) if rows else "\nNo hay evaluación V nueva todavía.\n"
    header = f"# V15 RANGE-READOUT\n\nEstado registrado: **{decision['status']}**.\n\n"
    links = """1. [Método y ejecución](01_METODO.md).
2. [Resultados y decisión](02_RESULTADOS.md).
3. [Muestreo, clipping y alcance del productor](03_DIAGNOSTICOS.md).
4. [Pruebas, fallos, recursos y coste](04_PRUEBAS_Y_COSTE.md).
5. [Reproducción y contenido del bundle](05_REPRODUCCION.md).
6. [Interpretación de los mecanismos](06_INTERPRETACION.md).
"""
    (OUT / "README.md").write_text(
        header
        + links
        + "\n"
        + result_table
        + f"Contabilidad: {ledger['scientific']} científicos + {ledger['technical']} técnicos; "
        f"{ledger['recovery_upper']} recuperación, {ledger['pending_upper']} inciertos. "
        "No se reasigna saldo. V14 permanece NO_CANDIDATE y V13 es de sólo lectura.\n",
        encoding="utf-8",
    )
    method = (ROOT / "docs/range_readout_integration_20261012.md").read_text(encoding="utf-8")
    (OUT / "01_METODO.md").write_text(method, encoding="utf-8")
    selection = (
        read(RUN / "SELECTION_FREEZE.json")
        if (RUN / "SELECTION_FREEZE.json").exists()
        else decision
    )
    gate_rows = []
    for family, entry in selection.get("gates", {}).items():
        for contrast in ("paired", "historical_V14_M0"):
            item = entry[contrast]
            gate_rows.append(
                {
                    "family": family,
                    "contrast": contrast,
                    "eligible": item["eligible"],
                    "failed": ", ".join(k for k, v in item["checks"].items() if not v) or "ninguno",
                }
            )
    gate_table = table(pd.DataFrame(gate_rows)) if gate_rows else "Selección pendiente.\n"
    contrasts = contrast_table()
    interpret(selection, contrasts)
    bootstrap_rows = []
    for path in sorted((RUN / "bootstrap").glob("*/SUMMARY.json")):
        value = read(path)
        bootstrap_rows.append(
            {
                "contrast": path.parent.name,
                "delta_MiDc": value.get("delta_MiDc"),
                "CI95_low": value.get("CI95", [None, None])[0],
                "CI95_high": value.get("CI95", [None, None])[1],
                "groups": value.get("groups"),
            }
        )
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    bootstrap_frame.to_csv(RUN / "BOOTSTRAP_TABLE.csv", index=False)
    (OUT / "02_RESULTADOS.md").write_text(
        "# Resultados y selección\n\n"
        + result_table
        + "## Gates registrados\n\n"
        + gate_table
        + "\n## Diferencias de los contrastes previstos\n\n"
        + (table(contrasts) if not contrasts.empty else "Pendientes.\n")
        + "\n## Bootstrap por grupos\n\n"
        + (table(bootstrap_frame) if not bootstrap_frame.empty else "Pendiente.\n")
        + "Los candidatos prospectivos son S-M1 y R-C20-CTX. Se aplican los gates V14 "
        "frente a su control pareado y al M0/L0 histórico. S-M0 y R-C05 no se promocionan "
        "a posteriori. Un intervalo favorable no es un gate adicional.\n\n"
        "Umbrales: MiDc ≤90 % del control y mejora en ≥60 % de sus grupos; overall "
        "sin aumento; MiDs/MiDl/MiDn como máximo control + max(2 puntos, 5 %); "
        "MAE ≤102 %; error de signo como máximo +0,5 puntos porcentuales; p95 de "
        "sobreestimación crucial ≤105 %; aviso tardío y falsa alerta como máximo "
        "+1 punto porcentual, con denominadores ≥30. Se exige cobertura completa. "
        "Cada candidato debe pasar frente a su pareja y al control histórico M0.\n\n"
        "MiD por banda promedia primero dentro de cada grupo y después entre grupos. "
        "Overall = 0,5 MiDc + 0,3 MiDs + 0,1 MiDl + 0,1 MiDn; permanece ausente si falta "
        "una banda. mean_MiD y MAE son medias por consulta. El bootstrap agrupa por escenario: "
        "1.000 draws, seed12345, descriptivo ante múltiples contrastes.\n\n"
        "El bootstrap conserva la lectura decimal CSV de la ruta V14. La reproducción "
        "exacta de métricas recupera primero los bits FP32 de la salida nativa; ambos "
        "procedimientos quedan identificados, sin cambiar predicciones ni gates.\n\n"
        "Las tablas completas por consulta y grupo están en evaluation/. Los siete "
        "contrastes iniciales se conservan en bootstrap/. Si hay réplica, se promedian "
        "pérdidas por consulta sobre seeds7/13/23; no se construye un ensemble TTC. "
        "El p95 agregado usa el error positivo medio por consulta antes de su cuantil.\n\n"
        "V es desarrollo interno expuesto; no es test oficial ni confirmación externa. "
        "La comparación histórica tiene seed7 y no se inventan sus seeds13/23.\n",
        encoding="utf-8",
    )
    diag = "# Diagnósticos\n\n"
    for filename, title in (
        ("H_GRADIENT_VARIANCE.csv", "Gradientes iniciales H"),
        ("CLIPPING.csv", "Clipping durante S"),
        ("CLIPPING_ALL_FITS.csv", "Clipping de todos los fits V15"),
        ("PRODUCER_REACH_SUMMARY.csv", "Alcance condicional P/H"),
        ("SHORT_PRODUCER_DIAGNOSTICS.csv", "Predicciones del productor en TTC ≤1 segundo"),
    ):
        path = RUN / filename
        if path.exists():
            diag += "## " + title + "\n" + table(pd.read_csv(path))
    diag += """La varianza se estima con 32 batches de diagnóstico y se contrasta con el gradiente exacto de todo H en la inicialización seed7. No garantiza menor varianza para todos los parámetros o durante todo el entrenamiento. La igualdad de riesgo esperado no implica igualdad tras clipping y AdamW.

La masa no se normaliza dentro del batch. El sampler usa probabilidades globales float64 y la corrección w/p. El cambio no crea escenas independientes. SAMPLER_EXPOSURE.csv conserva los draws por grupo y rango.

El alcance del productor mantiene la geometría fija y permite residuales máximos independientes en ambas parejas. Se marca como no disponible cuando se cruza un polo del transporte. No es un límite absoluto de PAIR/H8 ni de un encoder distinto. Se conservan predicciones, known, soporte, edades y correcciones por consulta en PRODUCER_REACH_P_H.csv.

EVENT_CROP_ROWS_P_H.csv y EVENT_CROP_SUMMARY.csv usan las coordenadas reales del índice de eventos congelado. La fuente identifica secuencias y tracks, pero no episodios independientes; estos últimos permanecen sin inventar.

PRODUCER_METRICS_P_H.json compara A5 original y ambos readouts sobre P/H. P es entrenamiento de readout; H entrena las cabezas. Estas cifras describen el mecanismo y no son selección ni validación independiente. Las normas de gradiente R proceden de un objetivo distinto del de S; sus valores absolutos no son comparables como una única pérdida.
"""
    for filename, caption in (
        ("S_SAMPLING_CURVES.png", "Muestreo S frente a sus referencias V14"),
        ("R_READOUT_CURVES.png", "Readouts A5 sobre P"),
        ("R_CONTEXT_CURVES.png", "Refinadores sobre H"),
    ):
        diag += f"\n![{caption}](../../artifacts/range_readout_20261012/figures/{filename})\n"
    (OUT / "03_DIAGNOSTICOS.md").write_text(diag, encoding="utf-8")
    qa = (
        "# Pruebas, contabilidad y coste\n\n"
        + table(pd.DataFrame(checks))
        + table(pd.DataFrame(costs))
    )
    profile_path = RUN / "A5_RAW_FORWARD_PROFILE.json"
    if profile_path.exists():
        profile = read(profile_path)
        qa += "## Forward raw de A5\n\n"
        qa += table(
            pd.DataFrame(
                [
                    {"modelo": name, "mediana_ms": value["median_ms"], "p95_ms": value["p95_ms"]}
                    for name, value in profile["results"].items()
                ]
            )
        )
        qa += (
            "CPU FP32, un thread, batch1 T3, misma consulta P, 3 warmups y 30 repeticiones "
            "por modelo, orden intercalado. Host compartido. Excluye lectura, preparación "
            "de voxels, C2F, PAIR, H8 e historial. No es latencia E2E.\n\n"
        )
    qa += f"Updates físicos superiores: {ledger['physical_upper']}. Máximos: 56.000 científicos, 500 técnicos, 3.500 recuperación y 60.000 físicos.\n\n"
    qa += """Los tests contractuales y logs se conservan en qa/. Las pruebas de continuidad comparan 10 updates sintéticos continuos frente a 5+restore+5. S consume 20 técnicos; R conserva una primera comprobación de 20 y otra de 20 tras añadir los guards explícitos. No se utilizan datos reales para consumir el presupuesto técnico.

Las paridades raw/caché y GPU/CPU verifican outputs, pérdidas y gradientes. La tolerancia 2e-5 se fijó antes de los fits R. Los módulos congelados se comparan exactamente tras el entrenamiento sintético y cada fit R. Los contratos T2/T3 y negativos usan fixtures; no son evidencia de precisión.

TERMINAL_RESUME_PROOF.json acredita la reanudación real de los seis endpoints ya completos: cero updates añadidos y hashes de pesos, curvas, predicciones y freezes conservados. METRICS_REPRODUCTION.json recupera los bits FP32 de las predicciones serializadas y reproduce todas las métricas guardadas. LOSSES_PER_QUERY.csv separa MiD del scorer FP64 y native_aux_MiD, que conserva la aritmética del mean_MiD auxiliar histórico. No se sustituyen predicciones ni se relajan tolerancias para la selección.

Los tiempos de fit proceden de RUN_PROVENANCE e incluyen trabajo vectorial/cabeza y persistencia durante el bucle; excluyen entrenar productores y la preparación de cachés. Los COST.json de V miden sólo forward de cabeza desde caché, batch128. No son latencias E2E ni batch1. PROGRESS.json de extracción registra el tiempo de esa invocación; cuando existe resume, no equivale al total de todos los intentos.

Se preservan el piloto CPU, la pausa controlada de extracción para validar la caché por filas, los fallos de colección iniciales y el ajuste del radio del fixture sintético. No se cambiaron los radios, datos ni parámetros científicos para resolver esos tests. Los artefactos históricos siguen identificados con sus versiones de código.

Ruff/Pyright se ejecutan sobre V15. No se afirma que la suite o CI completa del histórico esté verde. V13/R1 no se modifican; los snapshots de propietarios y recursos conservan el estado observado. No se cambian drivers, paging, energía ni afinidad.

Durante la ejecución cambió la admisión Git de V13, observada y conservada en OWNER_GIT_CHANGE_OBSERVATION.json. V15 no escribió ese archivo. Los padres event y la caché H vinculados por hash se volvieron a validar; la auditoría distingue estos activos científicos de la metadata viva de otra campaña.
"""
    (OUT / "04_PRUEBAS_Y_COSTE.md").write_text(qa, encoding="utf-8")
    reproduction = """# Reproducción y bundle

La raíz del worktree es `e-jepa-ttc-v15-range-readout`, junto a V13 y V14. Los resultados viven en `artifacts/range_readout_20261012/`.

```powershell
pwsh -NoProfile -File scripts/Resume-RangeReadout.ps1 -Help
pwsh -NoProfile -File scripts/Resume-RangeReadout.ps1
& '../e-jepa-ttc/.venv/Scripts/python.exe' scripts/build_range_readout_reports.py --bundle
```

No lance otra continuación mientras exista el coordinador activo. Los endpoints completos se restauran sin añadir updates. Un fallo requiere diagnosticar el log indicado antes de reanudar; no use reset, clean ni borre checkpoints.

El bundle contiene informes, código nuevo, protocolo recibido, pesos y estados nuevos, curvas, predicciones, diagnósticos, fragmentos vectoriales, features de las familias, pruebas y logs. Los archivos están relacionados mediante BUNDLE_MANIFEST.json y el SHA-256 externo. El verificador comprueba también duplicados y CRC.

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' scripts/build_range_readout_reports.py --verify-zip 'artifacts/V15_RANGE_READOUT_REPORTS_RESULTS_20261012.zip'
Get-FileHash -Algorithm SHA256 -LiteralPath 'artifacts/V15_RANGE_READOUT_REPORTS_RESULTS_20261012.zip'
```

El bundle no sustituye los datos sensoriales originales ni los tres padres matched V13. ADMISSION.json registra sus rutas y hashes; los manifiestos conservan los paths del host original. Repetir desde sensores requiere esas dependencias. La reproducción de tablas desde predicciones y la verificación de hashes no equivale a repetir inferencia ni entrenamiento.

Un checkpoint R requiere `load_endpoint(..., ResidualSpec(bound))`; su atributo `.model` conserva el forward raw del productor con el operador explícito. C20 no puede interpretarse como `0.20*tanh(s)` nativo. Los encoders son los mismos padres compartidos, no nuevas representaciones entrenadas con tres semillas.

No se hace push en esta campaña. Un resultado FCWD/Dev32, si la rama de transferencia llega a habilitarse, seguirá siendo desarrollo externo expuesto.
"""
    (OUT / "05_REPRODUCCION.md").write_text(reproduction, encoding="utf-8")
    (RUN / "FINAL_REPORT.md").write_text(
        header
        + result_table
        + gate_table
        + f"Informes detallados: ../../docs/range_readout_results_20261012/README.md\n\n"
        f"Contabilidad: {ledger['scientific']} científicos, {ledger['technical']} técnicos, "
        f"{ledger['recovery_upper']} recuperación.\n",
        encoding="utf-8",
    )
    plots()


def short_producer_table() -> None:
    """Separate rare short-case error/sign/quantiles from an optimistic reach bound."""
    path = RUN / "PRODUCER_REACH_P_H.csv"
    if not path.exists():
        return
    data = pd.read_csv(path)
    data = data[(data.target_ttc > 0.1) & (data.target_ttc <= 1)].copy()
    data["absolute_error_s"] = (data.prediction_ttc - data.target_ttc).abs()
    data["late_warning"] = ~((data.prediction_ttc > 0.1) & (data.prediction_ttc <= 1))
    data["sign_error"] = data.prediction_ttc < 0
    data.groupby(["role", "producer"]).agg(
        queries=("sample_token", "size"),
        target_mean_s=("target_ttc", "mean"),
        prediction_mean_s=("prediction_ttc", "mean"),
        prediction_median_s=("prediction_ttc", "median"),
        prediction_p95_s=("prediction_ttc", lambda x: x.quantile(0.95)),
        mae_s=("absolute_error_s", "mean"),
        absolute_error_p95_s=("absolute_error_s", lambda x: x.quantile(0.95)),
        late_warning_rate=("late_warning", "mean"),
        sign_error_rate=("sign_error", "mean"),
        known_rate=("known", "mean"),
    ).reset_index().to_csv(RUN / "SHORT_PRODUCER_DIAGNOSTICS.csv", index=False)


def contrast_table() -> pd.DataFrame:
    """Report every registered screen contrast; do not recompute or alter gates."""
    comparisons = [
        ("S-M0-L0-S7", "RT-M0-L0-S7"),
        ("S-M1-L0-S7", "RT-M1-L0-S7"),
        ("S-M1-L0-S7", "S-M0-L0-S7"),
        ("S-M1-L0-S7", "RT-M0-L0-S7"),
        ("R-C20-CTX-S7", "R-C05-CTX-S7"),
        ("R-C20-CTX-S7", "RT-M0-L0-S7"),
        ("R-C05-CTX-S7", "RT-M0-L0-S7"),
    ]
    rows = []
    for candidate, control in comparisons:
        candidate_path = RUN / "evaluation/screen" / candidate / "METRICS.json"
        root = RUN / "historical_V14" if control.startswith("RT-") else RUN
        reference_path = root / "evaluation/screen" / control / "METRICS.json"
        if not candidate_path.exists() or not reference_path.exists():
            continue
        a, b = read(reference_path), read(candidate_path)

        def percent(left: float | None, right: float | None) -> float | None:
            return 100 * (right / left - 1) if left and right is not None else None

        row: dict[str, str | float | None] = {"candidate": candidate, "control": control}
        for label, band in (
            ("MiDc", "crucial"),
            ("MiDs", "small"),
            ("MiDl", "large"),
            ("MiDn", "negative"),
        ):
            row[f"delta_{label}_pct"] = percent(
                a["bins"][band]["group_macro_mid"], b["bins"][band]["group_macro_mid"]
            )
        row["delta_overall_pct"] = percent(a["overall_MiD"], b["overall_MiD"])
        row["delta_MAE_pct"] = percent(a["diagnostics"]["mae_s"], b["diagnostics"]["mae_s"])
        row["groups_crucial_improved"] = sum(
            b["group_crucial_mid"].get(g, float("inf")) < value
            for g, value in a["group_crucial_mid"].items()
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(RUN / "CONTRASTS.csv", index=False)
    return frame


def interpret(selection: dict[str, Any], contrasts: pd.DataFrame) -> None:
    """Write evidence-derived conclusions without claiming unmeasured SOTA or causality."""
    body = "# Interpretación de V15\n\n"
    body += f"Decisión de cribado: **{selection['status']}**.\n\n"
    for family, candidate, control in (
        ("S", "S-M1-L0-S7", "S-M0-L0-S7"),
        ("R", "R-C20-CTX-S7", "R-C05-CTX-S7"),
    ):
        if contrasts.empty:
            continue
        row = contrasts[(contrasts.candidate == candidate) & (contrasts.control == control)].iloc[0]
        failed = [
            k for k, value in selection["gates"][family]["paired"]["checks"].items() if not value
        ]
        body += (
            f"{candidate} frente a {control}: MiDc cambia {row.delta_MiDc_pct:+.2f} %, "
            f"MiDl {row.delta_MiDl_pct:+.2f} %, MiDn {row.delta_MiDn_pct:+.2f} % y "
            f"MAE {row.delta_MAE_pct:+.2f} %. Los signos positivos indican empeoramiento. "
            f"Gates pareados incumplidos: {', '.join(failed) or 'ninguno'}. "
            "La tabla de resultados conserva también el contraste obligatorio con V14 M0.\n\n"
        )
    clip = pd.read_csv(RUN / "CLIPPING.csv").set_index("fit")
    old, new = [float(clip.at[name, "fraction_clipped"]) for name in ("RT-M1-L0-S7", "S-M1-L0-S7")]
    body += (
        f"En S-M1, el clipping pasa de {old:.2%} a {new:.2%}. La prueba de igualdad "
        "del riesgo esperado y de su gradiente impide atribuirlo a una segunda ponderación. "
        "La estimación de varianza usa 32 batches en la inicialización: describe ese punto "
        "y no prueba que el clipping sea la única causa de las diferencias finales. M0 "
        "no mostró la misma reducción de varianza inicial.\n\n"
    )
    short = pd.read_csv(RUN / "SHORT_PRODUCER_DIAGNOSTICS.csv").set_index(["role", "producer"])
    reach = pd.read_csv(RUN / "PRODUCER_REACH_SUMMARY.csv")
    for role in ("P", "H"):
        a, b = short.loc[(role, "R-C05-S7")], short.loc[(role, "R-C20-S7")]
        reach_role = reach[(reach.role == role) & (reach.band == "0.1-1")].set_index("producer")
        body += (
            f"En {role}, para {int(a['queries'])} consultas de TTC ≤1 s, el límite de "
            f"geometría fija deja fuera del alcance a {int(reach_role.at['R-C05-S7', 'overestimate_count'])} "
            f"targets con C05 y a {int(reach_role.at['R-C20-S7', 'overestimate_count'])} con C20. "
            f"Sin embargo, el MAE aprendido del productor cambia de {a['mae_s']:.3f} a "
            f"{b['mae_s']:.3f} s y su mediana TTC de {a['prediction_median_s']:.3f} a "
            f"{b['prediction_median_s']:.3f} s. El alcance disponible y el resultado "
            "aprendido son magnitudes distintas.\n\n"
        )
    body += (
        "La comparación C20/C05 mantiene el reset y los 6.000 updates del scorer en ambos "
        "brazos. El contraste con A5 o V14 históricos incluye esa nueva receta de "
        "entrenamiento; no se atribuye todo su efecto al cap. Los encoders permanecen "
        "compartidos. H es diagnóstico del readout y entrenamiento de la cabeza, no "
        "confirmación externa. Los avisos a 1 s son diagnósticos TTC, no validación AEB.\n\n"
        "Estos datos no acreditan SOTA universal. Una campaña posterior necesitaría "
        "un protocolo nuevo, presupuesto propio y comparaciones de la misma población, "
        "modalidad y coste. El saldo reservado a réplicas no autoriza nuevos coeficientes "
        "ni selección sobre FCWD/Dev32.\n"
    )
    (OUT / "06_INTERPRETACION.md").write_text(body, encoding="utf-8")


def snapshot_historical() -> None:
    """Carry the two referenced baselines so report reproduction needs no sibling V14."""
    source = ROOT.parent / "e-jepa-ttc-v14-range-transfer/artifacts/range_transfer_20261010"
    destination = RUN / "historical_V14"
    records = {}
    for mass in (0, 1):
        name = f"RT-M{mass}-L0-S7"
        relative_paths = [
            f"fits/{name}/CURVE.csv",
            f"fits/{name}/CHECKPOINT_RECEIPT.json",
            f"evaluation/screen/{name}/PREDICTIONS.csv",
            f"evaluation/screen/{name}/PREDICTIONS_RECEIPT.json",
            f"evaluation/screen/{name}/METRICS.json",
        ]
        for relative in relative_paths:
            target = destination / relative
            original = source / relative
            if target.exists():
                if original.exists() and sha(target) != sha(original):
                    raise ValueError("Historical snapshot changed: " + relative)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
            records[relative] = sha(target)
        csv = destination / f"evaluation/screen/{name}/PREDICTIONS.csv"
        if sha(csv) != read(csv.parent / "PREDICTIONS_RECEIPT.json")["sha256"]:
            raise ValueError("Historical prediction receipt mismatch")
    (destination / "SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )


def verify(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError("Invalid ZIP members/CRC")
        manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
        if set(names) != set(manifest) | {"BUNDLE_MANIFEST.json"}:
            raise ValueError("Manifest membership mismatch")
        for name, digest in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Hash mismatch: " + name)
    return {
        "status": "PASS",
        "files_verified": len(manifest),
        "bytes": path.stat().st_size,
        "sha256": sha(path),
    }


def bundle() -> None:
    if read(RUN / "NEXT_DECISION.json")["status"] not in (
        "NO_CANDIDATE",
        "NO_REPLICATED_CANDIDATE",
        "TRANSFER_COMPLETE",
    ):
        raise ValueError("Campaign is not closed; do not publish an incomplete final bundle")
    destination = ROOT / "artifacts/V15_RANGE_READOUT_REPORTS_RESULTS_20261012.zip"
    files = {}
    for folder in (
        RUN,
        OUT,
        ROOT / "operational/range_readout",
        ROOT / "src/e_jepa_ttc/range_readout",
        ROOT / "docs/range_readout_handoff_20261012",
    ):
        for path in folder.rglob("*"):
            if (
                path.is_file()
                and path.suffix not in (".zip", ".lock", ".pyc")
                and "__pycache__" not in path.parts
                and ".tmp" not in path.name
            ):
                files[path.relative_to(ROOT).as_posix()] = path
    for path in [
        *ROOT.glob("tests/test_range_readout*.py"),
        Path(__file__),
        ROOT / "scripts/Resume-RangeReadout.ps1",
        ROOT / "docs/range_readout_integration_20261012.md",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "README.md",
        ROOT / "LICENSE",
        ROOT / ".python-version",
    ]:
        files[path.relative_to(ROOT).as_posix()] = path
    # Preserve the native implementation imported by the new integration as well.
    for folder in (ROOT / "src", ROOT / "operational"):
        for path in folder.rglob("*.py"):
            files[path.relative_to(ROOT).as_posix()] = path
    for path in (ROOT / "configs").rglob("*"):
        if path.is_file() and path.suffix in (".yaml", ".yml", ".json"):
            files[path.relative_to(ROOT).as_posix()] = path
    manifest = {name: sha(path) for name, path in sorted(files.items())}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    metadata = json.dumps(
        {
            "git_head": commit,
            "base_commit": "1c251781ac845a5d661523e607e0f7c4a8d3af8a",
            "state": read(RUN / "NEXT_DECISION.json"),
            "optimizer_updates_during_packaging": 0,
        },
        indent=2,
    ).encode()
    manifest["BUNDLE_PROVENANCE.json"] = hashlib.sha256(metadata).hexdigest()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
            if sha(path) != manifest[name]:
                raise RuntimeError("Source changed while packaging: " + name)
        archive.writestr("BUNDLE_PROVENANCE.json", metadata)
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2))
    result = verify(destination)
    destination.with_suffix(".zip.sha256").write_text(
        result["sha256"] + "  " + destination.name + "\n", encoding="ascii"
    )
    destination.with_suffix(".zip.verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"bundle": str(destination), **result}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="store_true")
    parser.add_argument("--verify-zip", type=Path)
    args = parser.parse_args()
    if args.verify_zip:
        print(json.dumps(verify(args.verify_zip)))
        return
    generate()
    if args.bundle:
        bundle()


if __name__ == "__main__":
    main()
