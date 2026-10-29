"""Regenerate the capacity contrast report from complete cohorts and registered decisions."""

from __future__ import annotations

from operational.range_transfer.common import read, sha, write

from .capacity_train import CONTROL, DIRECTORY, FIT, REGISTRATION
from .common import ROOT, RUN


def build() -> dict:
    """Include every band and failed protection; distinguish P mechanism from H admission."""
    source = DIRECTORY / "evidence/COMPLETE.json"
    result, recipe = read(source), read(REGISTRATION)
    if result["status"] != "D1_READY_FOR_INTERPRETATION":
        raise ValueError("Complete D1 evidence required")
    decision = result["selection"]
    lines = [
        "# D1: resultado del contraste de capacidad",
        "",
        f"Estado: **{decision['status']}**. No implica promoción del sistema ni SOTA.",
        "",
        "Un único fit seed7 de 12.000 updates P duplica la anchura del predictor "
        "espacial y su MLP. La CNN nativa de 64 canales conserva su inicialización "
        "y se entrena realmente. El control PAIR-CORR-S7 se reutiliza con igual "
        "presupuesto de updates; su coste no se cuenta otra vez. Inputs, objetivo "
        "firmado, sampler y optimizador son los mismos.",
        "",
        f"Parámetros: {recipe['capacity']['control']:,} en el control y "
        f"{recipe['capacity']['wide']:,} en D1. El factor cambia la anchura de la "
        "torre y MLP; no se afirma que toda la arquitectura o el número de "
        "parámetros se hayan multiplicado exactamente por dos.",
        "",
        "## Cohortes completas",
        "",
        "| Población | Modelo | n | Agregado | MiDc | MiDs | MiDl | MiDn | MAE s | Signo |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, metric in result["cohorts"].items():
        role, name = key.split("/")
        folder = (DIRECTORY if name == FIT else RUN) / "predictions" / role / name
        proof = result["metric_proofs"][key]
        if not (
            proof["status"] == "PASS"
            and proof["prediction_sha256"] == sha(folder / "predictions.parquet")
            and proof["metrics_sha256"] == sha(folder / "METRICS.json")
            and metric == read(folder / "METRICS.json")
        ):
            raise ValueError("D1 report evidence changed")
        values = [metric["overall_MiD"]] + [
            metric["bins"][band]["group_macro_mid"]
            for band in ("crucial", "small", "large", "negative")
        ]
        values.append(metric["diagnostics"]["mae_s"])
        lines.append(
            f"| {role} | {name} | {metric['num_rows']} | "
            + " | ".join(f"{value:.3f}" for value in values)
            + f" | {100 * metric['diagnostics']['sign_error_rate']:.3f}% |"
        )
    p_base, p_wide = result["cohorts"][f"P/{CONTROL}"], result["cohorts"][f"P/{FIT}"]
    delta = 100 * (p_wide["overall_MiD"] / p_base["overall_MiD"] - 1)
    failed = [name for name, passed in decision["decision"].get("checks", {}).items() if not passed]
    lines += [
        "",
        "## Interpretación registrada",
        "",
        f"Cambio agregado P frente al control: {delta:+.3f}%. Señal de capacidad "
        f"según el criterio prospectivo: {result['P_capacity_signal']}. Este criterio "
        "requiere al menos 3% de mejora P y signo no peor. Describe el ajuste de "
        "entrenamiento; por sí solo no demuestra generalización.",
        "",
        "La admisión H conserva A5-P2-FROZEN y las protecciones originales. "
        f"Comprobaciones fallidas: {', '.join(failed) or 'ninguna de cobertura/bandas/signo'}. "
        "La decisión completa incluye además las rutas de mejora o coste y se "
        "conserva en `SELECTION_FINAL.json`.",
        "",
        "## Bootstrap y coste",
        "",
    ]
    for name, bootstrap in result["bootstrap"].items():
        interval = bootstrap["intervals"]["delta_overall"]["ci95_if_all_defined"]
        text = f"[{interval[0]:.3f}, {interval[1]:.3f}]" if interval else "no definido"
        lines.append(f"- {name}: IC95 del cambio agregado D1 − referencia: {text}.")
    lines += [
        "",
        "Se remuestrean grupos completos, no ventanas IID. Son intervalos de una "
        "semilla y no estiman la variabilidad entre entrenamientos.",
        "",
        "| Modelo | Mediana forward preparado, ms | p95, ms |",
        "|---|---:|---:|",
    ]
    for name, timing in result["prepared_cost"]["results"].items():
        lines.append(
            f"| {name} | {1000 * timing['p50_seconds']:.3f} | {1000 * timing['p95_seconds']:.3f} |"
        )
    lines += [
        "",
        "Estos tiempos empiezan con tensores event preparados en GPU y terminan "
        "en TTC firmado. Excluyen lectura sensorial, preprocessing, transferencias "
        "y carga de pesos: **no son E2E**. Se conserva el orden emparejado y fijo.",
        "",
        "El endpoint se comprueba sobre pesos, Adam, scheduler, encoder y 384.000 "
        "IDs de muestreo regenerados. Las predicciones completas tienen recálculo "
        "NumPy independiente. El export TorchScript incluye ejemplos CPU y paridad "
        "con el forward nativo; comienza en tensores P2, no en eventos crudos.",
        "",
        "No se abre V para rescatar un fallo H. La familia de réplica seleccionada "
        "sigue siendo C1-fusión; D1 es un contraste adicional de capacidad y no "
        "una segunda búsqueda de semillas. Una señal P favorable y una admisión "
        "H son decisiones diferentes.",
        "",
        "Regeneración: `python -B -m operational.joint_measurement.capacity_report`. "
        "Fuentes durables bajo `adaptive/D1_predictor_width`: `REGISTRATION.json`, "
        "`evaluation`, `evidence/COMPLETE.json`, `predictions`, `cost` y `exports`.",
    ]
    erratum = DIRECTORY / "evidence/COST_SCOPE_ERRATUM.json"
    if erratum.exists():
        lines += [
            "",
            "Errata de metadatos: el campo `decision.cost_scope` del JSON original "
            "heredó la palabra RGB del helper compartido. La medición registrada y "
            "ejecutada utiliza eventos P2, como indica este informe. Se conserva el "
            "JSON original y `evidence/COST_SCOPE_ERRATUM.json`; no cambia ninguna cifra.",
        ]
    path = ROOT / "docs/joint_measurement_20261012/D1_RESULTS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt = {
        "source_sha256": sha(source),
        "report_sha256": sha(path),
        "automatic_promotion": False,
    }
    write(DIRECTORY / "evidence/REPORT.json", receipt)
    return receipt


if __name__ == "__main__":
    print(build())
