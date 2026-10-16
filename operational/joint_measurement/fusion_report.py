"""Regenerate fusion results and their limits from frozen predictions and proofs."""

from __future__ import annotations

from operational.range_transfer.common import read, sha, write

from .common import ROOT
from .fusion_train import DIRECTORY


def build() -> dict:
    """Require all seven prediction tables and metric proofs before writing the report."""
    result = read(DIRECTORY / "evaluation/RESULT.json")
    rows = []
    for key, value in result["results"].items():
        role, name = key.split("/")
        folder = DIRECTORY / "predictions" / role / name
        proof = value["verification"]
        if not (
            proof["status"] == "PASS"
            and sha(folder / "predictions.parquet") == value["sha256"] == proof["prediction_sha256"]
            and sha(folder / "METRICS.json") == proof["metrics_sha256"]
        ):
            raise ValueError("Fusion report prediction or metric evidence changed")
        m = value["metrics"]
        rows.append(
            {
                "role": role,
                "model": name,
                "rows": m["num_rows"],
                "aggregate": m["overall_MiD"],
                **{band: entry["group_macro_mid"] for band, entry in m["bins"].items()},
                "mae": m["diagnostics"]["mae_s"],
                "sign": m["diagnostics"]["sign_error_rate"],
            }
        )
    lines = [
        "# V18: fusión C1 + R_CTX, semilla 7",
        "",
        "Estado: señal de desarrollo replicable; **NO_SYSTEM_PROMOTION**. "
        "No hay evidencia oficial ni afirmación SOTA.",
        "",
        "La medición espacial C1 se entrenó sobre P durante 12.000 updates. "
        "Su predictor independiente quedó rechazado por signo/banda negativa. "
        "Se reutilizan sus features espaciales nuevas (196 valores) como complemento "
        "de R_CTX (28 valores e historial RGB nativo), conservando el anchor firmado.",
        "",
        "TRUE y ZERO tienen la misma capacidad, inicialización y receta: 2.500 updates "
        "cada uno, batch 128, AdamW y pérdida de fase emitida más 0,1 de fase raw. "
        "ZERO anula todo el bloque event, incluido el tiempo. Si falta RGB, ambos "
        "copian exactamente el TTC de C1. No se promedian TTC de signos opuestos.",
        "",
        "R_CTX ya estaba entrenado en H. Estas cabezas usan H para ajuste secuencial "
        "supervisado; H no es OOF ni validación independiente. Los dos endpoints "
        "se congelaron antes de abrir V y antes de calcular sus diagnósticos H. "
        "V sigue siendo desarrollo interno expuesto.",
        "",
        "## Resultados completos",
        "",
        "| Población | Modelo | n | Agregado | MiDc | MiDs | MiDl | MiDn "
        "| MAE s | Error de signo |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['role']} | {row['model']} | {row['rows']} | {row['aggregate']:.3f} "
            f"| {row['crucial']:.3f} | {row['small']:.3f} | {row['large']:.3f} "
            f"| {row['negative']:.3f} | {row['mae']:.3f} | {100 * row['sign']:.3f}% |"
        )
    lines += ["", "## Contrastes y decisión", ""]
    true = result["results"]["V/FUSION-TRUE-S7"]["metrics"]
    for reference in ("FUSION-ZERO-S7", "R_CTX", "F_TRUE"):
        baseline = result["results"][f"V/{reference}"]["metrics"]
        key = f"FUSION-TRUE-S7_vs_{reference}"
        delta = 100 * (true["overall_MiD"] / baseline["overall_MiD"] - 1)
        crucial = 100 * (
            true["bins"]["crucial"]["group_macro_mid"]
            / baseline["bins"]["crucial"]["group_macro_mid"]
            - 1
        )
        ci = result["bootstrap"][key]["intervals"]["delta_overall"]["ci95_if_all_defined"]
        failed = [name for name, passed in result["gates"][key]["checks"].items() if not passed]
        lines.append(
            f"- Frente a {reference}: agregado {delta:+.2f}%, MiDc {crucial:+.2f}%; "
            f"IC95 por grupos del cambio agregado [{ci[0]:.3f}, {ci[1]:.3f}]. "
            f"Gate fallido: {', '.join(failed)}."
        )
    lines += [
        "",
        "El intervalo contra F_TRUE incluye cero: una semilla no establece "
        "superioridad frente a ese sistema. Contra ZERO y R_CTX hay una señal "
        "agregada, pero falta la mejora crucial del 10% exigida para promoción. "
        "La banda larga empeora frente a R_CTX aunque permanece dentro de su "
        "salvaguarda. Todas las regresiones se conservan.",
        "",
        "La señal controlada habilita la única familia de réplica: seeds 13 y 23, "
        "reentrenando encoder, lectura geométrica y las dos cabezas de cada semilla. "
        "Se mantienen A5/R_CTX históricos como inicialización/anchor compartidos; "
        "no es réplica del pretraining original. Son 34.000 updates científicos "
        "nuevos registrados antes de comenzar. Ningún resultado permite escoger "
        "la mejor semilla, reajustar la receta con V o promover C1 retrospectivamente.",
        "",
        "## Verificación y alcance",
        "",
        "- 18.422 filas H y 16.432 V por brazo, sin exclusión de consultas. "
        "En estas poblaciones existe RGB en todas las filas; el fallback se prueba "
        "también con máscaras sintéticas all/missing/mixed, no con fallos reales inventados.",
        "- Cachés FP32 sin pérdida, relojes nativos separados, normalizadores ajustados "
        "sólo en H. Cada salida del extractor R_CTX se contrasta exactamente con "
        "el forward histórico; V coincide además con su CSV original admitido.",
        "- Siete tablas recalculadas mediante NumPy independiente. Cinco contrastes "
        "con 2.000 resamples de grupos completos; no ventanas IID ni incertidumbre "
        "entre semillas disfrazada de bootstrap.",
        "- 320.000 IDs de muestreo regenerados por cabeza; Adam y scheduler en 2.500, "
        "ocho tensores aprendidos por brazo y estado final del sampler exacto. "
        "La QA CUDA de reanudación contó 40 updates técnicos antes de los científicos.",
        "- Dos exports TorchScript con normalizadores y fallback dinámico. "
        "Comienzan en features de productores: no contienen encoders ni preprocessing "
        "sensorial. Sus tiempos CPU son sólo de cabeza; el coste completo sigue pendiente.",
        "",
        "Las fuentes durables son `fusion/REGISTRATION.json`, `ENDPOINTS.json`, "
        "`evaluation/RESULT.json`, `inputs/{H,V}`, `predictions/{H,V}`, `bootstrap/V` "
        "y `exports`. Los pesos, curvas y estados completos permanecen en `fusion/fits`.",
        "",
        "Regeneración: `python -B -m operational.joint_measurement.fusion_report`.",
    ]
    supplementary = DIRECTORY / "v16_compare/SEED7.json"
    if supplementary.exists():
        contrast = read(supplementary)
        baseline = contrast["reference_metrics"]["7"]
        candidate = contrast["candidate_seed7_metrics"]
        interval = contrast["bootstrap_seed7"]["intervals"]["delta_overall"]["ci95_if_all_defined"]
        lines += [
            "", "## Comparador V16 conservado", "",
            "Tras congelar la decisión de réplica se volvió a comprobar el estado real "
            "de V16: su candidato PHASE_ONLY y sus réplicas están terminados; conserva "
            "6.762 predicciones test12 sin envío ni score oficial. No se repite ese trabajo.",
            "",
            f"En el contraste descriptivo de semilla 7 y cohorte V idéntica, PHASE_ONLY "
            f"obtiene agregado {baseline['overall_MiD']:.3f} y la nueva fusión "
            f"{candidate['overall_MiD']:.3f}. El IC95 del cambio es "
            f"[{interval[0]:.3f}, {interval[1]:.3f}] e incluye cero. La banda larga "
            f"empeora de {baseline['bins']['large']['group_macro_mid']:.3f} a "
            f"{candidate['bins']['large']['group_macro_mid']:.3f}, superando también "
            "esa salvaguarda V18. No se ha demostrado superioridad sobre el candidato V16.",
            "", "La comparación usa los CSV sin etiquetas sellados y sus checkpoints "
            "verificados; adjunta los targets V nativos sin modificar predicciones. "
            "No cambia los gates históricos de V16 ni la receta V18 ya registrada. "
            "La evidencia queda en `fusion/v16_compare`.",
        ]
    path = ROOT / "docs/joint_measurement_20261012/FUSION_RESULTS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "rows": rows,
        "evidence_sha256": sha(DIRECTORY / "evaluation/RESULT.json"),
        "system_promotion": False,
        "replication_selected": True,
    }
    write(DIRECTORY / "RESULTS.json", summary)
    return summary


if __name__ == "__main__":
    print(build())
