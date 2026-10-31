"""Regenerate the completed native-support versus capacity comparison."""

import argparse
import json
from pathlib import Path

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_campaign.data import read


def main() -> None:
    """Report primary gates and confounds without promoting an unverified candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("output/phase_native_support/cache_v1"))
    parser.add_argument(
        "--campaign", type=Path, default=Path("output/phase_native_support/campaign_v1")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/research_log_20261031_native_support")
    )
    args = parser.parse_args()
    sources = {
        "cache": args.cache / "COMPLETE.json",
        "selection": args.campaign / "SELECTION.json",
        "config": args.campaign / "CONFIG.json",
    }
    data = {name: read(path) for name, path in sources.items()}
    result = data["selection"]
    if result["fits_completed"] != 48:
        raise ValueError("Report requires all 48 preregistered fits")
    for name, key in (
        ("CONFIG.json", "config_sha256"),
        ("OOF_PREDICTIONS.npz", "oof_sha256"),
        ("RESTART_OOF_PREDICTIONS.npz", "restart_oof_sha256"),
    ):
        if file_sha(args.campaign / name) != result[key]:
            raise ValueError("Changed experiment evidence")
    evidence = args.output / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    for name, path in sources.items():
        (evidence / f"{name}.json").write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    lines = [
        "# PHASE-TTC: soporte temporal nativo y control de capacidad",
        "",
        "Esta comparación continúa tras el rechazo de la ampliación por reinicios "
        "de contexto. Se ejecuta sobre desarrollo con exposición histórica; no usa "
        "las consultas oficiales para ajustar ni constituye una evaluación SOTA.",
        "",
        "## Información nueva y procedencia",
        "",
        "Se recuperan diez valores existentes en el estado de 340 dimensiones: "
        "por modalidad, edad desde el último anchor, edad desde disponibilidad, "
        "último intervalo entre anchors, disponibilidad menos anchor y número de "
        "observaciones ejecutadas dividido por ocho. Son los índices 165:170 y "
        "335:340. No codifican duración completa de historia ni primer evento del sensor.",
        "",
        "Las tablas incluyen 2.608 consultas de desarrollo, 447 reinicios, 88.744 "
        "filas TRAIN40 y 6.762 filas lógicas Test12. Se recuperan sin ejecutar "
        "productores ni leer targets. Las 72 carreteras reextraídas y 76 consultas "
        "Slider disponibles reproducen exactamente las coordenadas históricas. "
        "Slider-500:76 y sus tres reinicios conservan ausencia explícita, con ceros "
        "como marcadores y máscaras de contexto ejecutado separadas para auditoría.",
        "",
        "TRAIN40 contiene estados upstream físicos OOF de cinco folds; Test12 "
        "contiene estados de despliegue y carece de etiquetas. Los recuentos de "
        "carretera son constantes 5 RGB / 4 eventos; TRAIN40 tiene predominantemente "
        "7 observaciones de eventos y Test12 predominantemente 4. Estas variables "
        "pueden informar del dominio: no interpretar correlación como causalidad.",
        "",
        "## Comparación fijada antes de entrenar",
        "",
        "IQ es el control original cuyo vector OOF se reprodujo bit a bit en la "
        "campaña anterior. F2 añade las diez entradas: 43→18→4, 868 parámetros. "
        "Fcap usa solo las 33 originales: 33→23→4, 878 parámetros. Fcap tiene "
        "10 parámetros más; no se afirma igualdad de capacidad funcional.",
        "",
        "Se ejecutan tres semillas (7,13,21), ocho folds por familia y 300 updates "
        "AdamW por ajuste. La pérdida conserva el algoritmo QA: mitad limpia, "
        "mitad con corrupción de un experto en el 25 % de las filas. Arquitecturas "
        "distintas consumen RNG distinto al inicializar: las perturbaciones tienen "
        "la misma distribución y semillas, no máscaras idénticas por actualización.",
        "",
        "Normalización y escala del MAE usan exclusivamente train limpio. En F2, "
        "las coordenadas nuevas con varianza de train menor de 1e-12 quedan "
        "enmascaradas a cero también en inferencia. Cada máscara se guarda por fold "
        "y semilla. Los reinicios son diagnóstico secundario y no entran en train.",
        "",
        "En el fold que reserva Slider, las carreteras de ajuste solo permiten "
        "aprender variación en el último intervalo RGB. Las otras nueve coordenadas "
        "nuevas, incluidos ambos recuentos y todo el soporte de eventos, quedan "
        "enmascaradas por ser constantes. Esta comparación no puede demostrar "
        "aprendizaje de una respuesta al arranque con pocos eventos en ese fold.",
        "",
        "## Resultados primarios limpios",
        "",
        "| Condición | RAE carretera macro (%) | MAE carretera macro (s) | "
        "RAE Slider-500 (%) | MAE Slider-500 (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in ("IQ", "F2", "Fcap"):
        road, slider = (result["report"][mode][k] for k in ("road", "slider"))
        lines.append(
            f"| {mode} | {road['macro_record_relative_error_pct']:.6f} "
            f"| {road['macro_record_mae_seconds']:.6f} "
            f"| {slider['macro_record_relative_error_pct']:.6f} "
            f"| {slider['macro_record_mae_seconds']:.6f} |"
        )
    lines += [
        "",
        f"Selección: `{result['selected']}`. Tiempo de campaña: {result['elapsed_seconds']:.3f} s.",
        "",
        "Cada candidato exige cobertura completa y las cuatro métricas no "
        "peores que IQ. Si ambos pasan, se selecciona la menor media de "
        "las cuatro razones candidato/IQ, con desempate F2. La tabla completa "
        "por semilla, criterios y diagnósticos está en los recibos JSON.",
        "",
        f"Dominancia de F2 sobre Fcap: `{result['support_dominates_capacity']}`. "
        "Incluso una dominancia es una asociación compatible con utilidad "
        "del soporte, no una prueba causal aislada de inicialización y dominio.",
        "",
        "## Límite de esta entrega experimental",
        "",
        "Esta campaña no exporta ni promueve automáticamente un modelo. "
        "Un seleccionado necesita refit y comprobación activa eAP frente a "
        "la entrega congelada; TRAIN40 sigue siendo resustitución. El diagnóstico "
        "Test12 sin etiquetas no certifica precisión. Resolver la discrepancia "
        "del evaluador Slider-750 y una evaluación comparable siguen siendo "
        "necesarios antes de afirmar SOTA.",
        "",
        "El informe se regenera con `scripts/report_phase_native_support.py`. "
        "Los datos crudos, matrices y checkpoints permanecen fuera de Git; "
        "se publican código, metodología y recibos verificables.",
        "",
    ]
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    (args.output / "evidence_manifest.json").write_text(
        json.dumps(
            {
                "generator_sha256": file_sha(Path(__file__)),
                "receipts": {
                    f"{name}.json": file_sha(evidence / f"{name}.json") for name in sources
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"REPORT {args.output}")


if __name__ == "__main__":
    main()
