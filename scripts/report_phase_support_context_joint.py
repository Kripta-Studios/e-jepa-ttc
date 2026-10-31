"""Regenerate the joint experiment and cross-campaign comparison from sealed receipts."""

import argparse
import json
from pathlib import Path

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_campaign.data import read


def main() -> None:
    """Publish measured outcomes, including the failed promotion gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign", type=Path, default=Path("output/phase_native_support/joint_v1")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/research_log_20261031_joint_support")
    )
    args = parser.parse_args()
    sources = {
        k: args.campaign / f"{k.upper()}.json" for k in ("selection", "config", "verification")
    }
    data = {k: read(p) for k, p in sources.items()}
    selection = data["selection"]
    if selection["fits_completed"] != 48 or data["verification"]["status"] != "PASS":
        raise ValueError("Require 48 completed and independently verified fits")
    for name, key in (
        ("OOF_PREDICTIONS.npz", "oof_sha256"),
        ("RESTART_OOF_PREDICTIONS.npz", "restart_oof_sha256"),
        ("CONFIG.json", "config_sha256"),
    ):
        if file_sha(args.campaign / name) != selection[key]:
            raise ValueError("Changed experiment evidence")
    evidence = args.output / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    for name, path in sources.items():
        (evidence / f"{name}.json").write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    lines = [
        "# PHASE-TTC: interacción entre soporte y reinicios de contexto",
        "",
        "Esta campaña evalúa una comparación adicional fijada antes de entrenar. "
        "Combina los reinicios reales con las entradas que describen sus relojes y "
        "recuentos. No se modificaron campañas anteriores ni se usó el benchmark "
        "para escoger configuraciones.",
        "",
        "## Qué cambia",
        "",
        "F2n conserva el router de 868 parámetros y Fcapn el control de 878. Ambos "
        "usan mitad de pérdida limpia y mitad con los 447 reinicios reales; no añaden "
        "corrupción multiplicativa QA. Los grupos reservados excluyen también todos "
        "sus reinicios. Se equilibran familias y consultas, dividiendo el peso entre "
        "las variantes admitidas de cada consulta.",
        "",
        "Los 33 inputs originales y la escala del MAE se normalizan con train limpio. "
        "Las diez entradas nuevas de F2n se normalizan y se habilitan por varianza "
        "usando la unión de soportes limpios y aumentados admitidos de train. "
        "Validación no interviene. Esto cambia tanto los ejemplos/objetivo como "
        "el conjunto que normaliza el soporte; no aísla solo el efecto de la pérdida.",
        "",
        "En el fold Slider, F2n puede aprender variación de ambos recuentos y de "
        "los intervalos RGB/eventos: máscara [0,0,1,0,1,0,0,1,0,1]. El experimento "
        "anterior con QA solo habilitaba el intervalo RGB. Esta diferencia estaba "
        "especificada antes de entrenar.",
        "",
        "Tres semillas 7,13,21, ocho folds y 300 actualizaciones por ajuste: 48 fits. "
        "El control de selección sigue siendo IQ; IQn/F2/Fcap previos son referencias "
        "informativas, verificadas por hashes e identidades. No seleccionan modelos.",
        "",
        "## Comparación de las condiciones ejecutadas",
        "",
        "Todas las cifras siguientes son OOF de desarrollo con exposición histórica. "
        "Se agregan las tres semillas y se promedian las métricas por grabación. "
        "No son resultados oficiales ni nuevas métricas del modelo entregable.",
        "",
        "| Condición | RAE carretera macro (%) | MAE carretera macro (s) | "
        "RAE Slider-500 (%) | MAE Slider-500 (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in ("IQ", "IQn", "F2", "Fcap", "F2n", "Fcapn"):
        road, slider = (selection["report"][mode][k] for k in ("road", "slider"))
        lines.append(
            f"| {mode} | {road['macro_record_relative_error_pct']:.6f} "
            f"| {road['macro_record_mae_seconds']:.6f} "
            f"| {slider['macro_record_relative_error_pct']:.6f} "
            f"| {slider['macro_record_mae_seconds']:.6f} |"
        )
    lines += [
        "",
        f"Selección conjunta: `{selection['selected']}`. "
        f"Tiempo: {selection['elapsed_seconds']:.3f} s. "
        f"Dominancia F2n/Fcapn: `{selection['support_dominates_capacity']}`.",
        "",
        "La selección exige cobertura completa y no empeorar ninguna de las "
        "cuatro métricas limpias frente a IQ. Mejorar solo errores relativos "
        "no basta si empeora MAE; no se relajan los criterios después de ver "
        "los resultados. La dominancia entre candidatos se interpreta como "
        "asociación experimental, no como prueba causal de generalización.",
        "",
        "## Verificación y consecuencia",
        "",
        "Se verificaron los 48 checkpoints, 300 pérdidas por fit, aislamiento "
        "de grupos, masas de variantes, normalizadores y máscaras recalculados "
        "por fold, referencias inmutables y reconstrucción exacta de predicciones "
        "por fold y promedios de semillas. Las predicciones de reinicio son "
        "diagnóstico secundario y no intervienen en selección.",
        "",
        "La campaña no ejecuta refit ni exportación automáticamente. Los "
        "candidatos rechazados no se evalúan en el benchmark para buscar "
        "una cifra favorable. eAP TRAIN40 activo seguiría siendo un control "
        "de resustitución y Test12 sin etiquetas no certificaría precisión.",
        "",
        "Las pruebas de fusión han identificado intercambios entre carretera, "
        "Slider, historia corta, error relativo y error en segundos. No demuestran "
        "que el modelo sea irreduciblemente óptimo ni que más capacidad garantice "
        "mejora. El siguiente diagnóstico debe justificar una intervención en "
        "sensores o productores antes de abrir otra búsqueda de cabezas.",
        "",
        "Este informe se regenera con `scripts/report_phase_support_context_joint.py`. "
        "Pesos, matrices y datos permanecen fuera de Git; se publican código "
        "y evidencia verificable. No se afirma SOTA.",
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
