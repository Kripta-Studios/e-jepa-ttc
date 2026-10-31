"""Regenerate canonical-PAIR diagnostic outcomes, including downstream regressions."""

import argparse
from pathlib import Path

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_campaign.data import read
from scripts.build_phase_native_context_augmentations import verify_artifacts
from scripts.report_phase_producer_timewarp import interval


def main() -> None:
    """Publish all context/head responses without using target accuracy for selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("output/phase_timewarp/canonical_v1"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/research_log_20261031_canonical_pair")
    )
    parser.add_argument(
        "--eap-audit",
        type=Path,
        default=Path("output/phase_timewarp/eap_current_pair_audit/verified_run_v2"),
    )
    args = parser.parse_args()
    complete = read(args.run / "COMPLETE.json")
    if (
        complete["status"] != "COMPLETE"
        or complete["targets_read"]
        or complete["optimizer_updates"] != 0
        or len(complete["reports"]) != 2
        or not all(q["baseline_exact"] for q in complete["reports"].values())
    ):
        raise ValueError("Require complete two-query no-label diagnostic and exact baseline")
    verify_artifacts(args.run, complete["artifacts"])
    if file_sha(args.run / "FREEZE.json") != complete["freeze_sha256"]:
        raise ValueError("Frozen intervention changed")
    evidence = args.output / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    for name in ("COMPLETE", "FREEZE"):
        (evidence / f"{name.lower()}.json").write_bytes(
            (args.run / f"{name}.json").read_bytes().replace(b"\r\n", b"\n")
        )
    lines = [
        "# PHASE-TTC: PAIR con conversión temporal explícita",
        "",
        "Intervención motivada por el diagnóstico previo de dos consultas. No se leen "
        "targets, no se actualizan pesos y no se modifica el modelo entregable. Se reutilizan "
        "los tensores sellados de carretera y Slider-500. El baseline original vuelve a "
        "reproducirse exactamente antes de introducir la intervención.",
        "",
        "PAIR recibe sus mismas representaciones y soportes, con sus tres variables de "
        "tiempo reconstruidas a 100 ms. Su fase cruda se convierte a inverso TTC, se escala "
        "por 0,1/dt_real y se convierte de nuevo a fase. La política original de emisión "
        "actúa después, una sola vez. Las filas cuyo dt es exactamente FP32(0,1) devuelven "
        "la fase original sin conversión. Los rechazos de dominio se conservan con trazas "
        "y sin sustituir argumentos inválidos por un epsilon.",
        "",
        f"Tiempo de ejecución, incluida carga y reproducción: "
        f"**{complete['elapsed_seconds']:.3f} s**. Actualizaciones de pesos: **0**.",
        "",
        "## Predicción limpia",
        "",
        "Son predicciones, no errores. El orden es F0 original de nueve componentes, "
        "contexto RGB y contexto de eventos. El router adaptado de la entrega no forma "
        "parte de este piloto.",
        "",
        "| Consulta | Bloque | Original (s) | PAIR corregido (s) | Diferencia (s) |",
        "|---|---|---:|---:|---:|",
    ]
    blocks = ("F0", "Contexto RGB", "Contexto eventos")
    for name, query in complete["reports"].items():
        for block, old, new in zip(
            blocks,
            query["original_clean_native_seconds"],
            query["clean_native_seconds"],
            strict=True,
        ):
            if old is None or new is None:
                values = f"{old} | {new} | no definido"
            else:
                values = f"{old:.9f} | {new:.9f} | {new - old:+.9f}"
            lines.append(f"| {name} | {block} | {values} |")
    lines += [
        "",
        "## Respuesta temporal completa",
        "",
        "El cociente a·T_transformado/T_control debería ser 1 en el caso ideal. Se "
        "utiliza el control limpio con las mismas observaciones y lotes. Menor distancia "
        "a 1 significa mayor consistencia ante esta transformación, no menor error real.",
        "",
        "| Consulta | Factor | Bloque | Original | Corregido | "
        "Cambio de distancia a 1 | Mismas observaciones admitidas |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for name, query in complete["reports"].items():
        for factor in query["factors"].values():
            for block, old, new in zip(
                blocks,
                factor["original_native_ttc_ratio"],
                factor["native_ttc_ratio_against_matched_control"],
                strict=True,
            ):
                if old is None or new is None:
                    values = f"{old} | {new} | no definido"
                else:
                    values = f"{old:.6f} | {new:.6f} | {abs(new - 1) - abs(old - 1):+.6f}"
                lines.append(
                    f"| {name} | {factor['speed']} | {block} | {values} | "
                    f"{factor['head_comparison_isolating']} |"
                )
    lines += [
        "",
        "## Respuesta de PAIR",
        "",
        "| Consulta | Factor | PAIR | Intervalo de a·T_transformado/T_control |",
        "|---|---:|---|---|",
    ]
    for name, query in complete["reports"].items():
        for factor in query["factors"].values():
            for block in ("PAIR_R", "PAIR_E_MATCHED"):
                ratios = [
                    v
                    for stages in factor["stage_by_observation"].values()
                    if block in stages
                    for v in stages[block]["ttc_ratio"]
                ]
                lines.append(f"| {name} | {factor['speed']} | {block} | {interval(ratios)} |")
    lines += [
        "",
        "## Interpretación y siguiente prueba",
        "",
        "La intervención corrige la sensibilidad temporal de PAIR observada en estas "
        "consultas. La tabla completa conserva también las respuestas de los contextos: "
        "la consistencia de un bloque intermedio no garantiza la del bloque posterior. "
        "En particular, el contexto de eventos de carretera al factor 0,5 empeora; no "
        "se oculta esa regresión detrás del comportamiento de la salida final.",
        "",
        "La siguiente decisión exige precisión y cobertura en desarrollo agrupado, "
        "recalculando las historias afectadas. Las dos consultas no permiten seleccionar "
        "un modelo por accuracy. La auditoría eAP de entradas actuales, cuando se informe "
        "por separado, tampoco sustituye el control completo de contexto y cabeza ni "
        "aporta etiquetas de evaluación. Se mantienen los criterios de promoción previos.",
        "",
        "No se exporta un nuevo modelo ni se afirma SOTA. El informe se regenera con "
        "`scripts/report_phase_canonical_pair.py`; sus tablas proceden de los artefactos "
        "verificados. Tensores y checkpoints permanecen fuera de Git.",
        "",
    ]
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    audit = read(args.eap_audit / "COMPLETE.json")
    if audit["status"] != "COMPLETE" or audit["targets_read"] or audit["optimizer_updates"] != 0:
        raise ValueError("Require complete label-blind eAP current-query audit")
    verify_artifacts(args.eap_audit, audit["artifacts"])
    if file_sha(args.eap_audit / "FREEZE.json") != audit["freeze_sha256"]:
        raise ValueError("eAP audit freeze changed")
    for name in ("COMPLETE", "FREEZE"):
        (evidence / f"eap_{name.lower()}.json").write_bytes(
            (args.eap_audit / f"{name}.json").read_bytes().replace(b"\r\n", b"\n")
        )
    for name in ("INPUTS_COMPLETE", "OUTPUT_FLOOR_AUDIT", "RUN_COMPLETE"):
        (evidence / f"eap_prior_{name.lower()}.json").write_bytes(
            (args.eap_audit.parent / f"{name}.json").read_bytes().replace(b"\r\n", b"\n")
        )
    eap_lines = [
        "# Auditoría eAP de PAIR en la observación actual",
        "",
        "Se usaron features de productores de despliegue TRAIN40, no los estados OOF "
        "de la auditoría de fusión activa. Solo se leyeron features, identidades y grupos; "
        "no se leyeron targets. Los padres de los productores y los miembros de los "
        "archivos locales quedaron vinculados por hashes.",
        "",
        "| Modalidad | Filas | Identidad exacta 100 ms | Batches rechazados | "
        "Cambios de signo | Cambio medio absoluto TTC (s) | Máximo (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in audit["modalities"].items():
        eap_lines.append(
            f"| {name} | {result['rows']} | {result['identity_rows']} | "
            f"{len(result['rejected_batches'])} | {result['sign_changes']} | "
            f"{result['ttc_abs_drift_mean']:.11f} | {result['ttc_abs_drift_max']:.11f} |"
        )
    eap_lines += [
        "",
        f"Tiempo medido después de cargar modelos: {audit['elapsed_seconds']:.6f} s.",
        "",
        "La identidad se verifica en bytes para fase cruda, fase emitida y TTC en "
        "las filas exactas de 100 ms. La política de suelo de magnitud del inverso "
        "permanece sin cambios. Un contador inicial que comprobaba TTC == 60 era "
        "incorrecto: el valor FP32 emitido es 59,999996. Se conserva el recibo inicial "
        "y la corrección explícita; el CLI definitivo usa el predicado de inverso.",
        "",
        "No se han reconstruido todas las observaciones históricas, los contextos ni "
        "la cabeza final. Estos cambios de salida no son errores contra ground truth "
        "ni certifican conservación de eAP. Test12 sigue sin targets disponibles.",
        "",
        "El CLI reproducible es `scripts/audit_phase_canonical_pair_source.py`. "
        "Requiere la extracción sellada en `output/phase_timewarp/eap_current_pair_audit` "
        "y un directorio de salida nuevo. Regenerar este documento no ejecuta inferencia.",
        "",
    ]
    (args.output / "EAP_CURRENT_QUERY_AUDIT.md").write_text(
        "\n".join(eap_lines), encoding="utf-8", newline="\n"
    )
    print(f"Report regenerated: {args.output}")


if __name__ == "__main__":
    main()
