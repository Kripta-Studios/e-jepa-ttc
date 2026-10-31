"""Regenerate the native context augmentation report from completed sealed runs."""

import argparse
import json
from pathlib import Path

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_campaign.data import read


def main() -> None:
    """Publish measured outcomes, preserving resubstitution and development limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/research_log_20261031_native_context")
    )
    args = parser.parse_args()
    sources = {
        "extraction": args.cache / "COMPLETE.json",
        "selection": args.campaign / "SELECTION.json",
        "training_config": args.campaign / "CONFIG.json",
        "control_reproduction": args.campaign / "IQ_REPRODUCTION.json",
        "source_closure": args.closure,
    }
    data = {key: read(path) for key, path in sources.items()}
    extraction, selection = data["extraction"], data["selection"]
    if (
        extraction["status"] != "COMPLETE"
        or selection["fits_completed"] != 48
        or not extraction["clean_exact"]
        or not extraction["disk_roundtrip_exact"]
    ):
        raise ValueError("Report requires completed extraction, clean parity and 48 fits")
    for filename, key in (
        ("OOF_PREDICTIONS.npz", "oof_sha256"),
        ("RESTART_OOF_PREDICTIONS.npz", "restart_oof_sha256"),
        ("CONFIG.json", "config_sha256"),
    ):
        if file_sha(args.campaign / filename) != selection[key]:
            raise ValueError("Changed training evidence")
    evidence = args.output / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        (evidence / f"{name}.json").write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
    lines = [
        "# PHASE-TTC: ampliación mediante reinicios del contexto nativo",
        "",
        "Informe generado desde una extracción y una comparación completadas. "
        "Los datos pertenecen al desarrollo con exposición histórica; los resultados "
        "no equivalen a una evaluación oficial ni a una reserva nueva.",
        "",
        "## Qué se construyó y comprobó",
        "",
        f"Se extrajeron {extraction['queries']} consultas: tres cuantiles temporales de "
        "cada una de las 24 grabaciones de carretera elegibles y las 77 de Slider-500. "
        "La selección no utilizó errores ni targets. Cada consulta conserva observaciones "
        "raw17/expert3, relojes, ruta de inferencia y tensores originales del contexto y cabeza.",
        "",
        "La reproducción limpia vuelve a leer NPZ y JSON verificados por hash y compara "
        "exactamente los tensores con la inferencia instrumentada. Se conservan las rutas "
        "de fusión, solo eventos y ausencia; las consultas no disponibles permanecen "
        "en los artefactos. El recibo complementario verifica las dependencias numéricas "
        "que no figuraban en el inventario inicial de fuentes.",
        "",
        f"Se generaron {extraction['augmentations']} variantes: últimas una, dos y tres "
        "observaciones aceptadas por modalidad, con relojes contextuales recalculados. "
        "La primera observación retenida tiene intervalo desde la anterior igual a cero. "
        "Los productores mantienen sus ventanas sensoriales originales: esto es reinicio "
        "de contexto, no arranque del sensor ni ampliación por cambio de velocidad.",
        "",
        f"Rutas conservadas: `{json.dumps(extraction['status_counts'], sort_keys=True)}`. "
        f"Tiempo instrumentado: {extraction['total_seconds']:.3f} s. Incluye extracción, "
        "guardado y comprobaciones; no es latencia de producción.",
        "",
        "## Entrenamiento y criterio de selección",
        "",
        "Se ejecutaron 48 ajustes: dos condiciones, ocho familias reservadas y semillas "
        "7,13,21. Se conservaron arquitectura de 688 parámetros, 300 actualizaciones AdamW, "
        "lr 0.01, weight decay 0.01 y recorte de gradiente a 1. Cada variante permanece "
        "en el grupo de su consulta. Normalización y escala de error usan solo train limpio.",
        "",
        "IQ reproduce el control original: mitad de pérdida limpia y mitad con corrupción "
        "de un experto en el 25 % de las filas. IQn sustituye ese segundo término por los "
        "reinicios nativos, equilibrando familias y consultas y dividiendo el peso de "
        "cada consulta entre sus variantes admitidas. Geometría y sus 20 variables de "
        "calidad permanecen iguales. No se añaden entradas de soporte nativo a la red.",
        "",
        "La selección exige cobertura completa y no empeorar MAE ni error relativo "
        "macro por grabación, tanto en carretera como en Slider-500. Las métricas de "
        "reinicio reservado son diagnósticas y no seleccionan el candidato.",
        "",
        "| Condición | RAE carretera macro (%) | MAE carretera macro (s) | "
        "RAE Slider-500 (%) | MAE Slider-500 (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in ("IQ", "IQn"):
        road, slider = (selection["report"][mode][k] for k in ("road", "slider"))
        lines.append(
            f"| {mode} | {road['macro_record_relative_error_pct']:.6f} "
            f"| {road['macro_record_mae_seconds']:.6f} "
            f"| {slider['macro_record_relative_error_pct']:.6f} "
            f"| {slider['macro_record_mae_seconds']:.6f} |"
        )
    lines += [
        "",
        f"Selección: `{selection['selected']}`. "
        f"Criterios: `{json.dumps(selection['gates'], sort_keys=True)}`.",
        "",
        "La campaña no promueve automáticamente un modelo. Un candidato admitido "
        "requiere refit y comprobación activa de conservación eAP; TRAIN40 es "
        "resustitución y Test12 sin targets solo permite medir deriva. Slider-750 "
        "sigue necesitando conciliación con el evaluador oficial. Ninguna cifra "
        "de este informe demuestra SOTA.",
        "",
        "## Reproducibilidad",
        "",
        "Los recibos publicados conservan configuración, grupos, semillas, hashes "
        "y resultados. El runner `train_phase_native_context_fusion.py` expone "
        "los argumentos con `--help`; requiere directorios nuevos y las cachés "
        "selladas. Datos crudos y pesos no se incorporan a Git.",
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
