"""Regenerate Spanish research report from completed, versioned experiment receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_campaign.data import read
from operational.phase_campaign.evaluation import metrics
from operational.phase_submission.runtime import load_submission_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", type=Path, default=Path("output/phase_safe_fusion"))
    parser.add_argument(
        "--submission", type=Path, default=Path("output/phase_submission/FINAL_MODEL")
    )
    parser.add_argument("--output", type=Path, default=Path("docs/research_log_20261031_sign_safe"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    source_path = args.research / "source_audit/POPULATION.npz"
    source_receipt = read(args.research / "source_audit/COMPLETE.json")
    if file_sha(source_path) != source_receipt["population_sha256"]:
        raise ValueError("Changed source audit population")
    with np.load(source_path, allow_pickle=False) as source:
        model = load_submission_model(args.submission)
        with torch.inference_mode():
            prediction = model(
                *(torch.from_numpy(source[k]) for k in ("native", "geometry", "quality"))
            ).numpy()
        native = source["native"][:, 0]
        target = source["target"]
        save_npz(
            args.research / "source_audit/CURRENT_ACTIVE_PREDICTIONS.npz",
            {"prediction": prediction, "token": source["token"]},
        )
    active = {
        "scope": "TRAIN40 resubstitution; active router; not held-out eAP generalization",
        "source_population_sha256": file_sha(source_path),
        "model_sha256": file_sha(args.submission / "MODEL.json"),
        "predictions_sha256": file_sha(
            args.research / "source_audit/CURRENT_ACTIVE_PREDICTIONS.npz"
        ),
        "F0": metrics(target, native),
        "current_active": metrics(target, prediction),
    }
    atomic_receipt(args.research / "source_audit/ACTIVE_SOURCE_REFERENCE.json", active)
    receipts = {
        "campaign_v1.json": args.research / "campaign_v1b/SELECTION.json",
        "campaign_v2.json": args.research / "campaign_v2/SELECTION.json",
        "evaluator_audit.json": args.research / "evaluator_audit/AUDIT.json",
        "source_audit.json": args.research / "source_audit/COMPLETE.json",
        "active_source_reference.json": args.research / "source_audit/ACTIVE_SOURCE_REFERENCE.json",
        "test12_audit.json": args.research / "test12_audit/COMPLETE.json",
        "augmentation.json": args.research / "augmentations/RECEIPT.json",
        "augmentation_clean_parity.json": args.research / "augmentations/CLEAN_PARITY.json",
        "slider750_investigation.json": args.research
        / "evaluator_audit/SLIDER750_INVESTIGATION.json",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = args.output / "evidence"
    evidence.mkdir(exist_ok=True)
    for name, path in receipts.items():
        # Git text normalization must not invalidate the published hash inventory.
        (evidence / name).write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    v1, v2 = read(receipts["campaign_v1.json"]), read(receipts["campaign_v2.json"])
    audit = read(receipts["evaluator_audit.json"])
    lines = [
        "# PHASE-TTC: continuación experimental del 31 de octubre de 2026",
        "",
        "Documento regenerado desde recibos de ejecución. Objetivo: mejorar EvTTC sin "
        "atribuir a SOTA resultados locales, sin ajustar con las grabaciones del benchmark "
        "y comprobando la conservación de eAP.",
        "",
        "## Resultado ejecutivo",
        "",
        "Se completaron 144 entrenamientos agrupados: 96 en v1b y 48 en v2. "
        "Ninguno de los candidatos nuevos pasó todos los criterios registrados. "
        "La versión congelada PHASE-TTC-Adapt-20261031 permanece como referencia. "
        "Otros seis ajustes del primer lanzamiento quedaron preservados tras un fallo "
        "de escritura de progreso; no se contabilizan como una campaña completa.",
        "",
        "El trabajo produjo además una corrección verificable de las fuentes del evaluador, "
        "la evaluación activa de eAP TRAIN40, una caché Test12 de despliegue sin etiquetas "
        "y cuatro aumentos causales del canal geométrico. No se envió correo al benchmark.",
        "",
        "## Protocolo y población",
        "",
        "Los ajustes usan 2.608 consultas de 25 grabaciones: 2.531 de carretera y 77 de "
        "Slider-500. Se excluyen las ocho grabaciones de carretera solapadas y Slider-750/1000. "
        "Hay ocho folds que reservan familias completas, con semillas 7,13,21, 300 actualizaciones "
        "AdamW, lr=0,01, weight_decay=0,01 y gradiente recortado a 1. Normalización y escala "
        "del error absoluto usan exclusivamente el fold de ajuste. La pérdida equilibra familias.",
        "",
        "Estos datos tienen exposición histórica. Las métricas OOF evitan ajuste directo sobre "
        "el grupo evaluado en este experimento, pero no constituyen una reserva nueva. La "
        "selección exige cobertura completa y no empeorar MAE ni error relativo macro por "
        "grabación, tanto en carretera como en Slider-500, frente a IQ. No se abrió el benchmark "
        "para escoger los candidatos; al rechazarse todos, no se generaron nuevas entregas.",
        "",
        "## Controles entrenados",
        "",
        "IQ reproduce la fusión inversa QA anterior bajo los mismos folds. LG selecciona "
        "signo y mezcla magnitudes logarítmicas; LA añade perturbaciones de expertos. GR "
        "corrige la geometría por un factor entre 0,5 y 2 sobre la entrada limitada a ±60 s. "
        "La revisión detectó que el promedio entre semillas podía volver a cancelar signos, "
        "y que una decisión de signo discreta carecía de gradiente directo en algunos casos.",
        "",
        "SG añade supervisión de fiabilidad exclusivamente durante entrenamiento: coste "
        "esperado de cada experto, limitado a 10, ponderado por 0,5. SGA añade aumentos del "
        "canal geométrico. Ambos usan mediana de tres semillas y reaplican el retorno exacto "
        "a F0 cuando no existe historia válida. Se conservan predicciones OOF de cada semilla. "
        "Es una comparación de sistemas; cambia también la agregación respecto a IQ.",
        "",
        "| Campaña/control | RAE carretera macro (%) | MAE carretera macro (s) | "
        "RAE Slider-500 (%) | MAE Slider-500 (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for campaign, receipt in (("v1b", v1), ("v2", v2)):
        for mode, result in receipt["report"].items():
            road, slider = result["road"], result["slider"]
            lines.append(
                f"| {campaign}/{mode} | {road['macro_record_relative_error_pct']:.4f} "
                f"| {road['macro_record_mae_seconds']:.6f} "
                f"| {slider['macro_record_relative_error_pct']:.4f} "
                f"| {slider['macro_record_mae_seconds']:.6f} |"
            )
    lines += [
        "",
        f"Selección v1b: `{v1['selected']}`. Selección v2: `{v2['selected']}`.",
        "",
        "## Aumentos realmente ejecutados",
        "",
        "Se recomputaron 10.432 combinaciones consulta/variante: últimas dos cajas, últimas "
        "cuatro, jitter independiente de ±2 píxeles y pérdida del 50 % de cajas. La geometría "
        "limpia y las 20 variables de calidad reproducen exactamente la caché original en "
        "las 2.608 consultas antes de generar variantes. Cada perturbación conserva causalidad, "
        "historia máxima de 650 ms y la identidad original. Native y targets permanecen iguales.",
        "",
        "Este experimento perturba solamente las observaciones del estimador geométrico. "
        "No ejecuta de nuevo los productores sobre sensores perturbados ni reproduce un "
        "arranque en frío de toda la cadena. SGA no pasó los criterios: no se presenta "
        "dataset augmentation como una mejora demostrada.",
        "",
        "## Corrección del evaluador",
        "",
        "Los enlaces de ground truth de CCRs1-low y CCRs1-medium en la página oficial "
        "corresponden a los archivos que localmente estaban asignados al revés. Se guardaron "
        "URL, bytes y hashes; se mantuvieron los offsets demostrados por identidad RGB. "
        "La reasignación reproduce 7,5688559 % y 9,4237158 % para la entrega original y "
        "recupera las once etiquetas ausentes. Ahora hay 434 targets públicos alineados.",
        "",
        "[Fuente oficial de descargas](https://nail-hnu.github.io/EvTTC/download/download/). "
        "Los ocho resultados de carretera y Slider-1000 reproducen el PDF a su precisión "
        "publicada. Esto no prueba igualdad del vector privado de targets. Slider-750 sigue "
        "discrepando: el CSV público da 283,931058 % para el original procesado y el PDF "
        "132 %. No se ajustó ningún offset ni transformación para forzar ese resultado.",
        "",
        "| Población | Original procesado RAE (%) | Versión congelada actual RAE (%) |",
        "|---|---:|---:|",
    ]
    for name in (
        "all434_public_diagnostic",
        "road365",
        "pdf_reproduced_sequences383",
        "Slider-750",
        "Slider-1000",
    ):
        values = audit["frozen_baseline_metrics"][name]
        old = values["original_email_processed"]["relative_error_pct"]
        new = values["frozen_submission_email_processed"]["relative_error_pct"]
        lines.append(f"| {name} | {old:.6f} | {new:.6f} |")
    lines += [
        "",
        "La tabla reevalúa modelos ya existentes. No atribuye estas diferencias a "
        "los nuevos ajustes rechazados. El agregado público de 434 consultas no es el score "
        "oficial mientras Slider-750 no esté resuelto.",
        "",
        "## Conservación de eAP y Test12",
        "",
        "Se construyó historia causal por consulta y por objeto a partir de los metadatos "
        "P/H/V firmados: 88.744 ejemplos, 40 secuencias y 1.812 tracks. Todas las consultas "
        "activan la fusión; 88.597 tienen geometría válida. Así se comprueba la rama activa, "
        "en vez de repetir únicamente la identidad del fallback sin cajas.",
        "",
        "| eAP TRAIN40 | MAE (s) | Error relativo (%) |",
        "|---|---:|---:|",
    ]
    for name in ("F0", "current_active"):
        value = active[name]
        lines.append(
            f"| {name} | {value['mae_seconds']:.6f} | {value['relative_absolute_error_pct']:.6f} |"
        )
    lines += [
        "",
        "Estas son métricas de conservación/resustitución TRAIN40 con features "
        "upstream OOF; la cabeza original ya se ajustó con TRAIN40. No son generalización "
        "independiente. La futura promoción exige no empeorar la referencia adaptada activa.",
        "",
        "Test12 aporta 6.762 consultas lógicas, 5.805 físicas y 12 secuencias, con estados "
        "reales de despliegue de 340 dimensiones y las cajas causales. Se verificaron archivos, "
        "aliases, sellos y reproducción de F0; el error máximo de TTC al repetir la cabeza "
        "es 4,577637e-5 s, dentro de la tolerancia registrada, no igualdad bit a bit. No hay "
        "targets Test12 disponibles en las fuentes locales auditadas. Por eso esta caché "
        "permite medir deriva de predicciones, no afirmar precisión ni ausencia de regresión.",
        "",
        "## Próxima hipótesis y límites",
        "",
        "Se abre un experto físico de expansión de eventos con referencia al final de la "
        "ventana. Una caja actual puede delimitar eventos pasados; no permite por sí sola "
        "medir expansión de cajas. Primero se exige recuperación sintética y luego desarrollo "
        "en Slider-500. No equivale a reproducir el CMax del leaderboard: no se encontró su "
        "solver exacto. El código público STRTTC usa convenciones temporales/ROI diferentes.",
        "",
        "Ningún resultado de esta continuación demuestra SOTA. Sigue pendiente la "
        "confirmación del evaluador Slider-750 y una comparación equivalente. Se preservan "
        "los resultados negativos y la versión previa, sin sustituirla por una variante "
        "que solo mejore una métrica.",
        "",
        "## Reproducción",
        "",
        "```powershell",
        "$env:PYTHONPATH='src;.'",
        ".venv-transfer/Scripts/python.exe scripts/report_phase_safe_research.py",
        "```",
        "",
        "Los scripts train_phase_safe_fusion.py y train_phase_safe_supervised.py documentan "
        "sus argumentos con --help. Cada campaña conserva CONFIG, código ejecutado, hashes, "
        "checkpoints, pérdidas y grupos de ajuste. Use directorios nuevos para repetir; las "
        "campañas completadas no se sobreescriben.",
        "",
    ]
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    manifest = {name: file_sha(evidence / name) for name in receipts}
    (args.output / "evidence_manifest.json").write_text(
        json.dumps({"generator_sha256": file_sha(Path(__file__)), "receipts": manifest}, indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"REPORT {args.output}", flush=True)


if __name__ == "__main__":
    main()
