"""Regenerate event-expansion research evidence without promoting rejected candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_campaign.data import read
from operational.phase_campaign.evaluation import metrics
from operational.phase_submission.runtime import load_submission_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/phase_event_expansion"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/research_log_20261031_event_expansion")
    )
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    evidence = out / "evidence"
    evidence.mkdir(exist_ok=True)
    sources = {
        "energy_original": "synthetic/run_v1/SUMMARY.json",
        "energy_midpoint": "synthetic/midpoint_energy_development_v2/SUMMARY.json",
        "ncc_development": "synthetic/endpoint_ncc_development_v2/SUMMARY.json",
        "ncc_confirmation": "synthetic/endpoint_ncc_confirmation_v2/SUMMARY.json",
        "real_slider500": "slider500_v1b/REPORT.json",
        "objective_profile": "objective_profile/run_v1b/SUMMARY.json",
        "accelerator": "objective_profile/accelerated_v1/SUMMARY.json",
        "background_sensitivity": "background_v1/SUMMARY.json",
        "cuda_all77": "objective_profile/all77_cuda_v1/SUMMARY.json",
        "cuda_all77_context": "objective_profile/all77_cuda_v1/RUN_CONTEXT.json",
        "hybrid_all77": "objective_profile/all77_hybrid_v1/SUMMARY.json",
        "hybrid_all77_context": "objective_profile/all77_hybrid_v1/RUN_CONTEXT.json",
    }
    data = {name: read(args.root / relative) for name, relative in sources.items()}
    for name, relative in sources.items():
        (evidence / f"{name}.json").write_bytes(
            (args.root / relative).read_bytes().replace(b"\r\n", b"\n")
        )
    population_path = Path("output/phase_submission/training/POPULATION.npz")
    prediction_path = args.root / "slider500_v1b/PREDICTIONS.npz"
    seal = read(args.root / "slider500_v1b/PREDICTIONS_SEALED.json")
    if file_sha(prediction_path) != seal["sha256"]:
        raise ValueError("Changed sealed real predictions")
    if file_sha(population_path) != data["real_slider500"]["population_sha256"]:
        raise ValueError("Changed eligible development population")
    torch.set_num_threads(2)
    with np.load(population_path, allow_pickle=False) as p, np.load(prediction_path) as e:
        lookup = {str(token): i for i, token in enumerate(p["token"])}
        index = np.array([lookup[str(token)] for token in e["token"]])
        if not np.all(p["group"][index] == "Slider-500"):
            raise ValueError("Unexpected development group")
        native, geo, quality, target = [
            p[k][index] for k in ("native", "geometry", "quality", "target")
        ]
        pred = e["prediction"]
        with torch.inference_mode():
            current = load_submission_model(Path("output/phase_submission/FINAL_MODEL"))(
                torch.from_numpy(native), torch.from_numpy(geo), torch.from_numpy(quality)
            ).numpy()
    valid = np.isfinite(pred) & np.isfinite(geo)
    comparison = {
        "scope": "postfreeze Slider500 development; current adapter fitted Slider500",
        "current_adapted": metrics(target, current),
        "event_then_current": metrics(target, np.where(np.isfinite(pred), pred, current)),
        "event_then_geometry_then_F0": metrics(
            target, np.where(np.isfinite(pred), pred, np.where(np.isfinite(geo), geo, native[:, 0]))
        ),
        "shared": int(valid.sum()),
        "event_better_than_geometry": int(
            (valid & (np.abs(pred - target) < np.abs(geo - target))).sum()
        ),
        "first10_event_valid": int(np.isfinite(pred[:10]).sum()),
        "prediction_sha256": file_sha(prediction_path),
        "population_sha256": file_sha(population_path),
    }
    (evidence / "full_coverage_comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    lines = [
        "# PHASE-TTC: experto causal de expansión de eventos",
        "",
        "Informe regenerado desde ejecuciones del 31 de octubre de 2026. Este experto "
        "es una hipótesis física adicional; todavía no sustituye al modelo congelado.",
        "",
        "## Qué se implementó",
        "",
        "Un estimador original de tres parámetros q,bx,by sobre eventos causales dentro "
        "de una ROI actual. La transformación p_ref = p + (q p − b)(t_ref − t) refiere "
        "q=1/TTC al instante de consulta. El foco de expansión es libre; no se impone "
        "que coincida con el centro de la caja. Se conservan polaridades separadas. "
        "El solver usa una rejilla de 81 candidatos, tres inicios Powell y 240 evaluaciones "
        "por inicio, con límites y criterios de ausencia explícitos.",
        "",
        "Las referencias matemáticas son [contraste de eventos](https://arxiv.org/abs/1804.01306) "
        "y [STRTTC](https://arxiv.org/abs/2407.07324). No se copió código externo. "
        "No se afirma reproducir el CMax de EvTTC: no se encontró su solver exacto; "
        "la entrada pública STRTTC usa otra referencia temporal y preparación de ROI.",
        "",
        "## Pruebas y resultados negativos",
        "",
        "La energía del histograma deformado favoreció contracciones espurias en "
        "escenas estacionarias. Perfiles con traslación conocida y cambios de raster "
        "demostraron sesgo del objetivo; aumentar solamente el presupuesto no lo corregía. "
        "También se probó una referencia interna al punto medio, con conversión exacta "
        "al endpoint, y se descartó una compensación por Jacobiano tras su diagnóstico.",
        "",
        "| Receta | Casos que pasan | Casos ejecutados |",
        "|---|---:|---:|",
    ]
    for name in ("energy_original", "energy_midpoint", "ncc_development", "ncc_confirmation"):
        rows = data[name]["rows"]
        lines.append(
            f"| {name} | {sum(bool(r['assessment']['passed']) for r in rows)} | {len(rows)} |"
        )
    lines += [
        "",
        "La receta aceptada compara imágenes de las mitades temprana y tardía "
        "mediante correlación normalizada de media cero. Mantiene el warp al endpoint, "
        "el raster128², Gaussianσ0,8, los límites y el presupuesto. Añade correlación mínima "
        "0,8 como requisito conservador; no relaja las tolerancias de recuperación. "
        "La confirmación usa semillas31/47, que no se habían abierto al elegir esta receta.",
        "",
        "Son contornos sintéticos ideales, no un simulador fotométrico completo de cámara "
        "de eventos. Superar estas pruebas permite investigar datos reales, no demostrar SOTA.",
        "",
        "## Mezcla de movimientos: sensibilidad al fondo",
        "",
        "Una prueba posterior fija las semillas nuevas 61 y 73 y añade un contorno "
        "estático independiente, manteniendo exactamente los eventos originales del "
        "objeto y su TTC. Se prueban proporciones de fondo 0,25,50,75 % del total, "
        "con el mismo solver, límites y criterios de aceptación. La máscara sintética "
        "de origen se usa solamente para verificar el fixture; nunca entra al estimador.",
        "",
        "| Semilla | Fondo (%) | q verdadero (1/s) | q estimado (1/s) | Válido | Recupera |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in data["background_sensitivity"]["cases"]:
        lines.append(
            f"| {row['seed']} | {100 * row['background_fraction']:.0f} "
            f"| {row['truth']['q']:.6f} | {row['result']['q']:.6f} "
            f"| {row['result']['valid']} | {row['assessment']['passed']} |"
        )
    lines += [
        "",
        "Con 50 % de fondo, ambos casos producen una estimación finita aceptada pero "
        "incorrecta. Con 75 % el solver se abstiene. Esto demuestra una limitación "
        "del criterio de confianza frente a movimientos mezclados; no identifica "
        "la causa del error real de Slider ni justifica recortar su ROI usando targets. "
        "No se ajustó el solver tras observar estos ocho resultados.",
        "",
        "## Slider-500 real",
        "",
        "La caché conserva77consultas y889.637eventos, ventana causal de100ms recortada a "
        "la cobertura disponible, ROIactual ampliada1,2× y hasta20.000eventos por consulta. "
        "El primer instante tiene226eventos; el último conserva su consulta aunque la "
        "cobertura termine262µs antes. Se registran cámaras, reloj, selección y calibración. "
        "Se corrige distorsión radtan antes del solver. En los bordes se ampliaron las "
        "iteraciones de inversión de12hasta64 para cumplir la tolerancia ya fijada de "
        "0,001píxeles; no se modificaron coeficientes ni targets. El intento parcial previo "
        "de74consultas se conserva.",
        "",
        "Las77predicciones se sellaron antes de abrir los targets. Slider-500 pertenece "
        "a desarrollo y la cabeza adaptada de referencia ya se ajustó con esta grabación. "
        "No es una comparación independiente ni una puntuación del benchmark.",
        "",
        "| Estimador/población | Cobertura | RAE (%) | MAE (s) |",
        "|---|---:|---:|---:|",
    ]
    real = data["real_slider500"]
    results = {
        "Eventos, condicionado a disponibilidad": real["expert_conditional"],
        "Geometría en las mismas62consultas": real["geometry_on_expert_valid_rows"],
        "Adaptador actual,77consultas": comparison["current_adapted"],
        "Eventos disponibles; si no, adaptador actual": comparison["event_then_current"],
        "Eventos; si no, geometría; si no, F0": comparison["event_then_geometry_then_F0"],
    }
    for name, m in results.items():
        lines.append(
            f"| {name} | {m['finite']}/{m['population']} | "
            f"{m['relative_absolute_error_pct']:.6f} | {m['mae_seconds']:.6f} |"
        )
    lines += [
        "",
        f"El experto de eventos gana a la geometría en "
        f"{comparison['event_better_than_geometry']}/{comparison['shared']} consultas compartidas, "
        "pero sus errores extremos hacen que la sustitución directa no mejore el sistema "
        "completo. En las primeras10consultas se abstiene10veces: esta implementación "
        "todavía no resuelve el arranque. Un score condicionado no sustituye a cobertura completa.",
        "",
        f"CPU: media {real['timing']['mean_seconds']:.6f}s/consulta; "
        f"p95 {real['timing']['p95_seconds']:.6f}s. Incluye lectura de caché, "
        "desdistorsión, solver y recibo; excluye extracción de sensores crudos.",
        "",
        "## GPU, precarga y CUDA Graphs",
        "",
        "Se implementó un objetivo PyTorch separado con precarga de puntos, edades, "
        "polaridad y bins temporales; acumulación bilineal; y padding gaussiano que reproduce "
        "el reflect de SciPy. Se capturaron grafos reales para lotes de1y81candidatos. "
        "La caché del solver mantiene una sola forma de eventos para limitar VRAM.",
        "",
        "Float64 reproduce el objetivo de referencia. Float32 se rechaza: el redondeo "
        "cerca del borde cambia admisión/pérdida de eventos y viola la tolerancia de1e-6. "
        "No se presenta mixed precision como una optimización válida para este solver.",
        "",
        "La comparación integrada abarca18casos sintéticos y5consultas reales sin etiquetas. "
        "Validez y causa coinciden exactamente, y q/b están dentro de1e-5. Cinco casos "
        "sintéticos cambian el número de evaluaciones Powell: equivalencia numérica no "
        "significa reproducción bit a bit ni del camino del optimizador.",
        "",
        "Los recibos objective_profile.json y accelerator.json contienen tiempos, memoria, "
        "capturas y replays medidos. El benchmark mide proceso CUDA inicializado y separa "
        "captura de grafos y reutilización; no confundirlo con arranque frío de la aplicación.",
        "",
        "La ampliación posterior a las 77 consultas falló la puerta estricta: "
        f"{sum(bool(r['parity']['passed']) for r in data['cuda_all77']['rows'])}/77 "
        "cumplen la tolerancia q/b de 1e-5. Las decisiones de validez y rechazo "
        "coinciden en las 77. Slider-500:15 es rechazado por ambos solvers como "
        "flat_objective, pero sus parámetros internos difieren más de lo permitido. "
        "Se conserva el resultado fallido; no se amplía retrospectivamente la tolerancia.",
        "",
        "Esta pasada tardó "
        f"{data['cuda_all77']['timing']['complete_wall_before_summary_seconds']:.3f} "
        f"s y reservó como máximo {data['cuda_all77']['peak_reserved_bytes'] / 1e6:.3f} MB. "
        "Hubo comprobaciones CPU breves simultáneas al principio: estos tiempos son "
        "observaciones de ejecución, no un benchmark aislado ni una prueba de aceleración "
        "del sistema PHASE completo. La comparación no abrió targets.",
        "",
        "Se registró después una política híbrida nueva: recalcular en CPU toda "
        "abstención de GPU y conservar su resultado exacto, sin consultar etiquetas. "
        "La comprobación exige además TTC finito coherente con 1/q para resultados "
        "válidos, ausencia de TTC en rechazos y tolerancia en segundos derivada de "
        "la diferencia de q. No impone una tolerancia absoluta en segundos ajena "
        "a la coordenada original. El fallo previo permanece conservado.",
        "",
        f"La pasada híbrida cumple "
        f"{sum(bool(r['parity']['passed']) for r in data['hybrid_all77']['rows'])}/77 "
        f"comparaciones con {data['hybrid_all77']['fallback_count']} retornos a CPU. "
        "RUN_CONTEXT registra la coordinación para evitar otros benchmarks de esta "
        "tarea, aunque no descarta actividad de fondo del sistema. No se ejecutó "
        "una referencia CPU contemporánea para medir una aceleración controlada.",
        "",
        "En esta segunda pasada también pasaron todas las estimaciones GPU antes "
        "del retorno a CPU: el caso fallido anterior muestra variación numérica "
        "del camino de Powell, no un fallo determinista que haya desaparecido. "
        "Estas pruebas no garantizan paridad universal ni ejecución bit a bit. "
        "La variante sigue siendo una herramienta experimental separada.",
        "",
        "## Decisión",
        "",
        "No se promueve una nueva entrega. Se conserva el experto como candidato a aportar "
        "información complementaria y se continúa investigando arranque y errores correlacionados. "
        "La precisión de eAP no se altera por añadir este módulo experimental separado. "
        "Cualquier integración requiere un nuevo protocolo, ajuste por grupos, control "
        "de conservación de eAP y evaluación con cobertura completa.",
        "",
        "## Reproducir este informe",
        "",
        "```powershell",
        "$env:PYTHONPATH='src;.'",
        ".venv-transfer/Scripts/python.exe scripts/report_phase_event_expansion.py",
        "```",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    manifest = {p.name: file_sha(p) for p in sorted(evidence.glob("*.json"))}
    (out / "evidence_manifest.json").write_text(
        json.dumps({"generator_sha256": file_sha(Path(__file__)), "receipts": manifest}, indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"REPORT {out}")


if __name__ == "__main__":
    main()
