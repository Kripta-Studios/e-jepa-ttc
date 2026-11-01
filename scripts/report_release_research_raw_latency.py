"""Publish the RAW latency boundary as a verified supplement to the campaign report."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from operational.release_research.protocol import ResearchRun
from scripts.release_research import read, verify_complete


def report(campaign: Path, parity: Path, latency: Path, output: Path) -> None:
    """Keep measured RAW service separate from precomputed-NPZ service."""
    for folder in (campaign, parity, latency):
        verify_complete(folder)
    raw = read(latency / "RESULTS.json")
    npz = read(campaign / "LATENCY.json")
    admission = read(parity / "RESULTS.json")
    if not raw["all_predictions_exact"] or not admission["all_exact"]:
        raise ValueError("Exact RAW preparation and prediction required")
    documents = {
        "RAW_LATENCY.json": latency / "RESULTS.json",
        "RAW_MEASUREMENTS.json": latency / "MEASUREMENTS.json",
        "RAW_PARITY.json": parity / "RESULTS.json",
        "RAW_PARITY_RECORDS.json": parity / "RECORDS.json",
        "PRECOMPUTED_LATENCY.json": campaign / "LATENCY.json",
    }
    run = ResearchRun(
        output,
        {
            **documents,
            "campaign": campaign / "COMPLETE.json",
            "parity": parity / "COMPLETE.json",
            "latency": latency / "COMPLETE.json",
            "source:scripts/report_release_research_raw_latency.py": Path(__file__),
        },
        {
            "experiment_id": "raw-latency-report-v1",
            "official_release_modified": False,
            "scope": "exposed data; no independent confirmation or official accuracy claim",
        },
    )
    try:
        for name, path in documents.items():
            shutil.copyfile(path, output / name)
        lines = [
            "# Latencia desde eventos RAW y cierre de la evidencia disponible",
            "",
            "Este suplemento completa la frontera de latencia del informe de campaña.",
            "El ensayo anterior lee representaciones de eventos precalculadas en NPZ;",
            "no incluye su construcción. El ensayo RAW sí incluye lectura HDF5, recorte,",
            "voxelización, transferencia y todos los productores, contextos y fusión,",
            "junto con la lectura y decodificación RGB. Las cajas vienen suministradas.",
            "No mide detección, adquisición física ni transporte del sensor.",
            "",
            f"Paridad: {admission['observations']} observaciones reconstruidas exactamente;",
            f"{admission['queries']} consultas físicas seleccionadas por prefijo temporal.",
            "Todas las predicciones cronometradas coinciden exactamente con el replay",
            "local completo. Se reinician lectores y cachés por secuencia y brazo; el",
            "caché del sistema operativo puede estar caliente. El orden de brazos alterna.",
            "",
            "| Frontera | Brazo | Consultas | p50 ms | p95 ms | p99 ms |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for name, result in (("NPZ preparado + RGB", npz), ("RAW + RGB", raw)):
            for arm, values in result["service"].items():
                p50, p95, p99 = values["p50_p95_p99_ms"]
                lines.append(
                    f"| {name} | {arm} | {values['queries']} | {p50:.3f} | {p95:.3f} | {p99:.3f} |"
                )
        ratio = raw["candidate_over_original_p95"]
        lines += [
            "",
            f"Cociente p95 canónico/original desde RAW: **{ratio:.6f}**.",
            f"Cumple el presupuesto descriptivo de +10%: **{ratio <= 1.1}**.",
            "Son mediciones separadas: no se suman percentiles ni se atribuye su",
            "diferencia al preprocesado aislado. No hay garantía de servicio a 50 Hz.",
            "",
            f"GPU: {raw['gpu']}. Máximo CUDA asignado: {raw['cuda_peak_allocated_bytes']} bytes;",
            f"reservado: {raw['cuda_peak_reserved_bytes']} bytes; RAM máxima del proceso:",
            f"{raw['process_peak_rss_bytes']} bytes. "
            "Las colas por secuencia están en RAW_LATENCY.json.",
            "",
            "## Qué permite concluir",
            "",
            "La cadencia tiene una señal favorable de desarrollo en Slider; el cambio",
            "de carretera es pequeño y su intervalo incluye cero. La comparación",
            "nativa completa mide sensibilidad local; no demuestra conservación del",
            "score oficial porque la paridad histórica intermedia falló y no se",
            "leyeron etiquetas privadas. El release oficial sigue intacto.",
            "",
            "Los controles de observabilidad, reloj, sensor y superficies están",
            "ejecutados con los datos disponibles. Sus resultados no admiten los",
            "entrenamientos condicionales O2/D4/S2/V2 ni JEPA definidos en el dossier.",
            "Eso no prueba que cualquier variante futura sea incapaz de mejorar.",
            "",
            "El usuario confirma que no existen adquisiciones reservadas ni calibración",
            "fotométrica adicional. Repartir de nuevo los mismos datos no crea una",
            "confirmación independiente. Quedan sin demostrar la generalización de",
            "la ganancia y la no inferioridad del score nativo; no se promociona el candidato.",
            "",
        ]
        (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--latency", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.campaign, args.parity, args.latency, args.output)


if __name__ == "__main__":
    main()
