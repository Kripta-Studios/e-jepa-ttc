"""Publish the completed mechanism/localization followups from sealed receipts."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from operational.release_research.protocol import ResearchRun, write_json
from scripts.release_research import sources, verify_complete


def report(root: Path, output: Path) -> None:
    """Copy audited results verbatim and retain machine-readable evidence for each table."""
    experiments = {
        "cadence_localization": "cadence_localization_analysis149_v1",
        "observability_failure": "observability_failure_audit_v1",
        "robust_measurement": "robust_measurement_analysis149_v1",
        "sensor_identifiability": "sensor_identifiability_v2",
        "clock_regimes": "clock_regimes_v1",
        "native_original_drift": "native_original_drift_profile_v1",
    }
    inputs = {**sources(), "source:scripts/report_release_research_followup.py": Path(__file__)}
    for name, experiment in experiments.items():
        folder = root / experiment
        verify_complete(folder)
        for filename in ("COMPLETE.json", "FREEZE.json", "METRICS.json", "RESULTS.md"):
            inputs[name + ":" + filename] = folder / filename
    run = ResearchRun(
        output,
        inputs,
        {
            "experiment_id": "release-research-localization-report-v1",
            "scope": "exposed development and fixtures; no promotion or independent confirmation",
            "optimizer_updates": 0,
        },
    )
    try:
        lines = [
            "# Seguimiento experimental: localización y controles",
            "",
            "Estos seis contrastes están terminados y sellados. Las tablas se copian de sus",
            "resultados regenerables; no se redactan métricas a mano.",
            "",
            "La corrección de cadencia tiene una señal favorable en el desarrollo de 149",
            "consultas. Los controles de observabilidad y lectura robusta delimitan por qué",
            "sus mediciones actuales no justifican un gate mayor. Los bancos de reloj y",
            "sensor comprueban mecanismos y limitaciones, no precisión del release.",
            "",
            "La ampliación de 2608 y la conservación nativa son etapas distintas. Este",
            "informe no las declara cerradas ni usa estos contrastes para cambiar su candidato.",
            "",
            "## Índice de evidencia",
            "",
        ]
        for name, experiment in experiments.items():
            destination = output / name
            destination.mkdir()
            for filename in ("COMPLETE.json", "FREEZE.json", "METRICS.json", "RESULTS.md"):
                shutil.copyfile(root / experiment / filename, destination / filename)
            lines.append(f"- [{name}]({name}/RESULTS.md); origen `{root / experiment}`.")
        lines += [
            "",
            "## Qué permiten decidir",
            "",
            "- Cadencia: atribución separada a eventos/RGB; candidato ampliado intacto.",
            "- Observabilidad: estimabilidad algebraica y residuo bajo no bastan; "
            "la lectura robusta histórica tampoco recupera una mejora en esta cohorte.",
            "- Sensor: resolver referencia/escala antes de ajustar el productor; "
            "los eventos idénticos del banco delimitan un problema de identificación.",
            "- Reloj: medir error y retraso ante maniobras; la ley constante no predice "
            "cambios futuros arbitrarios.",
            "- Nativo: la deriva ya existe en el original, antes del candidato; "
            "se conserva la comprobación CUDA y las tolerancias fijadas.",
            "",
            "No se reabren D4, O1/O2, S2/S3, V2/V3 ni JEPA mediante estos fixtures.",
            "S1 real sigue necesitando referencia anclada y escala identificada.",
            "",
            "## Reproducción",
            "",
            "```powershell",
            "python scripts/report_release_research_followup.py --output DIRECTORIO_NUEVO",
            "```",
            "",
            "Las copias COMPLETE de cada experimento se refieren a su carpeta original:",
            "sus artefactos completos, incluidas trazas y fuentes ejecutadas, permanecen allí.",
            "El COMPLETE de este informe autentica solo sus propias copias y fuentes.",
        ]
        (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(output / "EXPERIMENTS.json", {k: str(root / v) for k, v in experiments.items()})
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    """Regenerate the report into a fresh destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/release_research_20261031"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.root, args.output)


if __name__ == "__main__":
    main()
