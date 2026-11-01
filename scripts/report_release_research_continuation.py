"""Regenerate the continued research findings from immutable experiment receipts."""

from __future__ import annotations

import argparse
from pathlib import Path

from operational.phase_robust.bundle import verified_upstream_paths
from operational.release_research.protocol import ResearchRun, sha256, write_json
from scripts.release_research import read, sources, verify_complete


def report(root: Path, output: Path) -> None:
    """Publish measured negative results and prerequisite failures without promoting R0."""
    folders = {
        "O0": root / "observability_analysis149_v1",
        "V1": root / "surface_analysis149_v1",
        "S1_readiness": root / "sensor_inventory_v1",
        "native_profile": root / "native_pair_replay_profile_v1",
    }
    inputs = {**sources(), "source:scripts/report_release_research_continuation.py": Path(__file__)}
    for name, folder in folders.items():
        verify_complete(folder)
        inputs[name + ":complete"] = folder / "COMPLETE.json"
        for filename in (
            "METRICS.json",
            "DECISION.json",
            "RESULTS.json",
            "ORIGINAL_PARITY.json",
            "FITS.json",
        ):
            if (folder / filename).exists():
                inputs[name + ":" + filename] = folder / filename
    paths = verified_upstream_paths(Path("output/phase_submission/FINAL_MODEL/upstream"))
    if sha256(paths["MODEL_SEAL.json"]) != read(root / "R0_IDENTITY.json")["seal_sha256"]:
        raise ValueError("Research reference differs from official R0")
    inputs.update({"R0:" + name: path for name, path in paths.items()})
    run = ResearchRun(
        output,
        inputs,
        {
            "experiment_id": "continued-research-report-v1",
            "scope": "exposed development and native original profile; no new release",
        },
    )
    try:
        o0 = read(folders["O0"] / "METRICS.json")
        v1 = read(folders["V1"] / "METRICS.json")
        sensor = read(folders["S1_readiness"] / "DECISION.json")
        native = read(folders["native_profile"] / "RESULTS.json")
        parity = read(folders["native_profile"] / "ORIGINAL_PARITY.json")
        fit_count = sum(len(v) for v in read(folders["O0"] / "FITS.json").values())
        for name, data in (
            ("OBSERVABILITY", o0),
            ("SURFACES", v1),
            ("SENSOR_READINESS", sensor),
            ("NATIVE_PROFILE", native),
        ):
            write_json(output / (name + ".json"), data)
        ttc_passes = sum(
            all(
                field["passed"]
                for key, field in row["fields"].items()
                if key.endswith("output_ttc")
            )
            for row in parity.values()
        )
        lines = [
            "# Investigación continuada del release",
            "",
            "Los contrastes nuevos están ejecutados. No acreditan una mejora del release;",
            "localizan fallos de cobertura, geometría y paridad que condicionan el siguiente paso.",
            "",
            "## Observabilidad RAW O0",
            "",
            f"{o0['required']} consultas; {o0['labeled_pairs']} pares evaluables.",
            "Las seis variables físicas se extrajeron antes de unir targets.",
            f"Se hicieron {fit_count} ajustes ridge con adquisición excluida y cuatro brazos.",
            "El escalado y la imputación se ajustan solo en entrenamiento.",
            "Los padres R0 están expuestos: no es OOF de la cadena ni confirmación independiente.",
            "",
            "| Sonda de utilidad de la corrección | MSE macro |",
            "|---|---:|",
        ]
        lines += [f"| {name} | {data['macro_mse']:.9f} |" for name, data in o0["arms"].items()]
        difference = o0["physical_minus_control_mse"]["support"]
        lines += [
            "",
            f"Delta físico−soporte: {difference['mean']:.9f}; "
            f"CI95 por adquisición {difference['ci95']}.",
            f"Admisión del gate por la regla prefijada: **{o0['gate_pilot_admitted']}**.",
            "MSE corresponde a predecir gain, no al error TTC del release.",
            "",
            "| Indicador físico | Pares con medición |",
            "|---|---:|",
        ]
        lines += [
            f"| {name} | {data['available']} |"
            for name, data in o0["correlations_physical"].items()
        ]
        full = v1["all_queries_with_R0_fallback"]
        lines += [
            "",
            "## Coordenadas y superficies V0/V1",
            "",
            f"Medidas RGB representables: {v1['surface_available']}/149.",
            f"MAE macro R0: **{full['baseline']['macro_mae']:.9f} s**;",
            f"superficie con fallback R0: **{full['candidate']['macro_mae']:.9f} s**.",
            f"CI95 del delta por adquisición: {full['paired_macro_delta_ci95']}.",
            f"Fallback en {v1['fallback_queries']} consultas; ausencia original conservada.",
            f"Cambios de crop dentro del triplete: {v1['within_triplet_crop_changes']};",
            f"entre snapshots: {v1['history_crop_changes']}; "
            f"resize anisótropo: {v1['anisotropic_resize']}.",
            f"Coordenadas auditables en {149 - v1['missing_coordinate_audit']}/149 consultas.",
            "El resize es constante en el triplete; no aparece zoom artificial variable.",
            "Esto no descarta errores de oclusión, fondo, paralaje o identidad de superficie.",
            "No se admite V2/V3: la correspondencia consistente por sí sola no garantiza TTC.",
            "",
            "## Preparación del auxiliar de sensor",
            "",
            f"Se inspeccionó la calibración de {sensor['acquisitions']} adquisiciones.",
            "| Familia de campos | Adquisiciones con candidatos por nombre |",
            "|---|---:|",
        ]
        lines += [
            f"| {name} | {count} |"
            for name, count in sensor["field_candidate_acquisitions"].items()
        ]
        lines += [
            "",
            "La calibración geométrica y exposición RGB no identifican la escala fotométrica.",
            "Faltan referencia anclada, escala identificada y separar el error de alineación.",
            "E_C2F expone mapas densos mediante una opción de forward; PAIR133 no los conserva.",
            "S1 real y S2/S3 requieren resolver ese dato; no se inventa una calibración.",
            "",
            "## Captura de despliegue y paridad",
            "",
            "Se capturaron dos observaciones de cada modalidad desde medios verificados.",
            "El replay local de PAIR/PHASE17 es exacto. La comparación con el archivo histórico da",
            f"**{native['historical_parity']}** en "
            f"{native['historical_failed_queries']}/{native['queries']} consultas.",
            "El fallo afecta a tensores intermedios. "
            f"TTC pasa en {ttc_passes}/{native['queries']}.",
            "No se ejecutó el candidato. El siguiente control fijado repite el original en CUDA",
            "con las mismas tolerancias cuando termine la población y F0 quede admitido.",
            "Se conservan dos intentos técnicos fallidos: clave del inventario y contrato antiguo",
            "de máscara; el intento válido usa NativeObservation, como el despliegue archivado.",
            "",
            "## Continuación y límites",
            "",
            "El worker population2608 sigue siendo la única extracción ampliada. La continuación",
            "espera su COMPLETE, verifica fuentes, evalúa F0 y publica el informe. Solo si F0 pasa",
            "ejecuta el perfil CUDA; solo si su paridad pasa abre la captura nativa completa y",
            "exige nuevamente paridad original antes de la intervención canónica.",
            "O1/O2 y V2/V3 no quedan admitidos por estos resultados. D4 y JEPA tampoco se abren",
            "sin sus evidencias específicas. Ningún componente o normalizador de R0",
            "se ha actualizado. La campaña no está completa: siguen pendientes cadencia ampliada,",
            "conservación nativa, confirmación independiente y latencia completa del candidato.",
            "",
            "## Regeneración",
            "",
            "```powershell",
            "python scripts/report_release_research_continuation.py --output DIRECTORIO_NUEVO",
            "```",
        ]
        (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    """Generate tables only from validated machine-readable results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/release_research_20261031"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.root, args.output)


if __name__ == "__main__":
    main()
