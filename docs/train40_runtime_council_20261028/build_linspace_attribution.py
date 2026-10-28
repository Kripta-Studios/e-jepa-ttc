"""Regenerate the bounded linspace attribution note using only its JSON receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "artifacts/train40_runtime_council_20261028/compiled_pipeline/"
    "linspace_attribution_v1/summary.json"
)
OUTPUT = Path(__file__).with_name("LINSPACE_ATTRIBUTION.md")
CANDIDATE = "native_weighted_mean_and_linspace"


def different_elements(comparison: dict[str, Any]) -> int:
    """Count individual divergent scalars over both directions and all features."""
    return sum(
        int(feature["different_elements"])
        for output in comparison["outputs"]
        for feature in output["coordinates"]
    )


def render(source: Path) -> str:
    """Render the successful bounded experiment without admitting downstream TTC."""
    raw = source.read_bytes()
    report = json.loads(raw)
    rows = report["records"]
    required_true = (
        "diagnostic_gate_passed",
        "all_controls_reproduced_exactly",
        "all_candidates_exact",
        "standalone_opaque_exact",
    )
    required_false = (
        "frozen_model_checkpoints_loaded",
        "normalizers_loaded",
        "raw_sensors_loaded",
        "final_ttc_admitted",
        "complete_update_admitted",
        "latency_measured",
    )
    if (
        report["status"] != "COMPLETE"
        or report["scope"] != "BOUND_TRANSPORT_SUMMARY_LINSPACE_ONLY_DIAGNOSTIC"
        or len(rows) != 12
        or any(report[key] is not True for key in required_true)
        or any(report[key] is not False for key in required_false)
        or report["preservation"]["passed"] is not True
        or any(row["status"] != "COMPARED" for row in rows)
    ):
        raise ValueError("The source is not the complete preserved bounded linspace experiment")
    for row in rows:
        if (
            not row["variants"][CANDIDATE]["exact"]
            or different_elements(row["variants"][CANDIDATE]) != 0
            or any(
                not row["control_reproduction"][name]["exact"]
                for name in ("original", "native_weighted_mean")
            )
        ):
            raise ValueError("Individual candidate or control results contradict the gate")
    binding = report["binding"]
    intervention = binding["private_functions"]
    if intervention["replacements"] != 2 or intervention["scientific_globals_modified"]:
        raise ValueError("The intervention differs from the two private coordinate callsites")
    source_link = "../../" + source.relative_to(ROOT).as_posix()
    totals = sum(int(row["input_snapshot"]["tensors"]["previous"]["shape"][0]) * 36 for row in rows)
    differences = sum(different_elements(row["variants"]["native_weighted_mean"]) for row in rows)
    old_exact = sum(bool(row["variants"]["native_weighted_mean"]["exact"]) for row in rows)
    indices = ", ".join(str(value) for value in sorted({row["index"] for row in rows}))
    lines = [
        "# Atribución del residuo numérico de linspace",
        "",
        f"Resultado regenerado desde [el recibo JSON]({source_link}): **{len(rows)}/{len(rows)} "
        f"casos exactos**, {totals} escalares comparados en ambos sentidos y 18 features por par. "
        f"Los controles original y `native_weighted_mean` reprodujeron exactamente sus salidas "
        f"archivadas; se preservaron {report['preservation']['files']} archivos.",
        "",
        f"Consultas {indices}; productores E_A5_MATCHED, E_C2F_MATCHED, R_A5 y R_C2F, "
        "con escalas fine/coarse donde corresponden. Entradas congeladas, con sus shapes, "
        "strides y dtypes originales. No se cargaron RAW, checkpoints ni normalizadores.",
        "",
        f"El control anterior daba {old_exact}/{len(rows)} casos exactos y {differences} "
        "escalares diferentes, exclusivamente en divergencia x/y/isotrópica global y de "
        "foreground. El candidato mantuvo matching original y las fronteras opacas "
        "anteriores de muestreo inverso y media ponderada. La única intervención adicional "
        "sustituyó las dos construcciones x/y de `linspace(-0.5, 0.5, ...)` en una copia "
        "privada de `_summarise_one` por el operador ATen opaco, conservando dtype y device. "
        "Un guard AST y hashes vinculan esos dos callsites; no se alteraron globals científicos.",
        "",
        "| Escala | Casos | Control anterior exacto | Candidato exacto | "
        "Diferencias antes → después |",
        "|---|---:|---:|---:|---:|",
    ]
    for scale in ("fine", "coarse"):
        group = [row for row in rows if row["scale"] == scale]
        old = sum(bool(row["variants"]["native_weighted_mean"]["exact"]) for row in group)
        delta = sum(different_elements(row["variants"]["native_weighted_mean"]) for row in group)
        lines.append(f"| {scale} | {len(group)} | {old} | {len(group)} | {delta} → 0 |")
    lines.extend(
        [
            "",
            "El ensayo aislado de coordenadas CUDA comparó ATen original, linspace compilado "
            "y ATen opaco compilado:",
            "",
            "| Tamaño | Índices distintos en compilado | Máximo error absoluto | "
            "ATen opaco exacto |",
            "|---:|---|---:|---|",
        ]
    )
    for row in report["standalone_coordinates"]:
        compiled, opaque = row["variants"]["compiled"], row["variants"]["opaque"]
        changed = ", ".join(str(index) for index in compiled["different_indices"]) or "ninguno"
        lines.append(
            f"| {row['steps']} | {changed} | {compiled['max_abs']:.17g} | "
            f"{'sí' if opaque['exact'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Esta intervención atribuye el residuo coarse observado a la construcción compilada "
            "de coordenadas, condicionada a las fronteras anteriores ya preservadas. El patrón "
            "16/32 concuerda entre el ensayo aislado y los resúmenes completos. No prueba que "
            "linspace sea la única fuente de discrepancia de un pipeline totalmente compilado; "
            "el mecanismo concreto de redondeo o instrucciones de máquina no se aisló aquí.",
            "",
            "La cobertura se limita a estos tensores y a este entorno. No hay admisión de TTC "
            "final, modelos completos, otra población, 50 Hz ni latencia por actualización. "
            "Una integración futura requiere paridad y medición completas; el coste de las "
            "fronteras opacas tampoco se midió en este diagnóstico.",
            "",
            f"Entorno: commit `{binding['git_commit']}`, PyTorch `{binding['torch']}`, CUDA "
            f"`{binding['cuda']}`, {binding['gpu']}, `{binding['precision']}`, "
            f"seed {binding['seed']}. "
            f"Ejecución registrada: {report['start_time']} → {report['end_time']}.",
            "",
            f"SHA-256 del JSON fuente: `{hashlib.sha256(raw).hexdigest()}`. "
            "Opciones de compilación: "
            f"`{json.dumps(binding['compiler_options'], sort_keys=True)}`.",
            "",
            "Regeneración desde la raíz del repositorio (sólo JSON, sin GPU):",
            "",
            "```powershell",
            "python docs/train40_runtime_council_20261028/build_linspace_attribution.py",
            "python docs/train40_runtime_council_20261028/build_linspace_attribution.py --check",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Regenerate the note, or fail if the committed note differs from its receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render(SOURCE)
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != rendered:
            raise SystemExit("LINSPACE_ATTRIBUTION.md differs from its source JSON")
        print("LINSPACE_ATTRIBUTION.md matches its source JSON")
    else:
        OUTPUT.write_text(rendered, encoding="utf-8", newline="\n")
        print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
