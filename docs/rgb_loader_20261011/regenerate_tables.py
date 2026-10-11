"""Verify published JSON and regenerate TABLES.md using only Python stdlib."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    cpu = json.loads((root / "CPU_ABBA.json").read_text(encoding="utf-8"))
    guard = json.loads((root / "GUARD_EXPERIMENT.json").read_text(encoding="utf-8"))
    qa = json.loads((root / "QA.json").read_text(encoding="utf-8"))
    trials = cpu["trials"]
    assert [item["name"] for item in trials] == ["A1", "B1", "B2", "A2"]
    assert all(item["batch_sha256"] == trials[0]["batch_sha256"] for item in trials)
    assert all(item["rng_before"] == item["rng_after"] for item in trials)
    assert all(
        item["source_identity_record_sha256"] == trials[0]["source_identity_record_sha256"]
        for item in trials
    )
    assert all(
        item["batch_details_record_sha256"] == trials[0]["batch_details_record_sha256"]
        for item in trials
    )
    assert all(
        abs(sum(item["batch_source_seconds"]) - item["source_seconds"]) < 1e-8 for item in trials
    )
    assert cpu["checkpoint_pointer_unchanged"] and not cpu["cuda_initialized"]
    assert cpu["optimizer_updates"] == 0
    assert qa["tests_passed"] == len(qa["test_cases"]) == 68
    assert qa["status"] == qa["ruff"] == "PASSED"
    assert qa["pyright_errors"] == qa["pyright_warnings"] == 0
    baseline = statistics.mean(item["source_seconds"] for item in trials if not item["optimized"])
    optimized = statistics.mean(item["source_seconds"] for item in trials if item["optimized"])
    reduction = 100 * (1 - optimized / baseline)
    assert abs(reduction - cpu["source_time_reduction_percent"]) < 1e-8
    lines = [
        "# Tablas regeneradas: optimización RGB, 11 de octubre de 2026",
        "",
        "Generadas desde los JSON públicos por `regenerate_tables.py`. "
        "Unidades de tiempo: segundos.",
        "",
        "## Fuente CPU: comparación ABBA",
        "",
        "| Ensayo | Ruta | Ejemplos | Tiempo fuente | Lecturas DINO | Lecturas RGB |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for trial in trials:
        lines.append(
            f"| {trial['name']} | {'Optimizada' if trial['optimized'] else 'Original'} "
            f"| {trial['rows']} | {trial['source_seconds']:.4f} "
            f"| {trial['np_load_calls']['dino']} | {trial['np_load_calls']['rgb']} |"
        )
    lines += [
        "",
        f"Media original: **{baseline:.4f} s**. Media optimizada: **{optimized:.4f} s**. "
        f"Reducción: **{reduction:.2f} %**.",
        "",
        "Paridad exacta de todos los campos en los cuatro ensayos; estados aleatorios conservados.",
        "",
        "## Coste del control completo de recursos",
        "",
        "| Ronda | Variante | Tiempo | Decisión permite continuar | Procesos del proyecto |",
        "|---|---|---:|---|---:|",
    ]
    for item in guard["samples"]:
        lines.append(
            f"| {item['round']} | {item['variant']} | {item['elapsed_s']:.4f} "
            f"| {item['allowed']} | {item['project_process_count']} |"
        )
    original_guard = statistics.median(
        item["elapsed_s"] for item in guard["samples"] if item["variant"] == "original"
    )
    filtered_guard = statistics.median(
        item["elapsed_s"] for item in guard["samples"] if item["variant"] == "filtered"
    )
    lines += [
        "",
        f"Medianas: **{original_guard:.4f} → {filtered_guard:.4f} s** por llamada; "
        f"cociente **{original_guard / filtered_guard:.2f}×**.",
        "",
        "Las parejas de cada ronda identificaron los mismos procesos; "
        "hubo cambios de procesos entre rondas. Las seis decisiones observadas coincidieron.",
        "",
        "## QA",
        "",
        "| Comprobación | Resultado |",
        "|---|---|",
        f"| Pytest | {qa['tests_passed']} aprobadas |",
        f"| Ruff | {qa['ruff']} |",
        f"| Pyright | {qa['pyright_errors']} errores, {qa['pyright_warnings']} advertencias |",
        f"| Updates de optimizador de la intervención CPU | {cpu['optimizer_updates']} |",
        "",
        "Estos tiempos son de componentes CPU. "
        "No sustituyen una medición del entrenamiento completo tras reanudar.",
        "",
    ]
    (root / "TABLES.md").write_text("\n".join(lines), encoding="utf-8")
    print("Verified published evidence and regenerated TABLES.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
