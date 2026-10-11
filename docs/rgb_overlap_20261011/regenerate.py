"""Export compact allowlisted evidence and regenerate tables using only stdlib."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
GPU_FIELDS = (
    "status optimizer_updates device fit_id completed_updates checkpoint_sha256 "
    "recipe_identity_sha256 precision strict_exact_required covers_mixed_t2_t3 "
    "protected_files_unchanged baseline_repeat_exact candidate_trials_exact "
    "exact_gpu_gradient_parity median_seconds_per_batch speedup_ratio "
    "cuda_peak_allocated_bytes cuda_peak_reserved_bytes"
)


def load_record(root: Path, relative: str) -> tuple[dict[str, Any], str | None]:
    """Load one optional completed/partial record without exposing its local path."""
    file = root / relative
    if not file.is_file():
        return {}, None
    raw = file.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def select(value: dict[str, Any], names: str) -> dict[str, Any]:
    """Use an explicit allowlist, never recursively copy arbitrary raw metadata."""
    return {key: value[key] for key in names.split() if key in value}


def export(root: Path) -> dict[str, Any]:
    """Strip dataset paths, input IDs, individual tokens and machine identifiers."""
    cpu, cpu_sha = load_record(root, "CPU_DECODER/RESULT.json")
    gpu, gpu_sha = load_record(root, "COMPUTE_GPU_ABBA.json")
    isolated, isolated_sha = load_record(root, "COMPUTE_GPU_ISOLATED.json")
    extra, extra_sha = load_record(root, "CPU_DECODER/EXTRA_PARITY_RESULT.json")
    incomplete, incomplete_sha = load_record(
        root, "COMPUTE_GPU_ABBA_INCOMPLETE_snapshot_stride.json"
    )
    cpu_public = select(
        cpu,
        "status optimizer_updates cuda_initialized torch_threads origin_sha256 "
        "origin_unchanged origin_completed_updates sample_count exact_batch_parity "
        "cursor_rng_unchanged code_sha256",
    )
    cpu_public["raw_sha256"] = cpu_sha
    cpu_public["trials"] = [
        {
            **select(item, "name workers prefetch source_seconds batch_seconds np_load_calls"),
            "batch_digest_count": len(item.get("batch_sha256", [])),
            "batch_digest_list_sha256": hashlib.sha256(
                json.dumps(item.get("batch_sha256", []), separators=(",", ":")).encode()
            ).hexdigest(),
            "cursor_only_advanced_by_consumer": item.get("cursor_only_advanced_by_consumer"),
            "decoder": select(
                item.get("stats", {}).get("decoder") or {},
                "workers max_pending_bytes peak_pending_bytes peak_pending_reads parallel_reads "
                "fallback_reads hits decode_worker_seconds consumer_wait_seconds "
                "largest_decoded_shard_bytes rgb_cache_bytes",
            ),
        }
        for item in cpu.get("trials", [])
    ]
    gpu_public = select(gpu, GPU_FIELDS)
    gpu_public["raw_sha256"] = gpu_sha
    gpu_public["trials"] = [
        {
            "name": item["name"],
            "batch_seconds": [batch["seconds"] for batch in item.get("batches", [])],
        }
        for item in gpu.get("trials", [])
    ]
    extra_public = select(
        extra,
        "status exact_batch_parity cursor_rng_unchanged optimizer_updates cuda_initialized "
        "origin_sha256 origin_unchanged baseline_result_sha256 baseline_result_unchanged "
        "torch_threads workers prefetch_depth code_sha256 probe_sha256",
    )
    extra_public["raw_sha256"] = extra_sha
    extra_public["trial"] = select(extra.get("trial", {}), "name workers prefetch source_seconds")
    extra_public["decoder"] = select(
        extra.get("trial", {}).get("stats", {}).get("decoder") or {},
        "workers max_pending_bytes peak_pending_bytes peak_pending_reads",
    )
    isolated_public = select(isolated, GPU_FIELDS)
    isolated_public["raw_sha256"] = isolated_sha
    isolated_public["trials"] = [
        {
            "name": item["name"],
            "batch_seconds": [batch["seconds"] for batch in item.get("batches", [])],
        }
        for item in isolated.get("trials", [])
    ]
    qa = {}
    unique_cases = {}
    junit_records = []
    for junit in sorted(root.glob("*JUNIT*.xml")):
        raw = junit.read_bytes()
        suites = ET.fromstring(raw)
        for case in suites.iter("testcase"):
            unique_cases[(case.get("classname"), case.get("name"))] = case
        junit_records.append({"name": junit.name, "sha256": hashlib.sha256(raw).hexdigest()})
    if unique_cases:
        cases = list(unique_cases.values())
        qa = {
            "junit_records": junit_records,
            "total": len(cases),
            "new_overlap": sum("rgb_overlap" in case.get("classname", "") for case in cases),
            "failures": sum(len(case.findall("failure")) for case in cases),
            "errors": sum(len(case.findall("error")) for case in cases),
            "skipped": sum(len(case.findall("skipped")) for case in cases),
        }
    return {
        "schema": "rgb_overlap_public_component_evidence_v1",
        "qa": qa,
        "cpu": cpu_public,
        "gpu": gpu_public,
        "gpu_isolated": isolated_public,
        "extra_decoder4": extra_public,
        "incomplete_gpu_probe": {
            "present": bool(incomplete),
            "raw_sha256": incomplete_sha,
            "reason": "snapshot_instrumentation_singleton_stride",
            "optimizer_updates": incomplete.get("optimizer_updates"),
        },
        "whole_training_speedup_demonstrated_by_this_evidence": False,
    }


def number(value: float | int | None, digits: int = 3) -> str:
    return "pendiente" if value is None else f"{value:.{digits}f}"


def render(value: dict[str, Any]) -> str:
    """Render measured tables; partial records remain explicitly pending."""
    qa, cpu, gpu = value["qa"], value["cpu"], value["gpu"]
    lines = [
        "# Evidencia de componentes: overlap RGB",
        "",
        "Generado por `regenerate.py` desde `EVIDENCE.json`. No es un monitor en vivo.",
        "",
        "## Regresiones CPU",
        "",
    ]
    if qa:
        lines.extend(
            [
                "| Casos | Nuevos overlap | Fallos | Errores | Omitidos |",
                "|---:|---:|---:|---:|---:|",
                f"| {qa['total']} | {qa['new_overlap']} | {qa['failures']} | "
                f"{qa['errors']} | {qa['skipped']} |",
            ]
        )
    else:
        lines.append("JUnit pendiente de exportar.")
    lines.extend(["", "## Preparación de entradas CPU", ""])
    lines.append(f"Estado del archivo completo: **{cpu.get('status', 'PENDIENTE')}**.")
    if cpu.get("status") == "COMPLETE":
        trials = cpu["trials"]
        references = [item["source_seconds"] for item in trials if item["workers"] == 0]
        baseline = statistics.mean(references)
        lines.extend(
            [
                "",
                f"Muestras por caso: {cpu['sample_count']}. "
                f"Comparación desde update {cpu['origin_completed_updates']}.",
                "",
                "| Caso | Hilos decode | Prefetch | Espera acumulada (s) | "
                "Variación frente a media serial |",
                "|---|---:|---|---:|---:|",
            ]
        )
        for item in trials:
            if item["prefetch"]:
                continue
            delta = 100 * (item["source_seconds"] / baseline - 1)
            lines.append(
                f"| {item['name']} | {item['workers']} | {item['prefetch']} | "
                f"{item['source_seconds']:.3f} | {delta:+.1f}% |"
            )
        lines.extend(
            [
                "",
                f"Paridad exacta de batch: **{cpu['exact_batch_parity']}**. "
                f"Cursor/RNG preservados: **{cpu['cursor_rng_unchanged']}**. "
                f"Origen sin modificar: **{cpu['origin_unchanged']}**.",
            ]
        )
        lines.extend(
            [
                "",
                "### Pruebas de paridad con anticipación",
                "",
                "| Caso | Espera acumulada (s) |",
                "|---|---:|",
            ]
        )
        for item in trials:
            if item["prefetch"]:
                lines.append(f"| {item['name']} | {item['source_seconds']:.3f} |")
        extra = value.get("extra_decoder4", {})
        if extra.get("status") == "COMPLETE":
            item = extra["trial"]
            lines.append(f"| {item['name']} (prueba posterior) | {item['source_seconds']:.3f} |")
            lines.extend(
                [
                    "",
                    f"Paridad adicional con cuatro lectores: **{extra['exact_batch_parity']}**. "
                    f"Cursor/RNG: **{extra['cursor_rng_unchanged']}**.",
                    "Reserva pendiente máxima observada: "
                    f"**{extra['decoder']['peak_pending_bytes']} bytes**, "
                    f"límite **{extra['decoder']['max_pending_bytes']} bytes**.",
                ]
            )
    lines.extend(
        [
            "",
            "Dos referencias seriales; una ejecución por variante. Caché del sistema sin vaciar. "
            "Con prefetch, el cálculo de hashes puede solaparse con lecturas. "
            "Estos tiempos de espera no prueban aceleración del entrenamiento completo.",
            "",
            "## Cómputo GPU con entradas preparadas",
            "",
            "### Primer ABBA completo: coincidió con el benchmark CPU",
            "",
            f"Estado: **{gpu.get('status', 'PENDIENTE')}**.",
        ]
    )
    if gpu.get("status") in {"PASSED", "NOT_ADMITTED"}:
        lines.extend(
            [
                "",
                f"Paridad exacta GPU: **{gpu.get('exact_gpu_gradient_parity', False)}**. "
                "Archivos protegidos sin modificar: "
                f"**{gpu.get('protected_files_unchanged', False)}**.",
            ]
        )
        medians = gpu.get("median_seconds_per_batch", {})
        if medians:
            lines.extend(
                [
                    "",
                    "| Implementación | Mediana por batch (s) |",
                    "|---|---:|",
                    f"| Referencia A | {number(medians.get('A'))} |",
                    f"| Máscaras cacheadas B | {number(medians.get('B'))} |",
                    "",
                    f"Cociente de medianas A/B: **{number(gpu.get('speedup_ratio'))}×**.",
                ]
            )
            if gpu.get("speedup_ratio", 0) < 1:
                lines.extend(
                    [
                        "",
                        "**Resultado de rendimiento negativo:** la variante cacheada tardó más "
                        "en esta comparación. La paridad no demuestra una mejora de velocidad.",
                    ]
                )
    isolated = value.get("gpu_isolated", {})
    lines.extend(
        [
            "",
            "### Repetición sin entrenamiento ni benchmark CPU concurrente",
            "",
            f"Estado: **{isolated.get('status', 'PENDIENTE')}**.",
        ]
    )
    if isolated.get("status") in {"PASSED", "NOT_ADMITTED"}:
        medians = isolated.get("median_seconds_per_batch", {})
        lines.extend(
            [
                "",
                f"Paridad exacta GPU: **{isolated.get('exact_gpu_gradient_parity', False)}**.",
                "",
                "| Implementación | Mediana por batch (s) |",
                "|---|---:|",
                f"| Referencia A | {number(medians.get('A'))} |",
                f"| Máscaras cacheadas B | {number(medians.get('B'))} |",
                "",
                f"Cociente de medianas A/B: **{number(isolated.get('speedup_ratio'))}×**.",
            ]
        )
        if isolated.get("speedup_ratio", 0) < 1:
            lines.extend(
                [
                    "",
                    "La repetición tampoco demuestra aceleración. "
                    "La máscara cacheada queda **desactivada** en el runtime admitido.",
                ]
            )
    for label, item in (("Primero", gpu), ("Aislado", isolated)):
        if item.get("trials"):
            lines.extend(
                [
                    "",
                    f"Tiempos individuales ({label}); se conserva la variación entre A1 y A2:",
                    "",
                    "| Ronda | Segundos por batch |",
                    "|---|---|",
                ]
            )
            for trial in item["trials"]:
                timings = ", ".join(f"{seconds:.3f}" for seconds in trial["batch_seconds"])
                lines.append(f"| {trial['name']} | {timings} |")
    lines.extend(
        [
            "",
            "La variabilidad temporal impide atribuir causalmente todo el delta a la caché. "
            "La decisión conservadora es mantener el cómputo original.",
        ]
    )
    if value.get("incomplete_gpu_probe", {}).get("present"):
        lines.extend(
            [
                "",
                "Una ejecución anterior quedó incompleta por un error de instrumentación "
                "al serializar un tensor con stride singleton. Se conservó su evidencia; "
                "no se admite como prueba de paridad. "
                "El ajuste afectó al capturador del benchmark, "
                "y la comparación se repitió sin updates de optimizador.",
            ]
        )
    lines.extend(
        [
            "",
            "Orden ABBA tras calentamiento; forward, pérdida, backward y clipping. "
            "Excluye preparación, paso del optimizador y copias de QA. "
            "Las pruebas no consumen updates del optimizador de la campaña.",
            "",
            "## Alcance",
            "",
            "La admisión requiere todos los gates, aunque un componente sea más rápido. "
            "Faltan aquí observaciones de rendimiento sostenido de la campaña "
            "y métricas TTC nuevas. "
            "No se extrapola una ETA ni se afirma SOTA.",
            "La observación breve posterior del entrenamiento real está en [LIVE.md](LIVE.md).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check and args.evidence_root is not None:
        parser.error("--check consumes only the existing public evidence")
    path = HERE / "EVIDENCE.json"
    if args.evidence_root is not None:
        value = export(args.evidence_root)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    text = render(value)
    target = HERE / "TABLES.md"
    if args.check:
        if target.read_text(encoding="utf-8") != text:
            raise ValueError("TABLES.md differs from the public evidence")
        print("Public tables verified")
    else:
        target.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
