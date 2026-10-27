"""Publish compact runtime evidence and regenerate the handoff measurement tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def preserve(source: Path, destination: Path) -> None:
    """Copy only compact top-level receipts/source snapshots, excluding dependencies."""
    destination.mkdir(parents=True, exist_ok=False)
    inventory = []
    groups = (
        "oracle_profile",
        "pilot_v1",
        "integral_pilot_v1",
        "producer_probe",
        "encoding_probe",
        "compiler_probe",
        "transport_council",
        "closeout",
    )
    for group in groups:
        folder = source / group
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix not in {".json", ".jsonl", ".py", ".txt"}:
                continue
            if path.name == "pyright.json":
                continue
            if path.is_symlink() or path.stat().st_size > 2_000_000:
                raise ValueError(f"Unexpected evidence source: {path}")
            relative = Path(group) / (path.name + ".txt" if path.suffix == ".py" else path.name)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            inventory.append(
                {
                    "source": str(path.relative_to(source)),
                    "published": relative.as_posix(),
                    "sha256": _sha(target),
                    "bytes": target.stat().st_size,
                }
            )
    (destination / "MANIFEST.json").write_text(
        json.dumps(
            {
                "schema": "train40_runtime_handoff_evidence_v1",
                "files": inventory,
                "raw_data_weights_dependencies_included": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    generate_tables(destination, destination.parent / "RESULTS_GENERATED.md")


def generate_tables(evidence: Path, output: Path) -> None:
    """Regenerate every numeric table below from preserved JSON receipts."""

    def read(name: str) -> dict:
        return json.loads((evidence / name).read_text(encoding="utf-8"))

    pilot = read("integral_pilot_v1/summary.json")
    rows = [
        json.loads(line)
        for line in (evidence / "integral_pilot_v1/queries.jsonl").read_text().splitlines()
    ]
    text = [
        "# Mediciones regeneradas — TRAIN40, 27 de octubre de 2026",
        "",
        "Generado desde los JSON del directorio `evidence/`. Los relojes de estas tablas "
        "tienen alcances diferentes y no deben dividirse entre sí para afirmar un speedup.",
        "",
        "## Piloto completo, un objeto, tres consultas",
        "",
        "Desde paquete RAW CPU y frames RGB disponibles: incluye ROI, representación, "
        "transferencias, productores, contexto y TTC. Excluye reproducción de disco, "
        "inicialización del historial y priming sintético inicial. No es un stream sostenido.",
        "",
        "| Índice original | Actualización completa (ms) | Entrada exacta |",
        "|---:|---:|---|",
    ]
    for row in rows:
        text.append(
            f"| {row['index']} | {row['complete_wall_ms']:.4f} | {row['input_parity']['exact']} |"
        )
    latency = pilot["latency_ms"]
    text += [
        "",
        f"Estado: **{pilot['status']}**. Media {latency['mean_ms']:.4f} ms; "
        f"p95 {latency['p95_ms']:.4f} ms; p99 {latency['p99_ms']:.4f} ms. "
        "Solo tres observaciones: estos percentiles no estiman una cola estable.",
        "",
        "| Salida | Máxima diferencia absoluta respecto al original (s) |",
        "|---|---:|",
    ]
    for name, item in pilot["parity"]["components"].items():
        text.append(f"| {name} | {item['max_abs_error']:.10g} |")
    text += [
        "",
        "Tolerancia fijada: `atol=rtol=1e-5`. "
        f"Preservación: {pilot['originals']['files']} archivos, "
        f"{len(pilot['originals']['changed'])} cambiados.",
        "",
        "## Productores preparados con reutilización por objeto",
        "",
        "Tensores ya residentes en GPU. Incluye productores, PAIR, PHASE17 y guards; "
        "excluye sensores, H2D, contexto y TTC final. Carga multiobjeto artificial declarada.",
        "",
        "| Objetos | Paralelo mínimo (ms) | Paralelo máximo (ms) | Serial/paralelo exactos |",
        "|---:|---:|---:|---|",
    ]
    for row in read("producer_probe/grouped_multiobject_floor.json")["records"]:
        values = [value * 1000 for value in row["times"]["parallel"]]
        text.append(
            f"| {row['objects']} | {min(values):.3f} | {max(values):.3f} | "
            f"{row['serial_parallel_exact']} |"
        )
    text += [
        "",
        "El caso de dos objetos no es exacto y su deriva no se cuantificó al pausar. "
        "Ninguna fila demuestra paridad del TTC final multiobjeto contra el oracle.",
        "",
        "## Representación de eventos para veinte ROI",
        "",
    ]
    many = read("encoding_probe/integral_many_results.json")
    text += [
        f"Nuevo paquete de {many['packet_events']:,} eventos más veinte ROI: "
        f"**{many['new20ms_ingest_and_all20_roi_ms']:.3f} ms**. "
        "ROI distintas de carga, sin veinte objetos anotados. Modelos excluidos.",
        "",
        "## Compilación aislada de correlación",
        "",
        "B8, C64, 32×32, radio 1; features sintéticas. No es un productor completo.",
        "",
        "| Ejecución | Mediana (ms) | p95 (ms) |",
        "|---|---:|---:|",
    ]
    compiled = read("compiler_probe/compile_radius1_b8_toolchain.json")["records"][0]
    for name in ("eager", "eager_graph", "compiled"):
        item = compiled[name]
        # Preserve the probe's own reported summary units rather than infer throughput.
        median = item["median_ms"]
        p95 = item["p95_ms"]
        text.append(f"| {name} | {median:.4f} | {p95:.4f} |")
    text += [
        "",
        f"Compilación inicial: {compiled['compile_seconds']:.3f} s. "
        "Paridad del helper a 1e-5 aprobada, sin igualdad bit a bit. "
        "Falta paridad de TTC y compilación del core completo.",
        "",
        "## Resultado rechazado",
        "",
        "`pilot_v1/summary.json` es FAILED: un módulo aditivo cambió durante el ensayo. "
        "Se conserva para auditoría; no se admite su tabla de tiempos. "
        "El candidato congelado no cambió.",
        "",
    ]
    output.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    """Preserve fresh evidence or regenerate its tables without sensor inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    if args.tables_only:
        generate_tables(args.destination, args.destination.parent / "RESULTS_GENERATED.md")
    else:
        if args.source is None:
            parser.error("--source is required when preserving evidence")
        preserve(args.source, args.destination)


if __name__ == "__main__":
    main()
