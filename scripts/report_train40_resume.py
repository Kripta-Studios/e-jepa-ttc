"""Regenerate scoped TRAIN40 resume tables and optionally preserve compact evidence.

This command reads JSON only. It never imports Torch, opens sensors, loads weights,
changes a measurement, or infers sustained throughput from isolated latencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "artifacts/train40_runtime_resume_20261027"
DEFAULT_OUTPUT = ROOT / "docs/train40_runtime_resume_20261027/RESULTS_GENERATED.md"
MAX_RECEIPT_BYTES = 2_000_000
FAMILIES = (
    "producers",
    "prefix",
    "normalization",
    "multiobject",
    "transport",
    "encoders",
    "compiled_pipeline",
    "rgb_uint8",
    "scheduler",
)
SCOPES = {
    "producers": "Productores preparados en GPU; excluye sensores, H2D, contexto y TTC final.",
    "prefix": "Paquete RAW a representaciones de eventos; excluye modelos y TTC.",
    "normalization": "Normalización de vóxeles preparados; excluye sensores y modelos.",
    "multiobject": "Piloto completo con ROI de capacidad artificiales; no es un stream sostenido.",
    "transport": "Transporte sobre features sintéticas preparadas; excluye sensores y TTC.",
    "encoders": "Encoders de endpoints preparados; excluye transporte, contexto y TTC.",
    "compiled_pipeline": "Productores con transporte compilado e inputs GPU preparados; "
    "la paridad TTC se comprueba fuera del reloj. Excluye RAW, H2D y contexto del tiempo.",
    "rgb_uint8": "Preparación RGB sintética; incluye el tramo declarado en cada recibo. "
    "No incluye modelos ni TTC. Bytes propuestos no equivalen a transferencias ejecutadas.",
    "scheduler": "Control del planificador con RAW y predicciones ficticias. "
    "No ejecuta modelos ni GPU y no admite ningún objetivo TTC.",
}
COMPACT_NAMES = {
    "summary.json",
    "admission_analysis.json",
    "binding.json",
    "original_snapshot.json",
    "source_relocation.json",
    "service_startup.json",
    "setup_failure.json",
    "preservation.json",
    "command.json",
    "exit.json",
    "receipt.json",
    "cpu_tests.txt",
    "pytest.txt",
    "ruff.txt",
    "pyright.txt",
    "format.txt",
    "startup.json",
    "source_manifest.json",
    "source_snapshots.json",
    "MANIFEST.json",
    "NOTES.md",
    "IMPLEMENTATION.md",
    "IMPLEMENTATION_NOTES.md",
    "GPU_FINDINGS.md",
    "DESIGN_AND_COMMANDS.md",
    "CPU_BENCHMARK_NOTES.md",
    "CAUSAL_REUSE_AND_20MS.md",
    "causal_cache_metadata_v1.json",
    "scheduler_jitter_v1.json",
    "WINDOWS_SCHEDULER_JITTER.md",
    "PACED_RUNNER.md",
}
FORBIDDEN_PARTS = {
    "deps",
    "cache",
    "caches",
    "triton",
    "inductor",
    "checkpoints",
    "weights",
    ".venv",
    "venv",
    "site-packages",
    "__pycache__",
    "torch_compile_debug",
    "compiler_cache",
    "inductor_cache",
    "triton_cache",
}


def sha(data: bytes) -> str:
    """Return the byte-level SHA-256 used for every report source."""
    return hashlib.sha256(data).hexdigest()


def safe_path(path: Path, root: Path) -> Path:
    """Reject escaped paths and symbolic links when reading a receipt tree."""
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"Evidence path escapes source: {path}")
    current = path.absolute()
    while current != root and current != current.parent:
        if current.is_symlink():
            raise ValueError(f"Symbolic evidence path is not allowed: {path}")
        current = current.parent
    return resolved


def mapping(value: Any) -> dict[str, Any]:
    """Treat absent optional mappings as empty without inventing values."""
    return value if isinstance(value, dict) else {}


def cell(value: Any) -> str:
    """Format only observed scalar values, escaping Markdown table separators."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "sí" if value else "no"
    if isinstance(value, float):
        return f"{value:.9g}" if math.isfinite(value) else "NO_FINITO"
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    """Render a Markdown table with no estimated or interpolated measurements."""
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("Generated table row width differs from its header")
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(cell(value) for value in row) + " |" for row in rows),
        "",
    ]


@dataclass
class Receipt:
    """A parsed result and the optional binding that defines its scope."""

    relative: str
    result: dict[str, Any]
    binding: dict[str, Any]
    family: str

    @property
    def status(self) -> Any:
        if "status" in self.result:
            return self.result["status"]
        if isinstance(self.result.get("passed"), bool):
            return "PASSED" if self.result["passed"] else "FAILED"
        return "NO_DECLARADO"

    @property
    def scope(self) -> Any:
        return self.result.get("scope", self.binding.get("scope", "NO_DECLARADO"))

    @property
    def device(self) -> Any:
        return self.binding.get("device", self.result.get("device", self.result.get("gpu")))


class Sources:
    """Read each JSON once and bind all consumed bytes to a final manifest."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)
        self.files: dict[str, dict[str, Any]] = {}
        self.parsed: dict[str, dict[str, Any]] = {}

    def read(self, path: Path) -> dict[str, Any]:
        """Read a UTF-8 JSON object, retaining its exact bytes and digest."""
        path = safe_path(path, self.root)
        name = path.relative_to(self.root).as_posix()
        if name in self.parsed:
            return self.parsed[name]
        raw = path.read_bytes()
        if len(raw) > MAX_RECEIPT_BYTES:
            raise ValueError(f"Receipt exceeds compact limit: {name}")
        value = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object: {name}")
        self.files[name] = {"path": name, "sha256": sha(raw), "bytes": len(raw)}
        self.parsed[name] = value
        return value

    def verify(self) -> None:
        """Fail rather than publish tables from sources changed during generation."""
        for name, entry in self.files.items():
            path = safe_path(self.root / name, self.root)
            if sha(path.read_bytes()) != entry["sha256"]:
                raise ValueError(f"Source changed during report generation: {name}")

    def discover(self) -> list[Receipt]:
        """Discover completed or failed summaries; preserve their declared status."""
        receipts = []
        for path in sorted(self.root.rglob("*.json")):
            relative = path.relative_to(self.root)
            if any(part.lower() in FORBIDDEN_PARTS for part in relative.parts):
                continue
            if path.name not in {"summary.json", "admission_analysis.json", "setup_failure.json"}:
                continue
            if path.name == "summary.json" and path.with_name("admission_analysis.json").exists():
                continue
            group = relative.parts[0].lower()
            family = "baseline" if group.startswith(("baseline", "corrected_all")) else group
            if family.startswith("encoder") or family == "compiled_encoder":
                family = "encoders"
            result = self.read(path)
            if family == "compute_council" and "FAKE_RAW_FAKE_PREDICTIONS" in result.get(
                "scope", ""
            ):
                family = "scheduler"
            binding_path = path.with_name("binding.json")
            binding = self.read(binding_path) if binding_path.exists() else {}
            receipts.append(Receipt(relative.as_posix(), result, binding, family))
        return receipts


def preservation(result: dict[str, Any]) -> Any:
    """Read the explicit preservation outcome without deriving it from run status."""
    for name in ("originals", "preservation"):
        if isinstance(result.get(name), dict):
            return result[name].get("passed")
    if "sources_unchanged" in result:
        return result["sources_unchanged"]
    if isinstance(result.get("changed_sources"), list):
        return not result["changed_sources"]
    return None


def latency_rows(value: Any, prefix: str = "") -> list[tuple[str, dict[str, Any]]]:
    """Traverse nested latency summaries while keeping each clock route separate."""
    if not isinstance(value, dict):
        return []
    if any(key in value for key in ("mean_ms", "median_ms", "p50_ms")):
        return [(prefix or "total", value)]
    found = []
    for name, nested in value.items():
        found.extend(latency_rows(nested, f"{prefix}/{name}".strip("/")))
    return found


def timing_cells(
    label: Any, objects: Any, route: str, timing: dict[str, Any], gate: Any
) -> list[Any]:
    """Keep summary units in milliseconds and leave unavailable statistics blank."""
    count = timing.get("count")
    if count is None and isinstance(timing.get("wall_ms"), list):
        count = len(timing["wall_ms"])
    return [
        label,
        objects,
        route,
        count,
        timing.get("mean_ms"),
        timing.get("p50_ms", timing.get("median_ms")),
        timing.get("p95_ms"),
        timing.get("p99_ms"),
        timing.get("max_ms"),
        gate,
    ]


TIMING_HEADERS = [
    "Caso",
    "Objetos/ROI",
    "Ruta",
    "n",
    "Media ms",
    "p50 ms",
    "p95 ms",
    "p99 ms",
    "Máximo ms",
    "Paridad del tramo",
]


def baseline_section(receipt: Receipt) -> list[str]:
    """Render full-population parity and individual rejected queries."""
    data = dict(receipt.result)
    lines = [f"### `{receipt.relative}`", "", f"Estado: **{cell(receipt.status)}**.", ""]
    if "prediction_components" not in data and isinstance(data.get("parity"), dict):
        parity = data["parity"]
        queries = parity.get("per_query", [])
        data["prediction_components"] = {
            name: dict(
                item,
                passed_count=sum(
                    mapping(mapping(row).get("components")).get(name, {}).get("passed") is True
                    for row in queries
                ),
            )
            for name, item in mapping(parity.get("components")).items()
        }
        data["coverage"] = mapping(parity.get("candidate_coverage"))
        data["failed_queries"] = [row for row in queries if mapping(row).get("passed") is False]
        data["input_parity"] = {
            "passed_count": data["coverage"].get("records")
            if data.get("input_parity_passed") is True
            else None
        }
    if "prediction_components" not in data:
        lines += [cell(data.get("error", "Sin análisis de admisión disponible.")), ""]
        return lines
    coverage = mapping(data.get("coverage"))
    inputs = mapping(data.get("input_parity"))
    originals = mapping(data.get("originals"))
    components = mapping(data.get("prediction_components"))
    complete_parity = (
        bool(components)
        and coverage.get("passed") is True
        and all(
            mapping(item).get("passed_count") == coverage.get("expected")
            for item in components.values()
        )
    )
    lines += table(
        ["Cobertura", "Inputs pasan", "Inputs exactos", "Archivos preservados", "Intactos"],
        [
            [
                f"{coverage.get('records')}/{coverage.get('expected')}",
                inputs.get("passed_count"),
                inputs.get("exact_count"),
                originals.get("files"),
                originals.get("passed"),
            ]
        ],
    )
    lines += table(
        ["Salida", "Pasan", "Pares finitos", "Máx. error absoluto s", "Máx. relativo"],
        [
            [
                name,
                mapping(item).get("passed_count"),
                mapping(item).get("finite_pairs"),
                mapping(item).get("max_abs_error"),
                mapping(item).get("max_relative_error"),
            ]
            for name, item in mapping(data.get("prediction_components")).items()
        ],
    )
    failed = []
    for item in data.get("failed_queries", []):
        if not isinstance(item, dict):
            raise ValueError(f"Malformed failed query in {receipt.relative}")
        for name, component in mapping(item.get("components")).items():
            if mapping(component).get("passed") is False:
                failed.append(
                    [
                        item.get("index"),
                        name,
                        mapping(component).get("abs_error"),
                        mapping(component).get("allowed_abs_error"),
                    ]
                )
    if failed:
        lines += table(["Índice rechazado", "Salida", "Error absoluto s", "Límite s"], failed)
    lines += table(
        TIMING_HEADERS,
        [
            timing_cells(
                "cohorte completa",
                1,
                "RAW disponible → TTC",
                mapping(data.get("latency_ms")),
                complete_parity,
            )
        ],
    )
    lines += [cell(data.get("timing_caveat", "Inicialización de historial y disco excluidos.")), ""]
    return lines


def record_gate(record: dict[str, Any], family: str) -> Any:
    """Label parity according to the measured stage, never as a universal admission."""
    if family == "producers":
        return mapping(record.get("final_ttc_parity")).get("passed")
    if family == "prefix":
        return record.get("canonical_primary_final_passed")
    if family == "rgb_uint8":
        return mapping(record.get("parity")).get("passed")
    return record.get("all_encoder_parity_passed", record.get("passed_1e5", record.get("passed")))


def run_section(receipt: Receipt) -> list[str]:
    """Render known stage schemas and explicitly identify unmeasured/error cases."""
    result, binding = receipt.result, receipt.binding
    lines = [
        f"### `{receipt.relative}`",
        "",
        f"Estado: **{cell(receipt.status)}**. "
        f"Dispositivo: `{cell(receipt.device)}`. Alcance: `{cell(receipt.scope)}`.",
        "",
    ]
    keys = (
        "query_index",
        "encoder_layout",
        "transport_variant",
        "endpoint_reuse",
        "event_endpoint_reuse",
        "rgb_endpoint_reuse",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cuda_graphs",
        "normalization",
        "roi_size",
        "precision",
    )
    config = [[key, binding[key]] for key in keys if key in binding]
    if "input_population" in binding:
        config.append(["input_population", binding["input_population"]])
    population = mapping(binding.get("data"))
    config += [
        [f"data.{key}", population[key]]
        for key in ("synthetic", "index", "query_id", "additional_rois")
        if key in population
    ]
    if config:
        lines += table(["Configuración", "Valor"], config)
    records = result.get("records", result.get("results", result.get("cases", [])))
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError(f"Malformed record list: {receipt.relative}")
    rows, extra = [], []
    for index, record in enumerate(records):
        label = record.get("backend", record.get("model", record.get("variant", index)))
        if receipt.family == "scheduler":
            label = record.get("strategy", index)
        if "batch" in record:
            label = f"{label}; batch={record['batch']}"
        if "radius" in record:
            label = f"{label}; radio={record['radius']}"
        if receipt.family == "normalization":
            label = (
                f"{label}; ventanas={cell(record.get('windows'))}; "
                f"columnas={cell(record.get('columns'))}; "
                f"ocupación={cell(record.get('requested_occupancy'))}"
            )
        gate = record_gate(record, receipt.family)
        measured = latency_rows(record.get("latency_ms"))
        if receipt.family == "scheduler":
            for name in ("raw_availability_lateness_ms", "deadline_response_ms"):
                measured += latency_rows(record.get(name), name)
        if receipt.family == "rgb_uint8":
            measured += latency_rows(
                record.get("original_host_expanded_preparation_latency"), "original_preparation"
            )
            measured += latency_rows(record.get("rgb_preparation_latency"), "uint8_preparation")
        for route in ("eager", "graph"):
            measured += latency_rows(record.get(route), route)
        rows += [
            timing_cells(
                label,
                record.get("objects"),
                route,
                timing,
                mapping(mapping(record.get("parity")).get(route)).get("passed")
                if receipt.family == "compiled_pipeline"
                else gate,
            )
            for route, timing in measured
        ]
        if not measured:
            extra.append([label, record.get("status", "SIN_TIEMPOS"), record.get("error")])
    if isinstance(result.get("latency_by_count_ms"), dict):
        rows += [
            timing_cells(
                "piloto completo",
                count,
                "RAW disponible → último TTC",
                timing,
                result.get("passed"),
            )
            for count, timing in latency_rows(result["latency_by_count_ms"])
        ]
    if rows:
        lines += table(TIMING_HEADERS, rows)
    if extra:
        lines += table(["Caso", "Estado", "Error"], extra)
    if not records and not rows:
        lines += ["No hay mediciones de este tramo en el recibo.", ""]
    if receipt.family == "prefix":
        lines += [
            "La paridad canónica indicada corresponde a la ROI primaria; "
            "no certifica TTC ni veinte tracks reales.",
            "",
        ]
        lines += table(
            ["Backend", "ROI", "Exacto frente a buckets", "Máx. diferencia", "Bytes"],
            [
                [
                    r.get("backend"),
                    r.get("objects"),
                    r.get("prefix_vs_bucket_bit_exact"),
                    r.get("prefix_vs_bucket_max_abs"),
                    mapping(r.get("stats")).get("allocated_bytes"),
                ]
                for r in records
            ],
        )
    if receipt.family == "producers":
        lines += [
            "La columna de paridad corresponde a los TTC comprobados fuera del reloj "
            "de productores, frente a la referencia por objeto del probe.",
            "",
        ]
    if receipt.family == "compiled_pipeline":
        lines += [
            "La columna de paridad corresponde a las tres salidas TTC por ruta. "
            "El estado del ensayo conserva también fallos de auditoría. "
            "La paridad de features intermedias es un diagnóstico separado.",
            "",
        ]
    if receipt.family == "rgb_uint8":
        lines += table(
            [
                "Objetos",
                "Observaciones",
                "Frames/tripleta",
                "H2D ejecutados bytes",
                "H2D propuestos bytes",
                "Píxeles FP32 originales bytes",
            ],
            [
                [
                    r.get("objects"),
                    r.get("history_observations"),
                    r.get("steps"),
                    mapping(r.get("initial_receipt")).get("h2d_bytes"),
                    mapping(r.get("initial_receipt")).get("candidate_h2d_bytes"),
                    mapping(r.get("initial_receipt")).get("baseline_expanded_fp32_bytes"),
                ]
                for r in records
            ],
        )
    for key in ("error", "errors", "cleanup_errors"):
        if result.get(key):
            lines += [f"{key}: `{cell(result[key])}`", ""]
    return lines


def compact_overview(receipts: list[Receipt]) -> list[str]:
    """Group run coverage without combining incompatible timing distributions."""
    grouped: dict[tuple[str, str], list[Receipt]] = {}
    for receipt in receipts:
        grouped.setdefault((receipt.family, cell(receipt.device)), []).append(receipt)
    rows = []
    for (family, device), items in sorted(grouped.items()):
        counts = []
        for item in items:
            primary = item.binding.get("primary_objects_per_update")
            if isinstance(primary, int) and not isinstance(primary, bool):
                counts.append(primary)
            records = item.result.get("records", item.result.get("results", []))
            if isinstance(records, list):
                counts.extend(
                    value
                    for row in records
                    if isinstance(row, dict)
                    and isinstance(value := row.get("objects"), int)
                    and not isinstance(value, bool)
                )
            by_count = mapping(item.result.get("latency_by_count_ms"))
            counts.extend(int(value) for value in by_count if value.isdecimal())
        statuses = Counter(cell(item.status) for item in items)
        rows.append(
            [
                family,
                device,
                len(items),
                max(counts) if counts else None,
                "; ".join(f"{key}: {count}" for key, count in sorted(statuses.items())),
            ]
        )
    return table(
        ["Frente", "Dispositivo", "Recibos", "Máx. objetos/ROI ensayados", "Estados"], rows
    )


def write_atomic(path: Path, raw: bytes) -> None:
    """Replace a generated artifact without leaving a partial JSON/Markdown file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def generate(source: Path, output: Path) -> dict[str, Any]:
    """Generate tables and their source manifest, verifying input stability."""
    sources = Sources(source)
    output = output.resolve()
    if output.is_relative_to(sources.root):
        raise ValueError("Generated report must be outside the source evidence tree")
    receipts = sources.discover()
    if not receipts:
        raise ValueError("No supported summaries found; refusing an empty measurement report")
    lines = [
        "# Mediciones regeneradas — reanudación TRAIN40",
        "",
        "Generado exclusivamente desde recibos JSON. Cada sección conserva su frontera "
        "de medición; los tiempos de secciones distintas no se suman ni se convierten "
        "en una afirmación de 50 Hz sostenidos. CPU, GPU preparada y cadena completa "
        "son ensayos diferentes. Los resultados fallidos permanecen visibles.",
        "",
        "Fuentes y SHA-256: [RESULTS_SOURCES.json](RESULTS_SOURCES.json). "
        "`—` significa dato no declarado; no significa cero.",
        "",
        "## Lectura rápida",
        "",
    ]
    lines += compact_overview(receipts)
    lines += [
        "El máximo de objetos/ROI es cobertura del tramo indicado, no capacidad admitida "
        "de la cadena completa. Los encoders expresan batch de endpoints, no objetos. "
        "Cada tiempo y su paridad aparecen separados en el detalle.",
        "",
        "## Inventario",
        "",
    ]
    lines += table(
        ["Recibo", "Estado", "Dispositivo", "Preservación"],
        [[r.relative, r.status, r.device, preservation(r.result)] for r in receipts],
    )
    for family in ("baseline", *FAMILIES):
        selected = [r for r in receipts if r.family == family]
        lines += [f"## {family}", ""]
        if family == "baseline":
            lines += [
                "Una ROI por consulta original; historial reconstruido antes del reloj. "
                "Conserva el gate conjunto de las tres salidas.",
                "",
            ]
        else:
            lines += [SCOPES[family], ""]
        if not selected:
            lines += ["Sin recibos disponibles en esta generación.", ""]
        for receipt in selected:
            lines += baseline_section(receipt) if family == "baseline" else run_section(receipt)
    unknown = [r for r in receipts if r.family not in {"baseline", *FAMILIES}]
    if unknown:
        lines += [
            "## Recibos sin adaptador de tabla",
            "",
            "Se conservan en el manifiesto; no se interpreta su esquema.",
            "",
        ]
        lines += [f"- `{r.relative}`: {cell(r.status)}" for r in unknown]
        lines += [""]
    encoded = "\n".join(lines).encode("utf-8")
    sources.verify()
    manifest = {
        "schema": "train40_runtime_resume_report_sources_v1",
        "source_root": str(sources.root),
        "generator": str(Path(__file__).relative_to(ROOT)),
        "generator_sha256": sha(Path(__file__).read_bytes()),
        "report": output.name,
        "report_sha256": sha(encoded),
        "sources": list(sources.files.values()),
        "no_sensor_model_or_gpu_execution": True,
        "sustained_50hz_claim": False,
    }
    write_atomic(output, encoded)
    write_atomic(
        output.with_name("RESULTS_SOURCES.json"),
        (json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
    )
    return manifest


def executed_source_snapshots(
    receipts: list[tuple[Path, Path, bytes]],
) -> tuple[list[tuple[Path, Path, bytes]], list[dict[str, Any]]]:
    """Snapshot source files only when current bytes match an executed receipt SHA."""
    selected: dict[str, tuple[Path, Path, bytes]] = {}
    unavailable: dict[str, dict[str, Any]] = {}
    for _, relative, raw in receipts:
        if relative.name not in {"binding.json", "summary.json"}:
            continue
        value = mapping(json.loads(raw.decode("utf-8-sig")))
        digests = {**mapping(value.get("sources")), **mapping(value.get("source_hashes"))}
        for name, digest in digests.items():
            path = Path(name)
            if path.suffix != ".py" or not isinstance(digest, str) or len(digest) != 64:
                continue
            path = path if path.is_absolute() else ROOT / path
            key = f"{name}:{digest}"
            try:
                path = safe_path(path, ROOT)
            except (ValueError, FileNotFoundError):
                unavailable[key] = {
                    "source": name,
                    "expected_sha256": digest,
                    "reason": "not_available_inside_current_checkout",
                }
                continue
            data = path.read_bytes()
            if len(data) > MAX_RECEIPT_BYTES or sha(data) != digest:
                unavailable[key] = {
                    "source": name,
                    "expected_sha256": digest,
                    "reason": "current_bytes_differ_or_exceed_compact_limit",
                }
                continue
            data.decode("utf-8-sig")
            name_in_repo = path.relative_to(ROOT)
            target = Path("source_snapshots") / digest / name_in_repo.with_name(path.name + ".txt")
            selected[target.as_posix()] = (path, target, data)
    return list(selected.values()), list(unavailable.values())


def copy_evidence(source: Path, destination: Path) -> dict[str, Any]:
    """Copy a fresh compact receipt bundle; never copy weights, raw data or caches."""
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if (
        destination.exists()
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError("Evidence destination must be new and separate from the source tree")
    selected: list[tuple[Path, Path, bytes]] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(part.lower() in FORBIDDEN_PARTS for part in relative.parts):
            continue
        name = path.name
        wanted = (
            name in COMPACT_NAMES
            or name.endswith(("_command.json", "_exit.json", ".py.txt"))
            or path.suffix == ".py"
        )
        if not wanted:
            continue
        safe_path(path, source)
        raw = path.read_bytes()
        if len(raw) > MAX_RECEIPT_BYTES:
            raise ValueError(f"Selected receipt exceeds compact limit: {relative}")
        if path.suffix == ".json":
            json.loads(raw.decode("utf-8-sig"))
        else:
            raw.decode("utf-8-sig")
        published = relative.with_name(name + ".txt") if path.suffix == ".py" else relative
        selected.append((path, published, raw))
    if not selected:
        raise ValueError("No compact receipts found")
    snapshots, unavailable = executed_source_snapshots(selected)
    selected += snapshots
    if len({relative.as_posix() for _, relative, _ in selected}) != len(selected):
        raise ValueError("Duplicate evidence destination names")
    destination.mkdir(parents=True, exist_ok=False)
    inventory = []
    for path, relative, raw in selected:
        if sha(path.read_bytes()) != sha(raw):
            raise ValueError(f"Evidence source changed before copying: {path}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha(target.read_bytes()) != sha(raw):
            raise ValueError(f"Evidence changed while copying: {path}")
        inventory.append(
            {
                "source": path.relative_to(
                    source if path.is_relative_to(source) else ROOT
                ).as_posix(),
                "origin": "artifact" if path.is_relative_to(source) else "repository_source",
                "published": relative.as_posix(),
                "sha256": sha(raw),
                "bytes": len(raw),
            }
        )
    manifest = {
        "schema": "train40_runtime_resume_evidence_v1",
        "source_root": str(source),
        "repository_source_root": str(ROOT),
        "files": inventory,
        "unavailable_executed_source_snapshots": unavailable,
        "raw_data_weights_dependencies_included": False,
        "jsonl_query_streams_included": False,
        "maximum_file_bytes": MAX_RECEIPT_BYTES,
    }
    write_atomic(
        destination / "MANIFEST.json", (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    )
    return manifest


def main() -> None:
    """Regenerate directly, or explicitly copy a fresh evidence bundle first."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--copy-evidence", action="store_true")
    parser.add_argument("--evidence-destination", type=Path)
    args = parser.parse_args()
    source = args.source
    if args.evidence_destination is not None and not args.copy_evidence:
        parser.error("--evidence-destination requires --copy-evidence")
    if args.copy_evidence:
        destination = args.evidence_destination or args.output.parent / "evidence"
        copy_evidence(source, destination)
        source = destination
    manifest = generate(source, args.output)
    print(
        json.dumps(
            {
                "report": str(args.output),
                "sources": len(manifest["sources"]),
                "report_sha256": manifest["report_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
