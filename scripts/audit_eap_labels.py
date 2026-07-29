from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


EAP_ROOT = Path(r"E:\eAP_dataset")
LABEL_ROOT = EAP_ROOT / "data" / "train"
TRAIN_METADATA = EAP_ROOT / "data" / "train.parquet"

OUTPUT_DIR = Path("artifacts/audit/eap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_OUTPUT = OUTPUT_DIR / "eap_labels_audit.json"
MD_OUTPUT = OUTPUT_DIR / "eap_labels_audit.md"


def normalize_name(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def walk_arrow_type(
    data_type: pa.DataType,
    prefix: str,
    output: list[dict[str, str]],
) -> None:
    output.append({
        "path": prefix,
        "type": str(data_type),
    })

    if pa.types.is_struct(data_type):
        for child in data_type:
            walk_arrow_type(
                child.type,
                f"{prefix}.{child.name}",
                output,
            )

    elif (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
    ):
        value_field = data_type.value_field
        walk_arrow_type(
            value_field.type,
            f"{prefix}[]",
            output,
        )

    elif pa.types.is_map(data_type):
        walk_arrow_type(
            data_type.key_type,
            f"{prefix}.key",
            output,
        )
        walk_arrow_type(
            data_type.item_type,
            f"{prefix}.value",
            output,
        )


def schema_fields(schema: pa.Schema) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []

    for field in schema:
        walk_arrow_type(
            field.type,
            field.name,
            output,
        )

    return output


def compact_value(
    value: Any,
    depth: int = 0,
    max_depth: int = 5,
) -> Any:
    if depth >= max_depth:
        return f"<{type(value).__name__}>"

    if isinstance(value, dict):
        return {
            str(key): compact_value(
                item,
                depth + 1,
                max_depth,
            )
            for key, item in value.items()
        }

    if isinstance(value, list):
        compacted = [
            compact_value(
                item,
                depth + 1,
                max_depth,
            )
            for item in value[:3]
        ]

        if len(value) > 3:
            compacted.append(
                f"<... {len(value) - 3} elementos más>"
            )

        return compacted

    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"

    return value


def classify_path(path: str) -> set[str]:
    name = normalize_name(path)
    tokens = set(name.split("_"))
    categories: set[str] = set()

    direct_ttc_terms = (
        "ttc",
        "time_to_collision",
        "time_to_impact",
        "collision_time",
    )

    if any(term in name for term in direct_ttc_terms):
        categories.add("direct_ttc")

    if any(term in name for term in (
        "translation",
        "position",
        "location",
        "center",
        "distance",
        "depth",
        "range",
    )):
        categories.add("position_or_distance")

    if any(term in name for term in (
        "velocity",
        "speed",
        "relative_velocity",
        "closing_speed",
        "vel_mps",
    )) or tokens.intersection({"vx", "vy", "vz"}):
        categories.add("velocity")

    if any(term in name for term in (
        "track_id",
        "tracking_id",
        "instance_id",
        "object_id",
        "annotation_id",
    )):
        categories.add("tracking")

    if any(term in name for term in (
        "timestamp",
        "time_us",
        "time_ns",
        "frame_time",
        "exposure_start",
        "exposure_end",
    )):
        categories.add("timestamp")

    if any(term in name for term in (
        "size",
        "length",
        "width",
        "height",
        "extent",
        "dimensions",
    )):
        categories.add("object_size")

    if any(term in name for term in (
        "yaw",
        "rotation",
        "orientation",
        "quaternion",
    )):
        categories.add("orientation")

    if any(term in name for term in (
        "ego_velocity",
        "ego_speed",
        "vehicle_velocity",
        "platform_velocity",
    )):
        categories.add("ego_velocity")

    return categories


def schema_digest(schema: pa.Schema) -> str:
    return hashlib.sha256(
        str(schema).encode("utf-8")
    ).hexdigest()


label_files = sorted(
    LABEL_ROOT.rglob("labels.parquet")
)

if not label_files:
    raise SystemExit(
        f"No se encontraron labels.parquet en {LABEL_ROOT}"
    )

print("=" * 88)
print("AUDITORÍA DE LABELS eAP")
print("=" * 88)
print(f"Archivos encontrados: {len(label_files)}")
print()

schemas: dict[str, list[str]] = defaultdict(list)
field_types: dict[str, Counter[str]] = defaultdict(Counter)
category_paths: dict[str, set[str]] = defaultdict(set)
file_records: list[dict[str, Any]] = []

total_rows = 0

for index, path in enumerate(label_files, start=1):
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    digest = schema_digest(schema)
    rows = parquet.metadata.num_rows
    total_rows += rows

    relative_path = path.relative_to(EAP_ROOT).as_posix()
    sequence_id = path.parent.name

    schemas[digest].append(relative_path)

    fields = schema_fields(schema)

    for item in fields:
        field_types[item["path"]][item["type"]] += 1

        for category in classify_path(item["path"]):
            category_paths[category].add(item["path"])

    file_records.append({
        "sequence_id": sequence_id,
        "path": relative_path,
        "rows": rows,
        "row_groups": parquet.metadata.num_row_groups,
        "schema_sha256": digest,
    })

    print(
        f"[{index:02d}/{len(label_files):02d}] "
        f"{sequence_id}: {rows:,} filas"
    )

first_path = label_files[0]
first_table = pq.read_table(first_path).slice(0, 2)
sample_rows = [
    compact_value(row)
    for row in first_table.to_pylist()
]

metadata_info: dict[str, Any] = {}

if TRAIN_METADATA.exists():
    metadata_parquet = pq.ParquetFile(TRAIN_METADATA)
    metadata_info = {
        "path": TRAIN_METADATA.relative_to(
            EAP_ROOT
        ).as_posix(),
        "rows": metadata_parquet.metadata.num_rows,
        "schema": str(
            metadata_parquet.schema_arrow
        ),
        "sample_rows": [
            compact_value(row)
            for row in pq.read_table(
                TRAIN_METADATA
            ).slice(0, 2).to_pylist()
        ],
    }

direct_ttc = sorted(
    category_paths.get("direct_ttc", set())
)
position = sorted(
    category_paths.get(
        "position_or_distance",
        set(),
    )
)
velocity = sorted(
    category_paths.get("velocity", set())
)
tracking = sorted(
    category_paths.get("tracking", set())
)
timestamps = sorted(
    category_paths.get("timestamp", set())
)
object_size = sorted(
    category_paths.get("object_size", set())
)
ego_velocity = sorted(
    category_paths.get("ego_velocity", set())
)

if direct_ttc:
    verdict_code = "DIRECT_TTC_AVAILABLE"
    verdict = (
        "Existe al menos un campo cuyo nombre parece "
        "representar TTC directamente. Hay que verificar "
        "unidades, definición y valores antes de usarlo."
    )

elif position and velocity:
    verdict_code = "POTENTIALLY_DERIVABLE_POSITION_VELOCITY"
    verdict = (
        "No aparece un TTC directo, pero existen campos "
        "de posición/distancia y velocidad. TTC podría "
        "derivarse solo si ambas magnitudes están en el "
        "mismo sistema de coordenadas y la velocidad es "
        "relativa al ego o puede convertirse a relativa."
    )

elif position and tracking and timestamps:
    verdict_code = "POTENTIALLY_DERIVABLE_FROM_TRACKS"
    verdict = (
        "No aparece TTC ni una velocidad claramente "
        "disponible, pero hay posición, tracking y tiempo. "
        "Podría estimarse velocidad por diferencias "
        "temporales y después TTC, verificando coordenadas "
        "e identidad de tracks."
    )

else:
    verdict_code = "NO_TTC_SUPERVISION_IDENTIFIED"
    verdict = (
        "No se identificó TTC directo ni información "
        "suficiente por nombres de campos para derivarlo. "
        "eAP sería utilizable para percepción, detección, "
        "segmentación o pretraining autosupervisado, pero "
        "no como supervisión TTC sin información adicional."
    )

result = {
    "dataset_root": str(EAP_ROOT),
    "label_file_count": len(label_files),
    "total_label_rows": total_rows,
    "unique_schema_count": len(schemas),
    "schemas": schemas,
    "files": file_records,
    "field_types": {
        path: dict(counter)
        for path, counter in sorted(
            field_types.items()
        )
    },
    "candidate_paths": {
        "direct_ttc": direct_ttc,
        "position_or_distance": position,
        "velocity": velocity,
        "ego_velocity": ego_velocity,
        "tracking": tracking,
        "timestamps": timestamps,
        "object_size": object_size,
        "orientation": sorted(
            category_paths.get(
                "orientation",
                set(),
            )
        ),
    },
    "first_labels_file": (
        first_path.relative_to(EAP_ROOT).as_posix()
    ),
    "sample_label_rows": sample_rows,
    "train_metadata": metadata_info,
    "verdict_code": verdict_code,
    "verdict": verdict,
}

JSON_OUTPUT.write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    + "\n",
    encoding="utf-8",
)

markdown = [
    "# Auditoría de etiquetas eAP",
    "",
    f"- Archivos `labels.parquet`: {len(label_files)}",
    f"- Filas totales: {total_rows:,}",
    f"- Esquemas distintos: {len(schemas)}",
    f"- Veredicto: **{verdict_code}**",
    "",
    verdict,
    "",
    "## Campos TTC directos",
    "",
]

markdown.extend(
    [f"- `{item}`" for item in direct_ttc]
    or ["- Ninguno detectado."]
)

markdown.extend([
    "",
    "## Posición o distancia",
    "",
])

markdown.extend(
    [f"- `{item}`" for item in position]
    or ["- Ninguno detectado."]
)

markdown.extend([
    "",
    "## Velocidad",
    "",
])

markdown.extend(
    [f"- `{item}`" for item in velocity]
    or ["- Ninguno detectado."]
)

markdown.extend([
    "",
    "## Velocidad ego",
    "",
])

markdown.extend(
    [f"- `{item}`" for item in ego_velocity]
    or ["- Ninguno detectado."]
)

markdown.extend([
    "",
    "## Tracking",
    "",
])

markdown.extend(
    [f"- `{item}`" for item in tracking]
    or ["- Ninguno detectado."]
)

markdown.extend([
    "",
    "## Timestamps",
    "",
])

markdown.extend(
    [f"- `{item}`" for item in timestamps]
    or ["- Ninguno detectado."]
)

markdown.extend([
    "",
    "## Tamaño del objeto",
    "",
])

markdown.extend(
    [f"- `{item}`" for item in object_size]
    or ["- Ninguno detectado."]
)

MD_OUTPUT.write_text(
    "\n".join(markdown) + "\n",
    encoding="utf-8",
)

print()
print("=" * 88)
print("RESULTADO")
print("=" * 88)
print(f"Filas totales:      {total_rows:,}")
print(f"Esquemas distintos: {len(schemas)}")
print(f"Veredicto:          {verdict_code}")
print()
print(verdict)
print()
print("TTC directo:")
for item in direct_ttc or ["<ninguno>"]:
    print("  ", item)

print()
print("Posición/distancia:")
for item in position or ["<ninguno>"]:
    print("  ", item)

print()
print("Velocidad:")
for item in velocity or ["<ninguno>"]:
    print("  ", item)

print()
print("Velocidad ego:")
for item in ego_velocity or ["<ninguno>"]:
    print("  ", item)

print()
print("Tracking:")
for item in tracking or ["<ninguno>"]:
    print("  ", item)

print()
print("Timestamps:")
for item in timestamps or ["<ninguno>"]:
    print("  ", item)

print()
print(f"Informe JSON: {JSON_OUTPUT}")
print(f"Informe MD:   {MD_OUTPUT}")
