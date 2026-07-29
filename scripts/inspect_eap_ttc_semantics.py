from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(r"E:\eAP_dataset")
LABEL_ROOT = ROOT / "data" / "train"
METADATA_PATH = ROOT / "data" / "train.parquet"

label_files = sorted(LABEL_ROOT.rglob("labels.parquet"))

if not label_files:
    raise SystemExit("No se encontraron labels.parquet")

label_path = label_files[0]
sequence_id = label_path.parent.name

print("=" * 100)
print("ARCHIVO")
print("=" * 100)
print("Secuencia:", sequence_id)
print("Labels:", label_path)

labels = pq.read_table(label_path).to_pandas()
metadata = pq.read_table(METADATA_PATH).to_pandas()

print()
print("=" * 100)
print("COLUMNAS DE LABELS")
print("=" * 100)

for column in labels.columns:
    first_valid = next(
        (
            value for value in labels[column]
            if value is not None
        ),
        None,
    )

    if isinstance(first_valid, np.ndarray):
        description = (
            f"ndarray shape={first_valid.shape}, "
            f"dtype={first_valid.dtype}"
        )
    elif isinstance(first_valid, (list, tuple)):
        description = (
            f"{type(first_valid).__name__} "
            f"length={len(first_valid)}"
        )
    else:
        description = type(first_valid).__name__

    print(f"{column:35s} {str(labels[column].dtype):20s} {description}")

print()
print("=" * 100)
print("PRIMERA FILA COMPLETA")
print("=" * 100)

first_row = labels.iloc[0].to_dict()

def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value

print(
    json.dumps(
        {
            key: serializable(value)
            for key, value in first_row.items()
        },
        indent=2,
        ensure_ascii=False,
        default=str,
    )
)

print()
print("=" * 100)
print("COLUMNAS DE TRAIN.PARQUET")
print("=" * 100)

for column in metadata.columns:
    print(column)

sequence_meta = metadata[
    metadata["sequence_id"].astype(str) == sequence_id
].copy()

print()
print("Filas metadata de la secuencia:", len(sequence_meta))

timestamp_columns = [
    column
    for column in (
        "rgb_exposure_start_timestamp_us",
        "rgb_exposure_end_timestamp_us",
    )
    if column in sequence_meta.columns
]

if len(timestamp_columns) == 2:
    sequence_meta["timestamp_s"] = (
        sequence_meta[timestamp_columns].mean(axis=1) * 1e-6
    )

join_column = None

for candidate in (
    "sample_token",
    "frame_id",
    "sample_id",
    "token",
):
    if candidate in labels.columns and candidate in sequence_meta.columns:
        join_column = candidate
        break

print("Columna de unión:", join_column)

if join_column is None:
    print(
        "No se encontró una columna de unión directa. "
        "Se necesita inspeccionar la primera fila mostrada."
    )
    raise SystemExit(0)

keep_metadata = [
    join_column,
    "timestamp_s",
    "T_event_ego",
    "K_event",
]

keep_metadata = [
    column
    for column in keep_metadata
    if column in sequence_meta.columns
]

joined = labels.merge(
    sequence_meta[keep_metadata],
    on=join_column,
    how="left",
    validate="many_to_one",
)

print("Labels unidos:", len(joined))
print(
    "Labels con timestamp:",
    int(joined["timestamp_s"].notna().sum())
    if "timestamp_s" in joined.columns else 0,
)

track_column = None

for candidate in ("track_id", "instance_id"):
    if candidate in joined.columns:
        track_column = candidate
        break

if track_column is None or "timestamp_s" not in joined.columns:
    print("No se puede seleccionar una trayectoria temporal.")
    raise SystemExit(0)

track_counts = joined.groupby(track_column).size()
track_id = track_counts[track_counts >= 8].sort_values().index[-1]

track = joined[
    joined[track_column] == track_id
].sort_values("timestamp_s").copy()

print()
print("=" * 100)
print("TRAYECTORIA DE EJEMPLO")
print("=" * 100)
print("Track:", track_id)
print("Muestras:", len(track))

wanted = [
    "timestamp_s",
    "translation",
    "size",
    "yaw",
    "rotation",
    "velocity",
    "velocity_3d",
    "ego_translation",
]

wanted = [column for column in wanted if column in track.columns]

for _, row in track[wanted].head(12).iterrows():
    record = {
        key: serializable(value)
        for key, value in row.to_dict().items()
    }
    print(json.dumps(record, ensure_ascii=False, default=str))

if "translation" in track.columns:
    translations = np.stack(
        track["translation"].map(np.asarray).to_numpy()
    ).astype(np.float64)

    times = track["timestamp_s"].to_numpy(dtype=np.float64)
    delta_t = np.diff(times)

    valid = delta_t > 1e-6

    finite_difference = (
        np.diff(translations, axis=0)[valid]
        / delta_t[valid, None]
    )

    print()
    print("=" * 100)
    print("VELOCIDAD DERIVADA DE TRANSLATION")
    print("=" * 100)
    print("Mediana:", np.median(finite_difference, axis=0))
    print("Media:", np.mean(finite_difference, axis=0))
    print("Primeras cinco:")
    print(finite_difference[:5])

    for velocity_column in ("velocity", "velocity_3d"):
        if velocity_column not in track.columns:
            continue

        velocities = np.stack(
            track[velocity_column].map(np.asarray).to_numpy()
        ).astype(np.float64)

        velocities = velocities[1:][valid]

        dimensions = min(
            finite_difference.shape[1],
            velocities.shape[1],
        )

        error = (
            finite_difference[:, :dimensions]
            - velocities[:, :dimensions]
        )

        print()
        print(f"Comparación con {velocity_column}:")
        print(
            "MAE por eje:",
            np.mean(np.abs(error), axis=0),
        )
        print(
            "RMSE por eje:",
            np.sqrt(np.mean(error ** 2, axis=0)),
        )

output = Path("artifacts/audit/eap/eap_ttc_semantics.json")
output.parent.mkdir(parents=True, exist_ok=True)

output.write_text(
    json.dumps(
        {
            "sequence_id": sequence_id,
            "label_path": str(label_path),
            "label_columns": list(labels.columns),
            "metadata_columns": list(metadata.columns),
            "join_column": join_column,
            "track_column": track_column,
            "example_track": str(track_id),
        },
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

print()
print("Informe:", output)
print("PASS: inspección semántica completada.")
