from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EAP_ROOT = Path(r"E:\eAP_dataset")
LABEL_ROOT = EAP_ROOT / "data" / "train"
METADATA_PATH = EAP_ROOT / "data" / "train.parquet"

OUTPUT_DIR = Path("artifacts/audit/eap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "eap_velocity_semantics_global.json"
OUTPUT_CSV = OUTPUT_DIR / "eap_track_velocity_comparison.csv"


def stack_vectors(series: pd.Series, size: int) -> np.ndarray:
    rows: list[np.ndarray] = []

    for value in series:
        if value is None:
            rows.append(np.full(size, np.nan, dtype=np.float64))
            continue

        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size < size:
            rows.append(np.full(size, np.nan, dtype=np.float64))
        else:
            rows.append(array[:size])

    return np.stack(rows)


def smooth_signal(values: np.ndarray) -> np.ndarray:
    """Robust smoothing for noisy 3D-box trajectories."""
    return (
        pd.Series(values, dtype="float64")
        .rolling(window=9, center=True, min_periods=5)
        .median()
        .interpolate(limit_direction="both")
        .rolling(window=5, center=True, min_periods=3)
        .mean()
        .interpolate(limit_direction="both")
        .to_numpy(dtype=np.float64)
    )


def safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 20:
        return None

    av = a[valid]
    bv = b[valid]

    if np.std(av) < 1e-8 or np.std(bv) < 1e-8:
        return None

    return float(np.corrcoef(av, bv)[0, 1])


if not METADATA_PATH.exists():
    raise FileNotFoundError(METADATA_PATH)

label_files = sorted(LABEL_ROOT.rglob("labels.parquet"))
if len(label_files) != 40:
    raise RuntimeError(
        f"Se esperaban 40 labels.parquet y se encontraron {len(label_files)}."
    )

metadata = pq.read_table(
    METADATA_PATH,
    columns=[
        "sample_token",
        "sequence_id",
        "rgb_exposure_start_timestamp_us",
        "rgb_exposure_end_timestamp_us",
    ],
).to_pandas()

metadata["timestamp_s"] = (
    metadata["rgb_exposure_start_timestamp_us"].astype(np.float64)
    + metadata["rgb_exposure_end_timestamp_us"].astype(np.float64)
) * 0.5e-6

metadata = metadata[
    ["sample_token", "sequence_id", "timestamp_s"]
].drop_duplicates("sample_token")

track_records: list[dict[str, object]] = []
all_dxdt: list[np.ndarray] = []
all_closing: list[np.ndarray] = []
all_vx: list[np.ndarray] = []
all_near_depth: list[np.ndarray] = []

translation_equal_count = 0
translation_comparable_count = 0
size_hlw_count = 0
size_comparable_count = 0

for file_index, label_path in enumerate(label_files, start=1):
    sequence_id = label_path.parent.name

    labels = pq.read_table(
        label_path,
        columns=[
            "sample_token",
            "sequence_id",
            "track_id",
            "category",
            "bbox_3d_ego",
            "translation",
            "size",
            "yaw",
            "velocity",
            "velocity_3d",
            "ego_translation",
        ],
    ).to_pandas()

    sequence_metadata = metadata[
        metadata["sequence_id"].astype(str) == sequence_id
    ]

    joined = labels.merge(
        sequence_metadata,
        on=["sample_token", "sequence_id"],
        how="left",
        validate="many_to_one",
    )

    translations = stack_vectors(joined["translation"], 3)
    ego_translations = stack_vectors(joined["ego_translation"], 3)

    comparable_translation = (
        np.all(np.isfinite(translations), axis=1)
        & np.all(np.isfinite(ego_translations), axis=1)
    )
    translation_comparable_count += int(comparable_translation.sum())
    translation_equal_count += int(
        np.all(
            np.isclose(
                translations[comparable_translation],
                ego_translations[comparable_translation],
                atol=1e-8,
                rtol=0.0,
            ),
            axis=1,
        ).sum()
    )

    boxes = stack_vectors(joined["bbox_3d_ego"], 7)
    sizes = stack_vectors(joined["size"], 3)
    comparable_size = (
        np.all(np.isfinite(boxes), axis=1)
        & np.all(np.isfinite(sizes), axis=1)
    )

    # El release observado usa:
    # bbox_3d_ego = [x, y, z, length, width, height, yaw]
    # size        = [height, length, width]
    expected_hlw = np.column_stack([boxes[:, 5], boxes[:, 3], boxes[:, 4]])
    size_comparable_count += int(comparable_size.sum())
    size_hlw_count += int(
        np.all(
            np.isclose(
                sizes[comparable_size],
                expected_hlw[comparable_size],
                atol=1e-6,
                rtol=1e-5,
            ),
            axis=1,
        ).sum()
    )

    for track_id, track in joined.groupby("track_id", sort=False):
        track = (
            track.dropna(subset=["timestamp_s"])
            .sort_values("timestamp_s")
            .drop_duplicates("timestamp_s")
        )

        if len(track) < 30:
            continue

        times = track["timestamp_s"].to_numpy(dtype=np.float64)
        track_boxes = stack_vectors(track["bbox_3d_ego"], 7)
        track_velocities = stack_vectors(track["velocity_3d"], 3)

        valid_rows = (
            np.isfinite(times)
            & np.all(np.isfinite(track_boxes), axis=1)
            & np.all(np.isfinite(track_velocities), axis=1)
        )

        times = times[valid_rows]
        track_boxes = track_boxes[valid_rows]
        track_velocities = track_velocities[valid_rows]

        if len(times) < 30 or not np.all(np.diff(times) > 0):
            continue

        x = track_boxes[:, 0]
        length = track_boxes[:, 3]
        width = track_boxes[:, 4]
        yaw = track_boxes[:, 6]

        # Profundidad aproximada de la cara más próxima en el eje longitudinal ego.
        half_extent_x = 0.5 * (
            np.abs(length * np.cos(yaw))
            + np.abs(width * np.sin(yaw))
        )
        near_depth = x - half_extent_x

        x_smooth = smooth_signal(x)
        near_smooth = smooth_signal(near_depth)

        dxdt = np.gradient(x_smooth, times)
        closing_speed_track = -np.gradient(near_smooth, times)
        velocity_x_label = track_velocities[:, 0]

        interior = np.zeros(len(times), dtype=bool)
        if len(times) > 10:
            interior[5:-5] = True

        usable = (
            interior
            & np.isfinite(dxdt)
            & np.isfinite(closing_speed_track)
            & np.isfinite(velocity_x_label)
            & np.isfinite(near_smooth)
            & (near_smooth > 0.1)
        )

        if usable.sum() < 20:
            continue

        dxdt_u = dxdt[usable]
        closing_u = closing_speed_track[usable]
        vx_u = velocity_x_label[usable]
        depth_u = near_smooth[usable]

        all_dxdt.append(dxdt_u)
        all_closing.append(closing_u)
        all_vx.append(vx_u)
        all_near_depth.append(depth_u)

        category = str(track["category"].iloc[0])
        track_records.append(
            {
                "sequence_id": sequence_id,
                "track_id": str(track_id),
                "category": category,
                "samples": int(usable.sum()),
                "corr_dxdt_vs_label_vx": safe_corr(dxdt_u, vx_u),
                "corr_negative_dxdt_vs_label_vx": safe_corr(-dxdt_u, vx_u),
                "corr_track_closing_vs_label_vx": safe_corr(closing_u, vx_u),
                "mae_dxdt_vs_label_vx": float(np.mean(np.abs(dxdt_u - vx_u))),
                "mae_negative_dxdt_vs_label_vx": float(
                    np.mean(np.abs(-dxdt_u - vx_u))
                ),
                "median_near_depth_m": float(np.median(depth_u)),
                "median_dxdt_mps": float(np.median(dxdt_u)),
                "median_track_closing_mps": float(np.median(closing_u)),
                "median_label_vx_mps": float(np.median(vx_u)),
            }
        )

    print(
        f"[{file_index:02d}/{len(label_files):02d}] "
        f"{sequence_id}: tracks acumulados={len(track_records)}"
    )

if not track_records:
    raise RuntimeError("No se generó ninguna comparación de trayectorias.")

comparison = pd.DataFrame(track_records)
comparison.to_csv(OUTPUT_CSV, index=False)

dxdt_all = np.concatenate(all_dxdt)
closing_all = np.concatenate(all_closing)
vx_all = np.concatenate(all_vx)
depth_all = np.concatenate(all_near_depth)

valid_track_ttc = np.abs(closing_all) > 0.05
ttc_track = np.full_like(depth_all, np.nan)
ttc_track[valid_track_ttc] = (
    depth_all[valid_track_ttc] / closing_all[valid_track_ttc]
)

valid_label_ttc = np.abs(vx_all) > 0.05
ttc_from_label_vx = np.full_like(depth_all, np.nan)
ttc_from_label_vx[valid_label_ttc] = (
    depth_all[valid_label_ttc] / vx_all[valid_label_ttc]
)

summary = {
    "label_files": len(label_files),
    "tracks_compared": len(track_records),
    "samples_compared": int(dxdt_all.size),
    "translation_equals_ego_translation_fraction": (
        translation_equal_count / translation_comparable_count
        if translation_comparable_count
        else None
    ),
    "size_matches_height_length_width_fraction": (
        size_hlw_count / size_comparable_count
        if size_comparable_count
        else None
    ),
    "velocity_comparison": {
        "corr_dxdt_vs_label_vx": safe_corr(dxdt_all, vx_all),
        "corr_negative_dxdt_vs_label_vx": safe_corr(-dxdt_all, vx_all),
        "corr_track_closing_vs_label_vx": safe_corr(closing_all, vx_all),
        "mae_dxdt_vs_label_vx": float(np.mean(np.abs(dxdt_all - vx_all))),
        "mae_negative_dxdt_vs_label_vx": float(
            np.mean(np.abs(-dxdt_all - vx_all))
        ),
        "median_dxdt_mps": float(np.median(dxdt_all)),
        "median_track_closing_mps": float(np.median(closing_all)),
        "median_label_vx_mps": float(np.median(vx_all)),
    },
    "track_derived_ttc": {
        "finite_fraction": float(np.isfinite(ttc_track).mean()),
        "median_s": float(np.nanmedian(ttc_track)),
        "p01_s": float(np.nanpercentile(ttc_track, 1)),
        "p99_s": float(np.nanpercentile(ttc_track, 99)),
        "positive_fraction": float(np.nanmean(ttc_track > 0)),
    },
    "ttc_from_label_vx_candidate": {
        "finite_fraction": float(np.isfinite(ttc_from_label_vx).mean()),
        "median_s": float(np.nanmedian(ttc_from_label_vx)),
        "p01_s": float(np.nanpercentile(ttc_from_label_vx, 1)),
        "p99_s": float(np.nanpercentile(ttc_from_label_vx, 99)),
        "positive_fraction": float(np.nanmean(ttc_from_label_vx > 0)),
    },
    "interpretation_warning": (
        "The public Hugging Face release is a detection benchmark. "
        "Its velocity labels must not be treated as official TTC ground truth "
        "unless their coordinate frame and relation to ego velocity are documented."
    ),
}

OUTPUT_JSON.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print()
print("=" * 88)
print("RESULTADO GLOBAL")
print("=" * 88)
print(json.dumps(summary, indent=2, ensure_ascii=False))
print()
print(f"CSV:  {OUTPUT_CSV}")
print(f"JSON: {OUTPUT_JSON}")
print("PASS: auditoría global completada.")
