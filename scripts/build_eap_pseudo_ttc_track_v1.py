from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build confidence-filtered track-derived pseudo-TTC labels from the "
            "public eAP Hugging Face detection release. These are NOT official "
            "eAP TTC ground-truth labels."
        )
    )
    parser.add_argument(
        "--eap-root",
        type=Path,
        default=Path(r"E:\eAP_dataset"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:\eAP_dataset\derived\pseudo_ttc_track_v1"),
    )
    parser.add_argument("--min-track-frames", type=int, default=30)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--fit-window", type=int, default=11)
    parser.add_argument("--min-near-depth-m", type=float, default=1.0)
    parser.add_argument("--min-abs-range-rate-mps", type=float, default=0.5)
    parser.add_argument("--min-local-r2", type=float, default=0.70)
    parser.add_argument(
        "--max-abs-ttc-s",
        type=float,
        default=10.0,
        help=(
            "Training-window filter. Raw TTC is also retained. This is a "
            "configurable pseudo-label limit, not asserted as an official eAP limit."
        ),
    )
    parser.add_argument(
        "--max-gap-s",
        type=float,
        default=0.25,
        help="Reject local labels next to larger timestamp gaps.",
    )
    return parser.parse_args()


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


def robust_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window < 3 or window % 2 == 0:
        raise ValueError("--smooth-window must be an odd integer >= 3")

    min_periods = max(3, window // 2 + 1)
    return (
        pd.Series(values, dtype="float64")
        .rolling(window=window, center=True, min_periods=min_periods)
        .median()
        .interpolate(limit_direction="both")
        .rolling(window=5, center=True, min_periods=3)
        .mean()
        .interpolate(limit_direction="both")
        .to_numpy(dtype=np.float64)
    )


def local_linear_stats(
    times: np.ndarray,
    values: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    if window < 5 or window % 2 == 0:
        raise ValueError("--fit-window must be an odd integer >= 5")

    n = len(times)
    half = window // 2
    slope = np.full(n, np.nan, dtype=np.float64)
    r2 = np.full(n, np.nan, dtype=np.float64)

    for index in range(half, n - half):
        t = times[index - half : index + half + 1]
        y = values[index - half : index + half + 1]

        valid = np.isfinite(t) & np.isfinite(y)
        if valid.sum() < max(5, window - 2):
            continue

        t = t[valid]
        y = y[valid]
        t_centered = t - t.mean()

        denominator = float(np.dot(t_centered, t_centered))
        if denominator <= 1e-12:
            continue

        beta = float(np.dot(t_centered, y - y.mean()) / denominator)
        prediction = y.mean() + beta * t_centered

        residual_ss = float(np.sum((y - prediction) ** 2))
        total_ss = float(np.sum((y - y.mean()) ** 2))

        slope[index] = beta
        if total_ss <= 1e-12:
            r2[index] = 1.0 if residual_ss <= 1e-12 else 0.0
        else:
            r2[index] = max(0.0, min(1.0, 1.0 - residual_ss / total_ss))

    return slope, r2


def nearest_ego_x_from_box(boxes: np.ndarray) -> np.ndarray:
    # Public release observation, globally audited:
    # bbox_3d_ego = [x, y, z, length, width, height, yaw]
    x = boxes[:, 0]
    length = boxes[:, 3]
    width = boxes[:, 4]
    yaw = boxes[:, 6]

    projected_half_extent_x = 0.5 * (
        np.abs(length * np.cos(yaw))
        + np.abs(width * np.sin(yaw))
    )
    return x - projected_half_extent_x


def confidence_score(
    r2: np.ndarray,
    range_rate: np.ndarray,
    min_speed: float,
) -> np.ndarray:
    r2_component = np.clip(
        (r2 - 0.50) / 0.50,
        0.0,
        1.0,
    )
    speed_component = np.clip(
        (np.abs(range_rate) - min_speed) / max(2.0 - min_speed, 1e-6),
        0.0,
        1.0,
    )
    return np.sqrt(r2_component * speed_component)


def main() -> int:
    args = parse_args()

    eap_root = args.eap_root.resolve()
    metadata_path = eap_root / "data" / "train.parquet"
    label_root = eap_root / "data" / "train"
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    label_files = sorted(label_root.rglob("labels.parquet"))
    if len(label_files) != 40:
        raise RuntimeError(
            f"Expected 40 public train label files, found {len(label_files)}."
        )

    metadata = pq.read_table(
        metadata_path,
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

    summaries: list[dict[str, object]] = []
    total_rows = 0
    total_valid = 0

    for file_index, label_path in enumerate(label_files, start=1):
        sequence_id = label_path.parent.name

        labels = pq.read_table(
            label_path,
            columns=[
                "sample_token",
                "split",
                "sequence_id",
                "frame_name",
                "instance_id",
                "track_id",
                "category",
                "bbox_3d_ego",
            ],
        ).to_pandas()

        joined = labels.merge(
            metadata[metadata["sequence_id"].astype(str) == sequence_id],
            on=["sample_token", "sequence_id"],
            how="left",
            validate="many_to_one",
        )

        output_parts: list[pd.DataFrame] = []

        for track_id, track in joined.groupby("track_id", sort=False):
            track = (
                track.sort_values("timestamp_s")
                .drop_duplicates("timestamp_s")
                .reset_index(drop=True)
            )

            n = len(track)
            boxes = stack_vectors(track["bbox_3d_ego"], 7)
            times = track["timestamp_s"].to_numpy(dtype=np.float64)

            near_depth_raw = nearest_ego_x_from_box(boxes)
            near_depth_smooth = np.full(n, np.nan, dtype=np.float64)
            range_rate = np.full(n, np.nan, dtype=np.float64)
            local_r2 = np.full(n, np.nan, dtype=np.float64)

            enough_track = n >= args.min_track_frames
            finite_base = (
                np.isfinite(times)
                & np.all(np.isfinite(boxes), axis=1)
                & np.isfinite(near_depth_raw)
            )

            if enough_track and finite_base.sum() >= args.min_track_frames:
                near_depth_smooth = robust_smooth(
                    near_depth_raw,
                    args.smooth_window,
                )
                depth_slope, local_r2 = local_linear_stats(
                    times,
                    near_depth_smooth,
                    args.fit_window,
                )
                # Positive means approaching; negative means receding.
                range_rate = -depth_slope

            dt_previous = np.full(n, np.nan, dtype=np.float64)
            dt_next = np.full(n, np.nan, dtype=np.float64)
            if n > 1:
                dt = np.diff(times)
                dt_previous[1:] = dt
                dt_next[:-1] = dt

            contiguous = (
                (np.isnan(dt_previous) | (dt_previous <= args.max_gap_s))
                & (np.isnan(dt_next) | (dt_next <= args.max_gap_s))
            )

            ttc_raw = np.full(n, np.nan, dtype=np.float64)
            nonzero_speed = np.abs(range_rate) > 1e-9
            ttc_raw[nonzero_speed] = (
                near_depth_smooth[nonzero_speed]
                / range_rate[nonzero_speed]
            )

            valid = (
                enough_track
                & finite_base
                & contiguous
                & np.isfinite(near_depth_smooth)
                & np.isfinite(range_rate)
                & np.isfinite(local_r2)
                & np.isfinite(ttc_raw)
                & (near_depth_smooth >= args.min_near_depth_m)
                & (
                    np.abs(range_rate)
                    >= args.min_abs_range_rate_mps
                )
                & (local_r2 >= args.min_local_r2)
                & (np.abs(ttc_raw) <= args.max_abs_ttc_s)
            )

            confidence = confidence_score(
                local_r2,
                range_rate,
                args.min_abs_range_rate_mps,
            )
            confidence[~valid] = 0.0

            reason = np.full(n, "valid", dtype=object)
            reason[~enough_track] = "track_too_short"
            reason[enough_track & ~finite_base] = "non_finite_geometry"
            reason[enough_track & finite_base & ~contiguous] = "timestamp_gap"
            reason[
                enough_track
                & finite_base
                & contiguous
                & (
                    ~np.isfinite(range_rate)
                    | ~np.isfinite(local_r2)
                )
            ] = "window_edge_or_fit_failure"
            reason[
                enough_track
                & finite_base
                & contiguous
                & np.isfinite(range_rate)
                & np.isfinite(local_r2)
                & (near_depth_smooth < args.min_near_depth_m)
            ] = "near_depth_too_small"
            reason[
                enough_track
                & finite_base
                & contiguous
                & np.isfinite(range_rate)
                & np.isfinite(local_r2)
                & (
                    np.abs(range_rate)
                    < args.min_abs_range_rate_mps
                )
            ] = "range_rate_too_small"
            reason[
                enough_track
                & finite_base
                & contiguous
                & np.isfinite(range_rate)
                & np.isfinite(local_r2)
                & (
                    np.abs(range_rate)
                    >= args.min_abs_range_rate_mps
                )
                & (local_r2 < args.min_local_r2)
            ] = "local_fit_low_r2"
            reason[
                enough_track
                & finite_base
                & contiguous
                & np.isfinite(ttc_raw)
                & (np.abs(ttc_raw) > args.max_abs_ttc_s)
            ] = "outside_training_ttc_window"

            part = track[
                [
                    "sample_token",
                    "split",
                    "sequence_id",
                    "frame_name",
                    "instance_id",
                    "track_id",
                    "category",
                    "timestamp_s",
                ]
            ].copy()

            part["near_depth_ego_x_raw_m"] = near_depth_raw
            part["near_depth_ego_x_smooth_m"] = near_depth_smooth
            part["relative_range_rate_track_mps"] = range_rate
            part["local_linear_r2"] = local_r2
            part["pseudo_ttc_s"] = ttc_raw
            part["pseudo_ttc_valid"] = valid
            part["pseudo_ttc_confidence"] = confidence
            part["pseudo_ttc_reason"] = reason
            part["label_source"] = "track_derived_pseudo_ttc_v1"
            part["is_official_eap_ttc_ground_truth"] = False
            part["label_uses_future_context_offline"] = True

            output_parts.append(part)

        sequence_output = pd.concat(output_parts, ignore_index=True)
        sequence_output = sequence_output.sort_values(
            ["timestamp_s", "track_id"],
            kind="stable",
        ).reset_index(drop=True)

        destination = output_root / sequence_id / "pseudo_ttc.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_pandas(sequence_output, preserve_index=False)
        pq.write_table(
            table,
            destination,
            compression="zstd",
            compression_level=9,
        )

        valid_count = int(sequence_output["pseudo_ttc_valid"].sum())
        row_count = len(sequence_output)
        total_rows += row_count
        total_valid += valid_count

        summaries.append(
            {
                "sequence_id": sequence_id,
                "rows": row_count,
                "valid_pseudo_ttc_rows": valid_count,
                "valid_fraction": valid_count / row_count if row_count else 0.0,
                "output": str(destination),
            }
        )

        print(
            f"[{file_index:02d}/{len(label_files):02d}] {sequence_id}: "
            f"{valid_count:,}/{row_count:,} valid"
        )

    manifest = {
        "name": "eAP public detection train track-derived pseudo-TTC v1",
        "official_ground_truth": False,
        "method": (
            "nearest ego-longitudinal 3D-box face divided by an offline, "
            "robustly smoothed track-derived relative range rate"
        ),
        "warning": (
            "The public Hugging Face detection release does not expose the "
            "world-frame ego velocity and official smoothed TTC annotations "
            "described in the paper. Do not call these labels official eAP TTC GT."
        ),
        "parameters": {
            "min_track_frames": args.min_track_frames,
            "smooth_window": args.smooth_window,
            "fit_window": args.fit_window,
            "min_near_depth_m": args.min_near_depth_m,
            "min_abs_range_rate_mps": args.min_abs_range_rate_mps,
            "min_local_r2": args.min_local_r2,
            "max_abs_ttc_s": args.max_abs_ttc_s,
            "max_gap_s": args.max_gap_s,
        },
        "totals": {
            "rows": total_rows,
            "valid_pseudo_ttc_rows": total_valid,
            "valid_fraction": total_valid / total_rows if total_rows else 0.0,
        },
        "sequences": summaries,
    }

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("PSEUDO-TTC BUILD COMPLETE")
    print("=" * 88)
    print(f"Rows:          {total_rows:,}")
    print(f"Valid labels:  {total_valid:,}")
    print(
        f"Valid fraction: "
        f"{(total_valid / total_rows if total_rows else 0.0):.4f}"
    )
    print(f"Output root:   {output_root}")
    print(f"Manifest:      {manifest_path}")
    print("OFFICIAL GT:   False")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
