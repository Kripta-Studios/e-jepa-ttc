"""Audit the one-to-one linkage between GarlTTC parquets and eAP media.

This script verifies that the official GarlTTC annotations can be
deterministically joined to the GarlTTC data parquet using five exact
keys, and that every referenced eAP media file exists on disk. It
produces a reproducible JSON audit artifact with row counts, TTC
distribution statistics, SHA256 hashes, and a PASS/FAIL verdict.

IMPORTANT: This script never opens ``test_inputs.parquet``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash  # noqa: E402
from e_jepa_ttc.data.eap import EAPEventReader, build_eap_temporal_windows  # noqa: E402
from e_jepa_ttc.data.garlttc_eap import (  # noqa: E402
    GARLTTC_JOIN_KEYS,
    normalize_boxes_xyxy,
    normalize_event_windows_us,
    resolve_eap_events_path,
)
from e_jepa_ttc.utils.io import read_structured  # noqa: E402

DEFAULT_EXPECTED_TRAIN_ROWS = 88744


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_string(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git_commit() -> str | None:
    try:
        repo_root = Path(__file__).resolve().parents[3]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        return None


def _validate_boxes(boxes_raw: object) -> tuple[bool, str]:
    """Return (valid, reason) for a boxes_xyxy entry using the shared normalizer."""
    if boxes_raw is None:
        return False, "null"
    try:
        boxes = normalize_boxes_xyxy(boxes_raw)
    except ValueError as exc:
        return False, str(exc)

    if not boxes:
        return False, "empty"

    x0, y0, x1, y1 = boxes[-1]
    if not all(np.isfinite([x0, y0, x1, y1])):
        return False, "non_finite"
    if (x1 - x0) <= 0 or (y1 - y0) <= 0:
        return False, "non_positive_area"
    return True, "ok"


def audit(
    *,
    eap_root: Path,
    garlttc_root: Path,
    eap_split_path: Path,
    expected_train_rows: int = DEFAULT_EXPECTED_TRAIN_ROWS,
    allow_dataset_version_change: bool = False,
) -> dict[str, Any]:
    """Run the full GarlTTC ↔ eAP linkage audit and return the result dict."""

    errors: list[str] = []
    warnings: list[str] = []

    data_parquet = garlttc_root / "data" / "train.parquet"
    annotations_parquet = garlttc_root / "annotations" / "train.parquet"

    for required in (data_parquet, annotations_parquet):
        if not required.is_file():
            errors.append(f"Required file missing: {required}")

    test_inputs_opened = False

    if errors:
        return {
            "result": "FAIL",
            "errors": errors,
            "test_inputs_opened": test_inputs_opened,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }

    data_df = pd.read_parquet(data_parquet)
    ann_df = pd.read_parquet(annotations_parquet)

    data_rows = len(data_df)
    ann_rows = len(ann_df)

    # Pre-merge checks on required columns and join keys nulls
    for name, df in [("data", data_df), ("annotations", ann_df)]:
        for key in GARLTTC_JOIN_KEYS:
            if key not in df.columns:
                errors.append(f"Missing join key '{key}' in {name}/train.parquet")
            elif df[key].isnull().any():
                errors.append(f"Null values in join key '{key}' in {name}/train.parquet")

        if all(key in df.columns for key in GARLTTC_JOIN_KEYS):
            duplicates = df.duplicated(subset=GARLTTC_JOIN_KEYS, keep=False)
            n_dup = int(duplicates.sum())
            if n_dup > 0:
                errors.append(f"{name}/train.parquet has {n_dup} duplicate rows on join keys")

    if errors:
        return {
            "result": "FAIL",
            "errors": errors,
            "data_rows": data_rows,
            "annotations_rows": ann_rows,
            "test_inputs_opened": test_inputs_opened,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }

    # Strict outer merge diagnostic
    try:
        diagnostic = pd.merge(
            data_df,
            ann_df,
            on=GARLTTC_JOIN_KEYS,
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
    except pd.errors.MergeError as exc:
        errors.append(f"Outer merge validation failed: {exc}")
        return {
            "result": "FAIL",
            "errors": errors,
            "data_rows": data_rows,
            "annotations_rows": ann_rows,
            "test_inputs_opened": test_inputs_opened,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }

    left_only_count = int((diagnostic["_merge"] == "left_only").sum())
    right_only_count = int((diagnostic["_merge"] == "right_only").sum())

    if left_only_count > 0:
        errors.append(f"{left_only_count} rows in data.parquet unlinked to annotations")
    if right_only_count > 0:
        errors.append(f"{right_only_count} rows in annotations.parquet unlinked to data")

    merged = pd.merge(
        data_df,
        ann_df,
        on=GARLTTC_JOIN_KEYS,
        how="inner",
        validate="one_to_one",
    )
    merged_rows = len(merged)

    if "ttc" not in merged.columns and "ttc_time" in merged.columns:
        merged["ttc"] = merged["ttc_time"]

    raw_ttc = merged["ttc"]
    numeric_ttc = pd.to_numeric(raw_ttc, errors="coerce")
    non_numeric_mask = numeric_ttc.isna() & raw_ttc.notna()
    non_numeric_ttc_count = int(non_numeric_mask.sum())
    null_count = int(raw_ttc.isna().sum())

    if non_numeric_ttc_count > 0:
        errors.append(f"{non_numeric_ttc_count} non-numeric TTC values")
    if null_count > 0:
        errors.append(f"{null_count} null TTC values")

    ttc = numeric_ttc.to_numpy(dtype=np.float64)
    inf_count = int(np.isinf(ttc).sum())
    if inf_count > 0:
        errors.append(f"{inf_count} infinite TTC values")
    if inf_count > 0:
        errors.append(f"{inf_count} infinite TTC values")

    ttc_finite = ttc[np.isfinite(ttc)]
    ttc_tolerance = 1e-6
    out_of_range = int(
        np.sum((ttc_finite < -10.0 - ttc_tolerance) | (ttc_finite > 10.0 + ttc_tolerance))
    )
    if out_of_range > 0:
        errors.append(f"{out_of_range} TTC values outside [-10, 10] (tolerance {ttc_tolerance})")

    # eAP split train/validation disjoint check
    split_payload = read_structured(eap_split_path)
    if not verify_artifact_hash(split_payload):
        errors.append("eAP split artifact signature is invalid")
    assignments = split_payload.get("assignments", {})
    train_seqs = set(str(s) for s in assignments.get("train", []))
    val_seqs = set(str(s) for s in assignments.get("validation", []))

    if train_seqs & val_seqs:
        errors.append(f"Train/validation overlap in split: {train_seqs & val_seqs}")

    # Media existence and event window boundary checks
    missing_media: list[str] = []

    context_out_of_bounds_count = 0
    contexts_without_valid_future = 0
    valid_future_count_by_horizon = {100: 0, 250: 0, 500: 0}
    invalid_future_count_by_horizon = {100: 0, 250: 0, 500: 0}
    invalid_event_windows_count = 0

    context_rows = merged[
        ["sequence_id", "timestamp_us", "events_path", "event_windows_us"]
    ].drop_duplicates(subset=["sequence_id", "timestamp_us", "events_path"])

    for ep, group in context_rows.groupby("events_path"):
        try:
            resolved = resolve_eap_events_path(eap_root, ep)
            try:
                reader = EAPEventReader(resolved)
                reader.open()
                for _, r in group.iterrows():
                    parsed_windows = normalize_event_windows_us(r["event_windows_us"])
                    event_reference_end_us = int(parsed_windows[-1][1])
                    windows = build_eap_temporal_windows(
                        reference_end_us=event_reference_end_us,
                        event_window_ms=100,
                        horizons_ms=(100, 250, 500),
                    )

                    context_valid = (
                        windows.context_start_us >= reader.t_start_us
                        and windows.context_end_us <= reader.t_end_us
                    )

                    if not context_valid:
                        context_out_of_bounds_count += 1
                        continue

                    has_valid_future = False
                    for i, (start_us, end_us) in enumerate(windows.future_windows_us):
                        horizon = (100, 250, 500)[i]
                        valid = start_us >= reader.t_start_us and end_us <= reader.t_end_us
                        if valid:
                            valid_future_count_by_horizon[horizon] += 1
                            has_valid_future = True
                        else:
                            invalid_future_count_by_horizon[horizon] += 1

                    if not has_valid_future:
                        contexts_without_valid_future += 1

                reader.close()
            except Exception as e_err:
                errors.append(f"{ep}: Failed to read reader bounds ({e_err})")
        except FileNotFoundError:
            missing_media.append(str(ep))

    # Check physical event_windows_us format
    for _, r in merged.iterrows():
        try:
            normalize_event_windows_us(r.get("event_windows_us"))
        except Exception:
            invalid_event_windows_count += 1

    if missing_media:
        errors.append(f"{len(missing_media)} referenced media files not found on disk")
    if invalid_event_windows_count > 0:
        errors.append(f"{invalid_event_windows_count} rows have invalid event_windows_us")

    # Box validation
    bad_boxes: list[str] = []
    for idx in range(len(merged)):
        row = merged.iloc[idx]
        valid_b, reason_b = _validate_boxes(row.get("boxes_xyxy"))
        if not valid_b:
            bad_boxes.append(f"{row['sample_token']}: {reason_b}")
    if bad_boxes:
        errors.append(f"{len(bad_boxes)} samples with invalid boxes_xyxy")

    if merged_rows != expected_train_rows and not allow_dataset_version_change:
        errors.append(
            f"Expected {expected_train_rows} merged rows but got {merged_rows}. "
            "Pass --allow-dataset-version-change to override."
        )

    data_sha = _sha256_file(data_parquet)
    ann_sha = _sha256_file(annotations_parquet)
    import json

    full_merged_for_hash = merged.sort_values(
        GARLTTC_JOIN_KEYS,
        kind="mergesort",
        ignore_index=True,
    )
    full_join_key_lines = [
        json.dumps(
            [
                (
                    int(row[key])
                    if isinstance(
                        row[key],
                        np.integer,
                    )
                    else str(row[key])
                )
                for key in GARLTTC_JOIN_KEYS
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for _, row in full_merged_for_hash.iterrows()
    ]

    keys_sha = _sha256_string("\n".join(full_join_key_lines))

    unique_contexts = merged.groupby(["sequence_id", "timestamp_us"]).ngroups
    unique_tracks = merged["track_id"].nunique()

    verdict = "PASS" if not errors else "FAIL"

    return {
        "result": verdict,
        "errors": errors,
        "warnings": warnings,
        "data_rows": data_rows,
        "annotations_rows": ann_rows,
        "merged_rows": merged_rows,
        "left_only_count": left_only_count,
        "right_only_count": right_only_count,
        "unique_contexts_count": int(unique_contexts),
        "unique_tracks_count": int(unique_tracks),
        "context_out_of_bounds_count": context_out_of_bounds_count,
        "contexts_without_valid_future": contexts_without_valid_future,
        "valid_future_count_by_horizon": valid_future_count_by_horizon,
        "invalid_future_count_by_horizon": invalid_future_count_by_horizon,
        "invalid_event_windows_count": invalid_event_windows_count,
        "test_inputs_opened": test_inputs_opened,
        "garlttc_data_sha256": data_sha,
        "garlttc_annotations_sha256": ann_sha,
        "garlttc_join_keys_sha256": keys_sha,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eap-root", type=Path, default=Path(r"E:\eAP_dataset"))
    parser.add_argument("--garlttc-root", type=Path, default=Path(r"E:\GarlTTC_dataset"))
    parser.add_argument(
        "--eap-split",
        type=Path,
        default=Path("data/splits/eap_train40_v1.json"),
    )
    parser.add_argument("--expected-train-rows", type=int, default=DEFAULT_EXPECTED_TRAIN_ROWS)
    parser.add_argument("--allow-dataset-version-change", action="store_true")
    parser.add_argument("--output", type=Path, help="Path to save output JSON audit artifact")
    args = parser.parse_args()

    result = audit(
        eap_root=args.eap_root,
        garlttc_root=args.garlttc_root,
        eap_split_path=args.eap_split,
        expected_train_rows=args.expected_train_rows,
        allow_dataset_version_change=args.allow_dataset_version_change,
    )

    formatted_json = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted_json + "\n", encoding="utf-8")
        print(f"Audit report written to {args.output}")
    else:
        print(formatted_json)

    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
