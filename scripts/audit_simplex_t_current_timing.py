"""Audit D0 or pinned expansion exposure dependencies; never infer zero observation age."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.coordination import verified_ack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cohort = parser.add_mutually_exclusive_group()
    cohort.add_argument("--expansion-metadata", type=Path)
    cohort.add_argument("--dense-old-pool-plan", type=Path)
    args = parser.parse_args()
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    config = json.loads(
        Path("configs/experiment/simplex_t_coordination.json").read_text(encoding="utf-8")
    )
    ack = verified_ack(
        Path(paths["shared_coordination"]) / config["ack_filename"], config["ack_sha256"]
    )
    charter = json.loads(
        Path(ack["interfaces"]["time_charter"]["path"]).read_text(encoding="utf-8")
    )
    garl = Path(charter["metadata_source"]["path"])
    if sha256(garl) != charter["metadata_source"]["sha256"]:
        raise ValueError("charter input metadata changed")
    if args.dense_old_pool_plan is not None:
        if sha256(args.dense_old_pool_plan) != (
            "b0685050b799058e6090d6e3b7c47b653f2939ca7db2d75a93526ab236ec9e3c"
        ):
            raise ValueError("nominal matched pool plan changed")
        plan = json.loads(args.dense_old_pool_plan.read_text(encoding="utf-8"))
        allowed = set(ack["interfaces"]["role_manifest"]["roles"]["original"])
        tokens = set()
        for fold in plan["folds"]:
            selected = fold["pools"]["DENSE_OLD"]
            if not set(selected["sequences"]) <= allowed:
                raise ValueError("dense query plan includes a closed group")
            tokens.update(selected["tokens"])
        if not tokens:
            raise ValueError("empty dense-old pool")
    elif args.expansion_metadata is None:
        historical = Path(ack["producers"]["authoritative_historical_manifest"]["path"]).parent
        index = json.loads(
            (historical / "FROZEN_EXPERT_TABLE_INDEX.json").read_text(encoding="utf-8")
        )
        tokens = set()
        for role in ("inner_oof", "outer_dev"):
            record = next(r for r in index if r["outer_fold"] == 0 and r["role"] == role)
            path = historical / "tables" / f"outer0_{role}.csv"
            if sha256(path) != record["metadata_sha256"]:
                raise ValueError("historical D0 identity changed")
            tokens.update(
                pd.read_csv(path, usecols=lambda column: column == "sample_token").sample_token
            )
        if len(tokens) != 8192:
            raise ValueError("D0 cohort changed")
        allowed = set(ack["interfaces"]["role_manifest"]["roles"]["original"])
    else:
        if sha256(args.expansion_metadata) != (
            "e8514492952a3abd86e45e3b00c07e398d2e6def5b73062eb5b5f8f13a932064"
        ):
            raise ValueError("expansion metadata pin changed")
        selected = pd.read_parquet(args.expansion_metadata, columns=["sample_token", "sequence_id"])
        allowed = set(ack["interfaces"]["role_manifest"]["roles"]["expansion"])
        tokens = set(selected.sample_token)
        if len(tokens) != 27307 or len(selected) != len(tokens):
            raise ValueError("expansion cohort changed")
        if set(selected.sequence_id) != allowed:
            raise ValueError("expansion roles mismatch")
    pairs = (
        ds.dataset(garl)
        .to_table(
            columns=[
                "sample_token",
                "sequence_id",
                "frame_timestamps_us",
                "event_windows_us",
                "rgb_member_paths",
            ],
            filter=ds.field("sample_token").isin(tokens),
        )
        .to_pylist()
    )
    observed_tokens = [pair["sample_token"] for pair in pairs]
    if len(observed_tokens) != len(tokens) or set(observed_tokens) != tokens:
        raise ValueError("input metadata is missing, duplicated or unexpected")
    media_path = Path(paths["eap_root"]) / "data/train.parquet"
    media = (
        ds.dataset(media_path)
        .to_table(
            columns=[
                "sequence_id",
                "rgb_member_path",
                "rgb_exposure_start_timestamp_us",
                "rgb_exposure_end_timestamp_us",
            ],
            filter=ds.field("sequence_id").isin(allowed),
        )
        .to_pylist()
    )
    frames = {(r["sequence_id"], r["rgb_member_path"]): r for r in media}
    if len(frames) != len(media):
        raise ValueError("ambiguous media reference")
    rows = []
    for pair in pairs:
        if pair["sequence_id"] not in allowed:
            raise ValueError("closed-role query")
        windows, timestamps = pair["event_windows_us"], pair["frame_timestamps_us"]
        if len(windows) != 2 or len(timestamps) != 2:
            raise ValueError("unexpected frozen pair selection")
        offsets = [int(t) - int(w[1]) for t, w in zip(timestamps, windows, strict=True)]
        if abs(offsets[1] - offsets[0]) > 5:
            raise ValueError("frame/event conversion exceeds charter drift")
        anchor = int(windows[1][1])
        selected = [frames[(pair["sequence_id"], member)] for member in pair["rgb_member_paths"]]
        starts = [int(frame["rgb_exposure_start_timestamp_us"]) for frame in selected]
        ends = [int(frame["rgb_exposure_end_timestamp_us"]) for frame in selected]
        if starts != [int(w[1]) for w in windows] or any(
            e < s for s, e in zip(starts, ends, strict=True)
        ):
            raise ValueError("exposure/selected-window mapping mismatch")
        sensor_end = max(int(w[1]) for w in windows)
        dependency_end = max(sensor_end, *ends)
        rows.append(
            {
                "sample_token": pair["sample_token"],
                "anchor_us": anchor,
                "sensor_window_end_us": sensor_end,
                "selected_exposure_end_us": max(ends),
                "exposure_dependency_age_us": dependency_end - anchor,
                "h1_timing_if_exposure_end_is_cutoff": [
                    0.0,
                    0.0,
                    0.0,
                    (dependency_end - anchor) / 1e6,
                ],
            }
        )
    ages = np.array([r["exposure_dependency_age_us"] for r in rows])
    write_new_json(
        args.output,
        {
            "rows": rows,
            "queries": len(rows),
            "garl_metadata_sha256": charter["metadata_source"]["sha256"],
            "eap_frame_metadata_sha256": sha256(media_path),
            "age_us_min": int(ages.min()),
            "age_us_max": int(ages.max()),
            "age_us_median": float(np.median(ages)),
            "zero_age_queries": int((ages == 0).sum()),
            "scope": "EXPOSURE_DEPENDENCY_AUDIT_NOT_FINAL_PRODUCER_CUTOFF",
            "annotation_online_generation_latency_known": False,
            "historical_expert_cutoff_parity_validated": False,
            "scientific_updates": 0,
            "targets_read": False,
            "population": (
                "DENSE_OLD_UNION"
                if args.dense_old_pool_plan is not None
                else "D0"
                if args.expansion_metadata is None
                else "D1_EXPANSION_ONLY"
            ),
            "expansion_owner_time_ack_implied": False,
        },
    )
    print(
        json.dumps(
            {
                "queries": len(rows),
                "age_min_us": int(ages.min()),
                "age_max_us": int(ages.max()),
                "zero_age_queries": int((ages == 0).sum()),
            }
        )
    )


if __name__ == "__main__":
    main()
