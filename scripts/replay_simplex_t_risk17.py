"""Replay frozen RISK17 CPU weights on OLD development tables, without fitting."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.models.full_regret_router import FullRegretRidge
from e_jepa_ttc.simplex_t.coordination import shared_write_admission, verified_ack
from e_jepa_ttc.simplex_t.current_inputs import load_current_inputs
from e_jepa_ttc.simplex_t.lifecycle import ExclusiveLease, admitted


def main() -> None:
    """Require existing historical fits; emit unchanged selected experts and parity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    if args.other_reserved_bytes < 0 or args.output.exists():
        raise ValueError("nonnegative reservations and a new output directory required")
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    work = Path(paths["worktree"]).resolve(strict=True)
    if not args.output.resolve().is_relative_to(work):
        raise ValueError("write only inside companion worktree")
    ack_hash = "3e55ab3c6e9a57eecd862ad05e999627ea90957e58e329b2eb3652120e953318"
    ack = verified_ack(Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json", ack_hash)
    ancestry = ack["producers"]["authoritative_historical_manifest"]
    historical = Path(ancestry["path"]).parent
    ridge_root = historical / "frozen_audit/extracted_input/run/stage65"
    manifest_path = ridge_root / "ALL_RIDGE_FITS_FROZEN.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["evidence_type"] != "all_router_fits_frozen_before_evaluation":
        raise ValueError("historical ridge freeze evidence required")
    fits = [r for r in manifest["fits"] if r["model"] == "S65-RISK17"]
    if len(fits) != 3 or {r["outer_fold"] for r in fits} != {0, 1, 2}:
        raise ValueError("all three historical RISK17 producers required")
    original_path = ridge_root / "S65-RISK17_oof.csv"
    original = pd.read_csv(
        original_path,
        usecols=lambda name: name in {"sample_token", "prediction_ttc_s", "selected_expert"},
    )
    if len(original) != 8192 or original.sample_token.duplicated().any():
        raise ValueError("historical comparator must cover OLD8192 uniquely")
    original = original.set_index("sample_token")
    parts, bindings, timings, parsing_deltas = [], [], [], []

    def resources() -> None:
        snapshot = admitted([work])
        if not snapshot["has_headroom"] or not shared_write_admission(
            snapshot["written_volume_free_bytes"][0], args.other_reserved_bytes + 67_108_864
        ):
            raise RuntimeError(
                "RESOURCE_PAUSE: no optimizer state; replay may retry in a new directory"
            )

    with ExclusiveLease(args.output.with_suffix(".lock")):
        for record in sorted(fits, key=lambda r: r["outer_fold"]):
            resources()
            start = time.monotonic()
            fold = record["outer_fold"]
            checkpoint = ridge_root / f"outer{fold}/S65-RISK17.npz"
            if (
                sha256(checkpoint) != record["sha256"]
                or record["evidence_type"] != "train_only_frozen_router_fit"
            ):
                raise ValueError("historical router weights or fit role changed")
            table = load_current_inputs(
                historical,
                fold,
                "outer_dev",
                ancestry_sha256=ancestry["sha256"],
                allowed_sequences=set(ack["interfaces"]["role_manifest"]["roles"]["original"]),
            )
            with np.load(checkpoint, allow_pickle=False) as archive:
                model = FullRegretRidge(
                    archive["mean"],
                    archive["scale"],
                    archive["coefficient"],
                    archive["intercept"],
                    float(archive["ridge"]),
                )
            costs = model.predict_regret(table["arrays"]["features17"])
            selected = costs.argmin(1)
            experts = table["arrays"]["expert_ttc"]
            prediction = experts[np.arange(len(selected)), selected]
            metadata = table["metadata"]
            reference = original.loc[metadata.sample_token]
            if not np.array_equal(
                selected, reference.selected_expert.to_numpy()
            ) or not np.array_equal(
                prediction.astype(np.float32), reference.prediction_ttc_s.to_numpy(np.float32)
            ):
                raise ValueError(
                    json.dumps(
                        {
                            "failure_id": "RISK17_REPLAY_PARITY",
                            "fold": fold,
                            "selection_mismatches": int(
                                np.count_nonzero(selected != reference.selected_expert.to_numpy())
                            ),
                            "max_abs_ttc_difference": float(
                                np.max(np.abs(prediction - reference.prediction_ttc_s.to_numpy()))
                            ),
                            "exact_source_fp32": bool(
                                np.array_equal(
                                    prediction.astype(np.float32),
                                    reference.prediction_ttc_s.to_numpy(np.float32),
                                )
                            ),
                            "prediction_dtype": str(prediction.dtype),
                        }
                    )
                )
            frame = metadata[["sample_token", "sequence_id", "target_ttc"]].copy()
            parsing_deltas.append(
                float(np.max(np.abs(prediction - reference.prediction_ttc_s.to_numpy())))
            )
            tracks = pd.read_csv(
                historical / f"tables/outer{fold}_outer_dev.csv",
                usecols=lambda name: name in {"sample_token", "track_id"},
            )
            if not np.array_equal(tracks.sample_token, frame.sample_token):
                raise ValueError("track identity order differs from verified metadata")
            frame["track_id"] = tracks.track_id.to_numpy()
            frame["outer_fold"], frame["arm"], frame["seed"] = fold, "S65-RISK17", 7
            frame["prediction_ttc_s"], frame["selected_expert"] = prediction, selected
            for column, expert in enumerate(("a5", "c2f", "pair")):
                frame[f"{expert}_ttc_s"] = experts[:, column]
            parts.append(frame)
            bindings.append(
                {"fold": fold, "weights_sha256": record["sha256"], "table": table["reference"]}
            )
            timings.append(time.monotonic() - start)
        combined = pd.concat(parts).sort_values("sample_token").reset_index(drop=True)
        if len(combined) != 8192 or combined.sample_token.duplicated().any():
            raise ValueError("replayed OLD coverage differs")
        resources()
        args.output.mkdir()
        payload = args.output / "RISK17_OLD.parquet"
        combined.to_parquet(payload, index=False)
        write_new_json(
            args.output / "REPLAY.json",
            {
                "status": "HISTORICAL_RISK17_OLD_REPLAY_EXACT",
                "queries": 8192,
                "optimizer_updates": 0,
                "ack_sha256": ack_hash,
                "ancestry_sha256": ancestry["sha256"],
                "ridge_manifest_sha256": sha256(manifest_path),
                "historical_oof_sha256": sha256(original_path),
                "payload_sha256": sha256(payload),
                "bindings": bindings,
                "fold_seconds": timings,
                "runner_sha256": sha256(Path(__file__)),
                "parity": (
                    "Exact selection and original expert FP32 TTC representation; "
                    "emitted table values unchanged"
                ),
                "csv_float64_max_abs_difference_by_fold": parsing_deltas,
                "scientific_stage_authorized": False,
            },
        )
        print(
            json.dumps(
                {
                    "status": "HISTORICAL_RISK17_OLD_REPLAY_EXACT",
                    "queries": 8192,
                    "optimizer_updates": 0,
                }
            )
        )


if __name__ == "__main__":
    main()
