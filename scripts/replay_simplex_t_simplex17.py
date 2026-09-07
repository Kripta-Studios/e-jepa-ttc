"""Replay historical SIMPLEX17 endpoints; never fit or modify historical artifacts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.data.frozen_expert_tables_v10 import ModelInputs
from e_jepa_ttc.simplex_t.coordination import shared_write_admission, verified_ack
from e_jepa_ttc.simplex_t.current_inputs import load_current_inputs
from e_jepa_ttc.simplex_t.lifecycle import ExclusiveLease, admitted
from e_jepa_ttc.training.risk_router_v10 import predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.other_reserved_bytes < 0:
        raise ValueError("new output and explicit nonnegative reservations required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    work = Path(paths["worktree"]).resolve(strict=True)
    if not args.output.resolve().is_relative_to(work):
        raise ValueError("outputs must stay in companion worktree")
    ack_hash = "3e55ab3c6e9a57eecd862ad05e999627ea90957e58e329b2eb3652120e953318"
    ack = verified_ack(Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json", ack_hash)
    ancestry = ack["producers"]["authoritative_historical_manifest"]
    historical = Path(ancestry["path"]).parent
    phase = historical / "stage67_seeds7"
    endpoint_manifest = phase / "ALL_ENDPOINTS_FROZEN.json"
    endpoints = json.loads(endpoint_manifest.read_text(encoding="utf-8"))
    eval_manifest = phase / "ALL_EVALUATIONS.json"
    # Import only the frozen prediction file binding, not historical decision statistics.
    original_binding = json.loads(eval_manifest.read_text(encoding="utf-8"))["frames"][
        "S67-SIMPLEX17"
    ]
    original_path = phase / "S67-SIMPLEX17_OOF.csv"
    if sha256(original_path) != original_binding["sha256"]:
        raise ValueError("historical SIMPLEX17 prediction bytes changed")
    reference = pd.read_csv(
        original_path,
        usecols=lambda c: c in {"sample_token", "prediction_ttc_s", "selected_expert"},
    )
    if len(reference) != 8192 or reference.sample_token.duplicated().any():
        raise ValueError("historical SIMPLEX17 OLD8192 coverage differs")
    reference = reference.set_index("sample_token")
    parts, bindings, seconds, deltas = [], [], [], []

    def resource_check() -> None:
        snapshot = admitted([work])
        if not snapshot["has_headroom"] or not shared_write_admission(
            snapshot["written_volume_free_bytes"][0], args.other_reserved_bytes + 67_108_864
        ):
            raise RuntimeError("RESOURCE_PAUSE: no optimizer state; retry in a new directory")

    with ExclusiveLease(args.output.with_suffix(".lock")):
        for fold in range(3):
            resource_check()
            start = time.monotonic()
            key = f"S67-SIMPLEX17_seed7_outer{fold}"
            endpoint = endpoints[key]
            expected_path = (phase / key / "update1500.pt").resolve(strict=True)
            if (
                Path(endpoint["path"]).resolve(strict=True) != expected_path
                or sha256(expected_path) != endpoint["sha256"]
            ):
                raise ValueError("historical endpoint binding differs")
            table = load_current_inputs(
                historical,
                fold,
                "outer_dev",
                ancestry_sha256=ancestry["sha256"],
                allowed_sequences=set(ack["interfaces"]["role_manifest"]["roles"]["original"]),
            )
            arrays = table["arrays"]
            inputs = ModelInputs(arrays["features17"], arrays["expert_phase"], arrays["expert_ttc"])
            _, selected, prediction, _ = predict(endpoint, inputs)
            metadata = table["metadata"]
            old = reference.loc[metadata.sample_token]
            if not np.array_equal(selected, old.selected_expert.to_numpy()) or not np.array_equal(
                prediction.astype(np.float32), old.prediction_ttc_s.to_numpy(np.float32)
            ):
                raise ValueError(
                    "SIMPLEX17 historical selected-expert or original FP32 TTC mismatch"
                )
            frame = pd.read_csv(
                historical / f"tables/outer{fold}_outer_dev.csv",
                usecols=lambda c: (
                    c in {"sample_token", "sequence_id", "track_id", "target_ttc", "outer_fold"}
                ),
            )
            if not np.array_equal(frame.sample_token, metadata.sample_token):
                raise ValueError("historical identity ordering changed")
            frame["arm"], frame["seed"] = "S67-SIMPLEX17", 7
            frame["prediction_ttc_s"], frame["selected_expert"] = prediction, selected
            for column, expert in enumerate(("a5", "c2f", "pair")):
                frame[f"{expert}_ttc_s"] = arrays["expert_ttc"][:, column]
            parts.append(frame)
            bindings.append({"fold": fold, "endpoint": endpoint, "table": table["reference"]})
            seconds.append(time.monotonic() - start)
            deltas.append(float(np.max(np.abs(prediction - old.prediction_ttc_s.to_numpy()))))
        combined = pd.concat(parts).sort_values("sample_token").reset_index(drop=True)
        if len(combined) != 8192 or combined.sample_token.duplicated().any():
            raise ValueError("combined SIMPLEX17 OLD coverage differs")
        resource_check()
        args.output.mkdir()
        payload = args.output / "SIMPLEX17_OLD.parquet"
        combined.to_parquet(payload, index=False)
        write_new_json(
            args.output / "REPLAY.json",
            {
                "status": "HISTORICAL_SIMPLEX17_OLD_REPLAY_EXACT_FP32",
                "queries": 8192,
                "optimizer_updates": 0,
                "scientific_stage_authorized": False,
                "ack_sha256": ack_hash,
                "ancestry_sha256": ancestry["sha256"],
                "endpoint_manifest_sha256": sha256(endpoint_manifest),
                "evaluation_manifest_sha256": sha256(eval_manifest),
                "historical_oof_sha256": sha256(original_path),
                "payload_sha256": sha256(payload),
                "runner_sha256": sha256(Path(__file__)),
                "bindings": bindings,
                "fold_seconds": seconds,
                "csv_float64_max_abs_difference_by_fold": deltas,
            },
        )
        print(
            json.dumps(
                {
                    "status": "HISTORICAL_SIMPLEX17_OLD_REPLAY_EXACT_FP32",
                    "queries": 8192,
                    "optimizer_updates": 0,
                }
            )
        )


if __name__ == "__main__":
    main()
