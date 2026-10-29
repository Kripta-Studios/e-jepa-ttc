"""Generate fixed baselines from one verified original fold cache; never score or fit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.simplex_t.context_sources import load_context_sources
from e_jepa_ttc.simplex_t.coordination import shared_write_admission, verified_ack
from e_jepa_ttc.simplex_t.current_inputs import load_current_inputs
from e_jepa_ttc.simplex_t.fixed_baselines import predict_fixed_baselines
from e_jepa_ttc.simplex_t.lifecycle import ExclusiveLease, admitted


def main() -> None:
    """Publish input-only predictions, separately for TRAIN and OLD_DEV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--compiled-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.other_reserved_bytes < 0:
        raise ValueError("new output and nonnegative explicit reservations required")
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    work = Path(paths["worktree"]).resolve(strict=True)
    if not args.output.resolve().is_relative_to(work):
        raise ValueError("baseline output must remain inside companion worktree")
    temporal = work / "artifacts/simplex_t/T1"
    ack_path = Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json"
    ack_hash = "3e55ab3c6e9a57eecd862ad05e999627ea90957e58e329b2eb3652120e953318"
    ack = verified_ack(ack_path, ack_hash)
    ancestry = ack["producers"]["authoritative_historical_manifest"]
    root = Path(ancestry["path"]).parent
    roles = set(ack["interfaces"]["role_manifest"]["roles"]["original"])
    compiled_manifest = args.compiled / "COMPILED.json"
    observations = []

    def validate() -> None:
        verified_ack(ack_path, ack_hash)
        if compute_file_hash(str(compiled_manifest)) != args.compiled_sha256:
            raise ValueError("compiled source changed")
        if compute_file_hash(str(temporal / "query_context_index/INDEX_MANIFEST.json")) != (
            "93a4f62e5025c5046fc82fcb1428a428f8a8df869b34486b92d5c753b0f68a3b"
        ):
            raise ValueError("original context amendment changed")

    def resource_ok() -> bool:
        snapshot = admitted([work])
        observations.append(snapshot)
        return snapshot["has_headroom"] and shared_write_admission(
            snapshot["written_volume_free_bytes"][0], args.other_reserved_bytes + 67_108_864
        )

    validate()
    if not resource_ok():
        raise InterruptedError("fixed baseline preparation resource pause before source loading")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    started = time.perf_counter()
    outer = json.loads(compiled_manifest.read_text(encoding="utf-8"))["outer"]
    with ExclusiveLease(temporal / "FIXED_BASELINE_WRITER.lock"):
        sources = load_context_sources(
            args.compiled,
            temporal / "query_context_index",
            temporal / "query_context_dedup",
            root,
            compiled_manifest_sha256=args.compiled_sha256,
            ancestry_sha256=ancestry["sha256"],
            allowed_sequences=roles,
            feature_count=17,
        )
        args.output.mkdir(parents=True)
        receipts = {}
        for role, source in sources.items():
            predictions = predict_fixed_baselines(
                source,
                expected_source_sha256=source.identity_sha256,
                validate_prerequisites=validate,
                resource_ok=resource_ok,
            )
            table = load_current_inputs(
                root, outer, role, ancestry_sha256=ancestry["sha256"], allowed_sequences=roles
            )["metadata"]
            validate()
            if not resource_ok():
                raise InterruptedError("fixed baseline publication resource pause")
            destination = args.output / f"{role}.npz"
            with destination.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    sample_token=table.sample_token.to_numpy(dtype=str),
                    sequence_id=table.sequence_id.to_numpy(dtype=str),
                    history=source.history[:, -8:],
                    CURRENT_MEDIAN_prediction_phase=predictions["CURRENT_MEDIAN"][
                        "prediction_phase"
                    ],
                    CURRENT_MEDIAN_prediction_ttc_s=predictions["CURRENT_MEDIAN"][
                        "prediction_ttc_s"
                    ],
                    EWMA_0P3S_H8_prediction_phase=predictions["EWMA_0P3S_H8"]["prediction_phase"],
                    EWMA_0P3S_H8_prediction_ttc_s=predictions["EWMA_0P3S_H8"]["prediction_ttc_s"],
                )
            receipts[role] = {
                "path": destination.name,
                "sha256": compute_file_hash(str(destination)),
                "queries": source.population,
                "source_sha256": source.identity_sha256,
            }
        report = {
            "status": "FIXED_BASELINES_ONE_FOLD_COMPLETE_NOT_SCORED",
            "outer_fold": outer,
            "roles": receipts,
            "compiled_sha256": args.compiled_sha256,
            "ack_sha256": ack_hash,
            "ancestry": ancestry,
            "baseline_code_sha256": compute_file_hash(
                "src/e_jepa_ttc/simplex_t/fixed_baselines.py"
            ),
            "runner_sha256": compute_file_hash(__file__),
            "optimizer_updates": 0,
            "scores_computed": False,
            "ttc_emission_dtype": "float64",
            "seconds": time.perf_counter() - started,
            "resource_observations": observations,
        }
        write_new_json(args.output / "BASELINES.json", report)
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "status",
                        "outer_fold",
                        "roles",
                        "seconds",
                        "optimizer_updates",
                        "scores_computed",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
