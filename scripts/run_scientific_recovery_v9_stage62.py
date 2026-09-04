"""Execute Stage 62 matched GLOBAL/LOCAL/SHUFFLE arms and TIMESWAP."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact
from e_jepa_ttc.data.collision_clock_cache import load_canonical_supervision
from e_jepa_ttc.data.stage61_pair_feature_cache import load_feature_cache
from e_jepa_ttc.evaluation.collision_clock_bootstrap import paired_hierarchical_mid_bootstrap
from e_jepa_ttc.evaluation.stage61_nested_pair_router import (
    phase_from_ttc,
    prediction_frame,
    summarize_prediction_frame,
)
from e_jepa_ttc.evaluation.stage62_local_field import predict_local_field_ttc
from e_jepa_ttc.models.collision_clock_math import benchmark_phase_to_inverse_ttc
from e_jepa_ttc.models.local_temporal_phase_field import LocalTemporalPhaseField
from e_jepa_ttc.training.stage62_local_field import (
    LocalFieldTrainingConfig,
    derange_cross_track,
    global_pool_field,
    time_swap_field,
    train_local_field,
)


def _write(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    signed = sign_artifact(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(signed, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return signed


def _identity(path: Path, family: str) -> dict[str, str]:
    digest = compute_file_hash(str(path))
    return {
        "reference_family": family,
        "path": str(path),
        "file_sha256": digest,
        "artifact_sha256": digest,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    output = args.output_root
    output.mkdir(parents=True, exist_ok=False)
    reference = json.loads(
        (repo / "configs/protocol/scientific_recovery_v9_eclock_x0_reference.json").read_text()
    )
    supervision = load_canonical_supervision(reference, args.reference_root).rename(
        columns={"sample_token": "token_id"}
    )
    protocol = json.loads(
        (repo / "configs/protocol/scientific_recovery_v9_stage61_pair_router_x2.json").read_text()
    )
    arm_parts: dict[str, list[pd.DataFrame]] = {
        name: []
        for name in (
            "X2_A5_REPLAY",
            "X2_GLOBALPOOL",
            "X2_LOCALFIELD",
            "X2_SHUFFLEFIELD",
            "X2_TIMESWAP",
        )
    }
    initialization_hashes: list[str] = []
    for outer in range(3):
        arrays, metadata, manifest = load_feature_cache(
            args.feature_root / f"outer{outer}_final.npz"
        )
        joined = metadata.merge(
            supervision,
            left_on="sample_token",
            right_on="token_id",
            validate="one_to_one",
        )
        target_phase = phase_from_ttc(joined["target_ttc_s"].to_numpy(dtype=np.float64))
        train_mask = metadata["outer_fold"].to_numpy(dtype=int) != outer
        dev_mask = ~train_mask
        features = arrays["patch_features"].astype(np.float32)
        valid = arrays["patch_valid"].astype(bool)
        global_features = global_pool_field(features)
        shuffled_features = features.copy()
        for subset, offset in ((train_mask, 0), (dev_mask, 1)):
            shuffled, _ = derange_cross_track(
                features[subset],
                sequence_ids=metadata.loc[subset, "sequence_id"].astype(str).tolist(),
                track_ids=metadata.loc[subset, "track_id"].astype(str).tolist(),
                seed=args.seed + outer * 1009 + offset,
            )
            shuffled_features[subset] = shuffled
        torch.manual_seed(args.seed)
        initial_model = LocalTemporalPhaseField()
        initial_state = copy.deepcopy(initial_model.state_dict())
        models: dict[str, LocalTemporalPhaseField] = {}
        for arm, arm_features in (
            ("X2-GLOBALPOOL", global_features),
            ("X2-LOCALFIELD", features),
            ("X2-SHUFFLEFIELD", shuffled_features),
        ):
            model, training = train_local_field(
                patch_features=arm_features[train_mask],
                patch_valid=valid[train_mask],
                a5_phase=arrays["a5_phase"][train_mask],
                target_phase=target_phase[train_mask],
                sample_weight=joined.loc[train_mask, "sample_weight"].to_numpy(),
                sequence_ids=metadata.loc[train_mask, "sequence_id"].astype(str).tolist(),
                sample_tokens=metadata.loc[train_mask, "sample_token"].astype(str).tolist(),
                config=LocalFieldTrainingConfig(arm_id=arm, seed=args.seed),
                initial_state=initial_state,
                output_dir=output / f"outer_fold{outer}_seed{args.seed}" / arm / "train",
                device=args.device,
                identity={
                    "outer_fold": outer,
                    "feature_cache_artifact_sha256": manifest["artifact_sha256"],
                },
            )
            models[arm] = model
            initialization_hashes.append(training["identity"]["initialization_sha256"])
        predictions = {
            "X2_GLOBALPOOL": predict_local_field_ttc(
                models["X2-GLOBALPOOL"],
                features=global_features[dev_mask],
                valid=valid[dev_mask],
                a5_phase=arrays["a5_phase"][dev_mask],
                device=args.device,
            ),
            "X2_LOCALFIELD": predict_local_field_ttc(
                models["X2-LOCALFIELD"],
                features=features[dev_mask],
                valid=valid[dev_mask],
                a5_phase=arrays["a5_phase"][dev_mask],
                device=args.device,
            ),
            "X2_SHUFFLEFIELD": predict_local_field_ttc(
                models["X2-SHUFFLEFIELD"],
                features=shuffled_features[dev_mask],
                valid=valid[dev_mask],
                a5_phase=arrays["a5_phase"][dev_mask],
                device=args.device,
            ),
            "X2_TIMESWAP": predict_local_field_ttc(
                models["X2-LOCALFIELD"],
                features=time_swap_field(features[dev_mask]),
                valid=valid[dev_mask],
                a5_phase=arrays["a5_phase"][dev_mask],
                device=args.device,
            ),
        }
        phase = torch.as_tensor(arrays["a5_phase"][dev_mask], dtype=torch.float64)
        predictions["X2_A5_REPLAY"] = torch.reciprocal(
            benchmark_phase_to_inverse_ttc(phase, metric_delta_t_s=0.1)
        ).numpy()
        dev_metadata = pd.DataFrame(
            {
                "token_id": metadata.loc[dev_mask, "sample_token"].astype(str),
                "sequence_id": metadata.loc[dev_mask, "sequence_id"].astype(str),
                "track_id": metadata.loc[dev_mask, "track_id"].astype(str),
                "target_ttc": joined.loc[dev_mask, "target_ttc_s"].to_numpy(),
            }
        )
        for name, prediction in predictions.items():
            frame = prediction_frame(dev_metadata, prediction)
            frame["outer_fold"] = outer
            arm_parts[name].append(frame)
            path = output / f"outer_fold{outer}_seed{args.seed}" / f"{name}_outer_oof.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False, lineterminator="\n")
    if len(set(initialization_hashes)) != 1:
        raise ValueError("matched X2 arms/folds did not share initialization SHA")
    aggregate = output / "aggregate_seed7"
    aggregate.mkdir()
    frames: dict[str, pd.DataFrame] = {}
    paths: dict[str, Path] = {}
    for name, parts in arm_parts.items():
        frames[name] = (
            pd.concat(parts).sort_values("sample_token", kind="stable").reset_index(drop=True)
        )
        paths[name] = aggregate / f"{name}_oof.csv"
        frames[name].to_csv(paths[name], index=False, lineterminator="\n")
    router_raw = pd.read_csv(args.router_root / "aggregate_seed7/router_oof_predictions.csv")
    frames["RouterR"] = prediction_frame(router_raw, router_raw["prediction_ttc"].to_numpy())
    paths["RouterR"] = aggregate / "RouterR_oof.csv"
    frames["RouterR"].to_csv(paths["RouterR"], index=False, lineterminator="\n")
    comparison_pairs = {
        "LOCAL_minus_GLOBAL": ("X2_LOCALFIELD", "X2_GLOBALPOOL"),
        "LOCAL_minus_SHUFFLE": ("X2_LOCALFIELD", "X2_SHUFFLEFIELD"),
        "LOCAL_minus_TIMESWAP": ("X2_LOCALFIELD", "X2_TIMESWAP"),
        "LOCAL_minus_A5": ("X2_LOCALFIELD", "X2_A5_REPLAY"),
        "LOCAL_minus_RouterR": ("X2_LOCALFIELD", "RouterR"),
    }
    comparisons: dict[str, Any] = {}
    for name, (left, right) in comparison_pairs.items():
        artifact = paired_hierarchical_mid_bootstrap(
            frames[left],
            frames[right],
            protocol=protocol,
            candidate_identity=_identity(paths[left], left),
            reference_identity=_identity(paths[right], right),
        )
        comparisons[name] = artifact
        _write(aggregate / f"bootstrap_{name}.json", artifact)
    delta = {name: value["delta_candidate_minus_reference"] for name, value in comparisons.items()}
    local = frames["X2_LOCALFIELD"]
    integrity = (
        len(local) == 8192
        and local["sequence_id"].nunique() == 9
        and bool(local["finite"].all())
        and not bool(local["failure"].any())
    )
    gates = {
        "locality": delta["LOCAL_minus_GLOBAL"]["mean"] <= -1
        and delta["LOCAL_minus_GLOBAL"]["ci95_high"] < 0
        and delta["LOCAL_minus_GLOBAL"]["probability_delta_lt_zero"] >= 0.9,
        "association": delta["LOCAL_minus_SHUFFLE"]["mean"] < 0
        and delta["LOCAL_minus_SHUFFLE"]["ci95_high"] < 0
        and delta["LOCAL_minus_SHUFFLE"]["probability_delta_lt_zero"] >= 0.9,
        "time_order": delta["LOCAL_minus_TIMESWAP"]["mean"] < 0
        and delta["LOCAL_minus_TIMESWAP"]["ci95_high"] < 0,
        "utility": delta["LOCAL_minus_A5"]["mean"] <= -3
        and delta["LOCAL_minus_A5"]["ci95_high"] < 0
        and delta["LOCAL_minus_A5"]["probability_delta_lt_zero"] >= 0.9,
        "system": summarize_prediction_frame(local)["sequence_macro_paper_MiD_overall"]
        < summarize_prediction_frame(frames["RouterR"])["sequence_macro_paper_MiD_overall"]
        and delta["LOCAL_minus_RouterR"]["ci95_high"] < 0,
    }
    return _write(
        aggregate / "STAGE62_GATE.json",
        {
            "artifact_type": "scientific_recovery_v9_stage62_gate_v1",
            "status": "passed" if integrity and all(gates.values()) else "not_passed",
            "gate_passed": bool(integrity and all(gates.values())),
            "integrity_passed": integrity,
            "gates": gates,
            "summaries": {
                name: summarize_prediction_frame(frame) for name, frame in frames.items()
            },
            "comparisons": comparisons,
            "initialization_sha256": initialization_hashes[0],
            "outer_dev_evaluations_per_fold": 1,
            "outer_dev_used_for_selection": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
