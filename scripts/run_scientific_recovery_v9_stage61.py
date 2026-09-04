"""Execute the strict nested Stage 61 A5/C2F/PAIR router at seed 7."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact
from e_jepa_ttc.data.collision_clock_cache import load_canonical_supervision
from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch, load_feature_cache
from e_jepa_ttc.evaluation.collision_clock_bootstrap import paired_hierarchical_mid_bootstrap
from e_jepa_ttc.evaluation.stage61_nested_pair_router import (
    build_router_features,
    phase_from_ttc,
    prediction_frame,
    select_predictions,
    summarize_prediction_frame,
)
from e_jepa_ttc.models.three_expert_router import fit_three_expert_router
from e_jepa_ttc.training.stage61_pair_head import PairHeadTrainingConfig, train_pair_head


def _write_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    signed = sign_artifact(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(signed, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return signed


def _aligned(a5_path: Path, c2f_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    a5 = pd.read_csv(a5_path).sort_values("token_id", kind="stable").reset_index(drop=True)
    c2f = pd.read_csv(c2f_path).sort_values("token_id", kind="stable").reset_index(drop=True)
    if a5["token_id"].astype(str).tolist() != c2f["token_id"].astype(str).tolist():
        raise ValueError("V8 A5/C2F artifacts do not share exact tokens")
    return a5, c2f


def _permutation(frame: pd.DataFrame, seed: int) -> np.ndarray:
    result = np.arange(len(frame), dtype=np.int64)
    sequence = frame["sequence_id"].astype(str).to_numpy()
    track = frame["track_id"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    for name in sorted(np.unique(sequence).tolist()):
        indices = np.flatnonzero(sequence == name)
        for _ in range(10_000):
            candidate = rng.permutation(indices)
            if np.all(candidate != indices) and np.all(track[candidate] != track[indices]):
                result[indices] = candidate
                break
        else:
            raise ValueError(f"PAIR derangement impossible for sequence {name}")
    return result


def _predict_pair(model: torch.nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), 1024):
            batch = PairFeatureBatch(
                torch.as_tensor(features[start : start + 1024], dtype=torch.float32, device=device)
            )
            values.append(model.predict_ttc(batch).cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def _identity(path: Path, family: str) -> dict[str, str]:
    digest = compute_file_hash(str(path))
    return {
        "reference_family": family,
        "path": str(path),
        "file_sha256": digest,
        "artifact_sha256": digest,
    }


def _bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_path: Path,
    right_path: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    return paired_hierarchical_mid_bootstrap(
        left,
        right,
        protocol=protocol,
        candidate_identity=_identity(left_path, "stage61_candidate"),
        reference_identity=_identity(right_path, "stage61_reference"),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    if args.seed != 7:
        raise ValueError("seed 13/23 require missing matched A5/C2F producer universes")
    output = args.output_root
    output.mkdir(parents=True, exist_ok=False)
    reference_contract = json.loads(
        (repo / "configs/protocol/scientific_recovery_v9_eclock_x0_reference.json").read_text()
    )
    supervision = load_canonical_supervision(reference_contract, args.reference_root).rename(
        columns={"sample_token": "token_id"}
    )
    protocol = json.loads(
        (repo / "configs/protocol/scientific_recovery_v9_stage61_pair_router_x2.json").read_text()
    )
    pair_outer: list[pd.DataFrame] = []
    arm_outer: dict[str, list[pd.DataFrame]] = {name: [] for name in ("R1", "R2", "R2_SHUFFLE")}
    fold_records: list[dict[str, Any]] = []
    for outer in range(3):
        outer_root = output / f"outer_fold{outer}_seed{args.seed}"
        inner_pair_parts: list[pd.DataFrame] = []
        for inner in range(3):
            arrays, metadata, manifest = load_feature_cache(
                args.feature_root / f"outer{outer}_inner{inner}.npz"
            )
            nested_path = (
                args.router_root / f"outer_fold{outer}_seed7/a5/inner{inner}/nested_protocol.json"
            )
            nested = json.loads(nested_path.read_text(encoding="utf-8"))
            contract = nested["folds"][0]
            train_mask = metadata["sequence_id"].astype(str).isin(contract["train_sequence_ids"])
            dev_mask = metadata["sequence_id"].astype(str).isin(contract["dev_sequence_ids"])
            joined = metadata.merge(
                supervision, left_on="sample_token", right_on="token_id", validate="one_to_one"
            )
            if (
                joined["sample_token"].astype(str).tolist()
                != metadata["sample_token"].astype(str).tolist()
            ):
                raise ValueError("supervision merge changed cache row order")
            target_phase = phase_from_ttc(joined["target_ttc_s"].to_numpy(dtype=np.float64))
            model, summary = train_pair_head(
                features=arrays["pair_features"][train_mask],
                target_phase=target_phase[train_mask],
                sample_weight=joined.loc[train_mask, "sample_weight"].to_numpy(),
                sequence_ids=metadata.loc[train_mask, "sequence_id"].astype(str).tolist(),
                sample_tokens=metadata.loc[train_mask, "sample_token"].astype(str).tolist(),
                config=PairHeadTrainingConfig(seed=args.seed),
                output_dir=outer_root / "pair" / f"inner{inner}" / "train",
                device=args.device,
                identity={
                    "role": "inner_oof",
                    "outer_fold": outer,
                    "inner_fold": inner,
                    "feature_cache_artifact_sha256": manifest["artifact_sha256"],
                },
            )
            prediction = _predict_pair(model, arrays["pair_features"][dev_mask], args.device)
            part = metadata.loc[dev_mask].copy()
            part["token_id"] = part.pop("sample_token")
            part["prediction_ttc"] = prediction
            part = part.merge(supervision, on="token_id", validate="one_to_one")
            part["inner_fold"] = inner
            part["checkpoint_sha256"] = summary["checkpoint_sha256"]
            inner_path = outer_root / "pair" / f"inner{inner}" / "expert_oof.csv"
            inner_path.parent.mkdir(parents=True, exist_ok=True)
            part.to_csv(inner_path, index=False, lineterminator="\n")
            inner_pair_parts.append(part)
        pair_inner = (
            pd.concat(inner_pair_parts)
            .sort_values("token_id", kind="stable")
            .reset_index(drop=True)
        )
        pair_inner.to_csv(outer_root / "pair" / "inner_oof.csv", index=False, lineterminator="\n")
        a5_inner, c2f_inner = _aligned(
            args.router_root / f"outer_fold{outer}_seed7/a5/inner_oof.csv",
            args.router_root / f"outer_fold{outer}_seed7/c2f/inner_oof.csv",
        )
        pair_inner = (
            pair_inner.set_index("token_id").loc[a5_inner["token_id"].astype(str)].reset_index()
        )
        predictions_inner = np.column_stack(
            (a5_inner["prediction_ttc"], c2f_inner["prediction_ttc"], pair_inner["prediction_ttc"])
        )
        base_inner, phase_inner = build_router_features(
            a5_inner, c2f_inner, predictions_inner[:, 2]
        )
        tokens = tuple(a5_inner["token_id"].astype(str))
        common = {
            "target": a5_inner["target_ttc"].to_numpy(dtype=np.float64),
            "base_weights": a5_inner["sample_weight"].to_numpy(dtype=np.float64),
            "sample_tokens": tokens,
            "seed": args.seed,
        }
        r1 = fit_three_expert_router(
            base_inner, phase_features=False, predictions=predictions_inner, **common
        )
        r2 = fit_three_expert_router(
            phase_inner, phase_features=True, predictions=predictions_inner, **common
        )
        inner_perm = _permutation(a5_inner, args.seed + outer * 1009 + 61)
        shuffled_inner_predictions = predictions_inner.copy()
        shuffled_inner_predictions[:, 2] = predictions_inner[inner_perm, 2]
        _, shuffled_phase_inner = build_router_features(
            a5_inner, c2f_inner, shuffled_inner_predictions[:, 2]
        )
        shuffled = fit_three_expert_router(
            shuffled_phase_inner,
            phase_features=True,
            predictions=shuffled_inner_predictions,
            **common,
        )
        for name, fit in (("R1", r1), ("R2", r2), ("R2_SHUFFLE", shuffled)):
            _write_json(outer_root / "router" / f"{name}_signature.json", fit.signature)
        arrays, metadata, manifest = load_feature_cache(
            args.feature_root / f"outer{outer}_final.npz"
        )
        joined = metadata.merge(
            supervision, left_on="sample_token", right_on="token_id", validate="one_to_one"
        )
        target_phase = phase_from_ttc(joined["target_ttc_s"].to_numpy(dtype=np.float64))
        train_mask = metadata["outer_fold"].to_numpy(dtype=int) != outer
        dev_mask = ~train_mask
        pair_model, pair_summary = train_pair_head(
            features=arrays["pair_features"][train_mask],
            target_phase=target_phase[train_mask],
            sample_weight=joined.loc[train_mask, "sample_weight"].to_numpy(),
            sequence_ids=metadata.loc[train_mask, "sequence_id"].astype(str).tolist(),
            sample_tokens=metadata.loc[train_mask, "sample_token"].astype(str).tolist(),
            config=PairHeadTrainingConfig(seed=args.seed),
            output_dir=outer_root / "pair" / "outer_dev" / "train",
            device=args.device,
            identity={
                "role": "outer_dev",
                "outer_fold": outer,
                "feature_cache_artifact_sha256": manifest["artifact_sha256"],
            },
        )
        pair_prediction = _predict_pair(pair_model, arrays["pair_features"][dev_mask], args.device)
        pair_dev = metadata.loc[dev_mask].copy()
        pair_dev["token_id"] = pair_dev.pop("sample_token")
        pair_dev["prediction_ttc"] = pair_prediction
        pair_dev = pair_dev.merge(supervision, on="token_id", validate="one_to_one")
        pair_dev["checkpoint_sha256"] = pair_summary["checkpoint_sha256"]
        pair_dev.to_csv(outer_root / "pair" / "outer_dev_oof.csv", index=False, lineterminator="\n")
        pair_outer.append(pair_dev)
        a5_dev, c2f_dev = _aligned(
            args.router_root / f"outer_fold{outer}_seed7/a5/outer_dev/expert_oof.csv",
            args.router_root / f"outer_fold{outer}_seed7/c2f/outer_dev/expert_oof.csv",
        )
        pair_dev = pair_dev.set_index("token_id").loc[a5_dev["token_id"].astype(str)].reset_index()
        dev_predictions = np.column_stack(
            (a5_dev["prediction_ttc"], c2f_dev["prediction_ttc"], pair_dev["prediction_ttc"])
        )
        base_dev, phase_dev = build_router_features(a5_dev, c2f_dev, dev_predictions[:, 2])
        shuffle_perm = _permutation(a5_dev, args.seed + outer * 1009 + 62)
        shuffled_dev_predictions = dev_predictions.copy()
        shuffled_dev_predictions[:, 2] = dev_predictions[shuffle_perm, 2]
        _, shuffled_phase_dev = build_router_features(
            a5_dev, c2f_dev, shuffled_dev_predictions[:, 2]
        )
        selected = {
            "R1": select_predictions(r1, base_dev, dev_predictions),
            "R2": select_predictions(r2, phase_dev, dev_predictions),
            "R2_SHUFFLE": select_predictions(
                shuffled, shuffled_phase_dev, shuffled_dev_predictions
            ),
        }
        for name, (prediction, expert_index) in selected.items():
            frame = prediction_frame(a5_dev, prediction)
            frame["outer_fold"] = outer
            frame["selected_expert"] = expert_index
            arm_outer[name].append(frame)
            frame.to_csv(
                outer_root / "router" / f"{name}_outer_oof.csv", index=False, lineterminator="\n"
            )
        fold_records.append(
            {"outer_fold": outer, "inner_rows": len(a5_inner), "outer_rows": len(a5_dev)}
        )
    aggregate = output / "aggregate_seed7"
    aggregate.mkdir()
    paths: dict[str, Path] = {}
    frames: dict[str, pd.DataFrame] = {}
    for name, parts in arm_outer.items():
        frame = pd.concat(parts).sort_values("sample_token", kind="stable").reset_index(drop=True)
        path = aggregate / f"{name}_oof.csv"
        frame.to_csv(path, index=False, lineterminator="\n")
        paths[name], frames[name] = path, frame
    router_r_raw = pd.read_csv(args.router_root / "aggregate_seed7/router_oof_predictions.csv")
    router_r = prediction_frame(router_r_raw, router_r_raw["prediction_ttc"].to_numpy())
    paths["RouterR"] = aggregate / "RouterR_oof.csv"
    router_r.to_csv(paths["RouterR"], index=False, lineterminator="\n")
    frames["RouterR"] = router_r
    comparisons = {}
    for key, reference_name in (
        ("R2_minus_R1", "R1"),
        ("R2_minus_R2_SHUFFLE", "R2_SHUFFLE"),
        ("R2_minus_RouterR", "RouterR"),
    ):
        artifact = _bootstrap(
            frames["R2"],
            frames[reference_name],
            left_path=paths["R2"],
            right_path=paths[reference_name],
            protocol=protocol,
        )
        comparisons[key] = artifact
        _write_json(aggregate / f"bootstrap_{key}.json", artifact)
    delta = {key: value["delta_candidate_minus_reference"] for key, value in comparisons.items()}
    gate = (
        delta["R2_minus_R1"]["mean"] <= -1.0
        and delta["R2_minus_R1"]["ci95_high"] < 0
        and delta["R2_minus_R1"]["probability_delta_lt_zero"] >= 0.9
        and delta["R2_minus_R2_SHUFFLE"]["mean"] < 0
        and delta["R2_minus_R2_SHUFFLE"]["ci95_high"] < 0
        and delta["R2_minus_R2_SHUFFLE"]["probability_delta_lt_zero"] >= 0.9
        and delta["R2_minus_RouterR"]["mean"] <= -3.0
        and delta["R2_minus_RouterR"]["ci95_high"] < 0
        and delta["R2_minus_RouterR"]["probability_delta_lt_zero"] >= 0.9
    )
    summaries = {name: summarize_prediction_frame(frame) for name, frame in frames.items()}
    integrity = (
        summaries["R2"]["rows"] == 8192
        and summaries["R2"]["sequences"] == 9
        and summaries["R2"]["finite_fraction"] == 1.0
        and summaries["R2"]["failure_rate"] == 0.0
    )
    result = _write_json(
        aggregate / "STAGE61_GATE.json",
        {
            "artifact_type": "scientific_recovery_v9_stage61_gate_v1",
            "schema_version": "1.0",
            "evidence_type": "nested_outer_dev",
            "code_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "protocol_version": "stage61_stage62_v1",
            "protocol_sha256": protocol["artifact_sha256"],
            "created_at": datetime.now(UTC).isoformat(),
            "status": "passed" if gate and integrity else "not_passed",
            "seed": args.seed,
            "gate_passed": bool(gate and integrity),
            "integrity_passed": bool(integrity),
            "folds": fold_records,
            "summaries": summaries,
            "comparisons": comparisons,
            "replication_prerequisites": {"seed13": "missing", "seed23": "missing"},
            "outer_dev_evaluations_per_fold": 1,
            "outer_dev_used_for_selection": False,
        },
    )
    return result


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
