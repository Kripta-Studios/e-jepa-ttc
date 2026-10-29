"""Recover exact historical tables and apply fixed selector parameters, without fits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from run_scientific_recovery_v9_stage65 import (
    _expert_matrix,
    _load_experts,
    _load_fit,
    _load_pair,
    _loss_matrix,
    build_router_features,
    strict_macro_mass,
)


def sha(path: Path) -> str:
    """Hash a compact evidence file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Persist finite JSON atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    """Recover role-specific arrays and verify every legacy selection and prediction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-inputs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    local = json.loads(args.local_inputs.read_text(encoding="utf-8-sig"))
    output = args.output_root
    ancestry = json.loads((output / "NESTED_ANCESTRY_AUDIT.json").read_text())
    if ancestry["status"] != "passed":
        raise ValueError("nested ancestry prerequisite failed")
    reference = Path(local["reference_worktree"]) / "artifacts/scientific_recovery_v8/results"
    router = reference / "router"
    stage61 = (
        Path(local["stage61_worktree"]) / "artifacts/scientific_recovery_v9_stage61_stage62/stage61"
    )
    frozen = output / "frozen_audit/extracted_input/run"
    destination = output / "tables"
    destination.mkdir(exist_ok=False)
    index = []
    equivalence = []
    for fold in range(3):
        for role in ("inner_oof", "outer_dev"):
            a5, c2f = _load_experts(
                *[
                    router
                    / f"outer_fold{fold}_seed7"
                    / expert
                    / ("inner_oof.csv" if role == "inner_oof" else "outer_dev/expert_oof.csv")
                    for expert in ("a5", "c2f")
                ]
            )
            pair = _load_pair(
                stage61
                / f"outer_fold{fold}_seed7/pair"
                / ("inner_oof.csv" if role == "inner_oof" else "outer_dev_oof.csv"),
                cast(pd.Series, a5["token_id"]),
            )
            expert = _expert_matrix(a5, c2f, pair)
            _, features = build_router_features(a5, c2f, expert[:, 2])
            target = a5.target_ttc.to_numpy(np.float64)
            losses = _loss_matrix(target, expert)
            mass = strict_macro_mass(target, a5.sequence_id.astype(str).to_numpy())
            metadata = a5.copy().rename(columns={"token_id": "sample_token"})
            metadata["role"] = role
            metadata["producer_outer_fold"] = fold
            metadata["pair_checkpoint_sha256"] = pair["checkpoint_sha256"].to_numpy()
            stem = destination / f"outer{fold}_{role}"
            metadata.to_csv(stem.with_suffix(".csv"), index=False)
            arrays = dict(
                features17=features.to_numpy(np.float64),
                expert_ttc=expert,
                expert_phase=-np.log1p(-0.1 / expert),
                target_phase=-np.log1p(-0.1 / target),
                global_mass=mass,
                true_costs=losses,
            )
            np.savez(
                stem.with_suffix(".npz"),
                features17=arrays["features17"],
                expert_ttc=arrays["expert_ttc"],
                expert_phase=arrays["expert_phase"],
                target_phase=arrays["target_phase"],
                global_mass=arrays["global_mass"],
                true_costs=arrays["true_costs"],
            )
            record = dict(
                outer_fold=fold,
                role=role,
                rows=len(a5),
                ancestry_sha256=sha(output / "NESTED_ANCESTRY_AUDIT.json"),
                metadata_sha256=sha(stem.with_suffix(".csv")),
                arrays_sha256=sha(stem.with_suffix(".npz")),
                token_order_sha256=hashlib.sha256("\n".join(a5.token_id).encode()).hexdigest(),
                arrays={
                    k: dict(shape=list(v.shape), dtype=str(v.dtype), bytes=v.nbytes)
                    for k, v in arrays.items()
                },
            )
            write_json(stem.with_suffix(".json"), record)
            index.append(record)
            if role != "outer_dev":
                continue
            for arm in ("S65-RISK8", "S65-RISK17", "S65-CE17-REPLAY", "RouterR"):
                x = arrays["features17"]
                if arm.startswith("S65-RISK"):
                    fit = _load_fit(frozen / f"stage65/outer{fold}/{arm}.npz")
                    cost = fit.predict_regret(x[:, :8] if arm.endswith("8") else x)
                    selected = cost.argmin(1)
                    expected = pd.read_csv(frozen / f"final_audit/{arm}_oof.csv")
                else:
                    sigpath = (
                        stage61 / f"outer_fold{fold}_seed7/router/R2_signature.json"
                        if arm != "RouterR"
                        else reference / f"runs/router_fold{fold}_seed7/router_signature.json"
                    )
                    signature = json.loads(sigpath.read_text())
                    size = len(signature["scaler_mean"])
                    logits = (
                        (x[:, :size] - signature["scaler_mean"]) / signature["scaler_scale"]
                    ) @ np.array(signature["coef"]).T + signature["intercept"]
                    selected = (
                        (logits[:, 0] >= 0).astype(int) if arm == "RouterR" else logits.argmax(1)
                    )
                    cost = -logits
                    expected = pd.read_csv(
                        frozen
                        / (
                            "final_audit/S65-ROUTERR-REPLAY_original.csv"
                            if arm == "RouterR"
                            else "final_audit/S61-R2_original.csv"
                        )
                    )
                expected = expected.set_index("sample_token").loc[a5.token_id]
                prediction = expert[np.arange(len(expert)), selected]
                residual = np.abs(prediction - expected.prediction_ttc_s.to_numpy())
                permitted = np.maximum(1e-10, 1e-12 * np.abs(expected.prediction_ttc_s.to_numpy()))
                chosen_ok = (
                    np.array_equal(selected, expected.selected_expert.to_numpy())
                    if "selected_expert" in expected
                    else None
                )
                record = dict(
                    arm=arm,
                    outer_fold=fold,
                    rows=len(a5),
                    max_absolute_ttc_error=float(residual.max()),
                    ttc_matches=bool(np.all(residual <= permitted)),
                    selected_matches=chosen_ok,
                )
                equivalence.append(record)
                evidence = metadata[["sample_token", "sequence_id", "track_id"]].copy()
                evidence["target_ttc_s"] = target
                evidence["prediction_ttc_s"] = prediction
                evidence["selected_expert"] = selected
                evidence["loss"] = losses[np.arange(len(expert)), selected]
                for j in range(3):
                    evidence[f"expert{j}_ttc"] = expert[:, j]
                    evidence[f"expert{j}_true_cost"] = losses[:, j]
                for j in range(cost.shape[1]):
                    evidence[f"predicted_cost_or_negative_logit{j}"] = cost[:, j]
                evidence.to_csv(destination / f"outer{fold}_{arm}_replay.csv", index=False)
    write_json(output / "FROZEN_EXPERT_TABLE_INDEX.json", index)
    write_json(output / "LEGACY_EQUIVALENCE_ATTEMPT.json", equivalence)
    print(json.dumps(equivalence, indent=2))
    if any(not r["ttc_matches"] or r["selected_matches"] is False for r in equivalence):
        raise ValueError("legacy row equivalence failed; no new scientific fit authorized")


if __name__ == "__main__":
    main()
