"""Fixed-ridge, held-group-out PARENT feature probes, H only, no optimizer updates."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.range_transfer.analyze import metrics
from operational.range_transfer.common import digest, read, sha, write
from operational.representation_audit.run import OWNER, ROOT, RUN, cache, raw_side


def main() -> None:
    torch.set_num_threads(1)
    started = time.perf_counter()
    h = cache("H")
    tokens = h["sample_token"].astype(str)
    groups = h["group_id"].astype(str)
    unique = sorted(set(groups))
    if len(unique) != 8:
        raise ValueError("Expected original eight H groups")
    raw, binding = raw_side("H", tokens)
    g = raw[:, :384].reshape(-1, 3, 128).astype(np.float64)
    f = raw[:, 384:640].reshape(-1, 2, 128).astype(np.float64)
    r = raw[:, 640:].reshape(-1, 2, 128).astype(np.float64)
    directional = np.concatenate(
        (np.diff(g, axis=1).reshape(-1, 256), (f - r).reshape(-1, 256)), axis=1
    )
    # Undo the purely affine original normalization before fitting fold-local statistics.
    original_norm = read(OWNER / "features/H/E_NORMALIZER.json")
    base = h["features"][:, -1].astype(np.float64) * np.asarray(
        original_norm["scale"]
    ) + np.asarray(original_norm["mean"])
    features = {
        "R0": base,
        "RG": np.concatenate((base, raw), axis=1),
        "RD": np.concatenate((base, directional), axis=1),
    }
    anchor = np.median(h["expert_phase"].astype(np.float64), axis=1)
    truth = h["target_phase"].astype(np.float64)
    residual = truth - anchor
    target = -0.1 / np.expm1(-truth)
    mass = h["mass"].astype(np.float64)
    out = RUN / "group_probe"
    out.mkdir(parents=True, exist_ok=True)
    scores, fold_rows = {}, []
    for name, x in features.items():
        predicted = np.empty(len(tokens), np.float64)
        for group in unique:
            test = groups == group
            train = ~test
            xtrain = x[train]
            mean = xtrain.mean(axis=0)
            scale = np.maximum(xtrain.std(axis=0), 1e-6)
            normalized = (xtrain - mean) / scale
            weights = mass[train] / mass[train].sum()
            xmean = weights @ normalized
            ymean = float(weights @ residual[train])
            centered = normalized - xmean
            matrix = centered.T @ (weights[:, None] * centered) + np.eye(x.shape[1])
            coefficient = np.linalg.solve(
                matrix, centered.T @ (weights * (residual[train] - ymean))
            )
            predicted[test] = (
                anchor[test] + (((x[test] - mean) / scale) - xmean) @ coefficient + ymean
            )
            np.savez(
                out / f"{name}_{group}.npz",
                mean=mean,
                scale=scale,
                xmean=xmean,
                ymean=ymean,
                coefficient=coefficient,
            )
            fold_rows.append(
                {
                    "representation": name,
                    "held_group": group,
                    "train_n": int(train.sum()),
                    "test_n": int(test.sum()),
                    "train_groups": ",".join(g for g in unique if g != group),
                    "train_tokens_sha256": digest(tokens[train].tolist()),
                    "phase_mae": float(np.abs(predicted[test] - truth[test]).mean()),
                }
            )
        pred = phase_to_ttc(torch.from_numpy(predicted)).numpy()
        pd.DataFrame(
            {
                "sample_token": tokens,
                "group_id": groups,
                "target_ttc": target,
                "prediction_ttc": pred,
                "prediction_phase": predicted,
            }
        ).to_csv(out / f"{name}_OOF.csv", index=False)
        score = metrics(target, pred, groups)
        score["macro_phase_mae"] = float(
            np.mean([row["phase_mae"] for row in fold_rows if row["representation"] == name])
        )
        scores[name] = score
    pd.DataFrame(fold_rows).to_csv(out / "FOLDS.csv", index=False)
    write(out / "RESULTS.json", scores)
    write(
        out / "PROVENANCE.json",
        {
            "source_sha256": sha(ROOT / "scripts/probe_group_transfer_v17.py"),
            "vector_binding": binding,
            "history_sha256": sha(RUN / "cache/H.npz"),
            "optimizer_updates": 0,
            "closed_form_fits": 24,
            "ridge_lambda": 1.0,
            "cpu_seconds": time.perf_counter() - started,
            "gpu_seconds": 0,
            "scope": "H1_linear_OOF_H_diagnostic_not_head_selection",
        },
    )
    print(
        {
            name: {
                "phase_mae": score["macro_phase_mae"],
                "MiDc": score["bins"]["crucial"]["group_macro_mid"],
                "overall": score["overall_MiD"],
            }
            for name, score in scores.items()
        }
    )


if __name__ == "__main__":
    main()
