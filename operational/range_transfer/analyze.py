"""Fixed V-only gates, cohort-complete metrics and group bootstrap."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.rgb_port.evaluate import score_signed_ttc

from .common import freeze, read, sha, write
from .train_heads import initialize


def metrics(target: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    """Keep invalid rows and missing bands visible; never synthesize overall MiD."""
    result = score_signed_ttc(target, prediction, groups)
    valid = np.isfinite(prediction) & ((prediction < 0) | (prediction > 0.1))
    c = (target > 0.1) & (target <= 3)
    mid = np.full(len(target), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        mid[valid] = 10000 * np.abs(
            np.log1p(-0.1 / prediction[valid]) - np.log1p(-0.1 / target[valid])
        )
    complete = bool(np.isfinite(mid).all())
    result["mean_MiD"] = float(mid.mean()) if complete else None
    result["overall_MiD"] = result["group_macro_bucket_MiD"] if complete else None
    result["complete_prediction_and_mid_coverage"] = complete
    result["group_crucial_mid"] = {
        str(group): float(mid[c & (groups == group)].mean())
        for group in np.unique(groups[c])
        if np.isfinite(mid[c & (groups == group)]).all()
    }
    result["p95_positive_crucial_overestimate_s"] = (
        float(np.quantile(np.maximum(prediction[c] - target[c], 0), 0.95))
        if c.any() and valid[c].all()
        else None
    )
    # Preserve the existing revision's 1s urgent-warning definition.
    # A negative TTC is not a warning; a negative prediction on imminent contact is late.
    imminent = (target > 0.1) & (target <= 1)
    safe = (target < 0) | (target > 1)
    warning = (prediction > 0.1) & (prediction <= 1)
    result["warning_boundary_s"] = 1.0
    result["late_warning"] = {
        "denominator": int(imminent.sum()),
        "rate": float((~warning[imminent]).mean()) if imminent.any() else None,
    }
    result["false_alert"] = {
        "denominator": int(safe.sum()),
        "rate": float(warning[safe].mean()) if safe.any() else None,
    }
    return result


def gate(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Apply all preregistered safeguards without threshold rescue."""
    if reference["overall_MiD"] is None or candidate["overall_MiD"] is None:
        return {"eligible": False, "status": "LIMITED_VALIDATION_SUPPORT", "checks": {}}
    checks = {
        "coverage": bool(
            reference["complete_prediction_and_mid_coverage"]
            and candidate["complete_prediction_and_mid_coverage"]
        )
    }
    rb, cb = reference["bins"], candidate["bins"]
    checks["crucial_10pct"] = (
        cb["crucial"]["group_macro_mid"] <= 0.9 * rb["crucial"]["group_macro_mid"]
    )
    groups = reference["group_crucial_mid"]
    checks["crucial_groups_60pct"] = (
        bool(groups)
        and sum(candidate["group_crucial_mid"].get(g, float("inf")) < v for g, v in groups.items())
        / len(groups)
        >= 0.6
    )
    checks["aggregate"] = candidate["overall_MiD"] <= reference["overall_MiD"]
    for name in ("small", "large", "negative"):
        baseline = rb[name]["group_macro_mid"]
        checks[name] = cb[name]["group_macro_mid"] <= baseline + max(2.0, 0.05 * baseline)
    rd, cd = reference["diagnostics"], candidate["diagnostics"]
    checks["mae"] = cd["mae_s"] <= 1.02 * rd["mae_s"]
    checks["sign"] = cd["sign_error_rate"] <= rd["sign_error_rate"] + 0.005
    checks["crucial_over_p95"] = (
        candidate["p95_positive_crucial_overestimate_s"]
        <= 1.05 * reference["p95_positive_crucial_overestimate_s"]
    )
    limited = []
    for name in ("late_warning", "false_alert"):
        if min(reference[name]["denominator"], candidate[name]["denominator"]) < 30:
            limited.append(name)
        else:
            checks[name] = candidate[name]["rate"] <= reference[name]["rate"] + 0.01
    # Insufficient warning support remains descriptive, without relaxing gates.
    eligible = all(checks.values()) and not limited
    return {
        "eligible": eligible,
        "checks": checks,
        "limited_denominators": limited,
        "status": "ELIGIBLE"
        if eligible
        else ("LIMITED_VALIDATION_SUPPORT" if limited else "FAILED_GATES"),
    }


def predict(endpoint: Path, values: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    """Inference receives input fields only; timing covers cached CPU head forward."""
    import time

    if set(values) != {"features", "timing", "valid", "expert_phase"}:
        raise ValueError("Prediction may receive inputs only")
    payload = torch.load(endpoint, map_location="cpu", weights_only=False)
    model, _, _, _ = initialize(int(payload["identity"]["seed"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    output = []
    start = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(values["features"]), 128):
            batch = [
                torch.from_numpy(values[k][begin : begin + 128].copy())
                for k in ("features", "timing", "valid", "expert_phase")
            ]
            output.append(phase_to_ttc(model(*batch)["raw_location"]).numpy())
    return np.concatenate(output), time.perf_counter() - start


def evaluate_v(owner: Path, run: Path, endpoints: dict[str, Any], *, stage: str) -> dict[str, Any]:
    """Open V labels only after every relevant endpoint has been frozen."""
    freeze_path = run / (
        "SCREEN_ENDPOINTS.json" if stage == "screen" else "REPLICATION_ENDPOINTS.json"
    )
    if read(freeze_path) != endpoints:
        raise ValueError("Endpoint matrix is not frozen")
    cache = owner / "features/V/E_HISTORY.npz"
    receipt = read(cache.with_suffix(".json"))
    normalizer_sha = sha(owner / "features/H/E_NORMALIZER.json")
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("role") != "V"
        or receipt.get("modality") != "event"
        or receipt.get("output_sha256") != sha(cache)
        or receipt.get("normalizer_sha256") != normalizer_sha
    ):
        raise ValueError("V cache receipt/normalizer mismatch")
    with np.load(cache, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    if set(values["role"].astype(str)) != {"V"}:
        raise ValueError("V role mismatch")
    targets = -0.1 / np.expm1(-values["target_phase"].astype(np.float64))
    groups = values["group_id"].astype(str)
    tokens = values["sample_token"].astype(str)
    manifest = read(owner / "V_MANIFEST.json")
    if set(groups) != set(manifest["group_ids"]) or len(set(tokens)) != len(tokens):
        raise ValueError("V cohort/group contract mismatch")
    source_rows = Path(manifest["rows_path"])
    if sha(source_rows) != manifest["rows_sha256"]:
        raise ValueError("V metadata changed")
    original = pd.read_parquet(
        source_rows, columns=["sample_token", "group_id", "target_phase", "rgb_available"]
    )
    original = original.loc[original.rgb_available].set_index("sample_token", verify_integrity=True)
    if set(tokens) != set(original.index):
        raise ValueError("Incomplete V cohort")
    if not np.array_equal(
        original.loc[tokens].target_phase.to_numpy(np.float32), values["target_phase"]
    ):
        raise ValueError("V targets differ from native manifest")
    inputs = {k: values[k] for k in ("features", "timing", "valid", "expert_phase")}
    results = {}
    for name, item in endpoints.items():
        endpoint = Path(item["checkpoint_path"])
        if sha(endpoint) != item["checkpoint_sha256"]:
            raise ValueError("Frozen endpoint changed before V evaluation")
        out = run / "evaluation" / stage / name
        out.mkdir(parents=True, exist_ok=True)
        binding = {"endpoint_sha256": item["checkpoint_sha256"], "cache_sha256": sha(cache)}
        freeze(out / "BINDING.json", binding)
        if (out / "METRICS.json").exists():
            if sha(out / "PREDICTIONS.csv") != read(out / "PREDICTIONS_RECEIPT.json")["sha256"]:
                raise ValueError("Persisted predictions changed")
            results[name] = read(out / "METRICS.json")
            continue
        predictions, elapsed = predict(endpoint, inputs)
        frame = pd.DataFrame(
            {
                "sample_token": tokens,
                "group_id": groups,
                "target_ttc": targets,
                "prediction_ttc": predictions,
            }
        )
        frame.to_csv(out / "PREDICTIONS.csv", index=False)
        write(out / "PREDICTIONS_RECEIPT.json", {"sha256": sha(out / "PREDICTIONS.csv")})
        score = metrics(targets, predictions, groups)
        write(out / "METRICS.json", score)
        write(
            out / "GROUP_METRICS.json",
            {
                g: metrics(targets[groups == g], predictions[groups == g], groups[groups == g])
                for g in sorted(set(groups))
            },
        )
        write(
            out / "COST.json",
            {
                "scope": "CPU_cached_head_forward_batch128_including_output_conversion",
                "seconds": elapsed,
                "rows": len(targets),
                "raw_E2E": False,
            },
        )
        results[name] = score
    return results


def select(results: dict[str, Any], run: Path) -> dict[str, Any]:
    """Freeze at most one V-selected recipe before replication/transfer."""
    reference = results["RT-M0-L0-S7"]
    gates = {
        name: gate(reference, value) for name, value in results.items() if name != "RT-M0-L0-S7"
    }
    eligible = [name for name, value in gates.items() if value["eligible"]]
    winner = (
        min(
            eligible,
            key=lambda name: (
                results[name]["bins"]["crucial"]["group_macro_mid"],
                results[name]["overall_MiD"],
                name,
            ),
        )
        if eligible
        else None
    )
    result = {
        "selection_source": "V_ONLY",
        "candidate": winner,
        "candidate_recipe": [int(winner[4]), int(winner[7])] if winner else None,
        "gates": gates,
        "status": "SELECTED" if winner else "NO_CANDIDATE",
        "transfer_used_for_selection": False,
    }
    freeze(run / "SELECTION_FREEZE.json", result)
    return result


def bootstrap_pairs(reference: pd.DataFrame, candidate: pd.DataFrame, output: Path) -> None:
    """Preserve paired group-cluster bootstrap fragments; never sample windows IID."""
    if (
        not np.array_equal(reference.sample_token, candidate.sample_token)
        or not np.array_equal(reference.target_ttc, candidate.target_ttc)
        or not np.array_equal(reference.group_id, candidate.group_id)
    ):
        raise ValueError("Bootstrap cohorts must align by token and truth")
    for frame in (reference, candidate):
        pred = frame.prediction_ttc.to_numpy()
        if not (np.isfinite(pred) & ((pred < 0) | (pred > 0.1))).all():
            write(output / "SUMMARY.json", {"status": "INVALID_PREDICTION_COVERAGE"})
            return
    a, b = [], []
    for frame, result in ((reference, a), (candidate, b)):
        for _, group in frame.groupby("group_id", sort=True):
            mask = (group.target_ttc > 0.1) & (group.target_ttc <= 3)
            if mask.any():
                truth, pred = group.loc[mask, "target_ttc"], group.loc[mask, "prediction_ttc"]
                result.append(
                    float((10000 * np.abs(np.log1p(-0.1 / pred) - np.log1p(-0.1 / truth))).mean())
                )
    if not a or len(a) != len(b):
        write(output / "SUMMARY.json", {"status": "LIMITED_VALIDATION_SUPPORT"})
        return
    difference = np.asarray(b) - np.asarray(a)
    values = []
    for block in range(10):
        path = output / f"block_{block:02d}.json"
        rng = np.random.default_rng(7000 + block)
        indices = rng.integers(len(difference), size=(100, len(difference)))
        result = {
            "seed": 7000 + block,
            "scope": "paired_group_macro_crucial_MiD",
            "delta": difference[indices].mean(1).tolist(),
        }
        freeze(path, result)
        values.extend(result["delta"])
    write(
        output / "SUMMARY.json",
        {
            "delta_group_macro_MiDc": float(difference.mean()),
            "CI95": np.quantile(values, [0.025, 0.975]).tolist(),
            "groups": len(difference),
            "replicates": 1000,
        },
    )


def summarize_replicas(run: Path, recipe: tuple[int, int]) -> None:
    """Average per-query losses over fixed seeds; never ensemble TTC predictions."""
    from .common import fit_id

    reference_frame = None
    averaged = {}
    for label, (mass, loss) in (("control", (0, 0)), ("candidate", recipe)):
        frames = []
        for seed in (7, 13, 23):
            stage = "screen" if seed == 7 else "replication"
            path = run / "evaluation" / stage / fit_id(mass, loss, seed) / "PREDICTIONS.csv"
            frame = pd.read_csv(path)
            keys = frame[["sample_token", "group_id", "target_ttc"]]
            if reference_frame is None:
                reference_frame = keys
            elif not keys.equals(reference_frame):
                raise ValueError("Replication populations differ")
            prediction = frame.prediction_ttc.to_numpy()
            if not (np.isfinite(prediction) & ((prediction < 0) | (prediction > 0.1))).all():
                write(run / "replication/SUMMARY.json", {"status": "INVALID_PREDICTION_COVERAGE"})
                return
            target = frame.target_ttc.to_numpy()
            frames.append(10000 * np.abs(np.log1p(-0.1 / prediction) - np.log1p(-0.1 / target)))
        averaged[label] = np.stack(frames).mean(axis=0)
    assert reference_frame is not None
    table = reference_frame.copy()
    table["control_mean_loss"] = averaged["control"]
    table["candidate_mean_loss"] = averaged["candidate"]
    output = run / "replication"
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "MEAN_LOSS_PER_QUERY.csv", index=False)
    masks = {
        "crucial": (table.target_ttc > 0.1) & (table.target_ttc <= 3),
        "small": (table.target_ttc > 3) & (table.target_ttc <= 6),
        "large": (table.target_ttc > 6) & (table.target_ttc <= 10),
        "negative": table.target_ttc < 0,
    }
    summary = {}
    for band, mask in masks.items():
        means = (
            table.loc[mask].groupby("group_id")[["control_mean_loss", "candidate_mean_loss"]].mean()
        )
        if means.empty:
            summary[band] = {"status": "MISSING_BAND"}
            continue
        difference = (means.candidate_mean_loss - means.control_mean_loss).to_numpy()
        draws = []
        for block in range(10):
            rng = np.random.default_rng(7000 + block)
            indices = rng.integers(len(difference), size=(100, len(difference)))
            value = {"seed": 7000 + block, "delta": difference[indices].mean(axis=1).tolist()}
            freeze(output / band / f"block_{block:02d}.json", value)
            draws.extend(value["delta"])
        summary[band] = {
            "control": float(means.control_mean_loss.mean()),
            "candidate": float(means.candidate_mean_loss.mean()),
            "delta": float(difference.mean()),
            "CI95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "groups": len(means),
        }
    write(
        output / "SUMMARY.json",
        {
            "scope": "three_seed_mean_query_losses_group_cluster_bootstrap",
            "predictions_ensembled": False,
            "bands": summary,
        },
    )
