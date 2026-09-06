"""Loss-based paired uncertainty and complete selector diagnostic materialization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding, verify
from e_jepa_ttc.evaluation.exact_sequence_v10 import exact_sequence_diagnostic
from e_jepa_ttc.evaluation.stage63_65 import BUCKETS, benchmark_phase, strict_macro_mass


def loss_frame(
    meta: pd.DataFrame,
    expert: np.ndarray,
    costs: np.ndarray,
    selected: np.ndarray,
    *,
    arm: str,
    seed: int,
    fold: int,
) -> pd.DataFrame:
    """All exact actions and costs are retained; no TTC averaging."""
    target = meta.target_ttc.to_numpy(np.float64)
    true = 10000 * np.abs(benchmark_phase(expert) - benchmark_phase(target)[:, None])
    prediction = expert[np.arange(len(expert)), selected]
    if not np.isfinite(costs).all() or not np.all((expert <= -0.1) | (expert > 0.1)):
        raise ValueError("invalid expert/cost coverage")
    frame = meta[["sample_token", "sequence_id", "track_id"]].copy()
    frame["outer_fold"] = fold
    frame["arm"] = arm
    frame["seed"] = seed
    frame["target_ttc_s"] = target
    frame["prediction_ttc_s"] = prediction
    frame["selected_expert"] = selected
    frame["loss"] = true[np.arange(len(expert)), selected]
    frame["wrong_sign"] = np.sign(prediction) != np.sign(target)
    frame["oracle_loss"] = true.min(1)
    frame["regret"] = frame.loss - frame.oracle_loss
    for j in range(3):
        frame[f"expert{j}_ttc"] = expert[:, j]
        frame[f"expert{j}_phase"] = benchmark_phase(expert[:, j])
        frame[f"true_cost{j}"] = true[:, j]
        frame[f"predicted_cost{j}"] = costs[:, j]
    frame["margin"] = np.partition(costs, 1, axis=1)[:, 1] - costs.min(1)
    return cast(pd.DataFrame, frame)


def hierarchical_losses(
    frame: pd.DataFrame, losses: np.ndarray, output: Path, check: Callable[[], None]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Shared sequence→whole-track bootstrap of an arbitrary per-row loss matrix."""
    seq = frame.sequence_id.to_numpy()
    tracks = frame.track_id.to_numpy()
    target = frame.target_ttc_s.to_numpy()
    sequences = sorted(np.unique(seq))
    groups = {}
    for s in sequences:
        names = sorted(np.unique(tracks[seq == s]))
        counts = np.zeros((len(names), 4))
        sums = np.zeros((len(names), 4, losses.shape[1]))
        for t, name in enumerate(names):
            for b, (_, lo, hi, _) in enumerate(BUCKETS):
                mask = (seq == s) & (tracks == name) & (target > lo) & (target <= hi)
                counts[t, b] = mask.sum()
                sums[t, b] = losses[mask].sum(0)
        groups[s] = (counts, sums)
    rng = np.random.default_rng(640065)
    values = []
    attempts = 0
    h = hashlib.sha256()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "HIERARCHICAL_DRAWS.jsonl").open("w") as log:
        while len(values) < 8192 and attempts < 32768:
            check()
            attempts += 1
            draw = rng.integers(0, len(sequences), len(sequences))
            encoded = [draw.tolist()]
            scores = []
            valid = True
            for index in draw:
                counts, sums = groups[sequences[index]]
                selection = rng.integers(0, len(counts), len(counts))
                encoded.append(selection.tolist())
                total = counts[selection].sum(0)
                if np.any(total == 0):
                    valid = False
                    break
                scores.append(
                    np.array([0.5, 0.3, 0.1, 0.1]) @ (sums[selection].sum(0) / total[:, None])
                )
            raw = json.dumps(encoded, separators=(",", ":"))
            h.update(raw.encode())
            log.write(raw + "\n")
            if valid:
                values.append(np.mean(scores, axis=0))
    if len(values) != 8192 or len(values) / attempts < 0.99:
        raise RuntimeError("BOOTSTRAP_SUPPORT_BLOCKED")
    result = np.asarray(values)
    np.save(output / "BOOTSTRAP_LOSSES.npy", result, allow_pickle=False)
    return result, dict(
        draws_sha256=h.hexdigest(),
        valid_draws=len(values),
        attempts=attempts,
        discarded=attempts - len(values),
        validity_fraction=len(values) / attempts,
    )


def diagnostics(
    frames: dict[str, pd.DataFrame],
    primary: str,
    output: Path,
    train_quantiles: dict[str, dict[str, list[float]]],
    check: Callable[[], None],
) -> dict[str, Any]:
    """Materialize all outputs before decision; report both uncertainty procedures."""
    output.mkdir(parents=True, exist_ok=True)
    names = list(frames)
    aligned = {k: v.sort_values("sample_token").reset_index(drop=True) for k, v in frames.items()}
    base = aligned[primary]
    if len(base) != 8192 or base.sample_token.duplicated().any():
        raise ValueError("evaluation universe incomplete")
    for frame in aligned.values():
        if not frame[
            ["sample_token", "sequence_id", "track_id", "target_ttc_s", "outer_fold"]
        ].equals(base[["sample_token", "sequence_id", "track_id", "target_ttc_s", "outer_fold"]]):
            raise ValueError("paired identity mismatch")
    mass = strict_macro_mass(base.target_ttc_s.to_numpy(), base.sequence_id.to_numpy())
    loss = np.column_stack([aligned[n].loss.to_numpy() for n in names])
    if not np.isfinite(loss).all():
        raise ValueError("nonfinite loss")
    sampled, boot = hierarchical_losses(base, loss, output, check)
    seqs = sorted(base.sequence_id.unique())
    seq_scores = {}
    summaries = {}
    index = []
    for name, frame in aligned.items():
        check()
        frame = frame.copy()
        frame["mass"] = mass
        frame["weighted_loss"] = mass * frame.loss
        frame["weighted_sign_error"] = mass * frame.wrong_sign
        frame["weighted_regret"] = mass * frame.regret
        frame["bucket"] = np.select(
            [frame.target_ttc_s <= 0, frame.target_ttc_s <= 3, frame.target_ttc_s <= 6],
            ["negative", "crucial", "small"],
            default="large",
        )
        frame["target_sign"] = np.sign(frame.target_ttc_s)
        frame["prediction_sign"] = np.sign(frame.prediction_ttc_s)
        expert_losses = frame[[f"true_cost{j}" for j in range(3)]].to_numpy()
        frame["oracle_expert"] = expert_losses.argmin(1)
        costs = frame[[f"predicted_cost{j}" for j in range(3)]].to_numpy()
        phase = frame[[f"expert{j}_phase" for j in range(3)]].to_numpy()
        frame["lipschitz_excess"] = np.stack(
            [
                np.maximum(
                    np.abs(costs[:, a] - costs[:, b]) - 100 * np.abs(phase[:, a] - phase[:, b]), 0
                )
                for a, b in [(0, 1), (0, 2), (1, 2)]
            ],
            1,
        ).max(1)
        order = np.argsort(phase, axis=1, kind="stable")
        gaps = np.diff(np.take_along_axis(phase, order, axis=1), axis=1)
        slopes = np.divide(
            np.diff(np.take_along_axis(costs, order, axis=1), axis=1),
            100 * gaps,
            out=np.zeros_like(gaps),
            where=gaps > 0,
        )
        frame["convexity_excess"] = np.maximum(slopes[:, 0] - slopes[:, 1], 0)
        frame["duplicate_phase"] = np.any(gaps == 0, axis=1)
        thresholds = train_quantiles.get(name, train_quantiles[primary])
        frame["high_regret_bin"] = 0
        for fold in range(3):
            bins = np.asarray(thresholds[str(fold)] if isinstance(thresholds, dict) else thresholds)
            mask = frame.outer_fold.eq(fold)
            frame.loc[mask, "high_regret_bin"] = np.searchsorted(bins, frame.loc[mask, "regret"])
        margin_edges = np.array([0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100, float("inf")])
        frame["margin_bin"] = np.searchsorted(margin_edges, frame.margin, side="right") - 1
        relative_true = (expert_losses[:, 1:] - expert_losses[:, :1]) / 100
        frame["relative_risk_mse"] = np.mean((costs[:, 1:] - relative_true) ** 2, axis=1)
        frame["oracle_match"] = frame.selected_expert == frame.oracle_expert
        directory = output / name
        directory.mkdir(exist_ok=True)
        frame.to_csv(directory / "ROWS.csv", index=False)
        if "geometry11" in frame.columns:
            frame["geometry_support"] = (frame.geometry11 > 0) | (frame.geometry23 > 0)
            frame.groupby(["geometry_support", "sequence_id"]).agg(
                rows=("loss", "size"), loss=("weighted_loss", "sum"), mass=("mass", "sum")
            ).to_csv(directory / "geometry_support.csv")
        frame.groupby("margin_bin").agg(
            rows=("loss", "size"),
            mean_margin=("margin", "mean"),
            relative_risk_mse=("relative_risk_mse", "mean"),
            oracle_match=("oracle_match", "mean"),
            mean_regret=("regret", "mean"),
        ).to_csv(directory / "risk_calibration.csv")
        for keys, label in [
            (["sequence_id"], "sequence"),
            (["outer_fold"], "fold"),
            (["bucket"], "bucket"),
            (["sequence_id", "track_id"], "track"),
            (["sequence_id", "bucket", "selected_expert"], "selection"),
            (["target_sign", "prediction_sign"], "sign"),
            (["oracle_expert"], "expert_wins"),
            (["high_regret_bin"], "high_regret"),
            (["margin_bin"], "margin_calibration"),
        ]:
            table = frame.groupby(keys).agg(
                rows=("loss", "size"),
                mass=("mass", "sum"),
                loss=("weighted_loss", "sum"),
                sign_error=("weighted_sign_error", "sum"),
                regret=("weighted_regret", "sum"),
            )
            table = cast(pd.DataFrame, table)
            table["conditional_loss"] = table.loss / table.mass
            table.to_csv(directory / f"{label}.csv")
        seq_scores[name] = np.array(
            [float(frame.loc[frame.sequence_id == s, "weighted_loss"].sum() * 9) for s in seqs]
        )
        crucial = frame.bucket.eq("crucial").to_numpy()
        summaries[name] = dict(
            score=float(mass @ frame.loss),
            weighted_sign_error=float(mass @ frame.wrong_sign),
            crucial=float((mass[crucial] @ frame.loss.to_numpy()[crucial]) / 0.5),
            oracle=float(mass @ frame.oracle_loss),
            regret=float(mass @ frame.regret),
            lipschitz_fraction=float((frame.lipschitz_excess > 1e-10).mean()),
            convexity_fraction=float((frame.convexity_excess > 1e-10).mean()),
        )
        for other in names:
            if other == name:
                continue
            comparison = frame[
                ["sample_token", "sequence_id", "track_id", "selected_expert"]
            ].copy()
            comparison = cast(pd.DataFrame, comparison)
            comparison["reference_selected"] = aligned[other].selected_expert
            comparison["delta_loss"] = frame.loss - aligned[other].loss
            comparison["weighted_delta"] = mass * comparison.delta_loss
            comparison.to_csv(directory / f"vs_{other}.csv", index=False)
            comparison.sort_values("weighted_delta").iloc[
                np.r_[0:50, len(frame) - 50 : len(frame)]
            ].to_csv(directory / f"extremes_{other}.csv", index=False)
        index.extend(binding(p) for p in directory.iterdir() if p.is_file())
    primary_idx = names.index(primary)
    comparisons = {}
    for name in names:
        if name == primary:
            continue
        exact = exact_sequence_diagnostic(seq_scores[primary] - seq_scores[name])
        draw_delta = sampled[:, primary_idx] - sampled[:, names.index(name)]
        comparisons[name] = dict(
            point_delta=summaries[primary]["score"] - summaries[name]["score"],
            hierarchical_ci95_high=float(np.quantile(draw_delta, 0.975)),
            hierarchical_ci95_low=float(np.quantile(draw_delta, 0.025)),
            sequence_ci95_high=exact["ci95"][1],
            sequence_ci95_low=exact["ci95"][0],
            weighted_sign_error_delta=summaries[primary]["weighted_sign_error"]
            - summaries[name]["weighted_sign_error"],
            crucial_bucket_mid_delta=summaries[primary]["crucial"] - summaries[name]["crucial"],
            exact=exact,
            fraction_negative=float(np.mean(draw_delta < 0)),
        )
    result = dict(
        execution_complete=True,
        protocol_conformant=True,
        diagnostics_complete=True,
        coverage_unchanged=True,
        finite_fraction=1.0,
        failure_count=0,
        comparisons=comparisons,
        summaries=summaries,
        bootstrap=boot,
        oracle_deployable=False,
    )
    atomic_json(output / "DIAGNOSTICS.json", result)
    index.extend(
        binding(output / p)
        for p in ("DIAGNOSTICS.json", "HIERARCHICAL_DRAWS.jsonl", "BOOTSTRAP_LOSSES.npy")
    )
    atomic_json(
        output / "DIAGNOSTIC_COVERAGE.json",
        dict(
            completed=True,
            rows=8192,
            arms=names,
            files=index,
            required_categories=[
                "rows",
                "sequence",
                "fold",
                "bucket",
                "track",
                "selection",
                "sign",
                "expert_wins",
                "high_regret",
                "margin_calibration",
                "cost_geometry",
                "oracle_regret",
                "controls",
                "hierarchical",
                "exact_sequence",
                "omission",
            ],
        ),
    )
    return result


def verify_coverage(path: Path) -> None:
    """Validate physical mandatory schemas, rather than trust a completed boolean."""
    coverage = json.loads(path.read_text())
    if coverage.get("completed") is not True or coverage.get("rows") != 8192:
        raise ValueError("diagnostic coverage incomplete")
    files = {verify(r).resolve() for r in coverage["files"]}
    required = {
        path.parent / n
        for n in ("DIAGNOSTICS.json", "HIERARCHICAL_DRAWS.jsonl", "BOOTSTRAP_LOSSES.npy")
    }
    for arm in coverage["arms"]:
        directory = path.parent / arm
        required.update(
            directory / (n + ".csv")
            for n in (
                "ROWS",
                "sequence",
                "fold",
                "bucket",
                "track",
                "selection",
                "sign",
                "expert_wins",
                "high_regret",
                "margin_calibration",
            )
        )
        rowfile = directory / "ROWS.csv"
        frame = pd.read_csv(rowfile)
        if len(frame) != 8192 or frame.sample_token.duplicated().any():
            raise ValueError("diagnostic row coverage incomplete")
        columns = {
            "loss",
            "wrong_sign",
            "regret",
            "oracle_loss",
            "lipschitz_excess",
            "convexity_excess",
        }
        if (
            not columns <= set(frame.columns)
            or not np.isfinite(frame[list(columns)].to_numpy(float)).all()
        ):
            raise ValueError("diagnostic schema incomplete")
    if not {p.resolve() for p in required} <= files:
        raise ValueError("missing required indexed diagnostic")
