"""Post-training diagnostics of upstream experts, covariate shift and error concentration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.efficient_context.common import digest
from operational.train40_system.durable_io import atomic_json
from operational.ttc_revision.score import metrics

FEATURES = (
    "log_roi_count",
    "log_roi_rate",
    "a5_transport",
    "a5_confidence",
    "a5_log_variance",
    "c2f_transport",
    "c2f_confidence",
    "c2f_log_variance",
    "a5_phase",
    "c2f_phase",
    "pair_phase",
    "pair_minus_a5",
    "pair_minus_c2f",
    "c2f_minus_a5",
    "abs_pair_minus_a5",
    "abs_pair_minus_c2f",
    "abs_c2f_minus_a5",
)


def diagnose(campaign: Path, train_cache: Path) -> None:
    """Describe fixed predictions; never fit, select or modify a checkpoint."""
    torch.set_num_threads(2)
    with np.load(train_cache, allow_pickle=False) as data:
        current = data["history"][:, -1]
        if (current < 0).any():
            raise ValueError("training current observation must be valid")
        train = data["features"][current]
        history = data["history"]
        valid = history >= 0
        safe = np.maximum(history, 0)
        anchors, availability = data["anchor_us"], data["available_us"]
        lags = (anchors[current, None] - anchors[safe]) / 1e6
        time_one = (availability[current, None] - availability[safe]) / 1e6
        time_three = (availability[safe] - anchors[safe]) / 1e6
        timing = {}
        for name, difference in (
            ("channel1_train_minus_transfer_convention", time_one),
            ("channel3_train_minus_transfer_convention", time_three - lags),
        ):
            values = difference[valid]
            timing[name] = {
                "quantiles_seconds": dict(
                    zip(
                        ("minimum", "median", "p99", "maximum"),
                        np.quantile(values, (0, 0.5, 0.99, 1)).tolist(),
                        strict=True,
                    )
                ),
                "fraction_above_one_microsecond_abs": float((np.abs(values) > 1e-6).mean()),
            }
    atomic_json(
        campaign / "TIMING_DIAGNOSTICS.json",
        {
            "scope": (
                "Actual TRAIN40 timings minus transfer convention at the same lags; "
                "not causal attribution"
            ),
            "valid_history_observations": int(valid.sum()),
            "channels": timing,
        },
    )
    envelope = np.quantile(train, (0.005, 0.5, 0.995), axis=0)
    shifts, expert_metrics, concentration, numerical_impact = [], [], [], []
    identities = {"train_cache_sha256": digest(train_cache)}
    for cohort in ("dev32", "fcwd"):
        replay = campaign / f"replay_{cohort}"
        receipt = json.loads((replay / "REPLAY_COMPLETE.json").read_text(encoding="utf-8"))
        if receipt["status"] != "COMPLETE":
            raise ValueError("complete replay required")
        rows = []
        for name, sha in sorted(receipt["fragments"].items()):
            path = replay / "fragments" / name
            if digest(path) != sha:
                raise ValueError("replay fragment changed")
            with np.load(path, allow_pickle=False) as saved:
                rows.append(
                    {
                        "query_id": str(saved["query_id"]),
                        **dict(zip(FEATURES, saved["features"][-1].tolist(), strict=True)),
                        **{
                            f"numerical_{name}": float(np.median(saved[key]))
                            for name, key in (
                                ("H8_canonical", "canonical"),
                                ("H8_compact", "compact"),
                                ("Direct_canonical", "direct_gpu"),
                                ("Direct_compact", "direct_compact_gpu"),
                            )
                        },
                    }
                )
        features = pd.DataFrame(rows)
        raw = features[list(FEATURES)].to_numpy(float)
        quantiles = np.quantile(raw, (0.005, 0.5, 0.995), axis=0)
        for index, name in enumerate(FEATURES):
            shifts.append(
                {
                    "population": cohort,
                    "feature": name,
                    "n_train": len(train),
                    "n_transfer": len(raw),
                    "train_q005": envelope[0, index],
                    "train_median": envelope[1, index],
                    "train_q995": envelope[2, index],
                    "transfer_q005": quantiles[0, index],
                    "transfer_median": quantiles[1, index],
                    "transfer_q995": quantiles[2, index],
                    "outside_train_envelope_fraction": float(
                        (
                            (raw[:, index] < envelope[0, index])
                            | (raw[:, index] > envelope[2, index])
                        ).mean()
                    ),
                }
            )
        phases = torch.from_numpy(raw[:, 8:11].astype(np.float32))
        diagnostic_names = []
        for index, name in enumerate(("A5", "C2F", "PAIR")):
            column = f"{name}_phase_bound60"
            features[column] = phase_to_ttc(phases[:, index]).numpy()
            diagnostic_names.append(column)
        column = "current_phase_median_bound60"
        features[column] = phase_to_ttc(phases.median(-1).values).numpy()
        diagnostic_names.append(column)
        predictions = campaign / cohort / "ALL_PREDICTIONS.csv"
        scored = pd.read_csv(predictions)
        repeated = (set(scored.columns) & set(features.columns)) - {"query_id"}
        joined = scored.merge(
            features.drop(columns=sorted(repeated)), on="query_id", validate="one_to_one"
        )
        joined.to_csv(campaign / cohort / "UPSTREAM_DIAGNOSTICS.csv", index=False)
        truth = joined.truth_ttc_seconds.to_numpy(float)
        eligible = np.isfinite(truth) & (truth != 0)
        truth = truth[eligible]
        selected = joined[eligible]
        for name in ("H8", "Direct"):
            canonical = np.asarray(selected[f"numerical_{name}_canonical"], float)
            compact = np.asarray(selected[f"numerical_{name}_compact"], float)
            if not (np.isfinite(canonical).all() and np.isfinite(compact).all()):
                raise ValueError("finite predictions required for numerical metric impact")
            canonical_alarm = (canonical > 0) & (canonical <= 1)
            compact_alarm = (compact > 0) & (compact <= 1)
            urgent = (truth > 0) & (truth <= 1)
            numerical_impact.append(
                {
                    "population": cohort,
                    "method": name + "_median3",
                    "n": len(truth),
                    "mae_change_compact_minus_canonical": float(
                        (np.abs(compact - truth) - np.abs(canonical - truth)).mean()
                    ),
                    "max_prediction_change_seconds": float(np.max(np.abs(compact - canonical))),
                    "alarm_decision_changes": int((compact_alarm != canonical_alarm).sum()),
                    "urgent_miss_changes": int(((compact_alarm != canonical_alarm) & urgent).sum()),
                    "sign_changes": int((np.sign(compact) != np.sign(canonical)).sum()),
                }
            )
        buckets = {
            "all": np.ones(len(truth), bool),
            "positive_at_most_1": (truth > 0) & (truth <= 1),
            "positive_1_4": (truth > 1) & (truth < 4),
            "positive_4_8": (truth >= 4) & (truth < 8),
            "positive_at_least_8": truth >= 8,
            "negative": truth < 0,
        }
        for family in selected.scenario_family.unique():
            buckets[f"family:{family}"] = np.asarray(selected.scenario_family) == family
        for method in (*diagnostic_names, "H8_median3", "Direct_median3"):
            values = np.asarray(selected[method], float)
            ae = np.abs(values - truth)
            for bucket, mask in buckets.items():
                expert_metrics.append(
                    {
                        "population": cohort,
                        "method": method,
                        "cohort": bucket,
                        **metrics(truth[mask], values[mask]),
                    }
                )
                concentration.append(
                    {
                        "population": cohort,
                        "method": method,
                        "cohort": bucket,
                        "n": int(mask.sum()),
                        "query_fraction": float(mask.mean()),
                        "absolute_error_fraction": float(ae[mask].sum() / ae.sum())
                        if ae.sum()
                        else 0.0,
                    }
                )
        identities[cohort + "_prediction_sha256"] = digest(predictions)
        identities[cohort + "_replay_sha256"] = digest(replay / "REPLAY_COMPLETE.json")
    pd.DataFrame(shifts).to_csv(campaign / "FEATURE_SHIFT.csv", index=False)
    pd.DataFrame(expert_metrics).to_csv(campaign / "UPSTREAM_METRICS.csv", index=False)
    pd.DataFrame(concentration).to_csv(campaign / "ERROR_CONCENTRATION.csv", index=False)
    pd.DataFrame(numerical_impact).to_csv(campaign / "NUMERICAL_METRIC_IMPACT.csv", index=False)
    atomic_json(
        campaign / "DIAGNOSTICS.json",
        {
            "status": "COMPLETE",
            "source_sha256": digest(Path(__file__)),
            **identities,
            "scope": (
                "Post-training descriptive diagnostics; not hyperparameter or checkpoint selection"
            ),
            "expert_predictions": (
                "Reconstructed from stored FP32 phase and bounded by H8 decoder; "
                "not native producer outputs"
            ),
            "envelope": (
                "Unweighted current TRAIN40 observations, quantiles 0.005 and 0.995; "
                "not an OOD probability"
            ),
            "interpretation": (
                "Associations do not establish causality; transfer populations already exposed"
            ),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path("artifacts/ttc_revision_20261009"))
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=Path("artifacts/train40_system_20261005/H8_FEATURES.npz"),
    )
    args = parser.parse_args()
    diagnose(args.campaign, args.train_cache)
