"""Fixed causal transport control and grouped probes of frozen context features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from operational.final_refit.training import PhaseOnlyFinal
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_campaign.data import load_data
from operational.phase_campaign.evaluation import metrics, paired_group_bootstrap
from operational.phase_campaign.protocol import family_folds
from operational.phase_campaign.temporal import causal_transport_filter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "diagnosis", "protocol", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    values, rows, _ = load_data(args.data)
    folds = family_folds(yaml.safe_load(args.protocol.read_text())["records"])
    admitted = {name for fold in folds for name in fold["test"]}
    indices = np.flatnonzero(
        (values["domain"] == "evttc") & np.isin(values["group_id"], list(admitted))
    )
    metadata = [rows[str(values["sample_token"][i])] for i in indices]
    groups = values["group_id"][indices]
    clocks = np.array([r["anchor_us"] for r in metadata])
    with np.load(args.diagnosis / "predictions.npz") as source:
        lookup = {str(t): i for i, t in enumerate(source["sample_token"])}
        positions = [lookup[str(values["sample_token"][i])] for i in indices]
        target = source["target"][positions]
        baseline = source["F0"][positions]
    filtered = baseline.copy()
    innovations = []
    for name in sorted(set(groups)):
        selected = np.flatnonzero(groups == name)
        order = selected[np.argsort(clocks[selected])]
        filtered[order] = causal_transport_filter(baseline[order], clocks[order])
        dt = np.diff(clocks[order]) / 1e6
        innovations.append(
            {
                "recording": str(name),
                "baseline_innovation_median_s": float(
                    np.median(np.abs(np.diff(baseline[order]) + dt))
                ),
                "target_innovation_median_s": float(np.median(np.abs(np.diff(target[order]) + dt))),
            }
        )
    per_record = {
        str(g): {
            "F0": metrics(target[groups == g], baseline[groups == g]),
            "T0": metrics(target[groups == g], filtered[groups == g]),
        }
        for g in sorted(set(groups))
    }
    family_by_record = {g: f["family"] for f in folds for g in f["test"]}
    names = sorted(per_record)
    paired = paired_group_bootstrap(
        np.array([per_record[g]["F0"]["relative_absolute_error_pct"] for g in names]),
        np.array([per_record[g]["T0"]["relative_absolute_error_pct"] for g in names]),
        np.array([family_by_record[g] for g in names]),
    )
    times = np.arange(11) / 10
    distance = np.where(times < 0.5, 4 - times, 3.5 - 6 * (times - 0.5))
    truth = distance / np.where(times < 0.5, 1, 6)
    synthetic = causal_transport_filter(truth, (times * 1e6).astype(np.int64))
    crossing = lambda x: float(times[np.flatnonzero(x <= 0.5)[0]])  # noqa: E731
    temporal = {
        "scope": "fixed_exposed_diagnostic_no_parameter_search",
        "observation_weight": 0.5,
        "F0": metrics(target, baseline),
        "T0": metrics(target, filtered),
        "by_recording": per_record,
        "innovations": innovations,
        "paired_family_bootstrap": paired,
        "acceleration_fixture": {
            "timestamps": times.tolist(),
            "true_ttc": truth.tolist(),
            "filtered_ttc": synthetic.tolist(),
            "warning_delay_at_half_second_s": crossing(synthetic) - crossing(truth),
        },
    }
    save_npz(
        args.output / "temporal_predictions.npz",
        {"target": target, "F0": baseline, "T0": filtered, "group": groups, "timestamp_us": clocks},
    )
    atomic_receipt(args.output / "TEMPORAL.json", temporal)
    features = values["features"][indices]
    physical = PhaseOnlyFinal.inputs(torch.from_numpy(features)).numpy()
    probe_inputs = {
        "physical20": physical,
        "RGB_context_hidden160": features[:, :160],
        "event_context_hidden160": features[:, 170:330],
        "joint_context_hidden320": np.concatenate((features[:, :160], features[:, 170:330]), 1),
    }
    expansion = np.array(
        [
            np.log(
                (r["rgb_boxes"][-1][3] - r["rgb_boxes"][-1][1])
                / (r["rgb_boxes"][-2][3] - r["rgb_boxes"][-2][1])
            )
            / ((r["rgb_timestamps_us"][-1] - r["rgb_timestamps_us"][-2]) / 1e6)
            for r in metadata
        ]
    )
    targets = np.column_stack((1 / target, expansion))
    predictions = {key: np.empty_like(targets) for key in probe_inputs}
    constant = np.empty_like(targets)
    for fold in folds:
        train = np.isin(groups, fold["train"])
        test = np.isin(groups, fold["test"])
        constant[test] = targets[train].mean(0)
        for name, inputs in probe_inputs.items():
            probe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            probe.fit(inputs[train], targets[train])
            predictions[name][test] = probe.predict(inputs[test])
    report = {
        "scope": "context_feature_probe_not_an_encoder_or_JEPA_attribution",
        "alpha": 1.0,
        "folds": folds,
        "rows": len(indices),
        "negative_targets": int(np.sum(target < 0)),
        "metrics": {},
    }
    for name, pred in {"train_mean": constant, **predictions}.items():
        report["metrics"][name] = {
            label: {
                "mae": float(np.abs(pred[:, i] - targets[:, i]).mean()),
                "rmse": float(np.sqrt(np.mean((pred[:, i] - targets[:, i]) ** 2))),
            }
            for i, label in enumerate(("inverse_ttc", "bbox_log_expansion_per_second"))
        }
    save_npz(
        args.output / "probe_predictions.npz",
        {"target": targets, "group": groups, "train_mean": constant, **predictions},
    )
    atomic_receipt(args.output / "PROBES.json", report)
    print("COMPLETE: fixed temporal control and grouped frozen-context probes", flush=True)


if __name__ == "__main__":
    main()
