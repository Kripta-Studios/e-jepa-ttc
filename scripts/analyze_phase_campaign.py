"""Audit local targets and compare frozen branches, bbox geometry and exposed F1."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.evttc_transfer_audit.common import file_sha
from operational.final_refit.training import PhaseOnlyFinal
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_adapt_training import load_head
from operational.phase_campaign.data import load_data
from operational.phase_campaign.evaluation import bbox_ttc, metrics, phase_interval_metrics
from operational.phase_campaign.protocol import family_folds
from scripts.final_train40_transfer import HEAD, verify_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "candidate", "adapted", "protocol", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    values, rows, receipt = load_data(args.data)
    records = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))["records"]
    excluded = {r["sequence_id"] for r in records if r["role"] == "excluded"}
    selected = np.flatnonzero(values["domain"] == "evttc")
    f = torch.from_numpy(values["features"][selected])
    w = torch.from_numpy(values["rgb_width_logits"][selected])
    truth_phase = values["target_phase"][selected]
    groups = values["group_id"][selected]
    metadata = [rows[str(values["sample_token"][i])] for i in selected]
    paths, parent = verify_candidate(args.candidate)
    baseline = PhaseOnlyFinal().eval()
    baseline.load_state_dict(
        torch.load(paths[HEAD], map_location="cpu", weights_only=False)["model_state_dict"]
    )
    adapted = load_head(args.adapted, file_sha(args.adapted))
    with torch.inference_mode():
        y = phase_to_ttc(torch.from_numpy(truth_phase)).numpy()
        before, after = baseline(f, w), adapted(f, w)
        predictions = {
            "F0": before["ttc"].numpy(),
            "F1_all_data_resubstitution": after["ttc"].numpy(),
        }
        for name, index in {
            "RGB_context": 161,
            "event_context": 331,
            "RGB_A5": 162,
            "RGB_C2F": 163,
            "RGB_PAIR": 164,
            "event_A5": 332,
            "event_C2F": 333,
            "event_PAIR": 334,
        }.items():
            predictions[name] = phase_to_ttc(f[:, index]).numpy()
    for axis in ("height", "width", "area"):
        predictions[f"bbox_{axis}"] = np.array(
            [
                bbox_ttc(np.array(r["rgb_boxes"]), np.array(r["rgb_timestamps_us"]), axis)
                for r in metadata
            ]
        )
    quality = np.log1p(torch.nn.functional.softplus(w).sum(1).numpy())
    error = np.abs(predictions["F0"] - y) / np.abs(y)
    quality_report = []
    for group in sorted(set(groups)):
        m = groups == group
        correlation = spearmanr(quality[m], error[m])
        quality_report.append(
            {
                "recording": str(group),
                "rows": int(m.sum()),
                "spearman_width_error": float(correlation[0])
                if np.isfinite(correlation[0])
                else None,
            }
        )
    target_audit = []
    for label in sorted({r["ttc_csv_path"] for r in metadata}):
        table = np.loadtxt(label, ndmin=2)
        valid = np.isfinite(table[:, 2:5]).all(1) & (table[:, 3] != 0)
        expected = table[valid, 2] / table[valid, 3]
        difference = np.abs(expected - table[valid, 4])
        target_audit.append(
            {
                "path": label,
                "sha256": file_sha(Path(label)),
                "rows": len(table),
                "monotonic_time": bool(np.all(np.diff(table[:, 1]) > 0)),
                "max_abs_ttc_minus_depth_over_speed": float(difference.max())
                if len(difference)
                else None,
                "median_abs_ttc_minus_depth_over_speed": float(np.median(difference))
                if len(difference)
                else None,
            }
        )
    report = {
        "scope": "local_exposed_diagnostics_not_official_scoring",
        "parent": parent,
        "data": receipt,
        "folds": family_folds(records),
        "excluded_recordings": sorted(excluded),
        "target_audit": target_audit,
        "quality_width_vs_error": quality_report,
        "models": {},
        "intervals": {},
    }
    for name, pred in predictions.items():
        report["models"][name] = {
            "all": metrics(y, pred),
            "eligible24": metrics(
                y[~np.isin(groups, list(excluded))], pred[~np.isin(groups, list(excluded))]
            ),
            "recordings": {
                str(g): metrics(y[groups == g], pred[groups == g]) for g in sorted(set(groups))
            },
        }
    for name, prediction in (("F0", before), ("F1_all_data_resubstitution", after)):
        report["intervals"][name] = phase_interval_metrics(
            truth_phase, prediction["q10"].numpy(), prediction["q90"].numpy()
        )
    save_npz(
        args.output / "predictions.npz",
        {
            "target": y,
            "group": groups,
            "sample_token": values["sample_token"][selected],
            "quality_width": quality,
            **predictions,
        },
    )
    atomic_receipt(args.output / "REPORT.json", report)
    print(
        json.dumps(
            {name: value["eligible24"] for name, value in report["models"].items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
