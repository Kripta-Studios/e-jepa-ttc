"""Evaluate history truncation and A5 distillation on saved external producer features."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.efficient_context.common import digest
from operational.evttc_transfer.models import FrozenModels
from operational.garl_comparison.mid import load_scorer, scores
from operational.sota_evidence.run import json_safe
from operational.train40_system.durable_io import atomic_json

from .distill import load_student


def run(
    campaign: Path, revision: Path, student_path: Path, output: Path, scorer_path: Path
) -> None:
    """Use all existing external rows once; no fitting or endpoint selection here."""
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    frozen = FrozenModels(campaign, "cpu")
    student = load_student(student_path)
    quantized = load_student(student_path, int8_cpu=True)
    scorer = load_scorer(scorer_path)
    mean, scale = torch.from_numpy(frozen.mean), torch.from_numpy(frozen.scale)
    rows, parity = [], []
    for dataset in ("DEV32", "FCWD"):
        parent = pd.read_csv(revision / f"{dataset}_PREDICTIONS.csv")
        fragments = revision / f"replay_{dataset.lower()}" / "fragments"
        features, ids = [], []
        for path in sorted(fragments.glob("*.npz")):
            with np.load(path, allow_pickle=False) as stored:
                ids.append(str(stored["query_id"]))
                features.append(stored["compact_features"])
        if len(set(ids)) != len(ids) or set(ids) != set(parent.query_id):
            raise ValueError("external feature coverage differs from scored query set")
        parent = parent.set_index("query_id").loc[ids].reset_index()
        raw = torch.from_numpy(np.stack(features))
        for variant, length, replacement in (
            ("H8_reference", 8, None),
            ("H4_truncated", 4, None),
            ("H2_truncated", 2, None),
            ("H8_A5_student", 8, student),
            ("H8_A5_student_int8_cpu", 8, quantized),
        ):
            predictions = []
            with torch.inference_mode():
                for batch in raw.split(256):
                    values = batch[:, -length:]
                    if replacement is not None:
                        values = replacement(values)
                    times = torch.zeros((len(batch), length, 4))
                    lags = torch.arange(length - 1, -1, -1).float() / 20
                    times[:, :, 0], times[:, :, 3] = lags, lags
                    times[:, 1:, 2] = 0.05
                    args = (
                        ((values.double() - mean) / scale).float(),
                        times,
                        torch.ones((len(batch), length), dtype=torch.bool),
                        values[:, -1, 8:11],
                    )
                    outputs = torch.stack(
                        [phase_to_ttc(h(*args)["point_phase"]) for h in frozen.heads.values()]
                    )
                    predictions.append(outputs.median(0).values.numpy())
            prediction = np.concatenate(predictions)
            if variant == "H8_reference":
                difference = np.abs(prediction - parent.H8_median3.to_numpy(float))
                parity.append(
                    {
                        "dataset": dataset,
                        "max_abs_s": float(difference.max()),
                        "close": bool(
                            np.allclose(prediction, parent.H8_median3, atol=0.01, rtol=1e-4)
                        ),
                    }
                )
                if not parity[-1]["close"]:
                    raise ValueError("external head replay does not match reference")
            truth = parent.truth_ttc_seconds.to_numpy(float)
            eligible = np.isfinite(truth) & (truth != 0)
            metrics = scores(scorer, truth[eligible], prediction[eligible])
            residual = prediction[eligible] - truth[eligible]
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "eligible_n": int(eligible.sum()),
                    "MAE_s": float(np.abs(residual).mean()),
                    "signed_bias_s": float(residual.mean()),
                    "overestimate_fraction": float((residual > 0).mean()),
                    **metrics,
                }
            )
            parent[variant] = prediction
        parent.to_csv(output / f"{dataset}_PREDICTIONS.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "METRICS.csv", index=False)
    atomic_json(
        output / "RESULT.json",
        cast(
            dict,
            json_safe(
                {
                    "status": "COMPLETE_EXPLORATORY",
                    "head_parity": parity,
                    "metrics": rows,
                    "student_sha256": digest(student_path),
                    "source_sha256": digest(Path(__file__)),
                    "model_bindings": frozen.bindings,
                    "gpu_seconds": 0,
                    "optimizer_updates": 0,
                    "scope": (
                        "DEV32 and FCWD previously exposed; frozen heads with truncated histories"
                    ),
                    "selection": "no training or hyperparameter selection on these cohorts",
                    "limitations": [
                        "Does not measure live reuse or BF16 accuracy",
                        "INT8 timing is CPU student-only, not CUDA producer speed",
                        "H4 and H2 are inference ablations of H8 weights, not retrained models",
                    ],
                }
            ),
        ),
    )


def main() -> None:
    """Evaluate saved public-label cohorts without consuming GPU budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "revision", "student", "output", "scorer"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(args.campaign, args.revision, args.student, args.output, args.scorer)


if __name__ == "__main__":
    main()
