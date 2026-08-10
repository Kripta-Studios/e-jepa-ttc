"""Build a signed, token-exact comparison between two CausalScale arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402
from e_jepa_ttc.evaluation.bootstrap import (  # noqa: E402
    paired_sequence_bootstrap_difference,
)
from e_jepa_ttc.evaluation.garl_ttc_protocol import (  # noqa: E402
    sequence_macro_signed_metrics,
    signed_garl_metrics,
)

DELTA_T_S = 0.1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_predictions(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"sample_token", "sequence_id", "target_ttc_s", "prediction_ttc_s"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} predictions lack columns: {missing}")
    if frame["sample_token"].duplicated().any():
        raise ValueError(f"{label} predictions contain duplicate tokens")
    return frame[list(sorted(required))].copy()


def _arm_metrics(frame: pd.DataFrame, prediction: str) -> dict[str, Any]:
    target = frame["target_ttc_s"].to_numpy(dtype=np.float64)
    values = frame[prediction].to_numpy(dtype=np.float64)
    sequence = frame["sequence_id"].astype(str).to_numpy()
    return {
        "signed": signed_garl_metrics(target, values),
        "sequence_macro": sequence_macro_signed_metrics(target, values, sequence),
    }


def _mid_per_sample(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        target_eta = 1.0 - DELTA_T_S / target
        prediction_eta = 1.0 - DELTA_T_S / prediction
        return np.abs(np.log(target_eta) - np.log(prediction_eta)) * 1e4


def build_comparison(
    *,
    reference_predictions: Path,
    reference_summary: Path,
    candidate_predictions: Path,
    candidate_summary: Path,
    output_json: Path,
    reference_label: str,
    candidate_label: str,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 7,
) -> dict[str, Any]:
    """Compare public-validation arms with complete-sequence bootstrap."""

    for path in (
        reference_predictions,
        reference_summary,
        candidate_predictions,
        candidate_summary,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    reference = _read_predictions(reference_predictions, reference_label).rename(
        columns={
            "sequence_id": "reference_sequence_id",
            "target_ttc_s": "reference_target_ttc_s",
            "prediction_ttc_s": "reference_prediction_ttc_s",
        }
    )
    candidate = _read_predictions(candidate_predictions, candidate_label).rename(
        columns={
            "sequence_id": "candidate_sequence_id",
            "target_ttc_s": "candidate_target_ttc_s",
            "prediction_ttc_s": "candidate_prediction_ttc_s",
        }
    )
    aligned = reference.merge(candidate, on="sample_token", validate="one_to_one")
    if len(aligned) != len(reference) or len(aligned) != len(candidate):
        raise ValueError("arm prediction token sets differ")
    if not (
        aligned["reference_sequence_id"].astype(str)
        == aligned["candidate_sequence_id"].astype(str)
    ).all():
        raise ValueError("arm prediction sequence IDs differ")
    target = aligned["reference_target_ttc_s"].to_numpy(dtype=np.float64)
    if not np.allclose(
        target,
        aligned["candidate_target_ttc_s"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("arm prediction targets differ")
    aligned["sequence_id"] = aligned["reference_sequence_id"].astype(str)
    aligned["target_ttc_s"] = target
    reference_metrics = _arm_metrics(aligned, "reference_prediction_ttc_s")
    candidate_metrics = _arm_metrics(aligned, "candidate_prediction_ttc_s")

    def paper_mid(truth: np.ndarray, estimate: np.ndarray) -> float:
        return float(signed_garl_metrics(truth, estimate)["paper_MiD_overall"])

    reference_prediction = aligned["reference_prediction_ttc_s"].to_numpy(
        dtype=np.float64
    )
    candidate_prediction = aligned["candidate_prediction_ttc_s"].to_numpy(
        dtype=np.float64
    )
    bootstrap = paired_sequence_bootstrap_difference(
        target,
        reference_prediction,
        candidate_prediction,
        aligned["sequence_id"].to_numpy(),
        metric=paper_mid,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    reference_error = _mid_per_sample(target, reference_prediction)
    candidate_error = _mid_per_sample(target, candidate_prediction)
    finite = np.isfinite(reference_error) & np.isfinite(candidate_error)
    wins = candidate_error[finite] < reference_error[finite]
    result: dict[str, Any] = {
        "artifact_type": "causal_scale_eap_arm_comparison_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "completed_public_validation_only",
        "scope": {
            "sample_count": len(aligned),
            "sequence_count": int(aligned["sequence_id"].nunique()),
            "exact_token_equality_verified": True,
            "target_equality_verified": True,
            "private_test_opened": False,
            "codabench_opened": False,
            "evttc_test_opened": False,
        },
        "reference": {"label": reference_label, "metrics": reference_metrics},
        "candidate": {"label": candidate_label, "metrics": candidate_metrics},
        "paired": {
            "finite_count": int(finite.sum()),
            "candidate_win_count": int(wins.sum()),
            "candidate_win_rate": float(wins.mean()) if wins.size else float("nan"),
            "candidate_minus_reference_mean_mid": (
                float(np.mean(candidate_error[finite] - reference_error[finite]))
                if np.any(finite)
                else float("nan")
            ),
            "candidate_minus_reference_sequence_bootstrap_paper_MiD": bootstrap,
            "bootstrap_unit": "complete_sequence",
            "window_level_bootstrap_used": False,
        },
        "inputs": {
            "reference_predictions": {
                "path": str(reference_predictions.resolve()),
                "sha256": _sha256(reference_predictions),
            },
            "reference_summary": {
                "path": str(reference_summary.resolve()),
                "sha256": _sha256(reference_summary),
            },
            "candidate_predictions": {
                "path": str(candidate_predictions.resolve()),
                "sha256": _sha256(candidate_predictions),
            },
            "candidate_summary": {
                "path": str(candidate_summary.resolve()),
                "sha256": _sha256(candidate_summary),
            },
        },
    }
    sign_artifact(result)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        result = build_comparison(**vars(args))
    except Exception as error:
        parser.exit(2, f"arm comparison failed: {type(error).__name__}: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
