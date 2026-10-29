#!/usr/bin/env python
"""Paired cluster bootstrap for E-JEPA vs Garl on exactly matched public rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402
from e_jepa_ttc.evaluation.garl_ttc_protocol import (  # noqa: E402
    sequence_macro_signed_metrics,
    signed_garl_metrics,
)


def _read(path: Path) -> pd.DataFrame:
    return (
        pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    pred_col = next(
        (c for c in ("prediction_ttc_s", "predicted_ttc_s") if c in frame.columns), None
    )
    required = {"sample_token", "sequence_id", "target_ttc_s"}
    if pred_col is None or not required.issubset(frame.columns):
        raise ValueError(f"{prefix} predictions lack required columns")
    cols = ["sample_token", "sequence_id", "target_ttc_s", pred_col]
    if "track_id" in frame.columns:
        cols.append("track_id")
    out = frame[cols].copy()
    rename = {
        pred_col: f"prediction_{prefix}",
        "target_ttc_s": f"target_{prefix}",
    }
    if "track_id" in out.columns:
        rename["track_id"] = f"track_id_{prefix}"
    out = out.rename(columns=rename)
    if out["sample_token"].astype(str).duplicated().any():
        raise ValueError(f"{prefix} contains duplicate sample_token")
    return out


def _metrics(target: np.ndarray, pred: np.ndarray, sequence: np.ndarray) -> dict[str, float]:
    signed = signed_garl_metrics(target, pred)
    macro = sequence_macro_signed_metrics(target, pred, sequence)
    return {
        "sequence_macro_MiD": float(macro["sequence_macro_paper_MiD_overall"]),
        "sample_weighted_MiD": float(signed["paper_MiD_overall"]),
        "failure_pct": float(signed["failure_rate_pct"]),
        "finite_prediction_fraction": float(np.isfinite(pred).mean()),
    }


def run(
    ejepa_path: Path,
    garl_path: Path,
    output: Path,
    resamples: int,
    seed: int,
    cluster_metadata: Path | None = None,
    *,
    comparison_scope: str = "current_candidate_public_validation",
) -> dict[str, Any]:
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if comparison_scope not in {
        "current_candidate_public_validation",
        "diagnostic_only",
    }:
        raise ValueError(f"unsupported comparison scope: {comparison_scope}")
    e = _normalize(_read(ejepa_path), "ejepa")
    g = _normalize(_read(garl_path), "garl")
    if set(e["sample_token"].astype(str)) != set(g["sample_token"].astype(str)):
        raise ValueError("Prediction sample-token sets are not identical; comparison invalid")
    merged = e.merge(
        g, on=["sample_token", "sequence_id"], validate="one_to_one", suffixes=("_ejepa", "_garl")
    )
    if len(merged) != len(e):
        raise ValueError("Matched merge lost rows")
    if not np.allclose(
        merged["target_ejepa"], merged["target_garl"], rtol=0.0, atol=1e-5, equal_nan=False
    ):
        raise ValueError("Targets differ between E-JEPA and Garl predictions")
    prediction_track: pd.Series | None = None
    if "track_id_ejepa" in merged.columns and "track_id_garl" in merged.columns:
        ejepa_track = merged["track_id_ejepa"].astype(str)
        garl_track = merged["track_id_garl"].astype(str)
        if not (ejepa_track.to_numpy() == garl_track.to_numpy()).all():
            raise ValueError("Track IDs differ between E-JEPA and Garl predictions")
        prediction_track = ejepa_track
        clustering = "sequence_track"
    elif "track_id_ejepa" in merged.columns:
        prediction_track = merged["track_id_ejepa"].astype(str)
        clustering = "sequence_track_ejepa_only"
    elif "track_id_garl" in merged.columns:
        prediction_track = merged["track_id_garl"].astype(str)
        clustering = "sequence_track_garl_only"
    else:
        clustering = "sequence_only_fallback"

    if cluster_metadata is not None:
        metadata = _read(cluster_metadata)
        required_meta = {"sample_token", "sequence_id", "track_id"}
        if not required_meta.issubset(metadata.columns):
            raise ValueError("Cluster metadata lacks sample_token/sequence_id/track_id")
        metadata = metadata[["sample_token", "sequence_id", "track_id"]].copy()
        if metadata["sample_token"].astype(str).duplicated().any():
            raise ValueError("Cluster metadata contains duplicate sample_token")
        metadata = metadata.rename(columns={"track_id": "track_id_metadata"})
        merged = merged.merge(metadata, on=["sample_token", "sequence_id"], validate="one_to_one")
        if len(merged) != len(e):
            raise ValueError("Cluster metadata did not cover all paired rows")
        metadata_track = merged["track_id_metadata"].astype(str)
        if prediction_track is not None:
            if not (prediction_track.to_numpy() == metadata_track.to_numpy()).all():
                raise ValueError("Prediction and external metadata track IDs differ")
            clustering = f"{clustering}_verified_external_metadata"
        else:
            clustering = "sequence_track_external_metadata"
        merged["track_id"] = metadata_track
    elif prediction_track is not None:
        merged["track_id"] = prediction_track
    else:
        # Never use sample-level pseudo-clusters for temporally correlated windows.
        # With no track metadata, sequence is the conservative clustering unit.
        merged["track_id"] = "__sequence_only__"
        clustering = "sequence_only_fallback"
    merged["cluster"] = merged["sequence_id"].astype(str) + "::" + merged["track_id"].astype(str)
    groups = [
        idx.to_numpy(dtype=np.int64)
        for _, idx in merged.groupby("cluster", sort=True).groups.items()
    ]
    if len(groups) < 2:
        raise ValueError("Need at least two bootstrap clusters")
    target = merged["target_ejepa"].to_numpy(dtype=np.float64)
    pe = merged["prediction_ejepa"].to_numpy(dtype=np.float64)
    pg = merged["prediction_garl"].to_numpy(dtype=np.float64)
    seq = merged["sequence_id"].astype(str).to_numpy()
    point_e = _metrics(target, pe, seq)
    point_g = _metrics(target, pg, seq)
    point_delta = {k: point_e[k] - point_g[k] for k in point_e}
    token_digest = hashlib.sha256()
    for token in sorted(merged["sample_token"].astype(str).tolist()):
        token_digest.update(token.encode("utf-8"))
        token_digest.update(b"\0")
    rng = np.random.default_rng(seed)
    deltas_mid = np.empty(resamples, dtype=np.float64)
    deltas_failure = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        chosen = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[j] for j in chosen])
        # Preserve original sequence labels: resampling happens at sequence+track
        # cluster level, while the reported primary metric remains sequence-macro.
        seq_boot = seq[indices]
        me = _metrics(target[indices], pe[indices], seq_boot)
        mg = _metrics(target[indices], pg[indices], seq_boot)
        deltas_mid[i] = me["sequence_macro_MiD"] - mg["sequence_macro_MiD"]
        deltas_failure[i] = me["failure_pct"] - mg["failure_pct"]

    def ci(values: np.ndarray) -> dict[str, float]:
        return {
            "lower_95": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "upper_95": float(np.quantile(values, 0.975)),
        }

    report: dict[str, Any] = {
        "artifact_type": "paired_cluster_bootstrap_ejepa_vs_garl_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "completed_public_validation_only",
        "comparison_scope": comparison_scope,
        "rows": len(merged),
        "clusters": len(groups),
        "cluster_definition": clustering,
        "resamples": resamples,
        "seed": seed,
        "checks": {
            "exact_sample_tokens": True,
            "target_equality_atol_1e_5": True,
            "paired_evaluation": True,
            "input_provenance_complete": True,
            "private_test_opened": False,
        },
        "sample_contract": {
            "sample_token_count": len(merged),
            "sorted_sample_tokens_sha256": token_digest.hexdigest(),
        },
        "sources": {
            "ejepa_predictions": {
                "path": str(ejepa_path.resolve()),
                "sha256": _sha256(ejepa_path),
            },
            "garl_predictions": {
                "path": str(garl_path.resolve()),
                "sha256": _sha256(garl_path),
            },
            "cluster_metadata": (
                {
                    "path": str(cluster_metadata.resolve()),
                    "sha256": _sha256(cluster_metadata),
                }
                if cluster_metadata is not None
                else None
            ),
        },
        "ejepa": point_e,
        "garl": point_g,
        "delta_ejepa_minus_garl": point_delta,
        "bootstrap": {
            "sequence_macro_MiD_delta_ejepa_minus_garl": ci(deltas_mid),
            "failure_pct_delta_ejepa_minus_garl": ci(deltas_failure),
            "probability_ejepa_lower_MiD": float(np.mean(deltas_mid < 0.0)),
        },
        "sota_claim_authorized": False,
    }
    sign_artifact(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ejepa-predictions", type=Path, required=True)
    p.add_argument("--garl-predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resamples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260811)
    p.add_argument("--cluster-metadata", type=Path)
    p.add_argument(
        "--comparison-scope",
        choices=("current_candidate_public_validation", "diagnostic_only"),
        default="current_candidate_public_validation",
    )
    args = p.parse_args()
    try:
        report = run(
            args.ejepa_predictions.resolve(),
            args.garl_predictions.resolve(),
            args.output.resolve(),
            args.resamples,
            args.seed,
            args.cluster_metadata.resolve() if args.cluster_metadata else None,
            comparison_scope=args.comparison_scope,
        )
    except Exception as exc:
        print(f"paired bootstrap failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
