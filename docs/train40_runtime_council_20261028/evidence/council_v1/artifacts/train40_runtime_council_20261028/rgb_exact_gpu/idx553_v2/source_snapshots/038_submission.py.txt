"""Build an official-format JSON only from complete, real frozen predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_submission(tokens: list[str], frame: pd.DataFrame, *, checkpoint_sha256: str) -> dict:
    """Reject duplicates, missing/extra tokens, nonfinite values and anonymous weights."""
    if len(checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha256):
        raise ValueError("valid checkpoint SHA256 required")
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("reference tokens must be nonempty and unique")
    if not {"sample_token", "prediction"}.issubset(frame.columns):
        raise ValueError("sample_token and prediction columns required")
    if frame.sample_token.isna().any() or frame.sample_token.duplicated().any():
        raise ValueError("missing or duplicate prediction tokens")
    if set(frame.sample_token) != set(tokens):
        raise ValueError("prediction token set differs from the entire official population")
    values = np.asarray(frame.prediction, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite predictions cannot be exported")
    lookup = dict(zip(frame.sample_token, values, strict=True))
    return {
        "meta": {"format": "garlttc_prediction_v1", "checkpoint_sha256": checkpoint_sha256},
        "results": {token: {"ttc": float(lookup[token])} for token in tokens},
    }


def main() -> None:
    """Validate all official sample IDs and write locally; never submit externally."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.checkpoint.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    payload = build_submission(
        pd.read_parquet(args.inputs).sample_token.tolist(),
        pd.read_csv(args.predictions),
        checkpoint_sha256=checkpoint_hash,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
