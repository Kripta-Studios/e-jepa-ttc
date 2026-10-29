"""Reproduce descriptive P/H error attribution from frozen V15 predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root.parent
        / "e-jepa-ttc-v15-range-readout/artifacts/range_readout_20261012/PRODUCER_REACH_P_H.csv"
    )
    out = root / "artifacts/representation_audit_20261012/diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(source)
    if set(data.role) != {"P", "H"}:
        raise ValueError("Only P/H diagnostics allowed")
    data["band"] = np.select(
        [
            (data.target_ttc > 0) & (data.target_ttc <= 1),
            (data.target_ttc > 1) & (data.target_ttc <= 2),
            (data.target_ttc > 2) & (data.target_ttc <= 3),
            (data.target_ttc > 3) & (data.target_ttc <= 6),
            data.target_ttc > 6,
        ],
        ["0-1", "1-2", "2-3", "3-6", "6+"],
        default="negative",
    )
    data["absolute_error"] = (data.prediction_ttc - data.target_ttc).abs()
    data["wrong_sign"] = np.sign(data.prediction_ttc) != np.sign(data.target_ttc)
    summaries = []
    for keys, part in data.groupby(["role", "producer", "band"], sort=True):
        assert isinstance(keys, tuple)
        errors = part.absolute_error.to_numpy()
        good = part.loc[~part.wrong_sign]
        summaries.append(
            dict(zip(["role", "producer", "band"], keys, strict=True))
            | {
                "n": len(part),
                "groups": part.group_id.nunique(),
                "mae": errors.mean(),
                "median_ae": float(np.median(errors)),
                "p95_ae": float(np.quantile(errors, 0.95)),
                "top5_error_share": float(np.sort(errors)[-5:].sum() / errors.sum()),
                "wrong_sign_count": int(part.wrong_sign.sum()),
                "wrong_sign_error_share": float(
                    part.loc[part.wrong_sign, "absolute_error"].sum() / errors.sum()
                ),
                "correct_sign_mae": float(good.absolute_error.mean()) if len(good) else None,
                "median_prediction": float(part.prediction_ttc.median()),
                "mean_residual": float(part.residual_last.mean()),
                "positive_saturation_count": int((part.residual_over_cap > 0.95).sum()),
                "negative_saturation_count": int((part.residual_over_cap < -0.95).sum()),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "ERROR_ATTRIBUTION.csv", index=False)
    short = data.loc[data.band == "0-1"]
    short.groupby(["role", "producer", "group_id"]).agg(
        n=("sample_token", "size"),
        mae=("absolute_error", "mean"),
        median_prediction=("prediction_ttc", "median"),
        wrong_sign=("wrong_sign", "sum"),
    ).to_csv(out / "SHORT_BY_GROUP.csv")
    short.sort_values("absolute_error", ascending=False).to_csv(out / "SHORT_ROWS.csv", index=False)
    meta = {
        "source": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": len(data),
        "optimizer_updates": 0,
        "scope": "descriptive frozen P/H only; no new selection on V",
        "limitations": (
            "Queries are correlated; groups are not independent replicas; "
            "correct-sign subset is diagnostic only."
        ),
    }
    (out / "PROVENANCE.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(summary.loc[summary.band == "0-1"].to_string(index=False))


if __name__ == "__main__":
    main()
