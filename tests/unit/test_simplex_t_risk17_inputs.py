"""Synthetic publication validation; never fit a historical router."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.risk17_inputs import load_risk17_replay


@pytest.mark.parametrize("failure", ["", "prediction", "identity", "role", "bytes", "authority"])
def test_risk17_comparator_contract(tmp_path: Path, failure: str) -> None:
    ids = np.arange(8192)
    identity = pd.DataFrame(
        {
            "sample_token": [f"q{i}" for i in ids],
            "sequence_id": [f"s{i % 9}" for i in ids],
            "track_id": ids,
            "outer_fold": ids % 3,
            "target_ttc": np.ones(8192),
        }
    )
    frame = identity.assign(
        arm="S65-RISK17",
        seed=7,
        selected_expert=0,
        prediction_ttc_s=1.0,
        a5_ttc_s=1.0,
        c2f_ttc_s=2.0,
        pair_ttc_s=3.0,
    )
    if failure == "prediction":
        frame.loc[0, "prediction_ttc_s"] = 2.0
    if failure == "identity":
        frame.loc[0, "track_id"] = -1
    path = tmp_path / "RISK17_OLD.parquet"
    frame.to_parquet(path, index=False)
    bindings = [
        {
            "fold": fold,
            "table": {
                "outer_fold": fold,
                "role": "inner_oof" if failure == "role" else "outer_dev",
            },
        }
        for fold in range(3)
    ]
    report = {
        "status": "HISTORICAL_RISK17_OLD_REPLAY_EXACT",
        "queries": 8192,
        "optimizer_updates": 0,
        "scientific_stage_authorized": False,
        "ancestry_sha256": "a" * 64,
        "bindings": bindings,
        "payload_sha256": sha256(path),
    }
    manifest = tmp_path / "REPLAY.json"
    manifest.write_text(json.dumps(report), encoding="utf-8")
    pin = sha256(manifest)
    if failure == "bytes":
        path.write_bytes(b"changed")

    def validate():
        if failure == "authority":
            raise ValueError("unverified authority")

    arguments = dict(
        manifest_sha256=pin,
        expected_identity=identity,
        expected_bindings=bindings,
        ancestry_sha256="a" * 64,
        validate_prerequisites=validate,
    )
    if failure:
        with pytest.raises(ValueError):
            load_risk17_replay(manifest, **arguments)
    else:
        result = load_risk17_replay(manifest, **arguments)
        assert len(result) == 8192
        assert result.prediction_ttc_s.eq(1.0).all()
