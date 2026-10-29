"""Synthetic publication validation; never fit a historical router."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import risk17_inputs as module
from e_jepa_ttc.simplex_t.risk17_inputs import load_risk17_replay, load_selector_replay


@pytest.mark.parametrize(
    "failure", ["", "index", "weight", "ancestry", "fit_role", "fold", "resource", "cohort"]
)
def test_acknowledged_router_bindings(tmp_path: Path, monkeypatch, failure: str):
    ridge = tmp_path / "frozen_audit/extracted_input/run/stage65"
    (tmp_path / "tables").mkdir()
    ack = {
        "producers": {
            "authoritative_historical_manifest": {
                "path": str(tmp_path / "NESTED_ANCESTRY_AUDIT.json"),
                "sha256": "a" * 64,
            }
        },
        "interfaces": {"role_manifest": {"roles": {"original": ["old"]}}},
    }
    fits, tables = [], []
    for fold in range(3):
        checkpoint = ridge / f"outer{fold}/S65-RISK17.npz"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fixture")
        fit = {
            "model": "S65-RISK17",
            "outer_fold": fold,
            "sha256": sha256(checkpoint),
            "evidence_type": "train_only_frozen_router_fit",
        }
        if failure == "fit_role":
            fit["evidence_type"] = "unknown"
        if failure == "weight" and fold == 0:
            checkpoint.write_bytes(b"changed")
        fits.append(fit)
        table = {
            "outer_fold": fold,
            "role": "outer_dev",
            "ancestry_sha256": "b" * 64 if failure == "ancestry" else "a" * 64,
        }
        for suffix, key in (("csv", "metadata_sha256"), ("npz", "arrays_sha256")):
            path = tmp_path / "tables" / f"outer{fold}_outer_dev.{suffix}"
            path.write_bytes(b"fixture")
            table[key] = sha256(path)
        tables.append(table)
    if failure == "fold":
        fits.pop()
    index = tmp_path / "FROZEN_EXPERT_TABLE_INDEX.json"
    manifest = ridge / "ALL_RIDGE_FITS_FROZEN.json"
    index.write_text(json.dumps(tables), encoding="utf-8")
    manifest.write_text(
        json.dumps({"fits": fits, "evidence_type": "all_router_fits_frozen_before_evaluation"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "verified_ack", lambda *args: ack)
    calls = []

    def replay(*args, **kwargs):
        kwargs["validate_prerequisites"]()
        calls.append("verified")
        assert kwargs["expected_bindings"] == [
            {"fold": o, "weights_sha256": fits[o]["sha256"], "table": tables[o]} for o in range(3)
        ]
        return kwargs["expected_identity"]

    monkeypatch.setattr(module, "load_risk17_replay", replay)

    def run():
        return module.load_acknowledged_risk17(
            tmp_path / "REPLAY.json",
            manifest_sha256="f" * 64,
            ack_path=tmp_path / "ACK.json",
            ack_sha256="c" * 64,
            table_index_sha256="0" * 64 if failure == "index" else sha256(index),
            ridge_manifest_sha256=sha256(manifest),
            expected_identity=pd.DataFrame(
                {"sequence_id": ["closed" if failure == "cohort" else "old"]}
            ),
            resource_ok=lambda: failure != "resource",
        )

    if failure:
        with pytest.raises(InterruptedError if failure == "resource" else ValueError):
            run()
        assert not calls
    else:
        assert len(run()) == 1
        assert calls == ["verified"]


@pytest.mark.parametrize("failure", ["", "prediction", "identity", "role", "bytes", "authority"])
@pytest.mark.parametrize("comparator", ["S65-RISK17", "S67-SIMPLEX17"])
def test_risk17_comparator_contract(tmp_path: Path, failure: str, comparator: str) -> None:
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
        arm=comparator,
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
    path = tmp_path / (
        "RISK17_OLD.parquet" if comparator == "S65-RISK17" else "SIMPLEX17_OLD.parquet"
    )
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
        "status": "HISTORICAL_RISK17_OLD_REPLAY_EXACT"
        if comparator == "S65-RISK17"
        else "HISTORICAL_SIMPLEX17_OLD_REPLAY_EXACT_FP32",
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

    arguments: dict[str, Any] = dict(
        manifest_sha256=pin,
        expected_identity=identity,
        expected_bindings=bindings,
        ancestry_sha256="a" * 64,
        validate_prerequisites=validate,
    )
    if failure:
        with pytest.raises(ValueError):
            load_selector_replay(manifest, comparator=comparator, **arguments)
    else:
        result = load_selector_replay(manifest, comparator=comparator, **arguments)
        assert len(result) == 8192
        assert result.prediction_ttc_s.eq(1.0).all()
        if comparator == "S65-RISK17":
            assert load_risk17_replay(manifest, **arguments).equals(result)
        else:
            with pytest.raises(ValueError):
                load_risk17_replay(manifest, **arguments)
