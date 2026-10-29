from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from operational.final_refit.folds import (
    _audit_context_closure,
    audit_rotation0_reuse,
    iter_cache_lookups,
    select_rotation_rows,
    sha256,
    validate_rotation_contract,
)


def _fixture() -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    blocks: dict[str, list[str]] = {}
    source = 0
    for block in range(5):
        groups = [f"g{block}_{index}" for index in range(8)]
        blocks[f"B{block}"] = groups
        for group in groups:
            historical_role = "V" if block == 0 else "H" if block == 1 else "P"
            rows.append(
                {
                    "historical_local_ordinal": source + 100,
                    "source_ordinal": source,
                    "sample_token": f"token_{source}",
                    "group_id": group,
                    "historical_role": historical_role,
                }
            )
            source += 1
    frame = pd.DataFrame(rows)
    rotations = []
    all_groups = set(frame.group_id)
    for number in range(5):
        generation = blocks[f"B{number}"]
        refiner = blocks[f"B{(number + 1) % 5}"]
        producer = sorted(all_groups - set(generation) - set(refiner))
        rotations.append(
            {
                "rotation": number,
                "producer_groups": producer,
                "refiner_groups": refiner,
                "generation_groups": generation,
                "queries": {"producer_pair": 24, "refiner": 8, "generation": 8},
            }
        )
    review: dict[str, Any] = {
        "schema": "final_refit_nested_rotation_review_v1",
        "partition": "deterministic; no TTC labels or scores",
        "group_blocks": blocks,
        "rotations": rotations,
    }
    return frame, review


def test_contract_covers_each_token_once_and_preserves_cache_lookup() -> None:
    rows, review = _fixture()
    validate_rotation_contract(rows, review)
    rotation = review["rotations"][0]  # type: ignore[index]
    selected = select_rotation_rows(rows, rotation, "generation")
    assert len(selected) == 8
    assert list(iter_cache_lookups(selected)) == list(
        zip(
            selected["historical_role"].astype(str),
            selected["historical_local_ordinal"].astype(int),
            strict=True,
        )
    )
    assert selected.source_ordinal.is_monotonic_increasing


@pytest.mark.parametrize("mutation", ["overlap", "count", "policy", "token"])
def test_contract_fails_closed_on_leakage_or_identity_change(mutation: str) -> None:
    rows, review = _fixture()
    changed = copy.deepcopy(review)
    if mutation == "overlap":
        rotations = changed["rotations"]
        rotations[0]["producer_groups"][0] = rotations[0]["generation_groups"][0]
    elif mutation == "count":
        rotations = changed["rotations"]
        rotations[1]["queries"]["generation"] = 9
    elif mutation == "policy":
        changed["partition"] = "selected using score"
    else:
        rows.loc[1, "sample_token"] = rows.loc[0, "sample_token"]
    with pytest.raises(ValueError):
        validate_rotation_contract(rows, changed)


def test_unknown_role_is_rejected() -> None:
    rows, review = _fixture()
    rotation = review["rotations"][0]  # type: ignore[index]
    with pytest.raises(ValueError, match="Unknown rotation role"):
        select_rotation_rows(rows, rotation, "test")


def test_rotation0_reuse_is_not_admitted_when_receipts_are_missing(tmp_path) -> None:
    sources = {
        "P": {"manifest_sha256": "p"},
        "H": {"manifest_sha256": "h"},
        "V": {"manifest_sha256": "v"},
    }
    audit = audit_rotation0_reuse(tmp_path, sources)
    assert audit["schema"] == "final_refit_rotation0_reuse_audit_v2"
    assert audit["status"] == "NOT_ADMITTED"
    assert audit["reuse_admitted"] is False
    assert "FileNotFoundError" in audit["reason"]


def _context_closure_fixture(root: Path) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    feature_dir = root / "features" / "H"
    fit_dir = root / "fits" / "E_CTX_MATCHED"
    round_dir = root / "rounds"
    feature_dir.mkdir(parents=True)
    fit_dir.mkdir(parents=True)
    round_dir.mkdir(parents=True)
    checkpoint_hashes = {
        "E_A5_MATCHED": "1" * 64,
        "E_C2F_MATCHED": "2" * 64,
        "R_A5": "3" * 64,
        "R_C2F": "4" * 64,
        "PAIR_E_MATCHED": "5" * 64,
        "PAIR_R": "6" * 64,
    }
    endpoints = {
        name: {"checkpoint_sha256": digest, "historical_role": "P"}
        for name, digest in checkpoint_hashes.items()
    }
    freeze = {
        "schema": "rgb_port_parent_round_freeze_v1",
        "endpoints": {
            name: {"checkpoint_sha256": digest} for name, digest in checkpoint_hashes.items()
        },
    }
    freeze_path = round_dir / "PRODUCERS_PAIR_ENDPOINT_FREEZE.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    producer_sha = sha256(freeze_path)
    ids = np.asarray(["obs-a", "obs-b"])
    observation_path = feature_dir / "E_OBSERVATIONS.npz"
    np.savez(
        observation_path,
        observation_id=ids,
        features=np.zeros((2, 17), dtype=np.float32),
        anchor_us=np.asarray([10, 20], dtype=np.int64),
        available_us=np.asarray([11, 21], dtype=np.int64),
        role=np.asarray(["H", "H"]),
        schema_sha256=np.asarray("schema"),
        endpoint_freeze_sha256=np.asarray(producer_sha),
    )
    observation_receipt = {
        "schema": "rgb_port_observation_features_v1",
        "status": "COMPLETE",
        "role": "H",
        "modality": "event",
        "rows": 2,
        "schema_sha256": "schema",
        "endpoint_hashes": checkpoint_hashes,
        "output": str(observation_path),
        "output_sha256": sha256(observation_path),
    }
    (feature_dir / "E_OBSERVATIONS.json").write_text(
        json.dumps(observation_receipt), encoding="utf-8"
    )
    normalizer = {
        "schema": "rgb_port_normalizer_v1",
        "fit_role": "H",
        "modality": "event",
        "schema_sha256": "schema",
        "producer_sha256": producer_sha,
        "unique_ids_sha256": hashlib.sha256(b"obs-a\nobs-b").hexdigest(),
        "mean": [0.0] * 17,
        "scale": [1.0] * 17,
    }
    normalizer_path = feature_dir / "E_NORMALIZER.json"
    normalizer_path.write_text(json.dumps(normalizer), encoding="utf-8")
    history_path = feature_dir / "E_HISTORY.npz"
    np.savez(
        history_path,
        sample_token=np.asarray(["token-a", "token-b"]),
        group_id=np.asarray(["group-a", "group-b"]),
        role=np.asarray(["H", "H"]),
    )
    history_receipt = {
        "schema": "rgb_port_history_features_v1",
        "status": "COMPLETE",
        "role": "H",
        "modality": "event",
        "queries": 2,
        "source_sha256": sha256(observation_path),
        "normalizer_sha256": sha256(normalizer_path),
        "output": str(history_path),
        "output_sha256": sha256(history_path),
    }
    (feature_dir / "E_HISTORY.json").write_text(json.dumps(history_receipt), encoding="utf-8")
    (fit_dir / "RUN_PROVENANCE.json").write_text(
        json.dumps(
            {
                "source_sha256": sha256(history_path),
                "normalizer_sha256": sha256(normalizer_path),
            }
        ),
        encoding="utf-8",
    )
    h_rows = pd.DataFrame(
        {"sample_token": ["token-a", "token-b"], "group_id": ["group-a", "group-b"]}
    )
    return endpoints, h_rows


def test_context_closure_proves_transitive_endpoints_normalizer_and_h_groups(tmp_path) -> None:
    endpoints, h_rows = _context_closure_fixture(tmp_path)
    result = _audit_context_closure(
        tmp_path,
        fit_id="E_CTX_MATCHED",
        prefix="E",
        modality="event",
        endpoints=endpoints,
        h_rows=h_rows,
    )
    assert result["transitive_closure_verified"] is True
    assert result["h_groups"] == 2


@pytest.mark.parametrize("mutation", ["endpoint", "fit_role", "ids", "groups"])
def test_context_closure_rejects_incomplete_or_cross_role_evidence(tmp_path, mutation: str) -> None:
    endpoints, h_rows = _context_closure_fixture(tmp_path)
    feature_dir = tmp_path / "features" / "H"
    if mutation == "endpoint":
        path = feature_dir / "E_OBSERVATIONS.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["endpoint_hashes"]["E_A5_MATCHED"] = "f" * 64
        path.write_text(json.dumps(value), encoding="utf-8")
    elif mutation == "fit_role":
        path = feature_dir / "E_NORMALIZER.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["fit_role"] = "V"
        path.write_text(json.dumps(value), encoding="utf-8")
        history = json.loads((feature_dir / "E_HISTORY.json").read_text(encoding="utf-8"))
        history["normalizer_sha256"] = sha256(path)
        (feature_dir / "E_HISTORY.json").write_text(json.dumps(history), encoding="utf-8")
        provenance = tmp_path / "fits" / "E_CTX_MATCHED" / "RUN_PROVENANCE.json"
        value = json.loads(provenance.read_text(encoding="utf-8"))
        value["normalizer_sha256"] = sha256(path)
        provenance.write_text(json.dumps(value), encoding="utf-8")
    elif mutation == "ids":
        path = feature_dir / "E_NORMALIZER.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["unique_ids_sha256"] = "0" * 64
        path.write_text(json.dumps(value), encoding="utf-8")
        history = json.loads((feature_dir / "E_HISTORY.json").read_text(encoding="utf-8"))
        history["normalizer_sha256"] = sha256(path)
        (feature_dir / "E_HISTORY.json").write_text(json.dumps(history), encoding="utf-8")
        provenance = tmp_path / "fits" / "E_CTX_MATCHED" / "RUN_PROVENANCE.json"
        value = json.loads(provenance.read_text(encoding="utf-8"))
        value["normalizer_sha256"] = sha256(path)
        provenance.write_text(json.dumps(value), encoding="utf-8")
    else:
        h_rows.loc[1, "group_id"] = "group-a"
    with pytest.raises(ValueError):
        _audit_context_closure(
            tmp_path,
            fit_id="E_CTX_MATCHED",
            prefix="E",
            modality="event",
            endpoints=endpoints,
            h_rows=h_rows,
        )
