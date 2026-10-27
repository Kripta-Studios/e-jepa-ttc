"""Crash recovery around the frozen packager's ZIP/receipt publication boundary."""

import ast
import json
import os
import uuid
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

from scripts import runpod_test12_package_compat as compat
from scripts.runpod_test12_staged import canonical, digest


@pytest.fixture
def package_fixture(tmp_path):
    root = tmp_path / "final/test12"
    root.mkdir(parents=True)
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"physical_aliases": {"a": "a", "b": "b"}}))
    official = tmp_path / "official.parquet"
    pd.DataFrame({"sample_token": ["a", "b"]}).to_parquet(official)
    predictions = root / "predictions.csv"
    pd.DataFrame({"sample_token": ["a", "b"], "prediction": [1.5, -2.0]}).to_csv(
        predictions, index=False
    )
    seal = tmp_path / "final/MODEL_SEAL.json"
    seal.write_text('{"sealed":true}')
    read = lambda path: json.loads(path.read_text())  # noqa: E731
    cfg = {"test12_inputs": inputs, "official_test_inputs": official}
    namespace = {
        "Path": Path,
        "_config": lambda run: cfg,
        "_read": read,
        "sha256": digest,
        "_scientific_seal": lambda path: (read(path), digest(path)),
        "canonical_sha256": canonical,
        "OFFICIAL_TEST_QUERIES": 2,
        "self_candidate": lambda: "fixture_candidate",
        "_write_once": compat.durable_write_once,
        "pd": pd,
        "np": np,
        "json": json,
        "os": os,
        "uuid": uuid,
        "zipfile": zipfile,
        "cast": cast,
    }
    source_path = Path("operational/final_refit/scientific_oof_final.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "package_scientific_predictions"
    )
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(source_path),
            "exec",
            flags=__import__("__future__").annotations.compiler_flag,
        ),
        namespace,
    )
    final = SimpleNamespace(**namespace)
    final.TRUSTED_INPUTS_SHA256 = digest(inputs)
    final.TRUSTED_OFFICIAL_TEST_INPUTS_SHA256 = digest(official)
    final.PREDICTION_SCHEMA = "fixture_prediction_schema"
    prediction_receipt = {
        "schema": final.PREDICTION_SCHEMA,
        "candidate": "fixture_candidate",
        "status": "COMPLETE",
        "model_seal_sha256": digest(seal),
        "inputs_sha256": digest(inputs),
        "predictions_sha256": digest(predictions),
        "population": 2,
        "physical_queries": 2,
        "computed_under_seal": True,
        "private_targets_read": False,
        "external_submission": False,
        "run_binding_sha256": canonical(
            {"model_seal_sha256": digest(seal), "inputs_sha256": digest(inputs)}
        ),
    }
    compat.durable_write_once(root / "PREDICTION_RECEIPT.json", prediction_receipt)
    archive = root / "PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(
            "submission.json",
            json.dumps(
                {
                    "meta": {"format": "garlttc_prediction_v1"},
                    "results": {"a": {"ttc": 1.5}, "b": {"ttc": -2.0}},
                }
            ),
        )
    return final, tmp_path, archive


def test_missing_sidecar_is_reconstructed_without_changing_zip(package_fixture):
    final, run, archive = package_fixture
    original = archive.read_bytes()
    assert not archive.with_suffix(".json").exists()
    result = compat.recover_package_receipt(final, run)
    assert result["status"] == "PACKAGED_NOT_SUBMITTED"
    assert archive.read_bytes() == original
    assert final.package_scientific_predictions(run) == result
    sidecar_bytes = archive.with_suffix(".json").read_bytes()
    assert compat.recover_package_receipt(final, run) == result
    assert archive.with_suffix(".json").read_bytes() == sidecar_bytes


def test_wrong_zip_cannot_receive_repaired_receipt(package_fixture):
    final, run, archive = package_fixture
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("submission.json", '{"wrong":true}')
    with pytest.raises(ValueError, match="content differs"):
        compat.recover_package_receipt(final, run)
    assert not archive.with_suffix(".json").exists()


@pytest.mark.parametrize(
    "field,value",
    [("schema", "wrong"), ("computed_under_seal", False), ("external_submission", True)],
)
def test_invalid_prediction_claims_are_not_repaired(package_fixture, field, value):
    final, run, archive = package_fixture
    path = archive.parent / "PREDICTION_RECEIPT.json"
    receipt = json.loads(path.read_text())
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Prediction receipt"):
        compat.recover_package_receipt(final, run)
    assert not archive.with_suffix(".json").exists()


def test_existing_wrong_sidecar_is_never_overwritten(package_fixture):
    final, run, archive = package_fixture
    sidecar = archive.with_suffix(".json")
    sidecar.write_text('{"wrong":true}')
    before = sidecar.read_bytes()
    with pytest.raises(ValueError, match="package changed"):
        compat.recover_package_receipt(final, run)
    assert sidecar.read_bytes() == before
