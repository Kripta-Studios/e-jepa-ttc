"""Raw media release must require every exact sealed prediction."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "rgb_control", Path(__file__).parents[2] / "scripts/runpod_test12_rgb_controller_v1.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "FINAL", tmp_path)
    catalog = {
        "inputs_sha256": "i",
        "sequences": {"s": {"tokens": ["a", "b", "alias"], "canonical_tokens": ["a", "b"]}},
    }
    binding = module.canonical({"model_seal_sha256": "seal", "inputs_sha256": "i"})
    directory = tmp_path / "test12/staged_sequences"
    directory.mkdir(parents=True)
    fragments = tmp_path / "test12/fragments"
    fragments.mkdir()
    admission = directory / "s.admission.json"
    admission.write_text("{}")
    complete = directory / "s.complete.json"
    values = {"a": 1.0, "b": -2.0}
    complete.write_text(
        json.dumps(
            {
                "schema": "test12_staged_sequence_complete_v1",
                "status": "COMPLETE",
                "sequence_id": "s",
                "run_binding_sha256": binding,
                "private_targets_read": False,
                "external_submission": False,
                "canonical_tokens": ["a", "b"],
                "admission_sha256": module.digest(admission),
                "predictions_sha256": module.canonical(values),
            }
        )
    )
    paths = [admission, complete]
    for token, value in values.items():
        p = fragments / (hashlib.sha256(token.encode()).hexdigest() + ".json")
        p.write_text(
            json.dumps(
                {
                    "schema": "final_refit_test12_prediction_fragment_v1",
                    "sample_token": token,
                    "binding_sha256": binding,
                    "private_targets_read": False,
                    "prediction": value,
                }
            )
        )
        paths.append(p)
    receipt = {
        "schema": "test12_staged_rgb_consumed_v1",
        "status": "PASS",
        "sequence_id": "s",
        "private_targets_read": False,
        "source_receipts": {"tar": "receipt"},
        "run_binding_sha256": binding,
        "outputs": [
            {"path": str(p), "bytes": p.stat().st_size, "sha256": module.digest(p)} for p in paths
        ],
    }
    return catalog, {"source_receipts": {"tar": "receipt"}}, receipt, paths


def test_all_predictions_and_alias_coverage(tmp_path, monkeypatch):
    c, ready, receipt, _ = fixture(tmp_path, monkeypatch)
    result = module.verify_consumed("s", c, ready, receipt, "seal")
    assert result["predictions"] == 2 and result["all_tokens"] == 3


def test_missing_fragment_blocks_release(tmp_path, monkeypatch):
    c, ready, receipt, _ = fixture(tmp_path, monkeypatch)
    receipt["outputs"].pop()
    with pytest.raises(KeyError):
        module.verify_consumed("s", c, ready, receipt, "seal")


def test_changed_output_blocks_release(tmp_path, monkeypatch):
    c, ready, receipt, paths = fixture(tmp_path, monkeypatch)
    paths[-1].write_text("{}")
    with pytest.raises(ValueError, match="byte identity"):
        module.verify_consumed("s", c, ready, receipt, "seal")
