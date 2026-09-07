"""Exercise real lease, receipts and NPZ resume; mock only reference cohort/inference."""

import json
from contextlib import contextmanager

import numpy as np
import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import h16_qa_execution as module


@pytest.fixture
def rig(tmp_path, monkeypatch):
    cache = tmp_path / "artifacts/simplex_t/T1/context_features_fp32"
    cache.mkdir(parents=True)
    arrays = {
        key: np.zeros((16, 2), dtype=np.float32)
        for key in (
            "features145",
            "expert_ttc",
            "pair_features",
            "known",
            "observation_ids",
            "anchor_us",
            "available_us",
        )
    }
    stem = cache / "family00_query00000"
    np.savez(stem.with_suffix(".npz"), **arrays)
    stem.with_suffix(".json").write_text("{}", encoding="utf-8")
    item = {
        "reference_stem": stem.name,
        "family": 0,
        "query": 0,
        "receipt_sha256": sha256(stem.with_suffix(".json")),
        "payload_sha256": sha256(stem.with_suffix(".npz")),
    }
    bound = {"missing": [], "references": [item]}
    monkeypatch.setattr(module, "bind_h16_reference_receipts", lambda work: bound)
    loads = []

    @contextmanager
    def factory(family):
        loads.append(family)
        yield lambda query: {key: value.copy() for key, value in arrays.items()}

    args = dict(
        work=tmp_path,
        output=tmp_path / "artifacts/simplex_t/T0/qa",
        family_factory=factory,
        validate_prerequisites=lambda: None,
        resource_ok=lambda: True,
        execution_identity={"fixture": True},
    )
    return args, bound, arrays, loads


def test_pass_and_resume_does_not_reload_models(rig):
    args, _, _, loads = rig
    first = module.execute_h16_qa(**args)
    assert first["status"] == "SAME_LAYOUT_H16_EXACT_PASS"
    assert module.execute_h16_qa(**args) == first
    assert loads == [0]


@pytest.mark.parametrize("condition", ["missing", "lease", "resource"])
def test_preconditions_prevent_model_load(rig, condition):
    args, bound, _, loads = rig
    if condition == "missing":
        bound["missing"] = ["q"]
        with pytest.raises(ValueError, match="WAITING_H16"):
            module.execute_h16_qa(**args)
    elif condition == "lease":
        lease = args["work"] / "artifacts/simplex_t/T1/CURRENT_REPLAY.lock"
        lease.write_text("another owner", encoding="utf-8")
        with pytest.raises(FileExistsError):
            module.execute_h16_qa(**args)
        assert lease.read_text("utf-8") == "another owner"
    else:
        args["resource_ok"] = lambda: False
        assert module.execute_h16_qa(**args)["status"] == "RESOURCE_PAUSE"
    assert loads == []


def test_failure_is_preserved_and_not_retried(rig):
    args, _, arrays, loads = rig
    arrays["expert_ttc"][0, 0] = 1
    with pytest.raises(ValueError, match="expert_ttc"):
        module.execute_h16_qa(**args)
    receipt = args["output"] / "family00_query00000.json"
    assert json.loads(receipt.read_text("utf-8"))["status"] == "EXACT_FAIL"
    digest = sha256(receipt)
    with pytest.raises(ValueError, match="expert_ttc"):
        module.execute_h16_qa(**args)
    assert sha256(receipt) == digest and loads == [0]


def test_orphan_is_preserved_without_inference(rig):
    args, _, _, loads = rig
    args["output"].mkdir(parents=True)
    orphan = args["output"] / "family00_query00000.partial"
    orphan.write_bytes(b"interrupted payload")
    with pytest.raises(ValueError, match="orphan"):
        module.execute_h16_qa(**args)
    assert orphan.read_bytes() == b"interrupted payload" and loads == []


def test_changed_completed_payload_refuses_resume(rig):
    args, _, _, loads = rig
    module.execute_h16_qa(**args)
    payload = args["output"] / "family00_query00000.npz"
    with payload.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="saved H16 QA output changed"):
        module.execute_h16_qa(**args)
    assert loads == [0]
