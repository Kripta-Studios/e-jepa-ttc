"""Synthetic rejection fixtures: no optimizer and no real QA claims."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.model import TemporalConfig
from e_jepa_ttc.simplex_t.resume_evidence import RESUME_FILES, verify_real_cpu_resume
from e_jepa_ttc.simplex_t.training import atomic_checkpoint, state_digest


def fixture(root: Path, failure: str) -> dict:
    engine, script = root / "engine.py", root / "probe.py"
    engine.write_text("synthetic engine", encoding="utf-8")
    script.write_text("synthetic probe", encoding="utf-8")
    contract = {
        "scope": "TECHNICAL_REAL_QUERY_CONTEXT_NOT_SCIENTIFIC_FREEZE",
        "source_sha256": "a" * 64,
        "compiled_sha256": "b" * 64,
        "engine_sha256": sha256(engine),
        "script_sha256": sha256(script),
        "torch": str(torch.__version__),
        "updates_reserved": 20,
    }
    identity = {
        "source": "a" * 64,
        "freeze": state_digest(contract),
        "config": asdict(TemporalConfig()),
        "seed": 7,
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "batch": 128,
        "endpoint": 2500,
    }
    journal = {
        "schema": "simplex_t_physical_work_v1",
        "graph": {"continuous": 2500, "split": 2500},
        "fits": {},
        "events": [],
    }
    for branch in ("continuous", "split"):
        folder = root / branch
        folder.mkdir()
        state = {
            "identity": dict(identity),
            "completed_updates": 10,
            "status": "TECHNICAL_PARTIAL",
            "losses": [0.0] * 10,
            "sampler_hashes": ["synthetic"] * 10,
            "synthetic_tensor": torch.tensor([1.0]),
        }
        if failure == "states" and branch == "split":
            state["synthetic_tensor"] = torch.tensor([2.0])
        if failure == "device":
            state["identity"]["device"] = "cuda"
        checkpoint = folder / "checkpoint_last.pt"
        atomic_checkpoint(checkpoint, state)
        journal["fits"][branch] = {
            "completed": 10,
            "uncertain_lost_upper": 0,
            "pending": None,
            "checkpoint_sha256": sha256(checkpoint),
        }
        for count in [10] if branch == "continuous" or failure == "boundary" else [5, 10]:
            journal["events"].append({"operation": "checkpoint", "key": branch, "completed": count})
    if failure == "pending":
        journal["fits"]["split"]["pending"] = 10
    report = {
        "exact_complete_state_match": True,
        "source_sha256": "a" * 64,
        "executed_technical_updates": 20,
        "scientific_updates": 0,
        "nonzero_real_history": failure != "history",
        "train_h8_count": 8,
        "train_queries": 10,
        "max_sensor_age_seconds": 0.35,
    }
    for name, value in (
        ("CONTRACT.json", contract),
        ("RESUME_QA.json", report),
        ("TECHNICAL_JOURNAL.json", journal),
    ):
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    pins = {name: sha256(root / name) for name in RESUME_FILES}
    if failure == "bytes":
        (root / "RESUME_QA.json").write_text("{}", encoding="utf-8")
    if failure == "engine":
        engine.write_text("changed engine", encoding="utf-8")
    return dict(
        pins=pins,
        source_sha256="a" * 64,
        compiled_sha256="b" * 64,
        engine=engine,
        probe_script=script,
    )


@pytest.mark.parametrize(
    "failure", ["", "states", "device", "boundary", "pending", "history", "bytes", "engine"]
)
def test_resume_evidence(tmp_path: Path, failure: str) -> None:
    inputs = fixture(tmp_path, failure)
    if failure:
        with pytest.raises(ValueError):
            verify_real_cpu_resume(tmp_path, **inputs)
    else:
        result = verify_real_cpu_resume(tmp_path, **inputs)
        assert result["executed_updates_by_this_verification"] == 0
        assert result["scientific_stage_authorized"] is False
