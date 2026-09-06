"""Adversarial scientific engine, archive, and reference/production parity tests."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import psutil
import pytest
import torch

from e_jepa_ttc.artifacts.essential_bundle_v10 import create_bundle, safe_member
from e_jepa_ttc.artifacts.risk_geometry_v10 import CampaignOwner
from e_jepa_ttc.data.frozen_expert_tables_v10 import ModelInputs
from e_jepa_ttc.evaluation.risk_geometry_v10 import verify_coverage
from e_jepa_ttc.models.simplex_risk_router import costs_from_logits
from e_jepa_ttc.training.risk_router_v10 import train_steps


@pytest.mark.parametrize("name", ["../x", "/x", "C:/x", "a\\x", "a/../x"])
def test_zip_rejects_unsafe_names(name):
    with pytest.raises(ValueError):
        safe_member(name)


def test_bundle_roundtrip(tmp_path):
    p = tmp_path / "evidence.json"
    p.write_text('{"historical_acceptance":"INTEGRITY_BLOCKED"}')
    result = create_bundle({"run/evidence.json": p}, tmp_path / "result.zip")
    assert result["extracted_hashes_verified"] == 1 and result["crc_verified"]


def test_scientific_fit_without_real_ledger_rejected(tmp_path):
    x = np.zeros((8, 17))
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError, match="FITTING"):
        train_steps(
            inputs=ModelInputs(x, np.ones((8, 3)), np.ones((8, 3))),
            target_phase=np.ones(8),
            mass=np.ones(8) / 8,
            mean=np.zeros(17),
            std=np.ones(17),
            arm="S67-SIMPLEX17",
            seed=7,
            outer=0,
            output=tmp_path,
            identity={"purpose": "scientific"},
        )
    assert set(tmp_path.iterdir()) == before


def test_live_owner_rejected_stale_preserved(tmp_path):
    p = tmp_path / "ACTIVE_OWNER.json"
    p.write_text(json.dumps(dict(pid=os.getpid(), created=psutil.Process().create_time())))
    with pytest.raises(RuntimeError, match="live"):
        with CampaignOwner(tmp_path):
            pass
    p.write_text(json.dumps(dict(pid=99999999, created=0)))
    with CampaignOwner(tmp_path):
        assert p.exists()
    assert list(tmp_path.glob("STALE_OWNER*")) and list(tmp_path.glob("CLOSED_OWNER*"))


def test_forged_diagnostic_complete_flag_rejected(tmp_path):
    p = tmp_path / "DIAGNOSTIC_COVERAGE.json"
    p.write_text(json.dumps(dict(completed=True, rows=8192, arms=[], files=[])))
    with pytest.raises(ValueError):
        verify_coverage(p)


def test_reference_production_cost_and_gradient_parity():
    local = Path("artifacts/stage66_69_clean_v2/LOCAL_INPUTS.json")
    if not local.exists():
        pytest.skip("local verified reference package unavailable")
    package = Path(json.loads(local.read_text())["handoff_root"])
    spec = importlib.util.spec_from_file_location(
        "frozen_reference_simplex", package / "reference/simplex_router.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(7)
    phases = torch.randn(256, 3, dtype=torch.float64)
    logits = torch.randn(256, 2, dtype=torch.float64, requires_grad=True)
    for constrained in (True, False):
        a = costs_from_logits(phases, logits, constrained=constrained)
        b = module.costs_from_logits(phases, logits, constrained=constrained)
        assert torch.equal(a, b)
        ga = torch.autograd.grad(a.square().sum(), logits)[0]
        gb = torch.autograd.grad(b.square().sum(), logits)[0]
        assert torch.equal(ga, gb)
