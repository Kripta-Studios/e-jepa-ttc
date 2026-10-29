"""Preparation checks: no optimizer step, no scientific model evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_post_v18_refit as refit


def test_union_mass_balances_groups_not_roles() -> None:
    groups = np.array(["h1", "h1", "h1", "h2", "v1", "v1"])
    mass = refit.group_mass(groups)
    for group in np.unique(groups):
        assert mass[groups == group].sum() == pytest.approx(1 / 3)
    assert mass.sum() == pytest.approx(1)


def test_scaled_schedule_endpoints() -> None:
    plan = {
        "sampling": {"updates": 4730},
        "scheduler": {"warmup_updates": 190, "lr_max": 3e-4, "lr_min": 3e-5},
    }
    assert refit.learning_rate(1, plan) == pytest.approx(3e-4 / 190)
    assert refit.learning_rate(190, plan) == pytest.approx(3e-4)
    assert refit.learning_rate(4730, plan) == pytest.approx(3e-5)
    assert refit.learning_rate(191, plan) < refit.learning_rate(190, plan)
    with pytest.raises(ValueError):
        refit.learning_rate(0, plan)


def test_train_without_authorization_stops_before_any_io(tmp_path: Path) -> None:
    before = set(tmp_path.iterdir())
    with pytest.raises(PermissionError, match="Zero updates"):
        refit.train(tmp_path / "nonexistent_plan.json", None, resume=False)
    assert set(tmp_path.iterdir()) == before


def test_authorization_rejects_wrong_identity(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text('{"budget_proposed_only": {"scientific_updates": 4730}}')
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "post_v18_refit_authorization_v1",
                "plan_sha256": "wrong",
                "authorized": True,
                "user_instruction_reference": "synthetic",
                "budget": {"scientific_updates": 4730},
            }
        )
    )
    with pytest.raises(PermissionError, match="exact plan"):
        refit.authorize(plan, receipt)


def test_recovery_and_physical_caps_are_independent() -> None:
    plan = {"budget_proposed_only": {"recovery_update_cap": 250, "physical_update_cap": 4980}}
    refit.enforce_updates(
        SimpleNamespace(completed=4729, recovery_upper=250), plan, next_update=True
    )
    with pytest.raises(RuntimeError, match="recovery"):
        refit.enforce_updates(
            SimpleNamespace(completed=0, recovery_upper=251), plan, next_update=True
        )
    with pytest.raises(RuntimeError, match="physical"):
        refit.enforce_updates(
            SimpleNamespace(completed=4730, recovery_upper=250), plan, next_update=True
        )


@pytest.mark.skipif(not refit.V16.is_dir(), reason="Local frozen V16 worktree required")
def test_checkpoint_rng_recovery_and_export_without_updates(tmp_path: Path) -> None:
    # Isolated import avoids mixing this branch's modules with the frozen V16 implementation.
    program = r"""
import json, random, sys
from pathlib import Path
import numpy as np
import torch
from scripts import run_post_v18_refit as r
torch.set_num_threads(1)
anchor, _, checkpoint = r.load_runtime()
def forbidden(*args, **kwargs):
    raise AssertionError("Optimizer steps forbidden in preparation tests")
torch.optim.AdamW.step = forbidden
directory = Path(sys.argv[1])
model = anchor("EVENT")
assert sum(p.numel() for p in model.parameters()) == 1955
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
generator = torch.Generator().manual_seed(7)
identity = {"fit_id": "SYNTHETIC_NO_UPDATES", "identity_sha256": "synthetic"}
state = checkpoint(directory, identity, 4730)
state.save(model, optimizer, scheduler, generator, {"losses": []}, status="READY")
expected = torch.randint(1000, (10,), generator=generator)
expected_global = torch.rand(5)
expected_numpy = np.random.rand(5)
expected_python = random.random()
with torch.no_grad():
    next(model.parameters()).fill_(42)
state.restore(model, optimizer, scheduler, generator)
assert torch.equal(expected, torch.randint(1000, (10,), generator=generator))
assert torch.equal(expected_global, torch.rand(5))
assert np.array_equal(expected_numpy, np.random.rand(5))
assert expected_python == random.random()
assert not (next(model.parameters()) == 42).all()
assert state.completed == 0 and state.recovery_upper == 0
# Synthetic journal simulates a crash; no optimizer was called.
journal = r.read(state.journal)
journal["completed_updates"] = 2
journal["pending_update_upper"] = 1
r.atomic_json(state.journal, journal)
state.restore(model, optimizer, scheduler, generator)
assert state.completed == 0 and state.recovery_upper == 3
features = np.zeros((5, 340), dtype=np.float32)
features[:, 160] = np.array([-.1, -.01, 0, .01, .1], dtype=np.float32)
data = {"features": features, "rgb_width_logits": np.zeros((5, 2), dtype=np.float32),
        "sample_token": np.array(["synthetic_"+str(i) for i in range(5)])}
with torch.no_grad():
    model.output.weight.fill_(0.01)
    model.output.bias.fill_(0.001)
result = r.export_head(model, data, directory, identity)
assert result["max_abs_ttc_error_seconds"] <= 1e-5
pointer = r.read(state.pointer)
pointer["checkpoint_sha256"] = "corrupt"
r.atomic_json(state.pointer, pointer)
try:
    state.restore(model, optimizer, scheduler, generator)
except ValueError:
    pass
else:
    raise AssertionError("Corruption was accepted")
print(json.dumps({"optimizer_steps": 0,
                  "synthetic_export_parity": result["max_abs_ttc_error_seconds"]}))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", program, str(tmp_path)],
        cwd=refit.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"optimizer_steps": 0' in result.stdout
