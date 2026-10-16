"""CPU proofs of the V19 loss contrast and isolated durable runtime boundaries."""

from pathlib import Path

import numpy as np
import pytest
import torch

from e_jepa_ttc.joint_measurement.loss import training_mass as native_mass
from operational.decision_recovery import common, training
from operational.decision_recovery.losses import MASSES, training_mass
from operational.joint_measurement.captured_phase_loss import LossCore
from operational.joint_measurement.qa import exact
from operational.range_transfer.common import digest
from operational.rgb_port.train_producers import ProducerCheckpoint


def population():
    target = np.array([1, 2, 4, 8, -2, -5, 1, 4, 8, -2], dtype=float)
    groups = np.array(["a"] * 6 + ["b"] * 4)
    return target, groups


def test_l0_is_native_bitwise():
    target, groups = population()
    assert np.array_equal(
        training_mass(target, groups, "L0"), native_mass(target, groups, role="P")
    )


def test_lw_has_global_equal_groups_and_declared_band_mass():
    target, groups = population()
    mass = training_mass(target, groups, "LW")
    masks = ((target > 0) & (target <= 3), (target > 3) & (target <= 6), target > 6, target < 0)
    assert mass.sum() == pytest.approx(1)
    for group in ("a", "b"):
        assert mass[groups == group].sum() == pytest.approx(0.5)
        for mask, expected in zip(masks, MASSES["LW"], strict=True):
            assert mass[mask & (groups == group)].sum() == pytest.approx(expected / 2)
    assert mass[0] == mass[1]
    assert mass[0] < mass[6]  # More rows in a band divide its fixed mass.


def test_lw_missing_bands_renormalize_per_group_not_batch():
    target = np.array([1.0, 2.0, -2.0])
    groups = np.array(["a", "a", "b"])
    assert np.array_equal(training_mass(target, groups, "LW"), np.array([0.25, 0.25, 0.5]))


@pytest.mark.parametrize("target", [[0], [0.1], [11], [-11], [float("nan")]])
def test_invalid_populations_fail_closed(target):
    with pytest.raises(ValueError):
        training_mass(np.array(target), np.array(["a"]), "LW")


@pytest.mark.parametrize("arm", ["L0", "LW"])
def test_effective_batch_loss_and_gradients_are_additive_without_renormalizing(arm):
    target, groups = population()
    ids = [0, 0, 4, 7, 2, 9, 3, 6]
    weights = torch.tensor(training_mass(target, groups, arm)[ids], dtype=torch.float64)
    truth = torch.tensor(target[ids], dtype=torch.float64)
    raw = torch.tensor(
        [0.02, 0.04, -0.03, 0.05, 0.09, -0.04, 0.02, 0.1], dtype=torch.float64, requires_grad=True
    )
    core = LossCore(len(target), len(ids))
    total = core(raw, truth, weights)[0]
    total_grad = torch.autograd.grad(total, raw)[0]
    split = sum(
        core(raw[i : i + 2], truth[i : i + 2], weights[i : i + 2])[0] for i in range(0, 8, 2)
    )
    split_grad = torch.autograd.grad(split, raw)[0]
    torch.testing.assert_close(total, split, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(total_grad, split_grad, rtol=0, atol=0)


def test_mutable_paths_cannot_escape_or_point_to_v18(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RUN", tmp_path / "v19")
    monkeypatch.setattr(common, "SOURCE_ROOT", tmp_path / "v18")
    assert training.destination("training/result.json") == tmp_path / "v19/training/result.json"
    with pytest.raises(ValueError):
        training.destination("../v18/result.json")
    monkeypatch.setattr(common, "RUN", tmp_path / "v18/artifacts")
    with pytest.raises(ValueError):
        training.destination("training/result.json")


def test_runtime_patch_list_is_explicit_and_restored():
    from operational.joint_measurement import data, persistent_adaptation

    before = (
        data.cache_root,
        data.role_rows,
        persistent_adaptation.accounting,
        persistent_adaptation.require_resources,
        persistent_adaptation.backward,
    )
    assert len(training.PATCH_TARGETS) == 5
    with training.runtime_context():
        assert data.cache_root is training.cache_root
        assert data.role_rows is training.role_rows
        assert persistent_adaptation.accounting is training.accounting
        assert persistent_adaptation.require_resources is training.require_resources
    assert before == (
        data.cache_root,
        data.role_rows,
        persistent_adaptation.accounting,
        persistent_adaptation.require_resources,
        persistent_adaptation.backward,
    )


def test_v_is_never_opened():
    with pytest.raises(ValueError):
        training.cache_root("V")
    with pytest.raises(ValueError):
        training.manifest_path("V")


def test_accounting_counts_qa_recovery_and_pending_without_historical_work(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RUN", tmp_path / "v19")
    monkeypatch.setattr(common, "SOURCE_ROOT", tmp_path / "v18")
    for folder, updates, recovery, pending in (("fits/FULL-L0", 10, 2, 1), ("qa/L0", 4, 0, 0)):
        common.write_json(
            training.destination(f"training/{folder}/UPDATE_JOURNAL.json"),
            {
                "completed_updates": updates,
                "recovery_upper": recovery,
                "pending_update_upper": pending,
            },
        )
    result = training.accounting()
    assert result == {
        "scientific": 10,
        "technical": 4,
        "recovery_upper": 2,
        "pending_upper": 1,
        "physical_upper": 17,
    }
    monkeypatch.setattr(training, "UPDATE_CAP", 16)
    with pytest.raises(RuntimeError, match="budget"):
        training.accounting()


def test_existing_lock_does_not_change_v18_bytes(tmp_path):
    lock = tmp_path / "V18_GPU.lock"
    lock.write_bytes(b"0untouched")
    before = lock.read_bytes()
    with training.file_lock(lock, existing=True):
        with pytest.raises(OSError), training.file_lock(lock, existing=True):
            pytest.fail("An independent writer acquired the occupied byte")
    assert lock.read_bytes() == before


def trajectory(directory: Path, limit: int):
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Dropout(0.2), torch.nn.Linear(4, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    sampler = torch.Generator().manual_seed(7)
    identity = {"fit_id": directory.name, "population": "synthetic_cpu", "updates": 4}
    identity["identity_sha256"] = digest(identity)
    state = ProducerCheckpoint(directory, identity, 4)
    cursor = (
        state.restore(model, optimizer, scheduler, sampler)
        if state.pointer.exists()
        else {"curve": []}
    )
    while state.completed < limit:
        ids = torch.randint(8, (4,), generator=sampler)
        inputs = torch.stack((ids.float(), ids.float() / 3), dim=1)
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs).square().mean()
        loss.backward()
        state.begin()
        optimizer.step()
        scheduler.step()
        state.commit()
        cursor["curve"].append({"ids": ids, "loss": float(loss.detach())})
    state.save(
        model, optimizer, scheduler, sampler, cursor, status="COMPLETE" if limit == 4 else "PAUSED"
    )
    return torch.load(directory / "checkpoint_last.pt", map_location="cpu", weights_only=False)


def test_checkpoint_preserves_cpu_rng_sampler_optimizer_and_curve(tmp_path):
    direct = trajectory(tmp_path / "direct", 4)
    trajectory(tmp_path / "resumed", 2)
    resumed = trajectory(tmp_path / "resumed", 4)
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "sampler_generator_state",
        "torch_rng_state",
        "cursor",
    ):
        assert exact(direct[key], resumed[key]), key
    assert direct["recovery_upper"] == resumed["recovery_upper"] == 0
