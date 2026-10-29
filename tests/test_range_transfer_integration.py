"""Production-runner contracts without unaccounted optimizer updates."""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from operational.range_transfer import admission, analyze
from operational.range_transfer.cache import HeadCache
from operational.range_transfer.common import digest, freeze, read, write
from operational.range_transfer.qa import same, synthetic
from operational.range_transfer.train_heads import accounting, initialize, loss_on_batch
from operational.rgb_port.train_heads import _loss, _model, _recipe
from operational.rgb_port.train_producers import ProducerCheckpoint


@pytest.fixture
def cache(tmp_path):
    path = tmp_path / "synthetic.npz"
    synthetic(path)
    return HeadCache(path)


def test_native_model_recipe_initialization_and_l0_gradients(cache):
    torch.set_num_threads(1)
    model, optimizer, _, generator = initialize(7)
    torch.manual_seed(7)
    native = _model("E_CTX_MATCHED")
    assert same(model.state_dict(), native.state_dict())
    assert _recipe("E_CTX_MATCHED") == (2500, 128, 1e-3)
    assert optimizer.defaults["foreach"] is False
    ids = torch.randint(cache.population, (128,), generator=generator)
    loss = loss_on_batch(model, cache, ids, 0, 0)
    native_loss = _loss(native, "E_CTX_MATCHED", cache, ids, torch.device("cpu"))
    assert torch.equal(loss, native_loss)
    loss.backward()
    native_loss.backward()
    assert all(
        torch.equal(a.grad, b.grad)
        for a, b in zip(model.parameters(), native.parameters(), strict=True)
    )


def test_mass_changes_do_not_change_draw_sequence(cache):
    schedules = []
    for mass_id in range(3):
        _, _, _, generator = initialize(7)
        schedules.append(torch.randint(cache.population, (10, 128), generator=generator))
        assert cache.plans[mass_id].dtype == np.float32
    assert all(torch.equal(schedules[0], s) for s in schedules[1:])


def test_h_cache_is_readonly_and_retains_m0_bytes(cache):
    assert cache.plans[0].tobytes() == cache.values["mass"].tobytes()
    with pytest.raises(ValueError):
        cache.values["timing"][0, 0, 0] = 1


@pytest.mark.parametrize("mutation", ["role", "duplicate", "nonfinite", "domain", "v_labels"])
def test_cache_rejects_leakage_and_invalid_rows(tmp_path, mutation):
    path = tmp_path / "fixture.npz"
    synthetic(path)
    with np.load(path) as archive:
        values = dict(archive)
    if mutation == "role":
        values["role"][:] = "V"
    elif mutation == "duplicate":
        values["sample_token"][1] = values["sample_token"][0]
    elif mutation == "nonfinite":
        values["expert_phase"][0, 0] = np.nan
    elif mutation == "domain":
        values["target_phase"][0] = 0.0001
    else:
        values["v_labels"] = np.zeros(64)
    np.savez(path, **values)
    with pytest.raises(ValueError):
        HeadCache(path)


def test_missing_band_does_not_fill_overall():
    target = np.array([1.0, 2.0, 4.0, 7.0])
    result = analyze.metrics(target, target, np.array(["g"] * 4))
    assert result["overall_MiD"] is None
    assert analyze.gate(result, result)["status"] == "LIMITED_VALIDATION_SUPPORT"


def test_invalid_predictions_cannot_improve_selection():
    target = np.array([1.0, 4.0, 7.0, -2.0])
    prediction = target.copy()
    prediction[0] = np.nan
    result = analyze.metrics(target, prediction, np.array(["g"] * 4))
    assert result["mean_MiD"] is None
    assert result["overall_MiD"] is None
    assert not result["complete_prediction_and_mid_coverage"]


def test_unchanged_control_cannot_win_gate():
    target = np.tile([1.0, 4.0, 7.0, -2.0], 32)
    result = analyze.metrics(target, target * 1.1, np.array(["g"] * len(target)))
    assert not analyze.gate(result, result)["eligible"]


def test_gate_selects_improvement_and_freezes_once(tmp_path):
    target = np.tile([1.0, 4.0, 7.0, -2.0], 64)
    groups = np.array([f"g{i // 64}" for i in range(len(target))])
    base_pred = target * 1.1
    candidate = base_pred.copy()
    candidate[target == 1] = 1.02
    baseline = analyze.metrics(target, base_pred, groups)
    score = analyze.metrics(target, candidate, groups)
    assert analyze.gate(baseline, score)["eligible"]
    result = analyze.select({"RT-M0-L0-S7": baseline, "RT-M2-L1-S7": score}, tmp_path)
    assert result["candidate_recipe"] == [2, 1]
    with pytest.raises(ValueError):
        analyze.select({"RT-M0-L0-S7": baseline}, tmp_path)


def test_v_not_opened_before_endpoint_freeze(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze.evaluate_v(tmp_path / "owner", tmp_path, {}, stage="screen")


def test_role_overlap_is_rejected_without_loading_rows(tmp_path):
    for role in ("P", "H", "V"):
        write(
            tmp_path / f"{role}_MANIFEST.json",
            {
                "role": role,
                "status": "COMPLETE",
                "group_ids": ["same"],
                "population_size": 1,
                "split_assignment_sha256": "same",
                "rows_path": "must_not_open",
            },
        )
    with pytest.raises(ValueError, match="leakage"):
        admission.role_audit(tmp_path)


def test_accounting_counts_uncertain_and_enforces_separate_caps(tmp_path):
    journal = tmp_path / "fits/RT-M0-L0-S7/UPDATE_JOURNAL.json"
    write(
        journal,
        {
            "completed_updates": 200,
            "durable_updates": 100,
            "recovery_upper": 50,
            "pending_update_upper": 1,
        },
    )
    assert accounting(tmp_path)["physical_upper"] == 251
    write(
        journal,
        {
            "completed_updates": 25000,
            "durable_updates": 25000,
            "recovery_upper": 0,
            "pending_update_upper": 0,
        },
    )
    with pytest.raises(RuntimeError, match="budget"):
        accounting(tmp_path, prospective=1)


def test_atomic_checkpoint_failure_keeps_old_pointer(tmp_path):
    model, optimizer, scheduler, generator = initialize(7)
    identity = {"fit_id": "fault_injection_no_optimizer", "identity_sha256": digest("qa")}
    state = ProducerCheckpoint(tmp_path, identity, 10)
    state.save(model, optimizer, scheduler, generator, {"cursor": 0}, status="READY")
    pointer_before = read(state.pointer)
    # Inject before the pointer is advanced; no optimizer update is executed here.
    with patch("operational.rgb_port.train_producers.torch.save", side_effect=OSError("injected")):
        with pytest.raises(OSError, match="injected"):
            state.save(model, optimizer, scheduler, generator, {"cursor": 0}, status="READY")
    assert read(state.pointer) == pointer_before
    restored = ProducerCheckpoint(tmp_path, identity, 10)
    assert restored.restore(model, optimizer, scheduler, generator) == {"cursor": 0}


def test_restore_repairs_stale_receipt_from_durable_pointer(tmp_path):
    model, optimizer, scheduler, generator = initialize(7)
    identity = {"fit_id": "receipt_repair_no_optimizer", "identity_sha256": digest("qa2")}
    state = ProducerCheckpoint(tmp_path, identity, 10)
    state.save(model, optimizer, scheduler, generator, {"cursor": 0}, status="READY")
    write(state.receipt, {"completed_updates": -1, "checkpoint_sha256": "stale"})
    restored = ProducerCheckpoint(tmp_path, identity, 10)
    restored.restore(model, optimizer, scheduler, generator)
    assert read(state.receipt)["checkpoint_sha256"] == read(state.pointer)["checkpoint_sha256"]


def test_freeze_refuses_mutation(tmp_path):
    path = tmp_path / "freeze.json"
    freeze(path, {"a": 1})
    freeze(path, {"a": 1})
    with pytest.raises(ValueError):
        freeze(path, {"a": 2})


def test_bootstrap_reentry_is_exact_and_grouped(tmp_path):
    frame = pd.DataFrame(
        {
            "sample_token": ["a", "b", "c", "d"],
            "group_id": ["g1", "g1", "g2", "g2"],
            "target_ttc": [1.0, 2.0, 1.5, 2.5],
            "prediction_ttc": [1.2, 2.2, 1.6, 2.6],
        }
    )
    analyze.bootstrap_pairs(frame, frame.copy(), tmp_path)
    result = json.loads((tmp_path / "SUMMARY.json").read_text())
    assert result["groups"] == 2
    assert result["CI95"] == [0.0, 0.0]
    analyze.bootstrap_pairs(frame, frame.copy(), tmp_path)


def test_replication_averages_query_losses_not_ttc(tmp_path):
    for mass in (0, 1):
        for seed, prediction in ((7, 1.2), (13, 2.1), (23, 3.0)):
            stage = "screen" if seed == 7 else "replication"
            output = tmp_path / "evaluation" / stage / f"RT-M{mass}-L0-S{seed}"
            output.mkdir(parents=True)
            pd.DataFrame(
                {
                    "sample_token": ["a"],
                    "group_id": ["g"],
                    "target_ttc": [2.0],
                    "prediction_ttc": [prediction],
                }
            ).to_csv(output / "PREDICTIONS.csv", index=False)
    analyze.summarize_replicas(tmp_path, (1, 0))
    result = pd.read_csv(tmp_path / "replication/MEAN_LOSS_PER_QUERY.csv")
    expected = np.mean(
        10000 * np.abs(np.log1p(-0.1 / np.array([1.2, 2.1, 3.0])) - np.log1p(-0.1 / 2.0))
    )
    assert result.control_mean_loss.iloc[0] == pytest.approx(expected)
    summary = read(tmp_path / "replication/SUMMARY.json")
    assert summary["predictions_ensembled"] is False
    assert summary["bands"]["crucial"]["CI95"] == [0.0, 0.0]
    analyze.summarize_replicas(tmp_path, (1, 0))
