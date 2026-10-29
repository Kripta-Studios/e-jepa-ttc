"""Independent encoder-change evidence must be supported by actual retained weights."""

import copy
import json

import pytest
import torch

from operational.joint_measurement.replay import (
    file_hash,
    replay_encoder_change,
    replay_frozen_encoder,
)


@pytest.fixture
def witness(tmp_path):
    initial = {
        "identity": {"fixture": True},
        "identity_sha256": "same-fixture-identity",
        "completed_updates": 0,
        "model_state_dict": {"encoder.weight": torch.tensor([1.0, 2.0])},
    }
    trained = copy.deepcopy(initial)
    trained.update(completed_updates=2, cursor={"curves": [{"update": 1}, {"update": 2}]})
    trained["model_state_dict"]["encoder.weight"][0] += 0.25
    paths = [tmp_path / name for name in ("initial.pt", "trained.pt", "witness.json")]
    torch.save(initial, paths[0])
    torch.save(trained, paths[1])
    record = {
        "status": "PASS",
        "initial_checkpoint_sha256": file_hash(paths[0]),
        "trained_checkpoint_sha256": file_hash(paths[1]),
        "identity_sha256": "same-fixture-identity",
        "updates_compared": [0, 2],
        "encoder_tensors_checked": 1,
        "encoder_tensors_changed": 1,
        "encoder_maxabs": 0.25,
        "per_tensor_maxabs": {"encoder.weight": 0.25},
    }
    paths[2].write_text(json.dumps(record), encoding="utf-8")
    return paths, record, trained


def test_encoder_replay_reads_real_weights_without_optimizer(witness):
    paths, _, _ = witness
    result = replay_encoder_change(*paths)
    assert result["status"] == "PASS" and result["encoder_maxabs"] == 0.25
    assert result["optimizer_updates"] == 0 and not result["generalization_claim"]


@pytest.mark.parametrize("change", ["bytes", "identity", "unchanged_encoder", "curve", "claim"])
def test_encoder_replay_rejects_unsupported_claim(witness, change):
    paths, record, trained = witness
    if change == "bytes":
        paths[1].write_bytes(b"changed")
    elif change == "claim":
        record["encoder_maxabs"] = 0.5
    else:
        if change == "identity":
            trained["identity"] = {"fixture": "another-parent"}
        elif change == "unchanged_encoder":
            trained["model_state_dict"]["encoder.weight"][0] = 1.0
        else:
            trained["cursor"]["curves"] = [{"update": 1}, {"update": 1}]
        torch.save(trained, paths[1])
        record["trained_checkpoint_sha256"] = file_hash(paths[1])
    paths[2].write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        replay_encoder_change(*paths)


@pytest.fixture
def frozen_witness(tmp_path):
    initial = {
        "identity": {"fixture": "frozen"},
        "identity_sha256": "frozen-fixture-identity",
        "completed_updates": 0,
        "model_state_dict": {
            "encoder.weight": torch.tensor([0.0, 2.0]),
            "readout.weight": torch.tensor([1.0, 2.0]),
        },
    }
    trained = copy.deepcopy(initial)
    trained.update(
        completed_updates=2,
        cursor={"curves": [{"update": 1}, {"update": 2}]},
        optimizer_state_dict={"state": {0: {"step": torch.tensor(2)}}},
    )
    trained["model_state_dict"]["readout.weight"][0] += 0.25
    paths = [tmp_path / name for name in ("initial.pt", "trained.pt", "witness.json")]
    torch.save(initial, paths[0])
    torch.save(trained, paths[1])
    record = {
        "status": "PASS",
        "initial_checkpoint_sha256": file_hash(paths[0]),
        "trained_checkpoint_sha256": file_hash(paths[1]),
        "identity_sha256": "frozen-fixture-identity",
        "completed_updates": 2,
        "encoder_tensors_bitwise_unchanged": 1,
        "changed_readout_tensors": ["readout.weight"],
        "optimizer_steps": [2],
    }
    paths[2].write_text(json.dumps(record), encoding="utf-8")
    return paths, record, trained


def test_frozen_replay_proves_encoder_invariant_and_readout_change(frozen_witness):
    paths, _, _ = frozen_witness
    result = replay_frozen_encoder(*paths)
    assert result["encoder_tensors_bitwise_unchanged"] == 1
    assert result["readout_tensors_changed"] == 1
    assert result["optimizer_updates"] == 0 and not result["generalization_claim"]


@pytest.mark.parametrize(
    "change", ["encoder", "signed_zero", "readout", "identity", "curve", "optimizer", "nonfinite"]
)
def test_frozen_replay_rejects_unsupported_control(frozen_witness, change):
    paths, record, trained = frozen_witness
    if change == "encoder":
        trained["model_state_dict"]["encoder.weight"][1] += 0.25
    elif change == "signed_zero":
        trained["model_state_dict"]["encoder.weight"][0] = -0.0
    elif change == "readout":
        trained["model_state_dict"]["readout.weight"][0] = 1.0
    elif change == "identity":
        trained["identity"] = {"fixture": "different-parent"}
    elif change == "curve":
        trained["cursor"]["curves"] = [{"update": 1}, {"update": 1}]
    elif change == "optimizer":
        trained["optimizer_state_dict"]["state"][0]["step"] = torch.tensor(1)
    else:
        trained["model_state_dict"]["readout.weight"][0] = float("nan")
    torch.save(trained, paths[1])
    record["trained_checkpoint_sha256"] = file_hash(paths[1])
    paths[2].write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        replay_frozen_encoder(*paths)
