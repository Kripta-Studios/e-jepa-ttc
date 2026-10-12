"""Independent encoder-change evidence must be supported by actual retained weights."""

import copy
import json

import pytest
import torch

from operational.joint_measurement.replay import file_hash, replay_encoder_change


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
