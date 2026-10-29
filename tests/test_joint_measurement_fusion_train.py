"""Protect fixed paired endpoint admission and native H supervision."""

import numpy as np
import pandas as pd
import pytest

from operational.joint_measurement import fusion_train as train
from operational.range_transfer.common import sha, write


def test_signed_native_supervision_is_preserved_without_renormalization():
    rows = pd.DataFrame({"target_ttc": [-2.5, 0.6, 8.0], "mass": [0.25, 0.5, 0.25]})
    target, mass = train.supervision(rows)
    np.testing.assert_array_equal(target, np.array([-2.5, 0.6, 8.0], np.float32))
    np.testing.assert_array_equal(mass, rows.mass.to_numpy())


@pytest.mark.parametrize(
    "field,values",
    [("target_ttc", [0, 2]), ("target_ttc", [np.nan, 2]), ("mass", [0.5, 0.3]), ("mass", [-1, 2])],
)
def test_malformed_native_supervision_is_not_silently_repaired(field, values):
    rows = pd.DataFrame({"target_ttc": [-2.5, 0.6], "mass": [0.5, 0.5]})
    rows[field] = values
    with pytest.raises(ValueError, match="supervision"):
        train.supervision(rows)


def endpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "DIRECTORY", tmp_path)
    monkeypatch.setattr(train, "REGISTRATION", tmp_path / "REGISTRATION.json")
    record = {"recipes": {mode: {"mode": mode, "updates": 2500} for mode in train.MODES}}
    write(train.REGISTRATION, record)
    for mode in train.MODES:
        directory = tmp_path / "fits" / f"FUSION-{mode}-S7"
        version = directory / "checkpoint_versions/checkpoint_002500.pt"
        version.parent.mkdir(parents=True)
        version.write_bytes(f"{mode} fixture, not a model".encode())
        (directory / "checkpoint_last.pt").write_bytes(version.read_bytes())
        identity = {
            "recipe": record["recipes"][mode],
            "identity_sha256": mode,
            "binding": {"registration_sha256": sha(train.REGISTRATION)},
        }
        write(directory / "IDENTITY.json", identity)
        write(
            directory / "CHECKPOINT_RECEIPT.json",
            {
                "status": "COMPLETE",
                "scientific_endpoint": True,
                "completed_updates": 2500,
                "identity_sha256": mode,
                "checkpoint_sha256": sha(version),
            },
        )
        write(
            directory / "UPDATE_JOURNAL.json",
            {
                "completed_updates": 2500,
                "durable_updates": 2500,
                "pending_update_upper": 0,
                "identity_sha256": mode,
            },
        )
        write(
            directory / "CHECKPOINT_POINTER.json",
            {
                "completed_updates": 2500,
                "identity_sha256": mode,
                "checkpoint_sha256": sha(version),
                "version": version.name,
            },
        )
    return tmp_path / "fits/FUSION-ZERO-S7"


@pytest.mark.parametrize("failure", ["partial", "pending", "bytes", "recipe"])
def test_second_endpoint_must_pass_every_freeze_condition(tmp_path, monkeypatch, failure):
    directory = endpoints(tmp_path, monkeypatch)
    if failure == "bytes":
        (directory / "checkpoint_last.pt").write_bytes(b"changed")
    else:
        name = "IDENTITY.json" if failure == "recipe" else "UPDATE_JOURNAL.json"
        value = train.read(directory / name)
        if failure == "recipe":
            value["recipe"]["mode"] = "TRUE"
        else:
            value["durable_updates" if failure == "partial" else "pending_update_upper"] = 1
        write(directory / name, value)
    with pytest.raises(ValueError, match="Both fusion endpoints"):
        train.freeze_endpoints()
    assert not (tmp_path / "ENDPOINTS.json").exists()


def test_matching_pair_freezes_together(tmp_path, monkeypatch):
    endpoints(tmp_path, monkeypatch)
    result = train.freeze_endpoints()
    assert set(result["endpoints"]) == {"FUSION-TRUE-S7", "FUSION-ZERO-S7"}
