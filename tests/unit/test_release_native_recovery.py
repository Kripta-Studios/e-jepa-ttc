"""Interrupted-fragment recovery rejects identity changes and seals only complete pairs."""

import hashlib

import numpy as np
import pytest

from operational.release_research.protocol import sha256, write_json
from scripts.resume_release_native_research import inventory


def fixture_capture(tmp_path):
    prior = tmp_path / "prior"
    prior.mkdir()
    item = {"kind": "fixture_without_sequence_field"}
    public = tmp_path / "public.json"
    write_json(
        public,
        {
            "physical_aliases": {"q": "q"},
            "queries": [
                {
                    "sample_token": "q",
                    "sequence_id": "s",
                    "event_history": ["a", "b", "c"],
                    "rgb_history": [],
                }
            ],
            "event_observations": {k: item for k in "abc"},
            "rgb_observations": {},
        },
    )
    write_json(
        prior / "FREEZE.json",
        {
            "input_paths": {"public_inputs": str(public)},
            "inputs": {"public_inputs": sha256(public)},
        },
    )
    folders = []
    for identity in "abc":
        key = hashlib.sha256(("event:" + identity).encode()).hexdigest()[:24]
        folder = prior / "observations" / key
        folder.mkdir(parents=True)
        write_json(
            folder / "CALLS.json",
            {
                "source": item,
                "calls": [
                    {
                        "status": "accepted",
                        "prefix": "E",
                        "sensor_shape": [1, 3, 12, 128, 128],
                        "observations": [{"observation_id": identity}],
                    }
                ],
            },
        )
        arrays = {
            k: np.ones(1, np.float32)
            for k in (
                "call0.delta",
                "call0.raw17",
                "call0.expert3",
                "call0.PAIR_E_MATCHED.input133",
                "call0.PAIR_E_MATCHED.raw_phase",
            )
        }
        np.savez_compressed(folder / "PAIR_INPUTS.npz", allow_pickle=False, **arrays)
        folders.append(folder)
    return prior, public, folders


def test_recovery_seals_all_fragments_and_fixes_boundary_sentinels(tmp_path):
    prior, _, _ = fixture_capture(tmp_path)
    result = inventory(prior, tmp_path / "inventory.json")
    assert result["sentinels"] == ["event:a", "event:c"]
    assert len(result["records"]) == 3
    assert result["hashes_first_sealed_at_recovery"]
    assert not result["previous_complete_existed"]


def test_recovery_rejects_changed_original_input(tmp_path):
    prior, public, _ = fixture_capture(tmp_path)
    public.write_text("{}")
    with pytest.raises(ValueError, match="source/input changed"):
        inventory(prior, tmp_path / "inventory.json")


def test_recovery_rejects_wrong_observation_identity(tmp_path):
    prior, _, folders = fixture_capture(tmp_path)
    path = folders[0] / "CALLS.json"
    path.write_text(path.read_text().replace('"observation_id": "a"', '"observation_id": "wrong"'))
    with pytest.raises(ValueError, match="identity differs"):
        inventory(prior, tmp_path / "inventory.json")


@pytest.mark.parametrize("damage", ["missing_npz", "null_json"])
def test_incomplete_fragment_is_explicitly_recomputed(tmp_path, damage):
    prior, _, folders = fixture_capture(tmp_path)
    if damage == "missing_npz":
        (folders[-1] / "PAIR_INPUTS.npz").unlink()
    else:
        (folders[-1] / "CALLS.json").write_bytes(b"\x00" * 80)
    result = inventory(prior, tmp_path / "inventory.json")
    assert result["incomplete_recompute"] == [str(folders[-1])]
    assert len(result["records"]) == 2
    assert result["sentinels"] == ["event:a", "event:b"]
