"""Prospective CUDA QA must reject stale CPU evidence and never steal a live driver's gap."""

import pytest

from operational.joint_measurement import fusion_cuda_qa as qa
from operational.range_transfer.common import read, sha, write


def cpu_proof(tmp_path, monkeypatch):
    monkeypatch.setattr(qa, "CPU", tmp_path / "cpu")
    monkeypatch.setattr(qa, "OUT", tmp_path / "cuda")
    monkeypatch.setattr(qa, "committed_sources", lambda _: "test")
    write(qa.CPU / "REGISTRATION.json", {"sources": {}})
    write(
        qa.CPU / "PASS.json",
        {
            "status": "PASS",
            "registration_sha256": sha(qa.CPU / "REGISTRATION.json"),
            "checks": {
                mode: {key: True for key in (*qa.STATE_KEYS, "sample_ids_losses_norms")}
                for mode in ("TRUE", "ZERO")
            },
        },
    )


@pytest.mark.parametrize("change", ["status", "registration", "failed_check", "missing_state"])
def test_invalid_cpu_proof_rejects_cuda_registration(tmp_path, monkeypatch, change):
    cpu_proof(tmp_path, monkeypatch)
    proof = read(qa.CPU / "PASS.json")
    if change == "status":
        proof["status"] = "FAIL"
    elif change == "registration":
        proof["registration_sha256"] = "a" * 64
    elif change == "failed_check":
        proof["checks"]["TRUE"]["optimizer_state_dict"] = False
    else:
        del proof["checks"]["ZERO"]["sampler_generator_state"]
    write(qa.CPU / "PASS.json", proof)
    with pytest.raises(ValueError, match="CPU fusion resume proof"):
        qa.register()
    assert not (qa.OUT / "REGISTRATION.json").exists()


def test_registration_is_prospective_and_equal_capacity(tmp_path, monkeypatch):
    cpu_proof(tmp_path, monkeypatch)
    result = qa.register()
    true, zero = (result["recipes"][mode].copy() for mode in ("TRUE", "ZERO"))
    assert true.pop("mode") == "TRUE" and zero.pop("mode") == "ZERO" and true == zero
    assert result["physical_updates"] == 40 and result["scientific_updates"] == 0
    assert not result["scientific_recipe_registered"]
    assert not list(qa.OUT.rglob("UPDATE_JOURNAL.json"))


def test_live_driver_rejects_before_reservation_or_cuda_allocation(tmp_path, monkeypatch):
    monkeypatch.setattr(qa, "OUT", tmp_path / "cuda")
    monkeypatch.setattr(qa, "register", lambda: {})

    def active():
        raise RuntimeError("GPU_OWNER_WAIT: live training driver 40132")

    def forbidden(*args, **kwargs):
        raise AssertionError("Guard must run before allocating or reserving anything")

    monkeypatch.setattr(qa, "no_live_training_driver", active)
    monkeypatch.setattr(qa, "fixture", forbidden)
    monkeypatch.setattr(qa, "gpu_charge", forbidden)
    monkeypatch.setattr(qa, "require_authorization", forbidden)
    with pytest.raises(RuntimeError, match="live training driver"):
        qa.run()
    assert not qa.OUT.exists()
