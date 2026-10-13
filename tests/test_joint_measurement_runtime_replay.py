"""Independent runtime replay must reject altered state even with updated file receipts."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from operational.joint_measurement.replay_runtime import STATE_KEYS, replay_runtime


def write(path: Path, value: dict) -> None:
    """Write a tiny isolated witness fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def sha(path: Path) -> str:
    """Hash fixture bytes independently from the replay implementation."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root: Path, mutation: str | None = None) -> None:
    """Create matching graph states with six versus two physical updates."""
    runtime = root / "campaign/runtime/optimization_20261013"
    source = root / "source/operational/joint_measurement/runtime.py"
    source.parent.mkdir(parents=True)
    source.write_text("# retained runtime\n", encoding="utf-8")
    for name in ("ADMISSION", "NUMERICAL_GRAPH_POLICY", "RESUME_PROOF_REGISTRATION"):
        write(runtime / f"{name}.json", {"fixture": name})
    registration = sha(runtime / "RESUME_PROOF_REGISTRATION.json")
    state = {
        "model_state_dict": {"weight": torch.tensor([0.25])},
        "optimizer_state_dict": {"state": {0: {"step": torch.tensor(6137.0)}}},
        "scheduler_state_dict": {"last_epoch": 6137},
        "sampler_generator_state": torch.tensor([1, 2], dtype=torch.uint8),
        "torch_rng_state": torch.tensor([3], dtype=torch.uint8),
        "cuda_rng_state_all": [torch.tensor([4], dtype=torch.uint8)],
        "numpy_random_state": ("MT19937", np.array([5], dtype=np.uint32), 0),
        "python_random_state": (3, (6, 7), None),
        "cursor": {
            "curve": [
                {"ids": list(range(32)), "losses": {"loss": 0.5}, "norm": 1.0} for _ in range(6)
            ]
        },
    }
    hashes = {}
    for mode, count, offset in (("continuous", 6, 0), ("restored", 2, 4)):
        directory = root / "campaign/diagnostics/fast_runtime_20261013" / f"resume_{mode}"
        identity = {
            "prefix_offset": offset,
            "updates": count,
            "registration": registration,
            "origin_updates": 6131,
        }
        payload = {**copy.deepcopy(state), "identity": identity, "completed_updates": count}
        if mode == "restored" and mutation in STATE_KEYS:
            if mutation == "model_state_dict":
                payload[mutation]["weight"][0] += 0.01
            elif mutation == "sampler_generator_state":
                payload[mutation][0] += 1
            elif mutation == "cursor":
                payload[mutation]["curve"][0]["ids"][0] = 99
        path = directory / "checkpoint_versions/checkpoint.pt"
        path.parent.mkdir(parents=True)
        torch.save(payload, path)
        hashes[mode] = sha(path)
        common = {"checkpoint_sha256": sha(path), "completed_updates": count}
        write(directory / "IDENTITY.json", identity)
        write(directory / "CHECKPOINT_POINTER.json", {**common, "version": path.name})
        write(
            directory / "CHECKPOINT_RECEIPT.json",
            {
                **common,
                "status": "COMPLETE",
                "complete_state": True,
                "accumulation_index": 0,
            },
        )
        write(
            directory / "UPDATE_JOURNAL.json",
            {
                "completed_updates": count,
                "durable_updates": count,
                "pending_update_upper": int(mutation == "pending" and mode == "restored"),
                "recovery_upper": 0,
            },
        )
    write(
        runtime / "INDEPENDENT_RESUME_CHECK.json",
        {
            "reference_sha256": hashes["continuous"],
            "restored_sha256": hashes["restored"],
        },
    )
    write(
        runtime / "FINAL_ADMISSION.json",
        {
            "status": "PASS",
            "eager_bitwise_equivalence": False,
            "retained_initial_admission_sha256": sha(runtime / "ADMISSION.json"),
            "resume_registration_sha256": registration,
            "runtime_sources": {"runtime.py": sha(source)},
        },
    )
    write(
        runtime / "RUNTIME_BINDING.json",
        {
            "status": "ADMITTED",
            "graph": True,
            "eager_bitwise_equivalence": False,
            "admission_sha256": sha(runtime / "FINAL_ADMISSION.json"),
            "numerical_graph_policy_sha256": sha(runtime / "NUMERICAL_GRAPH_POLICY.json"),
            "runtime_sources": {"operational/joint_measurement/runtime.py": sha(source)},
        },
    )
    if mutation == "source":
        source.write_text("# changed\n", encoding="utf-8")


def test_replays_all_states_without_recounting_shared_prefix(tmp_path: Path) -> None:
    """Physical work is eight, logical endpoint six, and replay trains nothing."""
    fixture(tmp_path)
    result = replay_runtime(tmp_path)
    assert result["status"] == "PASS"
    assert len(result["checks"]) == 9 and all(result["checks"].values())
    assert result["retained_physical_updates"] == 8
    assert result["logical_updates"] == 6 and result["shared_prefix_updates"] == 4
    assert result["optimizer_updates"] == 0 and not result["eager_bitwise_equivalence"]
    assert not result["cuda_initialized"]


@pytest.mark.parametrize(
    "mutation",
    [
        "model_state_dict",
        "sampler_generator_state",
        "cursor",
        "pending",
        "source",
    ],
)
def test_rejects_altered_witness_even_with_matching_hashes(tmp_path: Path, mutation: str) -> None:
    """File-integrity success alone cannot establish scientific state equivalence."""
    fixture(tmp_path, mutation)
    with pytest.raises(ValueError):
        replay_runtime(tmp_path)
