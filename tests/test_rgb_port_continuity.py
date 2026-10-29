"""Exact durable state and lifecycle saves without technical optimizer updates."""

import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from operational.rgb_port.accounting import read_json_shared, sha256_file
from operational.rgb_port.train_producers import ProducerCheckpoint
from operational.rgb_port_continuity.checkpoint import installed


def fixture(tmp_path):
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    generator = torch.Generator().manual_seed(8)
    state = ProducerCheckpoint(
        tmp_path, {"identity_sha256": "fixture", "fit_id": "E_A5_MATCHED"}, 10
    )
    args = model, optimizer, scheduler, generator, {"position": 0, "order": torch.arange(3)}
    state.save(*args, status="READY")
    return state, args


def test_duplicate_pause_keeps_checkpoint_and_skips_all_auxiliary_wrappers(tmp_path):
    state, args = fixture(tmp_path)
    before = {p: sha256_file(p) for p in [state.path, state.pointer, *state.versions.glob("*.pt")]}
    with patch.object(ProducerCheckpoint, "save", side_effect=AssertionError("wrapper invoked")):
        with installed():
            state.save(*args, status="PAUSED_RESOURCE")
            state.save(*args, status="PAUSED_RESOURCE")
    assert before == {p: sha256_file(p) for p in before}
    assert read_json_shared(state.receipt)["status"] == "PAUSED_RESOURCE"
    assert read_json_shared(state.directory / "IDEMPOTENT_SAVE.json")["optimizer_updates"] == 0


@pytest.mark.parametrize("change", ["model", "rng", "cursor", "sampler", "scheduler"])
def test_changed_state_same_update_is_rejected_without_writes(tmp_path, change):
    state, args = fixture(tmp_path)
    before = {p: sha256_file(p) for p in [state.path, state.pointer, state.receipt, state.journal]}
    if change == "model":
        with torch.no_grad():
            args[0].weight.add_(1)
    elif change == "rng":
        torch.rand(1)
    elif change == "cursor":
        args[4]["position"] = 1
    elif change == "sampler":
        torch.rand(1, generator=args[3])
    else:
        args[2].last_epoch += 1
    with installed(), pytest.raises(RuntimeError, match="state changed"):
        state.save(*args, status="PAUSED_RESOURCE")
    assert before == {p: sha256_file(p) for p in before}


def test_new_update_delegates_and_pending_boundary_is_rejected(tmp_path):
    state, args = fixture(tmp_path)
    with installed(), patch.object(state, "pending", True):
        with pytest.raises(RuntimeError, match="optimizer boundary"):
            state.save(*args, status="PAUSED_RESOURCE")
    state.completed = 1  # Synthetic counter only; no optimizer update is executed.
    original = ProducerCheckpoint.save
    with patch.object(ProducerCheckpoint, "save", autospec=True, wraps=original) as delegate:
        with installed():
            state.save(*args, status="RUNNING")
        assert delegate.call_count == 1


def test_queue_routes_existing_delegates_through_guard(tmp_path):
    from operational.rgb_port_continuity import queue as guarded
    from operational.rgb_port_io_recovery import queue

    command = [
        "python",
        "-m",
        "operational.rgb_port_concurrent.producer",
        "--run",
        str(tmp_path),
        "--fit-id",
        "E_A5_MATCHED",
    ]
    with (
        patch.object(guarded, "validate"),
        patch.object(
            guarded, "validate_git", return_value={"original_base": "old", "admitted_head": "new"}
        ),
        patch("sys.argv", ["queue", "resume", "--run", str(tmp_path)]),
    ):

        def inner(args):
            routed = queue.route(command, tmp_path / "IO_RECOVERY_FREEZE.json")
            assert routed[2] == "operational.rgb_port_continuity.producer"
            assert routed[-4:] == command[-4:]
            inference = ["python", "-m", "operational.rgb_port.infer_experts", "observations"]
            assert queue.route(inference, tmp_path)[2] == "operational.rgb_port_streaming.infer"
            return 0

        with patch.object(queue, "main", side_effect=inner):
            assert guarded.main() == 0
    assert queue.PRODUCER_MODULE == "operational.rgb_port_io_recovery.producer"


def test_git_admission_rejects_unrecorded_commit(tmp_path):
    from operational.rgb_port.accounting import atomic_write_json
    from operational.rgb_port_continuity import git_admission

    atomic_write_json(tmp_path / git_admission.ADMISSION, {"head": "old"})
    with patch.object(git_admission, "inspect", return_value={"head": "new"}):
        with pytest.raises(ValueError, match="not been admitted"):
            git_admission.validate_git(tmp_path)


def test_runtime_commit_update_preserves_scientific_freeze_config(tmp_path):
    from operational.rgb_port_continuity import queue as guarded

    frozen = guarded.io_queue.frozen_queue
    config = {"base_commit": "old", "package": {"members": []}, "loss": "unchanged"}
    with (
        patch.object(guarded, "validate"),
        patch.object(
            guarded, "validate_git", return_value={"original_base": "old", "admitted_head": "new"}
        ),
        patch.object(frozen, "_load_config", return_value=config.copy()),
        patch.object(frozen, "_freeze_inputs", return_value={}) as freeze,
        patch("sys.argv", ["queue", "resume", "--run", str(tmp_path)]),
    ):

        def inner(args):
            runtime = frozen._load_config(tmp_path)
            assert runtime["base_commit"] == "new"
            assert "CONTINUITY_FREEZE.json" in runtime["package"]["members"]
            frozen._freeze_inputs(tmp_path, runtime, tmp_path)
            assert freeze.call_args.args[1]["base_commit"] == "old"
            assert freeze.call_args.args[1]["loss"] == "unchanged"
            return 0

        with patch.object(guarded.io_queue, "main", side_effect=inner):
            assert guarded.main() == 0
