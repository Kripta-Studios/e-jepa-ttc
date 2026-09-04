from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from e_jepa_ttc.models.raw_time_residual import (
    CausalTemporalBlock,
    RawModelBatch,
    RawTimeResidual,
    count_only_counts,
    normalize_arm_rates,
    permute_counts,
)
from e_jepa_ttc.training import raw_time_residual as training_module
from e_jepa_ttc.training.raw_time_residual import (
    RawTrainingConfig,
    TrainSupervision,
    fixed_derangements,
    train_raw_time_residual,
)


def _batch(size: int = 2) -> RawModelBatch:
    torch.manual_seed(7)
    return RawModelBatch(
        torch.randn(size, 2, 16, 2, 64, 64),
        torch.tensor([0.03, -0.02])[:size],
        torch.randn(size, 3),
        torch.randn(size, 6),
        torch.ones(size, 16, dtype=torch.bool),
    )


def test_model_matches_reference_zero_init_and_has_frozen_size() -> None:
    torch.manual_seed(17)
    model = RawTimeResidual()
    batch = _batch()
    with torch.no_grad():
        actual = model(batch).benchmark_phase
    torch.testing.assert_close(actual, batch.a5_phase, rtol=0, atol=0)
    assert sum(parameter.numel() for parameter in model.parameters()) == 30_138


def test_empty_mask_is_exact_a5_and_state_ignores_volume() -> None:
    model = RawTimeResidual()
    batch = _batch()
    empty = RawModelBatch(
        batch.normalized_rates,
        batch.a5_phase,
        batch.normalized_a5_state,
        batch.times,
        torch.zeros_like(batch.valid_patches),
    )
    with torch.no_grad():
        output = model(empty)
    assert torch.equal(output.benchmark_phase, batch.a5_phase)
    assert torch.equal(output.applied_phase_delta, torch.zeros_like(batch.a5_phase))


def test_count_and_perm_conserve_exact_mass() -> None:
    counts = torch.arange(2 * 2 * 16 * 2 * 2 * 2).reshape(2, 2, 16, 2, 2, 2).float()
    durations = torch.full((2, 2, 16), 0.00625)
    durations[:, :, 0] += 1e-6
    count = count_only_counts(counts, durations)
    torch.testing.assert_close(count.sum(dim=2), counts.sum(dim=2))
    permutation = torch.as_tensor(fixed_derangements(["a", "b"], 7))
    permuted = permute_counts(counts, permutation)
    torch.testing.assert_close(permuted.sum(dim=2), counts.sum(dim=2))


def test_state_arm_runs_same_shape_but_is_volume_invariant() -> None:
    counts = torch.ones(2, 2, 16, 2, 4, 4)
    duration = torch.full((2, 2, 16), 0.01)
    mean, std = torch.zeros(2), torch.ones(2)
    a = normalize_arm_rates(counts, duration, mean, std, arm="S64-STATE-L1")
    b = normalize_arm_rates(counts * 100, duration, mean, std, arm="S64-STATE-L1")
    assert torch.equal(a, b) and torch.count_nonzero(a) == 0


def test_tcn_is_strictly_causal() -> None:
    block = CausalTemporalBlock(32, 4)
    first = torch.randn(2, 32, 16)
    second = first.clone()
    second[..., 8:] = torch.randn_like(second[..., 8:])
    with torch.no_grad():
        a, b = block(first), block(second)
    assert torch.equal(a[..., :8], b[..., :8])


class _TinyResidual(torch.nn.Module):
    """Cheap deterministic stand-in that exercises the real resume machinery."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(self, batch: RawModelBatch) -> SimpleNamespace:
        return SimpleNamespace(
            benchmark_phase=batch.a5_phase + self.weight * batch.normalized_a5_state[:, 1]
        )


def _run_tiny_training(output: Path, *, resume: bool, stop_after: int | None) -> dict[str, object]:
    rows = 64
    state_axis = np.linspace(-1.0, 1.0, rows, dtype=np.float32)
    state = np.column_stack((state_axis, state_axis**2, np.sin(state_axis))).astype(np.float32)
    return train_raw_time_residual(
        counts=np.zeros((rows, 2, 16, 2, 64, 64), dtype=np.uint8),
        durations_s=np.full((rows, 2, 16), 0.01, dtype=np.float32),
        times=np.zeros((rows, 6), dtype=np.float32),
        valid_patches=np.ones((rows, 16), dtype=bool),
        a5_state=state,
        sample_tokens=[f"token-{index}" for index in range(rows)],
        supervision=TrainSupervision(state[:, 0] + 0.1 * state[:, 1], np.full(rows, 1 / rows)),
        rate_mean=np.zeros(2, dtype=np.float32),
        rate_std=np.ones(2, dtype=np.float32),
        arm="S64-STATE-L1",
        seed=7,
        outer_fold=0,
        output_dir=output,
        device=torch.device("cpu"),
        identity={"fixture": "resume-10-v1"},
        config=RawTrainingConfig(updates=10, checkpoint_interval=5),
        resume=resume,
        stop_after_updates=stop_after,
    )


def test_resume_10_matches_5_plus_5_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training_module, "RawTimeResidual", _TinyResidual)
    continuous = tmp_path / "continuous"
    resumed = tmp_path / "resumed"
    _run_tiny_training(continuous, resume=False, stop_after=None)
    partial = _run_tiny_training(resumed, resume=False, stop_after=5)
    assert partial["status"] == "checkpointed"
    _run_tiny_training(resumed, resume=True, stop_after=None)
    a = torch.load(continuous / "checkpoint_last.pt", weights_only=False)
    b = torch.load(resumed / "checkpoint_last.pt", weights_only=False)
    assert a["completed_updates"] == b["completed_updates"] == 10
    assert a["loss_history"] == b["loss_history"]
    assert a["schedule_sha256"] == b["schedule_sha256"]
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key])
    assert a["rng"]["python"] == b["rng"]["python"]
    assert np.array_equal(a["rng"]["numpy"][1], b["rng"]["numpy"][1])
    assert torch.equal(a["rng"]["torch_cpu"], b["rng"]["torch_cpu"])


def test_resume_rejects_corrupt_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training_module, "RawTimeResidual", _TinyResidual)
    output = tmp_path / "corrupt"
    output.mkdir()
    (output / "checkpoint_last.pt").write_bytes(b"not a torch checkpoint")
    with pytest.raises((pickle.UnpicklingError, RuntimeError, EOFError)):
        _run_tiny_training(output, resume=True, stop_after=None)
