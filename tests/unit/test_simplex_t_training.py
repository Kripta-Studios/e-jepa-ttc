"""Accounted 20-update CPU production-engine resume proof on synthetic inputs."""

import hashlib

import pytest
import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig
from e_jepa_ttc.simplex_t.training import fit, learning_rate


class SyntheticSource:
    population = 48

    def __init__(self):
        rng = torch.Generator().manual_seed(918)
        self.x = torch.randn(48, 8, 17, generator=rng)
        self.experts = 0.025 + 0.01 * torch.randn(48, 3, generator=rng)
        self.truth = self.experts.median(-1).values + 0.009 * torch.tanh(self.x[:, -3, 0])
        self.identity_sha256 = hashlib.sha256(self.x.numpy().tobytes()).hexdigest()

    def gather(self, ids):
        size = len(ids)
        timing = torch.zeros(size, 8, 4)
        timing[:, :, :2] = torch.arange(7, -1, -1)[None, :, None] * 0.1
        timing[:, 1:, 2] = 0.1
        return (
            self.x[ids],
            timing,
            torch.ones(size, 8, dtype=torch.bool),
            self.experts[ids],
            self.truth[ids],
            torch.full((size,), 1 / 48),
        )


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    else:
        assert left == right


def test_production_cpu_10_versus_5_plus_5(tmp_path):
    source = SyntheticSource()
    kwargs = dict(seed=7, freeze_sha256="f" * 64, resource_ok=lambda: True)
    fit(source, TemporalConfig(), tmp_path / "continuous", stop_after=10, **kwargs)
    fit(source, TemporalConfig(), tmp_path / "split", stop_after=5, **kwargs)
    result = fit(source, TemporalConfig(), tmp_path / "split", stop_after=10, resume=True, **kwargs)
    assert not result["scientific_endpoint"]
    first = torch.load(tmp_path / "continuous/checkpoint_last.pt", weights_only=True)
    second = torch.load(tmp_path / "split/checkpoint_last.pt", weights_only=True)
    assert_equal(first, second)


def test_busy_resource_saves_zero_update_state(tmp_path):
    result = fit(
        SyntheticSource(),
        TemporalConfig(),
        tmp_path,
        seed=7,
        freeze_sha256="f" * 64,
        resource_ok=lambda: False,
    )
    assert result["status"] == "PAUSED_RESOURCE"
    assert result["completed_updates"] == 0


def test_schedule_exact_endpoints():
    assert learning_rate(100) == 3e-4
    assert learning_rate(2500) == 3e-5
    assert learning_rate(1) == pytest.approx(3e-6, rel=1e-15)
