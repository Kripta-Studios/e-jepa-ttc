"""Smoke coverage for supervised TTC latency benchmarking."""

from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.models import build_regressor
from scripts.benchmark_supervised_latency import benchmark_supervised_latency


def test_supervised_latency_benchmark_is_explicitly_model_only(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    np.savez(cache_path, x=np.zeros((2, 2, 12, 16), dtype=np.float16))
    model = build_regressor("tiny-cnn", in_channels=2)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_name": "tiny-cnn",
            "model_state_dict": model.state_dict(),
            "epoch": 1,
        },
        checkpoint_path,
    )

    payload = benchmark_supervised_latency(
        cache_path=cache_path,
        checkpoint_path=checkpoint_path,
        output_path=tmp_path / "latency.json",
        device_name="cpu",
        warmup_iterations=1,
        measured_iterations=3,
    )

    assert payload["batch_size"] == 1
    assert payload["model_only"] is True
    assert payload["includes_event_voxelization"] is False
    assert payload["latency_ms"]["mean"] > 0.0
    assert payload["parameter_count"] > 0
    assert len(payload["artifact_sha256"]) == 64
