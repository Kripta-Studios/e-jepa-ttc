"""Deployment normalization/order parity and candidate checkpoint admission."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from operational.efficient_context.common import digest
from operational.simplex_t_shared_route.adapter import inputs
from operational.ttc_revision import head
from operational.ttc_revision.direct_runtime import DirectRuntime


def _campaign(path):
    contract = {
        "source_sha256": {"head.py": digest(Path(head.__file__))},
        "feature_manifest_sha256": "fixture",
        "recipe": {"updates": 2500},
    }
    (path / "TRAINING_FREEZE.json").write_text(json.dumps(contract))
    for seed in (7, 13, 23):
        torch.manual_seed(seed)
        checkpoint = path / f"direct_seed{seed}.pt"
        torch.save(
            {
                "model": head.DirectTTCHead().state_dict(),
                "contract": contract,
                "seed": seed,
                "update": 2500,
            },
            checkpoint,
        )
        checkpoint.with_suffix(".json").write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "sha256": digest(checkpoint),
                }
            )
        )
    return SimpleNamespace(
        device=torch.device("cpu"),
        mean=np.arange(17, dtype=float),
        scale=np.arange(1, 18, dtype=float),
        bindings={"normalization": {"manifest_sha256": "fixture"}},
    )


def test_direct_deployment_matches_scoring_adapter_and_seed_order(tmp_path):
    frozen = _campaign(tmp_path)
    runtime = DirectRuntime(frozen, tmp_path)
    raw = torch.arange(136, dtype=torch.float32).reshape(8, 17) / 10
    xs = inputs(
        "H8_SEED7",
        raw.numpy(),
        np.ones(8, bool),
        np.arange(350000, -1, -50000, dtype=np.int64),
        0,
        0,
        frozen.mean,
        frozen.scale,
    )[:3]
    with torch.inference_mode():
        expected = torch.cat([runtime.direct_heads[s](*xs) for s in (23, 7)])
    actual = runtime.predict_from_features(raw, seeds=(23, 7))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="finite"):
        runtime.predict_from_features(torch.full((8, 17), float("nan")))


def test_direct_checkpoint_change_is_rejected_before_loading(tmp_path):
    frozen = _campaign(tmp_path)
    (tmp_path / "direct_seed7.pt").write_bytes(b"modified checkpoint")
    with pytest.raises(ValueError, match="verified"):
        DirectRuntime(frozen, tmp_path)
