"""Four-update CPU admission for real RGB producer checkpoint equivalence.

This file is skipped unless the root QA supervisor explicitly admits and accounts
four technical optimizer updates.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from e_jepa_ttc.rgb_port.data import make_rgb_producer_source
from operational.rgb_port.accounting import atomic_write_json, sha256_file
from operational.rgb_port.recipe import resolved_recipe
from operational.rgb_port.train_producers import fit_producer

pytestmark = pytest.mark.skipif(
    os.environ.get("RGB_PORT_TECHNICAL_RESUME_QA") != "1",
    reason="requires root_qa --allow-updates 4 accounting",
)

CONFIG = Path("configs/rgb_port/producers.json")
P_MANIFEST = Path("artifacts/rgb_port_20261008/P_MANIFEST.json")


class _OneRealRGBSource:
    """One-row deterministic view retaining the actual RGB/DINO tensors."""

    def __init__(self, source: Any, index: int = 0) -> None:
        self.source = source
        self.index = int(index)
        self.population_size = 1
        self.frame_counts: Sequence[int] = (int(source.frame_counts[self.index]),)
        self.identity: Mapping[str, Any] = {
            **dict(source.identity),
            "population_size": 1,
            "technical_subset_index": self.index,
            "technical_subset_only": True,
        }

    def batch(self, indices: Sequence[int], modality: str) -> ObjectEventV4Batch:
        if list(indices) != [0]:
            raise ValueError("technical source contains exactly local row zero")
        return self.source.batch([self.index], modality)


def _payload(run: Path) -> dict[str, Any]:
    path = run / "fits" / "R_A5" / "checkpoint_last.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, Mapping):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_real_rgb_two_step_resume_is_bit_exact_to_uninterrupted(tmp_path: Path) -> None:
    """Consume exactly four accounted CPU updates across the two executions."""
    torch.set_num_threads(2)
    manifest = P_MANIFEST.resolve(strict=True)
    real_source = make_rgb_producer_source(manifest)
    source = _OneRealRGBSource(real_source)
    recipe = resolved_recipe(
        CONFIG,
        fit_id="R_A5",
        producer_population=1,
        role_manifest_sha256=sha256_file(manifest),
        microbatch_size=1,
    )
    freeze = {
        "schema": "rgb_port_technical_source_freeze_v1",
        "base_commit": "technical_resume_admission",
        "producer_source_identity": dict(source.identity),
    }
    resumed = tmp_path / "resumed"
    continuous = tmp_path / "continuous"
    for run in (resumed, continuous):
        atomic_write_json(run / "SOURCE_FREEZE.json", freeze)

    first = fit_producer(source, recipe, resumed, device="cpu", max_updates_this_call=1)
    gc.collect()
    second = fit_producer(source, recipe, resumed, device="cpu", max_updates_this_call=1)
    gc.collect()
    direct = fit_producer(source, recipe, continuous, device="cpu", max_updates_this_call=2)

    assert first["completed_updates"] == 1
    assert second["completed_updates"] == direct["completed_updates"] == 2
    resumed_payload, direct_payload = _payload(resumed), _payload(continuous)
    for field in (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "sampler_generator_state",
        "torch_rng_state",
        "numpy_random_state",
        "python_random_state",
        "cursor",
        "completed_updates",
        "accumulation_index",
    ):
        _assert_nested_equal(resumed_payload[field], direct_payload[field])
