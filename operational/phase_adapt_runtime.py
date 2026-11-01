"""Cached sensor IO and captured producers with the original fusion state ABI."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from torch import Tensor

from operational.rgb_port import transfer as engine
from operational.train40_runtime.inputs import PersistentSensorInputs
from operational.train40_runtime.producers import ProducerRuntime
from scripts.final_train40_transfer import FinalTransferPredictor


class CachedAdaptPredictor(FinalTransferPredictor):
    """Single-owner offline extractor; preserve native batches and context forward."""

    def __init__(self, paths: dict[str, Path], device: str = "cuda") -> None:
        super().__init__(paths, device)
        self.inputs = PersistentSensorInputs(rgb_workers=4)
        self.producers = ProducerRuntime(self.models, self.device, graphs=device == "cuda")

    def predict(self, row: dict[str, Any]) -> dict[str, Any]:
        """Replay captured producer cores while retaining original head hooks."""

        def expert(
            _models: dict[str, Any], modality: str, sensor: Tensor, delta: Tensor
        ) -> tuple[Tensor, Tensor]:
            return self.producers.expert_block(modality, sensor, delta)

        with (
            patch.object(engine, "prepare_event", self.inputs.prepare_event),
            patch.object(engine, "raw_rgb_observations", self.inputs.raw_rgb_observations),
            patch.object(engine, "_expert_block", expert),
        ):
            result = super().predict(row)
        result["sensor_cache"] = dict(self.inputs.stats)
        result["execution"] = "cached_io_native_batches_cuda_producer_cores"
        return result

    def close(self) -> None:
        """Close bounded raw caches and crop workers."""
        self.inputs.close()
