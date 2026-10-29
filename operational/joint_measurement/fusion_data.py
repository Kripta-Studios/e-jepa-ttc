"""Resident, label-free fusion inputs from admitted new producer feature caches.

The caller owns GPU admission and the training sampler. This module neither
selects endpoints nor fits statistics, and performs no optimizer updates.
"""

from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.rgb_port.normalization import FrozenNormalizer

from .fusion_features import FeatureBlock


class FusionTensorBank:
    """Normalize once and retain immutable copies instead of decoding each batch.

    The query manifest and ordered IDs must come from the registered population,
    independently of either cache. Cache bytes and feature identities must first
    pass load_feature_cache admission. Every source row remains present, including
    missing RGB observations that take the event fallback.
    """

    def __init__(
        self,
        rgb: FeatureBlock,
        event: FeatureBlock,
        rgb_normalizer: FrozenNormalizer,
        event_normalizer: FrozenNormalizer,
        *,
        role: str,
        manifest_sha256: str,
        sample_tokens: np.ndarray,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.role = role
        self.population = len(sample_tokens)
        self.sample_tokens = sample_tokens.copy()
        self.manifest_sha256 = manifest_sha256
        self.inputs: dict[str, torch.Tensor] = {}
        self.sensor_pair_us: dict[str, np.ndarray] = {}
        for modality, block, normalizer in (
            ("rgb", rgb, rgb_normalizer),
            ("event", event, event_normalizer),
        ):
            if block.identity.modality != modality:
                raise ValueError("Fusion modalities were swapped")
            block.validate(
                expected_identity=block.identity,
                role=role,
                manifest_sha256=manifest_sha256,
                sample_tokens=sample_tokens,
                feature_count=block.features.shape[-1],
            )
            width = block.features.shape[1]
            if (
                normalizer.mean.shape != (width,)
                or normalizer.scale.shape != (width,)
                or not np.isfinite(normalizer.mean).all()
                or not np.isfinite(normalizer.scale).all()
                or np.any(normalizer.scale <= 0)
            ):
                raise ValueError("Invalid fusion normalizer dimensions or values")
            normalized = block.normalized(normalizer)
            if not np.isfinite(normalized).all():
                raise ValueError("Fusion normalization overflow")
            # torch.tensor owns storage, including on CPU: later cache mutation
            # cannot silently modify a run that already passed admission.
            self.inputs[f"{modality}_features"] = torch.tensor(
                normalized, dtype=torch.float32, device=self.device
            )
            self.inputs[f"{modality}_ttc"] = torch.tensor(
                block.prediction_ttc, dtype=torch.float32, device=self.device
            )
            self.sensor_pair_us[modality] = block.sensor_pair_us.copy()
        self.inputs["rgb_available"] = torch.tensor(
            rgb.available, dtype=torch.bool, device=self.device
        )
        self.storage_bytes = sum(
            value.numel() * value.element_size() for value in self.inputs.values()
        )

    def batch(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """Gather declared indices; preserve repeats, order and the caller's RNG.

        The sampler supplies CPU int64 indices, just as the resumable producers
        do. Only those small indices transfer per batch when the bank is on GPU.
        No data labels, weights, role codes or absolute timestamps enter forward.
        """
        if (
            indices.device.type != "cpu"
            or indices.dtype != torch.int64
            or indices.ndim != 1
            or indices.numel() == 0
            or (indices < 0).any()
            or (indices >= self.population).any()
        ):
            raise ValueError("Fusion batch requires nonempty in-range CPU int64 indices")
        selected = indices.to(self.device)
        return {name: value.index_select(0, selected) for name, value in self.inputs.items()}
