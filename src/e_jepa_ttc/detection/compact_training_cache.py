"""Lossless prepared training residency with bounded CPU/GPU image slabs.

Pillow RGB resize yields uint8 pixels. Normalization maps each channel to 256
float32 values. Store their indices, then gather the original CPU-computed values
on the training device. Every insertion verifies bitwise round-trip parity.
Binary object-region targets are packed; original raw images are not retained.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.detection.campaign_training import CampaignConfig, ResumablePermutationSampler
from e_jepa_ttc.detection.eap_rgb_data import RawRGBSample
from e_jepa_ttc.detection.monodgp_eap_adapter import PreparedSample
from e_jepa_ttc.detection.resident_input_cache import SampleReader


def normalization_table() -> torch.Tensor:
    """Reproduce the adapter's float32 arithmetic for all possible RGB values."""
    values = torch.arange(256, dtype=torch.float32)[None, :] / 255
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None]
    return (values - mean) / std


def encode_image(image: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Encode only exactly representable normalized RGB images, rejecting drift."""
    if image.device.type != "cpu" or image.dtype != torch.float32 or image.shape[0] != 3:
        raise ValueError("Compact image encoding requires CPU float32 RGB")
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    approximate = ((image * std + mean) * 255).round()
    if (
        not torch.isfinite(approximate).all()
        or not ((approximate >= 0) & (approximate <= 255)).all()
    ):
        raise ValueError("Prepared image is outside the uint8 RGB normalization table")
    codes = approximate.to(torch.uint8).contiguous()
    recovered = decode_image(codes, table)
    if not torch.equal(image, recovered):
        raise ValueError("Compact image representation would change prepared inputs")
    return codes


def decode_image(codes: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Gather stored float32 values without recomputing normalization on CUDA."""
    indices = codes.to(table.device).reshape(3, -1).long()
    return torch.gather(table, 1, indices).reshape(codes.shape)


def peek_permutation_batch(sampler: ResumablePermutationSampler, count: int) -> list[int]:
    """Peek across epoch boundaries without advancing sampler or global RNG state."""
    if not 0 < count <= sampler.size:
        raise ValueError("Peek size must fit one population")
    result = sampler.order[sampler.position : sampler.position + count]
    if len(result) < count:
        generator = torch.Generator()
        generator.set_state(sampler.generator.get_state())
        following = torch.randperm(sampler.size, generator=generator).tolist()
        result += following[: count - len(result)]
    return result


@dataclass(frozen=True)
class TrainingSampleIdentity:
    """Training bookkeeping only; deliberately does not masquerade as a raw image."""

    sample_token: str
    member_sha256: str


class CompactTrainingCache:
    """Retain the complete finite training population without FP32/raw-image copies."""

    def __init__(
        self,
        dataset: SampleReader,
        config: CampaignConfig,
        member_sha256: dict[str, str],
        *,
        prepare: Callable[[RawRGBSample], PreparedSample],
        device: str = "cuda",
        gpu_image_budget: int = 28 * 2**30,
        cpu_image_budget: int = 30 * 2**30,
        metadata_budget: int = 4 * 2**30,
    ) -> None:
        self.dataset = dataset
        if (
            device not in {"cuda", "cpu"}
            or min(gpu_image_budget, cpu_image_budget, metadata_budget) < 0
        ):
            raise ValueError("Invalid compact cache device or memory budget")
        self.config = config
        self.member_sha256 = member_sha256
        self.prepare = prepare
        self.label_free_model_input = False
        self.device = device
        self.capacity = len(dataset)
        self.height, self.width = config.model.height, config.model.width
        image_bytes = 3 * self.height * self.width
        self.gpu_count = (
            min(self.capacity, gpu_image_budget // image_bytes) if device == "cuda" else 0
        )
        self.cpu_count = self.capacity - self.gpu_count
        if self.capacity < 1 or self.cpu_count * image_bytes > cpu_image_budget:
            raise ValueError("Full compact training population exceeds declared CPU image budget")
        # Single allocations avoid per-image CUDA allocator fragmentation.
        self.gpu_images = torch.empty(
            (self.gpu_count, 3, self.height, self.width), dtype=torch.uint8, device=device
        )
        self.cpu_images = torch.empty(
            (self.cpu_count, 3, self.height, self.width), dtype=torch.uint8, device="cpu"
        )
        self.cpu_table = normalization_table()
        self.table = self.cpu_table.to(device)
        self.entries: dict[int, tuple[PreparedSample, np.ndarray, TrainingSampleIdentity]] = {}
        self.metadata_budget = metadata_budget
        self.metadata_bytes = 0
        self.hits = 0
        self.misses = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="train_cpu_prepare")
        self._pending: dict[
            int,
            Future[tuple[torch.Tensor, PreparedSample, np.ndarray, TrainingSampleIdentity, int]],
        ] = {}

    def _prepare(
        self, index: int
    ) -> tuple[torch.Tensor, PreparedSample, np.ndarray, TrainingSampleIdentity, int]:
        """Own reader access on one CPU worker; never touch CUDA or sampler state."""
        raw = self.dataset[index]
        if self.member_sha256.get(raw.sample_token) != raw.member_sha256:
            raise ValueError(f"Training RGB member differs from admission: {raw.sample_token}")
        prepared = self.prepare(raw)
        if tuple(prepared.image.shape) != (3, self.height, self.width):
            raise ValueError("Prepared training image shape differs")
        codes = encode_image(prepared.image, self.cpu_table)
        region = prepared.target["obj_region"]
        if tuple(region.shape) != (self.height, self.width) or region.dtype != torch.float32:
            raise ValueError("Object region shape/dtype differs")
        array = region.numpy()
        if not np.all((array == 0) | (array == 1)):
            raise ValueError("Object region is not binary; refusing lossy packing")
        packed = np.packbits(array.astype(np.uint8), axis=None)
        target = {key: value for key, value in prepared.target.items() if key != "obj_region"}
        size = packed.nbytes + sum(v.numel() * v.element_size() for v in target.values())
        size += prepared.calibration.numel() * prepared.calibration.element_size()
        size += len(json.dumps([prepared.report, prepared.native_ground_truth]).encode())
        stored = replace(prepared, image=torch.empty(0), target=target)
        identity = TrainingSampleIdentity(raw.sample_token, raw.member_sha256)
        return codes, stored, packed, identity, size

    def prefetch(self, indices: list[int]) -> None:
        """Queue no more than two four-sample batches; retain main-thread CUDA ownership."""
        for index in indices:
            if not 0 <= index < self.capacity:
                raise IndexError(index)
            if index not in self.entries and index not in self._pending:
                if len(self._pending) >= 8:
                    raise ValueError("Training preparation queue exceeded its bound")
                self._pending[index] = self._executor.submit(self._prepare, index)

    def close(self) -> None:
        """Finish CPU work before the owning dataset closes its TAR handles."""
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._pending.clear()

    def get(self, index: int) -> tuple[PreparedSample, TrainingSampleIdentity]:
        """Return identical image/target tensors; decode raw media only on the first visit."""
        if not 0 <= index < self.capacity:
            raise IndexError(index)
        storage = (
            self.gpu_images[index]
            if index < self.gpu_count
            else self.cpu_images[index - self.gpu_count]
        )
        if index not in self.entries:
            self.prefetch([index])
            codes, stored, packed, identity, metadata_bytes = self._pending.pop(index).result()
            if self.metadata_bytes + metadata_bytes > self.metadata_budget:
                raise ValueError("Compact target metadata exceeds declared budget")
            storage.copy_(codes)
            self.entries[index] = stored, packed, identity
            self.metadata_bytes += metadata_bytes
            self.misses += 1
        else:
            self.hits += 1
        stored, packed, identity = self.entries[index]
        region = torch.from_numpy(
            np.unpackbits(packed, count=self.height * self.width).reshape(self.height, self.width)
        ).float()
        prepared = replace(
            stored,
            image=decode_image(storage, self.table),
            target=stored.target | {"obj_region": region},
        )
        return prepared, identity

    def statistics(self) -> dict[str, Any]:
        """Report allocated compact slabs and populated training entries."""
        return {
            "schema": "phase_lossless_compact_training_cache_v1",
            "device": self.device,
            "capacity": self.capacity,
            "resident_samples": len(self.entries),
            "gpu_image_bytes": self.gpu_images.numel(),
            "cpu_image_bytes": self.cpu_images.numel(),
            "target_metadata_payload_bytes": self.metadata_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "raw_images_retained": 0,
            "pending_cpu_preparations": len(self._pending),
        }
