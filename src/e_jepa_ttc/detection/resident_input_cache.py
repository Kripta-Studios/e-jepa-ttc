"""Bounded exact prepared-image residency, with unchanged targets and RNG ordering."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Protocol

from e_jepa_ttc.detection.eap_rgb_data import RawRGBSample


class SampleReader(Protocol):
    """Minimal synchronous raw-image reader."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> RawRGBSample: ...


class ResidentInputCache:
    """Cache deterministic preparation; move only training images to the chosen device.

    Loading is synchronous: the caller retains sole sampler and CUDA ownership.
    Evaluation always keeps its annotation-empty prepared input on CPU. Raw samples,
    geometry, targets, and their original order are retained without transformation.
    """

    def __init__(
        self,
        dataset: SampleReader,
        config: object,
        member_sha256: dict[str, str],
        *,
        prepare: Callable[[Any, bool], Any],
        label_free_model_input: bool = False,
        train_capacity: int = 4096,
        evaluation_capacity: int = 8,
        device: str = "cuda",
        maximum_image_bytes: int = 13 * 2**30,
    ) -> None:
        if min(train_capacity, evaluation_capacity, maximum_image_bytes) <= 0:
            raise ValueError("Residency capacities must be positive")
        self.dataset = dataset
        self.config = config
        self.member_sha256 = member_sha256
        self.label_free_model_input = label_free_model_input
        self.prepare = prepare
        self.device = "cpu" if label_free_model_input else device
        self.capacity = min(
            len(dataset), evaluation_capacity if label_free_model_input else train_capacity
        )
        self.maximum_image_bytes = maximum_image_bytes
        self.image_bytes = 0
        self.peak_image_bytes = 0
        self.cache: OrderedDict[int, tuple[Any, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, index: int) -> tuple[Any, Any]:
        """Validate consumed image identity before preparation or device transfer."""
        if index in self.cache:
            self.hits += 1
            self.cache.move_to_end(index)
            return self.cache[index]
        raw = self.dataset[index]
        expected = self.member_sha256.get(raw.sample_token)
        if expected is not None and expected != raw.member_sha256:
            raise ValueError(f"RGB member bytes changed: {raw.sample_token}")
        prepared = self.prepare(raw, self.label_free_model_input)
        size = prepared.image.numel() * prepared.image.element_size()
        if size > self.maximum_image_bytes:
            raise ValueError("A prepared image exceeds the residency budget")
        if self.capacity < 1:
            raise ValueError("Cannot cache an empty dataset")
        while self.cache and (
            len(self.cache) >= self.capacity or self.image_bytes + size > self.maximum_image_bytes
        ):
            evicted, _ = self.cache.popitem(last=False)[1]
            self.image_bytes -= evicted.image.numel() * evicted.image.element_size()
        prepared = replace(prepared, image=prepared.image.to(self.device))
        self.member_sha256[raw.sample_token] = raw.member_sha256
        self.cache[index] = prepared, raw
        self.image_bytes += size
        self.peak_image_bytes = max(self.peak_image_bytes, self.image_bytes)
        self.misses += 1
        return prepared, raw

    def statistics(self) -> dict[str, Any]:
        """Return operational residency evidence, separately from scientific settings."""
        return {
            "device": self.device,
            "capacity": self.capacity,
            "resident_samples": len(self.cache),
            "image_bytes": self.image_bytes,
            "peak_image_bytes": self.peak_image_bytes,
            "maximum_image_bytes": self.maximum_image_bytes,
            "hits": self.hits,
            "misses": self.misses,
        }
