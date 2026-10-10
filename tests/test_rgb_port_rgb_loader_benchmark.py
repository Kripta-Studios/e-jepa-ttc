"""The CPU benchmark digest must detect scientific batch contract changes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from operational.rgb_port_rgb_loader.benchmark import fingerprint, rng_fingerprints


@dataclass
class Batch:
    values: torch.Tensor
    tokens: list[str]


def test_fingerprint_covers_values_shape_dtype_and_metadata() -> None:
    original = Batch(torch.tensor([[1.0, 2.0]]), ["sample-a"])
    digest = fingerprint(original)
    assert digest == fingerprint(Batch(original.values.clone(), ["sample-a"]))
    variants = [
        Batch(torch.tensor([[1.0, 3.0]]), ["sample-a"]),
        Batch(original.values.reshape(2, 1), ["sample-a"]),
        Batch(original.values.double(), ["sample-a"]),
        Batch(original.values.clone(), ["sample-b"]),
    ]
    assert all(fingerprint(variant) != digest for variant in variants)


def test_fingerprint_preserves_all_cpu_rng_states() -> None:
    batch = Batch(torch.tensor([1.0, 2.0]), ["sample-a", "sample-b"])
    before = rng_fingerprints()
    fingerprint([batch, batch])
    assert before == rng_fingerprints()
