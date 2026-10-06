"""Verify real TRAIN40 input parity on CPU without touching optimizer or CUDA state."""

from __future__ import annotations

import argparse
import gc
import hashlib
from dataclasses import fields
from pathlib import Path

import torch

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from operational.efficient_context.common import Lease, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.engine_overlap import SealedInputs


def batch_identity(batch: ObjectEventV4Batch) -> dict:
    """Hash exact typed tensor bytes and retain every non-tensor batch field."""
    identity = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError("Input admission must remain on CPU")
            array = value.contiguous().numpy()
            identity[field.name] = {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        else:
            identity[field.name] = value
    return identity


def run(output: Path) -> None:
    """Compare full and tail batches against the frozen collator after a safe pause."""
    from operational.train40_system.adaptive_cache import AdaptiveSealedInputs

    verify_sources(read(output / "COORDINATION_FREEZE.json"))
    if not (output / "COORDINATION_PAUSE_REQUEST.json").is_file():
        raise ValueError("Admission requires a recoverable trainer pause")
    checkpoint = output / "fits/a5_seed7/checkpoint_last.pt"
    receipt = read(checkpoint.parent / "CHECKPOINT_RECEIPT.json")
    checkpoint_sha = digest(checkpoint)
    if receipt["status"] != "PAUSED_COORDINATION" or receipt["sha256"] != checkpoint_sha:
        raise ValueError("Verified paused checkpoint required")
    if torch.cuda.is_initialized():
        raise ValueError("CPU-only admission unexpectedly initialized CUDA")
    torch.set_num_threads(1)
    rng = torch.get_rng_state().clone()
    cases = [list(range(3, 227, 7)), list(range(88736, 88744))]
    controls = []
    for ids in cases:
        source = SealedInputs(output)
        try:
            controls.append(batch_identity(source.batch(ids)))
        finally:
            source.close()
            del source
            gc.collect()
    candidate = AdaptiveSealedInputs(output)
    try:
        for ids, expected in zip(cases, controls, strict=True):
            for _ in range(2):
                actual = batch_identity(candidate.batch(ids))
                if actual != expected:
                    raise ValueError("Adaptive cache changed real batch tensor bytes or metadata")
        counters = candidate.snapshot()
    finally:
        candidate.close()
    if not torch.equal(rng, torch.get_rng_state()) or torch.cuda.is_initialized():
        raise ValueError("CPU input caching changed RNG or initialized CUDA")
    if digest(checkpoint) != checkpoint_sha:
        raise ValueError("Input admission changed the training checkpoint")
    atomic_json(
        output / "REAL_ADAPTIVE_CACHE_ADMISSION.json",
        {
            "status": "PASSED",
            "source_sha256": digest(Path(__file__)),
            "cache_source_sha256": digest(Path(__file__).with_name("adaptive_cache.py")),
            "coordination_freeze_sha256": digest(output / "COORDINATION_FREEZE.json"),
            "input_manifest_sha256": digest(output / "INPUT_MANIFEST.json"),
            "teacher_manifest_sha256": digest(output / "TEACHER_MANIFEST.json"),
            "checkpoint_sha256": checkpoint_sha,
            "optimizer_updates": 0,
            "training_checkpoint_modified": False,
            "cuda_initialized": False,
            "exact_all_tensor_bytes_and_metadata": True,
            "CPU_RNG_unchanged": True,
            "batch_sizes": [len(ids) for ids in cases],
            "ordinals": cases,
            "batch_identities": controls,
            "candidate_counters": counters,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with Lease(args.output.resolve()):
        run(args.output.resolve())
