"""CPU-only real TRAIN40 batch parity across group boundaries and resumed cursors."""

from __future__ import annotations

import argparse
from pathlib import Path


def run(output: Path) -> dict:
    """Compare every tensor byte and metadata field without an optimizer or CUDA."""
    import hashlib
    import random
    import time
    from dataclasses import fields

    import numpy as np
    import psutil
    import torch

    from operational.efficient_context.common import ROOT, digest
    from operational.train40_system.checkpoint import payload_digest
    from operational.train40_system.contracts import read
    from operational.train40_system.durable_io import atomic_json
    from operational.train40_system.engine_overlap import SealedInputs
    from operational.train40_system.freeze import dependency_files
    from operational.train40_system.models import epoch_order
    from operational.train40_system.process_inputs_fast_2 import ProcessInputs

    lock = output / "WRITER.lock"
    if lock.exists():
        lease = read(lock)
        try:
            owner = psutil.Process(lease["pid"])
            if abs(owner.create_time() - lease["create_time"]) < 0.001:
                raise RuntimeError("Pause the authenticated trainer before full CPU admission")
        except psutil.NoSuchProcess:
            pass
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU parity must run without a CUDA context")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    protected = [
        output / "fits" / f"{arm}_seed7" / name
        for arm in ("a5", "c2f")
        for name in ("checkpoint_last.pt", "CHECKPOINT_RECEIPT.json", "UPDATE_JOURNAL.json")
    ]
    before_files = {str(path): digest(path) for path in protected if path.exists()}
    generator = torch.Generator().manual_seed(20261006)
    order = epoch_order(88744, generator)
    before_rng = {
        "torch": torch.get_rng_state().clone(),
        "numpy": payload_digest(np.random.get_state()),
        "python": payload_digest(random.getstate()),
        "sampler": generator.get_state().clone(),
    }
    rows = []
    reference = SealedInputs(output)
    candidate = ProcessInputs(output)
    try:
        # Warmup is permitted before the trainer supplies any sampler hints.
        cases: list[tuple[str, list[int], int | None]] = [
            ("warmup_without_order", order[:32].tolist(), None)
        ]
        cases.extend(
            ("first_two_groups", order[start : start + 32].tolist(), start)
            for start in range(0, 512, 32)
        )
        short_group_start = next(i for i, ordinal in enumerate(order.tolist()) if ordinal >= 88576)
        cases.extend(
            ("short_group_boundary", order[start : start + 32].tolist(), start)
            for start in range(
                max(0, short_group_start - 32), min(88744, short_group_start + 224), 32
            )
        )
        cases.extend(
            ("epoch_tail", order[start : start + 32].tolist(), start)
            for start in range(88672, 88744, 32)
        )
        for kind, ids, position in cases:
            if position is not None:
                candidate.prepare_order(order, position, 32)
            begun = time.perf_counter()
            expected = reference.batch(ids)
            reference_seconds = time.perf_counter() - begun
            begun = time.perf_counter()
            actual = candidate.batch(ids)
            candidate_seconds = time.perf_counter() - begun
            hashes = {}
            for field in fields(expected):
                left, right = getattr(expected, field.name), getattr(actual, field.name)
                if isinstance(left, torch.Tensor):
                    if not isinstance(right, torch.Tensor) or left.dtype != right.dtype:
                        raise ValueError(f"Dtype parity failure: {field.name}")
                    if left.shape != right.shape or not torch.equal(left, right):
                        raise ValueError(f"Tensor parity failure: {field.name}")
                    left_bytes = left.contiguous().view(torch.uint8).numpy().tobytes()
                    right_bytes = right.contiguous().view(torch.uint8).numpy().tobytes()
                    if left_bytes != right_bytes:
                        raise ValueError(f"Byte parity failure: {field.name}")
                    hashes[field.name] = hashlib.sha256(left_bytes).hexdigest()
                elif left != right:
                    raise ValueError(f"Metadata parity failure: {field.name}")
            rows.append(
                {
                    "case": kind,
                    "position": position,
                    "ids": ids,
                    "tensor_sha256": hashes,
                    "reference_seconds": reference_seconds,
                    "candidate_seconds": candidate_seconds,
                }
            )
            del expected, actual
        candidate.close()
        # Recreate a worker at a nonzero cursor, as a resumed trainer does.
        candidate = ProcessInputs(output)
        candidate.prepare_order(order, 288, 32)
        expected, actual = (
            reference.batch(order[288:320].tolist()),
            candidate.batch(order[288:320].tolist()),
        )
        for field in fields(expected):
            left, right = getattr(expected, field.name), getattr(actual, field.name)
            if isinstance(left, torch.Tensor):
                if (
                    left.dtype != right.dtype
                    or left.shape != right.shape
                    or not torch.equal(left, right)
                ):
                    raise ValueError(f"Resumed batch parity failure: {field.name}")
            elif left != right:
                raise ValueError(f"Resumed metadata parity failure: {field.name}")
    finally:
        reference.close()
        candidate.close()
    rng_equal = (
        torch.equal(torch.get_rng_state(), before_rng["torch"])
        and payload_digest(np.random.get_state()) == before_rng["numpy"]
        and payload_digest(random.getstate()) == before_rng["python"]
        and torch.equal(generator.get_state(), before_rng["sampler"])
    )
    if not rng_equal:
        raise ValueError("CPU batch preparation changed the parent's random state")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU admission unexpectedly initialized CUDA")
    if any(digest(Path(path)) != value for path, value in before_files.items()):
        raise ValueError("CPU admission changed protected training state")
    result = {
        "schema": "train40_host_fast_2_real_parity_v1",
        "status": "PASSED",
        "all_batch_tensors_and_metadata_exact": True,
        "parent_rng_unchanged": rng_equal,
        "CUDA_initialized": False,
        "additional_optimizer_updates": 0,
        "cases": rows,
        "recreated_worker_at_nonzero_cursor_exact": True,
        "protected_state_sha256": before_files,
        "input_manifest_sha256": digest(output / "INPUT_MANIFEST.json"),
        "teacher_manifest_sha256": digest(output / "TEACHER_MANIFEST.json"),
        "files": dependency_files(
            [ROOT / "operational/train40_system/process_inputs_fast_2.py", Path(__file__)]
        ),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "timing_caveat": (
            "Sequential CPU parity checks, not a throughput benchmark or causal comparison."
        ),
    }
    path = output / "HOST_FAST_2_REAL_PARITY.json"
    if path.exists():
        raise FileExistsError("Preserve previous real parity admission")
    atomic_json(path, result)
    return result


if __name__ == "__main__":
    from operational.train40_system.data_audit import OUTPUT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(args.output.resolve())
    print(
        {
            key: value
            for key, value in result.items()
            if key not in {"cases", "protected_state_sha256"}
        }
    )
