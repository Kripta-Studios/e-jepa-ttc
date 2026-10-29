"""Fragment-resumable published Garl inference on TRAIN40, never a holdout evaluation."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch

from operational.efficient_context.common import ROOT, Lease, digest
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json, replace
from operational.train40_system.models import resource_guard

_pool: Any = None


class NativeModel(Protocol):
    """Typed boundary for the pinned upstream model, without replacing its implementation."""

    dT: float  # noqa: N815 - pinned upstream API

    def to(self, device: str) -> NativeModel: ...

    def forward_test(self, sensor: torch.Tensor) -> tuple[torch.Tensor, Any]: ...


def worker_init() -> None:
    """Retain bounded read-only HDF5 readers and one CPU thread per worker."""
    from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool

    global _pool
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _pool = ReaderPool()


def sensor_batch(task: dict) -> np.ndarray:
    """Encode exactly the native40-plane two-endpoint input, with no target fields."""
    from e_jepa_ttc.efficient_context.garl_input import inference_record

    root = Path(task["raw_root"])
    tensors = []
    for row in task["rows"]:
        raw = root / "data/train" / row["sequence_id"] / "events.h5"
        stat = raw.stat()
        pin = task["pins"][row["sequence_id"]]
        if (stat.st_size, stat.st_mtime_ns) != (pin["bytes"], pin["mtime_ns"]):
            raise ValueError("Published Garl raw TRAIN source changed")
        tensors.append(inference_record(row, _pool, root / "data/train").numpy())
    return np.stack(tensors)


def prepare_jobs(output: Path, raw_root: Path) -> list[dict]:
    """Select the already audited sensor endpoints without reading TTC or 3D labels."""
    import pandas as pd

    audit = read(output / "DATA_AUDIT.json")
    for name, key in (
        ("TRAIN40_INDEX.npz", "index_sha256"),
        ("TRAIN40_ROWS.parquet", "rows_sha256"),
    ):
        if digest(output / name) != audit[key]:
            raise ValueError("Canonical TRAIN40 identities changed")
    with np.load(output / "TRAIN40_INDEX.npz", allow_pickle=False) as stored:
        sequences, windows, endpoints = (
            stored["sequences"],
            stored["windows_us"],
            stored["endpoint_indices"],
        )
    boxes = pd.read_parquet(output / "TRAIN40_ROWS.parquet", columns=["boxes_xyxy"])
    pins = {}
    for sequence in audit["sequences"]:
        path = output / "raw_receipts" / (sequence + ".json")
        if not path.exists():
            path = (
                ROOT / f"artifacts/efficient_context_20261004/data_recovery/files/{sequence}.json"
            )
        pin = read(path)
        if pin["status"] != "VERIFIED":
            raise ValueError("Verified raw TRAIN source required")
        pins[sequence] = pin
    rows = [
        {
            "sequence_id": str(sequences[row]),
            "event_windows_us": windows[row, 1:].tolist(),
            "boxes_xyxy": np.asarray(boxes.iloc[row, 0].tolist())[endpoints[row]].tolist(),
        }
        for row in range(88744)
    ]
    return [
        {"start": start, "rows": rows[start : start + 8], "pins": pins, "raw_root": str(raw_root)}
        for start in range(0, 88744, 8)
    ]


def native_model(output: Path, code_root: Path) -> NativeModel:
    """Strictly load every published parameter; no initialization checkpoint side effects."""
    import yaml

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code_root))
    configuration = yaml.safe_load(
        (output / "public_garl/configs/ablation/event_lhr.yaml").read_text(encoding="utf-8")
    )
    for name in ("pretrained_ckpt_event", "pretrained_ckpt_rgb"):
        configuration["model"].pop(name, None)
    constructor = importlib.import_module("garl_ttc.models.ttc_network").TTCNetwork
    model = constructor(configuration, is_train=False).float().eval().requires_grad_(False)
    state = torch.load(
        output / "public_garl/paper_event_only_lhr.pth", weights_only=True, map_location="cpu"
    )
    model.load_state_dict(state, strict=True)
    return cast(NativeModel, model)


def release_ttc(heights: np.ndarray, delta: float) -> np.ndarray:
    """Preserve the release conversion without clipping or deleting failed predictions."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return delta / (1.0 - heights[:, 0] / heights[:, 1])


def run(output: Path, raw_root: Path, code_root: Path) -> None:
    """Finish or recover all native predictions using four bounded preparation workers."""
    freeze_path = output / "DELIVERY_FREEZE.json"
    freeze = read(freeze_path)
    verify_sources(freeze)
    for item in freeze["native_source_files"]:
        if digest(code_root / item["path"]) != item["sha256"]:
            raise ValueError("Pinned native Garl inference source changed")
    checkpoint = output / "public_garl/paper_event_only_lhr.pth"
    if digest(checkpoint) != freeze["public_checkpoint_sha256"]:
        raise ValueError("Published comparator checkpoint changed")
    if digest(output / "public_garl/configs/ablation/event_lhr.yaml") != freeze["config_sha256"]:
        raise ValueError("Published comparator configuration changed")
    if read(output / "TRAINING_QUEUE_COMPLETION.json")["status"] != "COMPLETE":
        raise ValueError("Primary local training queue has priority")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = native_model(output, code_root)
    jobs = prepare_jobs(output, raw_root)
    directory = output / "garl_train_predictions"
    directory.mkdir(exist_ok=True)
    binding = {
        "delivery_freeze_sha256": digest(freeze_path),
        "checkpoint_sha256": freeze["public_checkpoint_sha256"],
        "index_sha256": digest(output / "TRAIN40_INDEX.npz"),
        "role": "TRAIN_FIT_DIAGNOSTIC_NOT_GENERALIZATION",
        "actual_checkpoint_training_sequence_manifest_verified": False,
        "native_dT_seconds": float(model.dT),
        "input": "native40_float32_sensor_ROI_only",
        "batch_size": 8,
        "no_prediction_clipping_or_query_dropping": True,
    }
    binding_path = directory / "BINDING.json"
    if binding_path.exists() and read(binding_path) != binding:
        raise ValueError("Preserve saved native prediction lineage")
    atomic_json(binding_path, binding)
    binding_sha = digest(binding_path)
    predicted = np.empty(88744, np.float32)
    missing = []
    for job in jobs:
        start, count = job["start"], len(job["rows"])
        path = directory / f"batch_{start:06d}.npz"
        receipt_path = path.with_suffix(".json")
        if receipt_path.exists():
            receipt = read(receipt_path)
            if digest(path) != receipt["sha256"] or receipt["binding_sha256"] != binding_sha:
                raise ValueError("Saved native prediction fragment changed")
            with np.load(path, allow_pickle=False) as stored:
                if not np.array_equal(stored["ordinals"], np.arange(start, start + count)):
                    raise ValueError("Native prediction fragment identity changed")
                predicted[start : start + count] = stored["ttc"]
        else:
            missing.append(job)
    model = model.to("cuda")
    parity = output / "PUBLIC_GARL_CPU_GPU_PARITY.json"
    complete = 88744 - sum(len(job["rows"]) for job in missing)
    with ProcessPoolExecutor(max_workers=4, initializer=worker_init) as pool:
        pending, iterator = {}, iter(missing)
        with torch.inference_mode():
            while True:
                allowed, resource = resource_guard(output)
                if not allowed:
                    atomic_json(output / "PUBLIC_GARL_PREDICTION_PAUSE.json", resource)
                    return
                while len(pending) < 4:
                    job = next(iterator, None)
                    if job is None:
                        break
                    pending[job["start"]] = (job, pool.submit(sensor_batch, job))
                if not pending:
                    break
                start = min(pending)
                job, future = pending.pop(start)
                begun = time.perf_counter()
                sensor = torch.from_numpy(future.result())
                torch.cuda.synchronize()
                compute_start = time.perf_counter()
                heights, _ = model.forward_test(sensor.to("cuda"))
                heights = heights.float().cpu().numpy()
                torch.cuda.synchronize()
                compute_seconds = time.perf_counter() - compute_start
                if not parity.exists():
                    cpu_model = native_model(output, code_root)
                    cpu_heights, _ = cpu_model.forward_test(sensor)
                    reference = cpu_heights.numpy()
                    passed = np.allclose(heights, reference, rtol=1e-4, atol=1e-4)
                    atomic_json(
                        parity,
                        {
                            "status": "PASSED" if passed else "FAILED",
                            "TRAIN_ordinal_start": start,
                            "population": len(sensor),
                            "max_absolute_height_difference": float(
                                np.abs(heights - reference).max()
                            ),
                            "rtol": 1e-4,
                            "atol": 1e-4,
                            "optimizer_updates": 0,
                            "delivery_freeze_sha256": digest(freeze_path),
                        },
                    )
                    del cpu_model
                    if not passed:
                        raise ValueError("Published native FP32 CPU/GPU parity failed")
                else:
                    receipt = read(parity)
                    if receipt["status"] != "PASSED" or receipt["delivery_freeze_sha256"] != digest(
                        freeze_path
                    ):
                        raise ValueError("Published native FP32 CPU/GPU parity is not admitted")
                count = len(job["rows"])
                values = release_ttc(heights, float(model.dT))
                path = directory / f"batch_{start:06d}.npz"
                temporary = path.with_suffix(".pending.npz")
                np.savez_compressed(
                    temporary, heights=heights, ttc=values, ordinals=np.arange(start, start + count)
                )
                with temporary.open("rb+") as handle:
                    os.fsync(handle.fileno())
                replace(temporary, path)
                predicted[start : start + count] = values
                complete += count
                atomic_json(
                    path.with_suffix(".json"),
                    {
                        "sha256": digest(path),
                        "binding_sha256": binding_sha,
                        "rows": count,
                        "seconds": time.perf_counter() - begun,
                        "forward_and_transfer_seconds": compute_seconds,
                        "optimizer_updates": 0,
                    },
                )
                atomic_json(
                    output / "PUBLIC_GARL_PREDICTION_PROGRESS.json",
                    {
                        "status": "RUNNING",
                        "completed_rows": complete,
                        "total_rows": 88744,
                        "optimizer_updates": 0,
                    },
                )
    destination = output / "PUBLIC_GARL_TRAIN_PREDICTIONS.npz"
    temporary = destination.with_suffix(".pending.npz")
    np.savez_compressed(temporary, ttc=predicted, ordinals=np.arange(88744))
    replace(temporary, destination)
    atomic_json(
        output / "PUBLIC_GARL_PREDICTION_MANIFEST.json",
        {
            "status": "COMPLETE_VERIFIED",
            "row_count": 88744,
            "files": [{"path": destination.name, "sha256": digest(destination)}],
            "binding": binding,
            "optimizer_updates": 0,
            "not_a_reproduction_of_published_test_results": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    args = parser.parse_args()
    with Lease(args.output.resolve()):
        run(args.output.resolve(), args.raw_root.resolve(), args.code_root.resolve())
