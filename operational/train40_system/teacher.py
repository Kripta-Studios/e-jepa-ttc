"""Resumable TRAIN40 DINO supervision using the original frozen RGB teacher."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image

from operational.efficient_context.common import Lease, atomic_json, digest
from operational.train40_system.data_audit import OUTPUT, safe_media_path

WEIGHT_SHA = "96ad081d5863ba14707d2e3bb734b7fa8dfbf3248cb67ddfcc74d86dd160d628"
MODEL_REVISION = "e959efa74c867491dcfe3ec3e4f97382e39025b3"
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
_tars: OrderedDict[Path, tuple[tarfile.TarFile, Lock]] = OrderedDict()
_lock = Lock()


def rgb_crop(path: Path, member: str, square: np.ndarray) -> tuple[np.ndarray, str]:
    """Reuse bounded TAR indexes while preserving the historical crop byte for byte."""
    with _lock:
        if path not in _tars:
            while len(_tars) >= 4:
                # All references to an evicted file must finish before closure.
                _, (old, mutex) = _tars.popitem(last=False)
                with mutex:
                    old.close()
            opened = tarfile.open(path, "r")
            opened.getmembers()
            _tars[path] = (opened, Lock())
        tar, mutex = _tars[path]
        _tars.move_to_end(path)
        # Reserve before releasing the cache lock to prevent concurrent eviction.
        mutex.acquire()
    try:
        file = tar.extractfile(member)
        if file is None:
            raise FileNotFoundError(member)
        with file:
            raw = file.read()
    finally:
        mutex.release()
    with Image.open(io.BytesIO(raw)) as original:
        image = original.convert("RGB")
    x1, y1 = max(0, int(square[0])), max(0, int(square[1]))
    x2, y2 = min(image.width, int(square[2])), min(image.height, int(square[3]))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid original-coordinate teacher crop")
    image = image.crop((x1, y1, x2, y2)).resize((256, 256), Image.Resampling.BILINEAR)
    value = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (value - MEAN) / STD, hashlib.sha256(raw).hexdigest()


def run(output: Path, raw_root: Path, model_path: Path) -> None:
    """Generate only available verified TRAIN RGB fragments, never reading TTC arrays."""
    import psutil
    import torch
    from transformers import AutoModel

    from e_jepa_ttc.distillation.dinov3_relational import local_cosine_relation_maps
    from operational.train40_system.models import resource_guard

    freeze = json.loads((output / "TEACHER_FREEZE.json").read_text(encoding="utf-8"))
    if freeze["source_sha256"] != digest(Path(__file__)):
        raise ValueError("Teacher source differs from QA freeze")
    audit = json.loads((output / "DATA_AUDIT.json").read_text(encoding="utf-8"))
    if digest(output / "TRAIN40_INDEX.npz") != audit["index_sha256"]:
        raise ValueError("TRAIN40 RGB endpoint contract changed")
    # Load only event/RGB metadata, keeping targets outside this materializer.
    names = ("tokens", "sequences", "rgb_shards", "rgb_members", "square_xyxy")
    with np.load(output / "TRAIN40_INDEX.npz", allow_pickle=False) as stored:
        index = {name: stored[name] for name in names}
    receipts = output / "rgb_receipts"
    directory = output / "teacher"
    directory.mkdir(exist_ok=True)
    eligible = []
    completed = 0
    for start in range(0, len(index["tokens"]), 32):
        stop = min(start + 32, len(index["tokens"]))
        shard = directory / f"shard_{start // 32:05d}.npz"
        receipt_path = shard.with_suffix(".json")
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["freeze_sha256"] != digest(output / "TEACHER_FREEZE.json"):
                raise ValueError("Teacher fragment recipe changed")
            if not shard.is_file() or digest(shard) != receipt["sha256"]:
                raise ValueError("Completed teacher fragment changed")
            completed += stop - start
            continue
        paths = set(index["rgb_shards"][start:stop].flatten().tolist())
        available = True
        for relative in paths:
            sequence = relative.split("/")[2]
            path = safe_media_path(relative, sequence, raw_root)
            receipt_path = receipts / (hashlib.sha256(relative.encode()).hexdigest() + ".json")
            if not path.is_file() or not receipt_path.is_file():
                available = False
                break
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                receipt["status"] != "VERIFIED"
                or path.stat().st_size != receipt["bytes"]
                or path.stat().st_mtime_ns != receipt["mtime_ns"]
            ):
                raise ValueError("Verified RGB receipt is stale")
        if available:
            eligible.append((start, stop, shard))
    if not eligible:
        atomic_json(
            output / "TEACHER_PROGRESS.json",
            {
                "status": "COMPLETE" if completed == 88744 else "PARTIAL_PRESERVED",
                "completed_rows": completed,
                "total_rows": 88744,
                "optimizer_updates": 0,
            },
        )
        return
    allowed, resource = resource_guard(output)
    if not allowed:
        atomic_json(output / "TEACHER_RESOURCE_PAUSE.json", resource)
        return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = (
        AutoModel.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
        .eval()
        .requires_grad_(False)
    )
    weights = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        weights.update(name.encode())
        weights.update(tensor.cpu().numpy().tobytes())
    if weights.hexdigest() != WEIGHT_SHA:
        raise ValueError("DINO teacher does not match the original frozen weights")
    model = model.to("cuda")
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start, stop, shard in eligible:
            allowed, resource = resource_guard(output)
            if not allowed or psutil.virtual_memory().available < 4 * 1024**3:
                atomic_json(output / "TEACHER_RESOURCE_PAUSE.json", resource)
                break
            begun = time.perf_counter()
            values, validity, raw_hashes = [], [], []
            for first in range(start, stop, 8):
                last = min(first + 8, stop)
                futures = []
                for row in range(first, last):
                    for endpoint in range(2):
                        path = safe_media_path(
                            str(index["rgb_shards"][row, endpoint]),
                            str(index["sequences"][row]),
                            raw_root,
                        )
                        futures.append(
                            pool.submit(
                                rgb_crop,
                                path,
                                str(index["rgb_members"][row, endpoint]),
                                index["square_xyxy"][row].astype(np.float32),
                            )
                        )
                inputs = [future.result() for future in futures]
                pixels = torch.from_numpy(np.stack([x[0] for x in inputs])).to("cuda")
                with torch.inference_mode():
                    features = model(pixels, output_hidden_states=True).hidden_states[2]
                    if features.shape[1:] != (384, 32, 32):
                        raise ValueError("Teacher hidden-state selection drifted")
                    maps = local_cosine_relation_maps(features)
                values.append(
                    maps.values.cpu().numpy().astype(np.float16).reshape(last - first, 2, 6, 32, 32)
                )
                validity.append(maps.valid.cpu().numpy().reshape(last - first, 2, 6, 32, 32))
                raw_hashes.extend(x[1] for x in inputs)
            temporary = shard.with_suffix(".pending.npz")
            np.savez_compressed(
                temporary,
                ordinals=np.arange(start, stop),
                tokens=index["tokens"][start:stop],
                relation_targets=np.concatenate(values),
                relation_valid=np.concatenate(validity),
                rgb_sha256=np.asarray(raw_hashes).reshape(stop - start, 2),
            )
            os.replace(temporary, shard)
            atomic_json(
                shard.with_suffix(".json"),
                {
                    "status": "VERIFIED",
                    "sha256": digest(shard),
                    "rows": stop - start,
                    "freeze_sha256": digest(output / "TEACHER_FREEZE.json"),
                    "teacher_weights_sha256": WEIGHT_SHA,
                    "elapsed_seconds": time.perf_counter() - begun,
                    "optimizer_updates": 0,
                    "TTC_targets_read": False,
                },
            )
            completed += stop - start
            atomic_json(
                output / "TEACHER_PROGRESS.json",
                {
                    "status": "RUNNING",
                    "completed_rows": completed,
                    "total_rows": 88744,
                    "optimizer_updates": 0,
                    "TTC_targets_read": False,
                },
            )
    atomic_json(
        output / "TEACHER_PROGRESS.json",
        {
            "status": "COMPLETE" if completed == 88744 else "PARTIAL_PRESERVED",
            "completed_rows": completed,
            "total_rows": 88744,
            "optimizer_updates": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    with Lease(args.output.resolve()):
        run(args.output.resolve(), args.raw_root.resolve(), args.model_path.resolve())
