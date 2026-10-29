"""Verify CPU worker parity against existing native TRAIN tensors before any runtime amendment."""

from __future__ import annotations

import hashlib
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import cast

from .common import ROOT, Campaign, atomic_json, digest, read
from .compressed_cache import CompressedInputCache
from .garl_train import records
from .parallel_inputs import prepare_record, worker_init


def run(c: Campaign) -> dict:
    """Keep source arithmetic, dtypes, layouts, targets and parent RNG exactly unchanged."""
    import numpy as np
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    c.freeze()
    c.require_resources()
    proto = c.out / "garl/PROTOCOL.json"
    for pin in read(proto)["science_files"]:
        if digest(Path(pin["path"])) != pin["sha256"]:
            raise ValueError("native scientific sources changed before CPU worker QA")
    rows = records(c)
    native_binding = digest(proto)
    keymap = {}
    stats = {}
    for token, row in rows.items():
        sequence = row["sequence_id"]
        if sequence not in stats:
            s = (c.raw / sequence / "events.h5").stat()
            stats[sequence] = s.st_size, s.st_mtime_ns
        size, mtime = stats[sequence]
        key = hashlib.sha256(f"{native_binding}:{token}:{size}:{mtime}".encode()).hexdigest()
        keymap[key + ".npz"] = token
    chosen = []
    for source in sorted((c.out / "garl/native_cache").glob("*.npz")):
        try:
            payload = source.read_bytes()
        except FileNotFoundError:
            continue
        token = keymap[source.name]
        chosen.append(
            (token, CompressedInputCache.unpack(payload), hashlib.sha256(payload).hexdigest())
        )
        if len(chosen) == 32:
            break
    if len(chosen) != 32:
        raise ValueError("32 existing TRAIN tensors required for worker parity")
    torch_rng = torch.get_rng_state().clone()
    numpy_rng, python_rng = cast(tuple, np.random.get_state()), random.getstate()
    if not isinstance(numpy_rng, tuple):
        raise TypeError("legacy NumPy RNG state required")
    cases = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4, initializer=worker_init) as executor:
        futures = [executor.submit(prepare_record, rows[t], str(c.raw)) for t, _, _ in chosen]
        for (token, expected, sha), future in zip(chosen, futures, strict=True):
            x, visible, target = future.result()
            exact = np.array_equal(x, expected[0].numpy()) and np.array_equal(
                visible, expected[1].numpy()
            )
            exact = exact and target == expected[2]
            layout = x.flags.c_contiguous and visible.flags.c_contiguous
            if not exact or not layout or x.dtype != np.float32 or visible.dtype != np.float32:
                raise ValueError("parallel CPU preparation changed a native TRAIN input")
            c.require_resources()
            cases.append({"token": token, "reference_sha256": sha, "all_fields_exact": True})
    nr = cast(tuple, np.random.get_state())
    rng_unchanged = (
        torch.equal(torch_rng, torch.get_rng_state())
        and python_rng == random.getstate()
        and isinstance(nr, tuple)
        and numpy_rng[0] == nr[0]
        and np.array_equal(numpy_rng[1], nr[1])
        and numpy_rng[2:] == nr[2:]
    )
    if not rng_unchanged:
        raise ValueError("CPU worker preparation consumed a parent training RNG")
    result = {
        "status": "PASSED",
        "native_protocol_sha256": digest(proto),
        "resource_authorization_sha256": digest(c.out / "RESOURCE_AUTHORIZATION_V2.json"),
        "implementation_sha256": digest(Path(__file__).with_name("parallel_inputs.py")),
        "QA_implementation_sha256": digest(Path(__file__)),
        "cases": cases,
        "selection": "first32 readable existing TRAIN cache keys in SHA order; no GT selection",
        "CPU_workers": 4,
        "CPU_threads_per_worker": 4,
        "CPU_interop_threads_per_worker": 2,
        "original_FP32_arrays_targets_and_layouts_exact": True,
        "parent_Torch_NumPy_Python_RNG_unchanged": True,
        "raw_encodings_for_QA": len(cases),
        "optimizer_updates": 0,
        "models_or_CUDA_in_workers": False,
        "elapsed_seconds_including_spawn_and_contended_raw": time.perf_counter() - started,
        "production_inference_latency_claimed": False,
    }
    atomic_json(c.out / "garl/PARALLEL_INPUT_QA.json", result)
    return result


if __name__ == "__main__":
    print(run(Campaign(ROOT / "configs/campaign/efficient_context_v1.json"))["status"], flush=True)
