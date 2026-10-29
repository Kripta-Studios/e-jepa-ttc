"""Read-only parity on64 original cached TRAIN tensors; no raw events or optimizer updates."""

from __future__ import annotations

import hashlib
import os
import random
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

from .common import ROOT, Campaign, atomic_json, digest


def main() -> int:
    """Consume actual signed native cache bytes in order and compare original FP32 outputs."""
    import numpy as np
    import torch

    from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool

    from .compressed_cache import CompressedInputCache
    from .garl_train import records
    from .parallel_decode import DecodingInputCache

    c = Campaign(ROOT / "configs/campaign/efficient_context_v1.json")
    c.freeze()
    rows = records(c)
    binding = digest(c.out / "garl/PROTOCOL.json")
    stats = {}
    keys = {}
    for token, row in rows.items():
        sequence = row["sequence_id"]
        if sequence not in stats:
            stat = (c.raw / sequence / "events.h5").stat()
            stats[sequence] = stat.st_size, stat.st_mtime_ns
        size, mtime = stats[sequence]
        keys[hashlib.sha256(f"{binding}:{token}:{size}:{mtime}".encode()).hexdigest() + ".npz"] = (
            token
        )
    root = c.out / "garl/native_cache"
    copied = OrderedDict()
    with os.scandir(root) as entries:
        for entry in entries:
            token = keys.get(entry.name)
            if token is None:
                continue
            try:
                copied[token] = Path(entry.path).read_bytes()
            except FileNotFoundError:
                continue
            if len(copied) == 64:
                break
    if len(copied) != 64:
        raise ValueError("decoding admission requires64 existing original TRAIN cache records")
    cache = DecodingInputCache.__new__(DecodingInputCache)
    cache.c = c
    cache.rows = rows
    cache.root = root
    cache.token_keys = {token: key for key, token in keys.items()}
    cache.key_tokens = keys
    cache.sequence_stats = stats
    cache.values = copied
    cache.files = OrderedDict()
    cache.bytes = sum(map(len, copied.values()))
    cache.disk = 0
    cache.limit = 32_000_000_000
    cache.memory_limit = 8 * 1024**3
    cache.pool = ReaderPool()
    cache.executor = None
    cache.active = False
    cache.upcoming = iter(())
    cache.reads = cache.hits = cache.decoded_hits = 0
    cache.decode_executor = ThreadPoolExecutor(max_workers=4)
    cache.decoding = {}
    tokens = list(copied)
    order = tokens + tokens[::8]  # Duplicate lookahead also remains strictly sampler-ordered.
    cache.pending = deque((token, None) for token in order)
    torch_rng = torch.get_rng_state()
    numpy_rng = cast(tuple, np.random.get_state())
    python_rng = random.getstate()
    cache.fill()
    receipts = []
    try:
        for token in order:
            payload = copied[token]
            reference = CompressedInputCache.unpack(payload)
            actual = cache.get(token)
            if (
                not torch.equal(reference[0], actual[0])
                or not torch.equal(reference[1], actual[1])
                or reference[2] != actual[2]
            ):
                raise ValueError("parallel decoding differs from the frozen original NPZ decoder")
            receipts.append(
                {"sample_token": token, "original_NPZ_sha256": hashlib.sha256(payload).hexdigest()}
            )
        after = cast(tuple, np.random.get_state())
        if (
            not torch.equal(torch_rng, torch.get_rng_state())
            or numpy_rng[0] != after[0]
            or not np.array_equal(numpy_rng[1], after[1])
            or numpy_rng[2:] != after[2:]
            or python_rng != random.getstate()
        ):
            raise ValueError("parallel cache decoding consumed parent scientific RNG")
        atomic_json(
            c.out / "garl/DECODE_INPUT_QA.json",
            {
                "status": "PASSED",
                "native_protocol_sha256": binding,
                "implementation_sha256": digest(Path(__file__).with_name("parallel_decode.py")),
                "QA_implementation_sha256": digest(Path(__file__)),
                "samples": receipts,
                "unique_original_TRAIN_records": 64,
                "ordered_checks_including_duplicates": len(order),
                "exact_tensor_target_and_parent_RNG_parity": True,
                "cache_decoding_threads": 4,
                "raw_event_reads": 0,
                "optimizer_updates": 0,
                "neural_models_constructed": 0,
                "live_cache_or_trainer_modified": False,
            },
        )
        print("DECODE_INPUT_QA_PASSED", len(order), flush=True)
    finally:
        cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
