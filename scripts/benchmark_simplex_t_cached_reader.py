"""Compare real OLD input tensors exactly, CPU only, without expert forwards."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.data.eap import EAPEventReader
from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool
from e_jepa_ttc.simplex_t.context_raw_union import encode_context_union


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = json.loads(args.local_paths.read_text("utf-8"))
    work = Path(paths["worktree"])
    index_path = work / "artifacts/simplex_t/T1/query_context_index/query_context_index.npz"
    expected = "0fe7d7bb597dc768073a2940795442417ab6dba0be6110cd356d3ff0b9f656bf"
    if sha256(index_path) != expected:
        raise ValueError("OLD input index changed")
    with np.load(index_path, allow_pickle=False) as data:
        index = {
            key: data[key]
            for key in ("sequences", "base_windows_us", "lag_us", "valid", "square_xyxy")
        }
    prep_path = work.parent / (
        "e-jepa-ttc/artifacts/cache/garl_object_event_common_roi_train8192_v1/manifest.json"
    )
    if sha256(prep_path) != "063980fdae5fda0b2836befc662fdd1cd5659bf06f10d9760dfc0d566fac8e39":
        raise ValueError("preprocessing changed")
    prep = json.loads(prep_path.read_text("utf-8"))["config"]
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    records = []
    pool = ReaderPool()
    try:
        for query in (2522, 1343):
            sequence = str(index["sequences"][query])
            root = (Path(paths["eap_root"]) / "data/train").resolve(strict=True)
            path = (root / sequence / "events.h5").resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError("path escapes TRAIN")
            outputs = []
            elapsed = []
            for variant in ("historical", "cached"):
                started = time.perf_counter()
                reader = EAPEventReader(path) if variant == "historical" else pool.get(path)
                try:
                    value = encode_context_union(
                        reader,
                        index["base_windows_us"][query],
                        index["lag_us"],
                        index["valid"][query],
                        tuple(index["square_xyxy"][query]),
                        sequence_id=sequence,
                        roi_size=prep["roi_size"],
                        event_pixel_diff=prep["event_pixel_diff"],
                    )
                    outputs.append(value)
                finally:
                    if variant == "historical":
                        reader.close()
                elapsed.append(time.perf_counter() - started)
            equal = torch.equal(*outputs)
            record = {
                "query": query,
                "tensor_equal": equal,
                "tensor_sha256": [hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in outputs],
                "historical_seconds": elapsed[0],
                "cached_seconds": elapsed[1],
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            if not equal:
                raise ValueError("real tensor parity failed")
    finally:
        pool.close()
    write_new_json(
        args.output,
        {
            "schema": "simplex_t_cached_reader_real_qa_v1",
            "status": "EXACT_PASS",
            "records": records,
            "index_sha256": expected,
            "reader_sha256": sha256(work / "src/e_jepa_ttc/simplex_t/cached_event_reader.py"),
            "benchmark_sha256": sha256(Path(__file__)),
            "expert_forwards": 0,
            "optimizer_updates": 0,
            "limitation": (
                "CPU input parity only; sequential warm-cache timings, not a GPU speedup claim"
            ),
        },
    )


if __name__ == "__main__":
    main()
