"""Exact input-only regression for the real query that exceeded the union buffer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.data.eap import EAPEventReader
from e_jepa_ttc.simplex_t.context_raw_union import encode_context_union
from e_jepa_ttc.simplex_t.lifecycle import ExclusiveLease, admitted
from e_jepa_ttc.simplex_t.query_context_voxel import encode_query_window


def main() -> None:
    """Compare bounded, large-union and original separate-window tensors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--raw-train", type=Path, required=True)
    parser.add_argument("--preprocessing-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pins = {
        args.index: "0fe7d7bb597dc768073a2940795442417ab6dba0be6110cd356d3ff0b9f656bf",
        args.preprocessing_manifest: (
            "063980fdae5fda0b2836befc662fdd1cd5659bf06f10d9760dfc0d566fac8e39"
        ),
    }
    if args.output.exists():
        raise FileExistsError("preserve capacity regression evidence")
    for path, digest in pins.items():
        if compute_file_hash(str(path)) != digest:
            raise ValueError("input pin changed")
    with np.load(args.index, allow_pickle=False) as archive:
        index = {key: archive[key] for key in archive.files}
    prep = json.loads(args.preprocessing_manifest.read_text(encoding="utf-8"))["config"]
    qi = 4981  # Operational failure ID selected before any scores.
    sequence = str(index["sequences"][qi])
    raw_root = args.raw_train.resolve(strict=True)
    raw_path = (raw_root / sequence / "events.h5").resolve(strict=True)
    if not raw_path.is_relative_to(raw_root):
        raise ValueError("raw path escapes TRAIN")
    windows, lag, valid = index["base_windows_us"][qi], index["lag_us"], index["valid"][qi]
    square = tuple(index["square_xyxy"][qi])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    outputs, seconds = {}, {}
    with ExclusiveLease(args.output.parent / "CURRENT_REPLAY.lock"):
        with EAPEventReader(raw_path) as reader:
            for mode in ("bounded_256MiB", "union_512MiB", "independent"):
                if not admitted([args.output.parent])["has_headroom"]:
                    raise RuntimeError("RESOURCE_PAUSE")
                started = time.perf_counter()
                if mode != "independent":
                    tensor = encode_context_union(
                        reader,
                        windows,
                        lag,
                        valid,
                        square,
                        sequence_id=sequence,
                        roi_size=prep["roi_size"],
                        event_pixel_diff=prep["event_pixel_diff"],
                        retained_bytes_max=(256 if mode == "bounded_256MiB" else 512) * 1024**2,
                    )
                else:
                    tensor = torch.zeros(16, 3, 12, prep["roi_size"], prep["roi_size"])
                    for slot in np.flatnonzero(valid):
                        for w, (start, end) in enumerate(windows - lag[slot]):
                            tensor[slot, w] = encode_query_window(
                                reader.read_window(int(start), int(end)),
                                square_xyxy=square,
                                start_us=int(start),
                                end_us=int(end),
                                sequence_id=sequence,
                                roi_size=prep["roi_size"],
                                bins_per_polarity=5,
                                event_pixel_diff=prep["event_pixel_diff"],
                            )
                outputs[mode] = tensor
                seconds[mode] = time.perf_counter() - started
                print(json.dumps({"mode": mode, "seconds": seconds[mode]}), flush=True)
    exact = all(torch.equal(value, outputs["independent"]) for value in outputs.values())
    memory = psutil.Process().memory_info()
    write_new_json(
        args.output,
        {
            "status": "REAL_OVERSIZED_UNION_INPUT_PARITY",
            "query_index": qi,
            "sequence": sequence,
            "windows": int(valid.sum()) * 3,
            "all_exact": exact,
            "seconds": seconds,
            "process_memory": memory._asdict(),
            "resources": admitted([args.output.parent]),
            "union_source_sha256": compute_file_hash(
                "src/e_jepa_ttc/simplex_t/context_raw_union.py"
            ),
            "source_pins": {str(path): digest for path, digest in pins.items()},
            "optimizer_updates": 0,
            "targets_read": False,
            "timing_caveat": "Same process; filesystem caches not cleared",
        },
    )
    if not exact:
        raise ValueError("input tensor parity failed; do not amend production cache")


if __name__ == "__main__":
    main()
