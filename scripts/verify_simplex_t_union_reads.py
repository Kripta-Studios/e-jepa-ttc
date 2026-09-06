"""Compare union and independent-window preprocessing on the fixed64 TRAIN cohort."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.data.eap import EAPEventReader
from e_jepa_ttc.simplex_t.context_raw_union import encode_context_union
from e_jepa_ttc.simplex_t.lifecycle import ExclusiveLease, admitted
from e_jepa_ttc.simplex_t.query_context_voxel import encode_query_window


def main() -> None:
    """No inference, labels or optimizer; preserve failed equality evidence too."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--raw-train", type=Path, required=True)
    parser.add_argument("--preprocessing-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous union-read proof")
    bindings = {
        args.index: "0fe7d7bb597dc768073a2940795442417ab6dba0be6110cd356d3ff0b9f656bf",
        args.cohort: "1783d658d3a50a383fe18d4abb3e295d120c6a91e1847f902282372553ab592e",
        args.preprocessing_manifest: (
            "063980fdae5fda0b2836befc662fdd1cd5659bf06f10d9760dfc0d566fac8e39"
        ),
    }
    for path, digest in bindings.items():
        if compute_file_hash(str(path)) != digest:
            raise ValueError("pinned input changed")
    with np.load(args.index, allow_pickle=False) as archive:
        index = {key: archive[key] for key in archive.files}
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    selected = [token for tokens in cohort["selection"].values() for token in tokens]
    positions = {str(token): i for i, token in enumerate(index["tokens"])}
    prep = json.loads(args.preprocessing_manifest.read_text(encoding="utf-8"))["config"]
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    results = []
    with ExclusiveLease(args.output.parent / "CURRENT_REPLAY.lock"):
        for number, token in enumerate(selected):
            if not admitted([args.output.parent])["has_headroom"]:
                raise RuntimeError("RESOURCE_PAUSE")
            qi = positions[token]
            sequence = str(index["sequences"][qi])
            raw_root = args.raw_train.resolve(strict=True)
            raw_path = (raw_root / sequence / "events.h5").resolve(strict=True)
            if not raw_path.is_relative_to(raw_root):
                raise ValueError("raw path escapes allowed TRAIN")
            windows, lag, valid = index["base_windows_us"][qi], index["lag_us"], index["valid"][qi]
            square = tuple(index["square_xyxy"][qi])
            outputs, timings = {}, {}
            order = ("union", "independent") if number % 2 else ("independent", "union")
            with EAPEventReader(raw_path) as reader:
                for method in order:
                    started = time.perf_counter()
                    if method == "union":
                        tensor = encode_context_union(
                            reader,
                            windows,
                            lag,
                            valid,
                            square,
                            sequence_id=sequence,
                            roi_size=prep["roi_size"],
                            event_pixel_diff=prep["event_pixel_diff"],
                        )
                    else:
                        tensor = torch.zeros(16, 3, 12, 128, 128)
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
                    timings[method] = time.perf_counter() - started
                    outputs[method] = tensor
            exact = torch.equal(outputs["union"], outputs["independent"])
            result = {
                "query": token,
                "windows": int(valid.sum()) * 3,
                "exact": exact,
                "seconds": timings,
                "first_method": order[0],
                "max_abs_difference": float(
                    (outputs["union"] - outputs["independent"]).abs().max()
                ),
                "resources": admitted([args.output.parent]),
            }
            results.append(result)
            print(
                json.dumps({"completed": len(results), "exact": exact, "seconds": timings}),
                flush=True,
            )
    write_new_json(
        args.output,
        {
            "status": "REAL64_TEMPORAL_UNION_READ_PARITY_AUDIT",
            "all_exact": all(row["exact"] for row in results),
            "queries": len(results),
            "windows": sum(row["windows"] for row in results),
            "results": results,
            "sources": {str(path): digest for path, digest in bindings.items()},
            "union_source_sha256": compute_file_hash(
                "src/e_jepa_ttc/simplex_t/context_raw_union.py"
            ),
            "optimizer_updates": 0,
            "targets_read": False,
            "timing_caveat": (
                "Same-process alternating method order, host filesystem caches not cleared"
            ),
        },
    )


if __name__ == "__main__":
    main()
