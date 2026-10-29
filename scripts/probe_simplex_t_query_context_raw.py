"""Compare a real input-only query crop with its frozen three-window tensor."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.data.eap import EAPEventReader
from e_jepa_ttc.data.event_v4_geometry import shifted_precontext_window
from e_jepa_ttc.data.raw_event_binding import EXPECTED_SEQUENCES
from e_jepa_ttc.simplex_t.lifecycle import admitted
from e_jepa_ttc.simplex_t.query_context_voxel import encode_query_window


def main() -> None:
    """Use the first input-hash-selected TRAIN token; no score/target selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--raw-train", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-selected", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve raw probe")
    binding_hash = compute_file_hash(str(args.binding))
    if binding_hash != "47b3ee83d61654cf716399a4f564043d2f2289b6785d8c26c456b941e25e70f1":
        raise ValueError("historical input binding changed")
    receipt = json.loads((args.replay / "INPUTS_READY.json").read_text(encoding="utf-8"))
    if compute_file_hash(str(args.replay / "inputs.pt")) != receipt["sha256"]:
        raise ValueError("replay input changed")
    saved = torch.load(args.replay / "inputs.pt", map_location="cpu", weights_only=True)
    binding = pd.read_csv(args.binding)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    indices = range(len(saved["tokens"])) if args.all_selected else range(1)
    results = []
    started = time.perf_counter()
    for query_index in indices:
        resource = admitted([args.output.parent])
        if not resource["has_headroom"]:
            raise RuntimeError(f"RESOURCE_PAUSE:{resource}")
        record = probe(args, saved, binding, query_index)
        results.append(record)
        print(
            json.dumps(
                {
                    "query_index": query_index,
                    "token": record["token"],
                    "all_windows_bit_identical": all(
                        row["bit_identical"] for row in record["results"]
                    ),
                }
            ),
            flush=True,
        )
    write_new_json(
        args.output,
        {
            "queries": results,
            "binding_sha256": binding_hash,
            "input_sha256": receipt["sha256"],
            "optimizer_updates": 0,
            "scores_read": False,
            "seconds": time.perf_counter() - started,
            "all_windows_bit_identical": all(
                window["bit_identical"] for query in results for window in query["results"]
            ),
        },
    )


def probe(
    args: argparse.Namespace, saved: dict[str, Any], binding: pd.DataFrame, query_index: int
) -> dict[str, Any]:
    """Replay exact frozen crop/windows for one authorized original TRAIN query."""
    token = saved["tokens"][query_index]
    rows = binding.loc[binding.sample_token == token].sort_values("window_id")
    if len(rows) != 2 or rows.window_id.tolist() != [0, 1]:
        raise ValueError("ambiguous original query endpoints")
    first, second = rows.to_dict("records")
    if first["sequence_id"] not in EXPECTED_SEQUENCES:
        raise ValueError("probe is restricted to the acknowledged original groups")
    path = (args.raw_train / first["events_path_relative"]).resolve(strict=True)
    if not path.is_relative_to(args.raw_train.resolve(strict=True)):
        raise ValueError("raw path escapes TRAIN root")
    if (
        compute_file_hash(str(args.cache_manifest))
        != "063980fdae5fda0b2836befc662fdd1cd5659bf06f10d9760dfc0d566fac8e39"
    ):
        raise ValueError("producer cache manifest changed")
    config = json.loads(args.cache_manifest.read_text(encoding="utf-8"))["config"]
    square = tuple(first[f"roi_{axis}"] for axis in ("x0", "y0", "x1", "y1"))
    windows = [
        shifted_precontext_window(
            (first["window_start_us"], first["window_end_us"]),
            shift_s=config["jepa_context_delta_t_s"],
        ),
        (first["window_start_us"], first["window_end_us"]),
        (second["window_start_us"], second["window_end_us"]),
    ]
    results = []
    with EAPEventReader(path) as reader:
        for index, (start, end) in enumerate(windows):
            raw = reader.read_window(start, end)
            tensor = encode_query_window(
                raw,
                square_xyxy=square,
                start_us=start,
                end_us=end,
                sequence_id=first["sequence_id"],
                roi_size=config["roi_size"],
                bins_per_polarity=(saved["inputs"].shape[2] - 2) // 2,
                event_pixel_diff=config["event_pixel_diff"],
            )
            expected = saved["inputs"][query_index, index]
            results.append(
                {
                    "window": index,
                    "start_us": start,
                    "end_us": end,
                    "raw_events": len(raw["t"]),
                    "bit_identical": torch.equal(tensor, expected),
                    "max_abs_by_channel": (tensor - expected).abs().flatten(1).amax(1).tolist(),
                }
            )
    return {
        "token": token,
        "results": results,
        "raw_path": str(path),
        "raw_hash_from_historical_binding": first["h5_file_sha256"],
        "raw_full_file_rehashed_this_probe": False,
        "optimizer_updates": 0,
        "scores_read": False,
    }


if __name__ == "__main__":
    main()
