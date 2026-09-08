"""Measure fixed input-only sequence representatives without expert inference."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t import context_raw_union as union
from e_jepa_ttc.simplex_t.cached_event_reader import CachedEventReader
from e_jepa_ttc.simplex_t.lifecycle import admitted


class TimedReader(CachedEventReader):
    """Attribute time spent advancing the HDF5 iterator, excluding its consumer."""

    read_seconds = 0.0
    events_read = 0

    def iter_window_chunks(
        self, start_us: int, end_us: int, *, chunk_events: int = 250_000
    ) -> Iterator[dict[str, np.ndarray]]:
        iterator = super().iter_window_chunks(start_us, end_us, chunk_events=chunk_events)
        while True:
            started = time.perf_counter()
            try:
                chunk = next(iterator)
            except StopIteration:
                self.read_seconds += time.perf_counter() - started
                return
            self.read_seconds += time.perf_counter() - started
            self.events_read += len(chunk["t"])
            yield chunk


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
    index_hash = sha256(index_path)
    if index_hash != "0fe7d7bb597dc768073a2940795442417ab6dba0be6110cd356d3ff0b9f656bf":
        raise ValueError("OLD index changed")
    with np.load(index_path, allow_pickle=False) as archive:
        index = {
            key: archive[key]
            for key in ("sequences", "base_windows_us", "lag_us", "valid", "square_xyxy")
        }
    prep_path = (
        work.parent
        / "e-jepa-ttc/artifacts/cache/garl_object_event_common_roi_train8192_v1/manifest.json"
    )
    prep_hash = sha256(prep_path)
    if prep_hash != "063980fdae5fda0b2836befc662fdd1cd5659bf06f10d9760dfc0d566fac8e39":
        raise ValueError("historical preprocessing changed")
    prep = json.loads(prep_path.read_text("utf-8"))["config"]
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    records = []
    for sequence in sorted(set(index["sequences"].tolist())):
        ids = np.flatnonzero(index["sequences"] == sequence)
        query = int(ids[len(ids) // 2])
        root = (Path(paths["eap_root"]) / "data/train").resolve(strict=True)
        raw_path = (root / sequence / "events.h5").resolve(strict=True)
        if not raw_path.is_relative_to(root):
            raise ValueError("path escapes permitted TRAIN")
        for repeat in range(2):
            resource = admitted([args.output.parent])
            if not resource["has_headroom"]:
                raise RuntimeError("RESOURCE_PAUSE before profiling query")
            reader = TimedReader(raw_path)
            voxel_seconds = 0.0
            calls = 0
            original = union.encode_query_window

            def measured(
                *positional: Any,  # noqa: ANN401 -- transparent instrumentation wrapper
                _original: Callable[..., torch.Tensor] = original,
                **keywords: Any,  # noqa: ANN401 -- preserve frozen encoder arguments
            ) -> torch.Tensor:
                nonlocal voxel_seconds, calls
                started = time.perf_counter()
                result = _original(*positional, **keywords)
                voxel_seconds += time.perf_counter() - started
                calls += 1
                return result

            started = time.perf_counter()
            try:
                with patch.object(union, "encode_query_window", measured):
                    value = union.encode_context_union(
                        reader,
                        index["base_windows_us"][query],
                        index["lag_us"],
                        index["valid"][query],
                        tuple(index["square_xyxy"][query]),
                        sequence_id=sequence,
                        roi_size=prep["roi_size"],
                        event_pixel_diff=prep["event_pixel_diff"],
                    )
            finally:
                reader.close()
            elapsed = time.perf_counter() - started
            row = dict(
                sequence=sequence,
                query=query,
                repeat=repeat,
                total_input_seconds=elapsed,
                read_decompress_seconds=reader.read_seconds,
                voxel_seconds=voxel_seconds,
                crop_assembly_other_seconds=elapsed - reader.read_seconds - voxel_seconds,
                events_read=reader.events_read,
                voxel_calls=calls,
                tensor_bytes=value.numel() * value.element_size(),
            )
            records.append(row)
            print(json.dumps(row), flush=True)
            del value
    write_new_json(
        args.output,
        dict(
            schema="simplex_t_input_stage_profile_v1",
            records=records,
            selection="median input-index row per sequence, two repetitions; no targets or scores",
            limitations=(
                "Concurrent D0 replay; warm OS cache; read includes HDF5 decompression "
                "and slicing; no GPU timing"
            ),
            index_sha256=index_hash,
            preprocessing_sha256=prep_hash,
            script_sha256=sha256(Path(__file__)),
            optimizer_updates=0,
            expert_forwards=0,
        ),
    )


if __name__ == "__main__":
    main()
