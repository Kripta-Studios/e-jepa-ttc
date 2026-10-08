"""Admit all 64 fixed TRAIN windows without repeating irrelevant gap ingestion."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import cast

from .common import ROOT, Campaign, atomic_json, digest
from .r1_profile import ARMS, RuntimeCampaign, load_inputs, parity, schedule
from .r1_reader import ResidentReplay


def run(output: Path) -> None:
    """Compare canonical HDF5 tensors with bounded resident-window preparation."""
    import numpy as np
    import torch

    from e_jepa_ttc.data.eap import EAPEventReader
    from e_jepa_ttc.efficient_context.mapped_union import encode_union
    from e_jepa_ttc.efficient_context.sparse_history import WIDE_SLOTS
    from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool
    from e_jepa_ttc.simplex_t.context_raw_union import encode_context_union

    from .common import read
    from .profile_safe import verify_raw

    output.mkdir(parents=True, exist_ok=True)
    base = Campaign(ROOT / "configs/campaign/efficient_context_v1.json")
    base.freeze()
    scoped = RuntimeCampaign(base, output)
    protocol, indexes, prep = load_inputs(base)
    files = [
        Path(__file__),
        Path(__file__).with_name("r1_reader.py"),
        Path(__file__).with_name("r1_profile.py"),
    ]
    binding = dict(
        sources={str(p.relative_to(ROOT)): digest(p) for p in files},
        purpose="exact real tensor parity, not R1 cost measurement",
        queries=64,
        arms=list(ARMS),
        optimizer_updates=0,
    )
    freeze = output / "PROTOCOL.json"
    if freeze.exists() and read(freeze) != binding:
        raise ValueError("CPU window admission freeze changed")
    atomic_json(freeze, binding)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    pool = ReaderPool()
    start_time = time.perf_counter()
    cases = []
    try:
        for qi, query in schedule(protocol, indexes):
            idx, i = indexes[query["pool"]], query["index_row"]
            if str(idx["tokens"][i]) != query["sample_token"]:
                raise ValueError("TRAIN query mapping changed")
            path = base.raw / query["sequence_id"] / "events.h5"
            expected = next(r for r in protocol["raw"] if Path(r["path"]) == path)
            raw_binding = verify_raw(base, path, expected)
            source = pool.get(path)
            for label, length in ARMS.items():
                scoped.require_resources()
                receipt = output / "fragments" / f"{qi:02d}_{label}.json"
                if receipt.exists():
                    cases.append(read(receipt))
                    continue
                valid = idx["valid"][i].copy()
                if label == "WIDE":
                    valid &= np.isin(np.arange(16), WIDE_SLOTS)
                else:
                    valid[:-length] = False
                intervals = idx["base_windows_us"][i][None] - idx["lag_us"][:, None, None]
                first, cutoff = int(intervals[valid].min()), int(intervals[valid].max())
                resident = ResidentReplay(source)
                resident.advance(first, cutoff)
                options = dict(
                    sequence_id=query["sequence_id"],
                    roi_size=prep["roi_size"],
                    event_pixel_diff=prep["event_pixel_diff"],
                )
                reference = encode_context_union(
                    source,
                    idx["base_windows_us"][i],
                    idx["lag_us"],
                    valid,
                    tuple(idx["square_xyxy"][i]),
                    **options,
                )
                mapped = encode_union(
                    cast(EAPEventReader, resident),
                    idx["base_windows_us"][i],
                    idx["lag_us"],
                    valid,
                    tuple(idx["square_xyxy"][i]),
                    **options,
                )
                evidence = parity(reference, mapped, {}, {})
                evidence.update(
                    original_query=qi,
                    query=query["sample_token"],
                    arm=label,
                    raw_binding=raw_binding,
                    resident_stats=resident.stats,
                )
                atomic_json(receipt, evidence)
                cases.append(evidence)
                atomic_json(
                    output / "PROGRESS.json",
                    dict(
                        completed=len(cases),
                        total=256,
                        elapsed_s=time.perf_counter() - start_time,
                        optimizer_updates=0,
                    ),
                )
                print(f"CPU_WINDOW_PARITY {len(cases)}/256 {evidence['status']}", flush=True)
        atomic_json(
            output / "RESULT.json",
            dict(status="EXACT", cases=cases, tensors=256, optimizer_updates=0),
        )
    finally:
        pool.close()


def main() -> int:
    """Run the independent CPU admission phase, resumable per exact pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/efficient_context_20261004/r1_20261008/window_parity",
    )
    run(parser.parse_args().output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
