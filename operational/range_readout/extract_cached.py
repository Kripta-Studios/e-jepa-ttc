"""Lossless native row-cache adapter for the same admitted vector extraction."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import torch

from operational.range_transfer.common import ownership
from operational.range_transfer.qa import same
from operational.rgb_port.train_producers import EventProducerSource
from operational.rgb_port_revision.cache import GroupRowCache

from . import cache_readout
from .common import OWNER, RUN, freeze, read, sha, utc, write

_original_factory = cache_readout.make_event_producer_source


def row_source(manifest_path: str | Path, *, event_source_root: str | Path) -> EventProducerSource:
    """Reuse the admitted V13 row cache with unchanged input order and tensors."""
    source = _original_factory(manifest_path, event_source_root=event_source_root)
    source._inputs.limit = 384 * 1024**2
    source._audit_row_cache = GroupRowCache(source)
    return source


def validate() -> None:
    root = Path(
        read(OWNER / "fits/E_A5_MATCHED/RUN_PROVENANCE.json")["source_identity"][
            "event_source_root"
        ]
    )
    original = _original_factory(OWNER / "P_MANIFEST.json", event_source_root=root)
    original._inputs.limit = 384 * 1024**2
    row = row_source(OWNER / "P_MANIFEST.json", event_source_root=root)
    results = []
    for begin in (0, 8, 120):
        ids = list(range(begin, begin + 8))
        a = original.batch(ids, "event")
        b = row.batch(ids, "event")
        checks = {f.name: same(getattr(a, f.name), getattr(b, f.name)) for f in fields(a)}
        assert all(checks.values()), checks
        results.append({"indices": ids, "fields": checks})
    original.close()
    row.close()
    write(
        RUN / "qa/ROW_CACHE_PARITY.json",
        {
            "status": "PASS",
            "optimizer_updates": 0,
            "checks": results,
            "utc": utc(),
            "row_cache_source": sha(Path(__import__("inspect").getfile(GroupRowCache))),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=["P", "H", "V", "validate"])
    parser.add_argument("--max-shards", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.role == "validate":
        validate()
        return
    if read(RUN / "qa/ROW_CACHE_PARITY.json")["status"] != "PASS":
        raise ValueError("Row parity absent")
    if args.role == "V" and not (RUN / "SCREEN_ENDPOINTS.json").exists():
        raise ValueError("V queue not frozen")
    directory = RUN / "vector_cache" / args.role
    freeze(
        directory / "IO_ADAPTER.json",
        {
            "scope": "lossless row-cache eviction only",
            "source_sha256": sha(Path(__file__)),
            "parity_sha256": sha(RUN / "qa/ROW_CACHE_PARITY.json"),
            "original_vector_binding_unchanged": True,
            "optimizer_updates": 0,
        },
    )
    with ownership(RUN / "vector_extraction"):
        cache_readout.make_event_producer_source = row_source
        cache_readout.extract(args.role, "cuda", max_shards=args.max_shards)


if __name__ == "__main__":
    main()
