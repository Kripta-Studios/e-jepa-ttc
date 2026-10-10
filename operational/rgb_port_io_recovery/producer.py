"""Run the admitted producer with bounded read retries and graceful I/O pauses."""

# ruff: noqa: ANN401

from __future__ import annotations

import argparse
import errno
import importlib
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from operational.rgb_port.accounting import atomic_write_json, sha256_file

from .common import FREEZE_NAME, retry_input_read, validate_freeze, write_event

DELEGATES = {
    "operational.rgb_port_concurrent.producer",
    "operational.rgb_port_c2f_graph.producer",
    "operational.rgb_port_acceleration.producer",
}
FIT_IDS = {"E_A5_MATCHED", "E_C2F_MATCHED", "R_A5", "R_C2F"}


@contextmanager
def installed(run: Path, fit_id: str) -> Iterator[None]:
    """Wrap complete input reads; leave successful arrays and model code exact."""
    from e_jepa_ttc.rgb_port.data import RGBProducerSource
    from operational.rgb_port_revision.cache import GroupRowCache

    original_decode = GroupRowCache._decode
    original_rgb_batch = RGBProducerSource.batch

    def decode(cache: Any, number: int) -> Any:
        return retry_input_read(
            lambda: original_decode(cache, number),
            lambda event: write_event(run, fit_id, event),
            scope=f"event_shard:{number:05d}",
        )

    def batch(source: Any, indices: Any, modality: str) -> Any:
        return retry_input_read(
            lambda: original_rgb_batch(source, indices, modality),
            lambda event: write_event(run, fit_id, event),
            scope="rgb_producer_batch",
        )

    with ExitStack() as stack:
        stack.enter_context(patch.object(GroupRowCache, "_decode", decode))
        stack.enter_context(patch.object(RGBProducerSource, "batch", batch))
        yield


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--io-freeze", type=Path, required=True)
    parser.add_argument("--delegate-module", choices=sorted(DELEGATES), required=True)
    parser.add_argument("forward", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.forward or args.forward[0] != "--":
        parser.error("original producer arguments must follow --")
    forwarded = args.forward[1:]
    if "--fit-id" not in forwarded or "--run" not in forwarded:
        parser.error("producer fit and run are required")
    fit_id = forwarded[forwarded.index("--fit-id") + 1]
    run = Path(forwarded[forwarded.index("--run") + 1]).resolve(strict=True)
    if fit_id not in FIT_IDS or args.io_freeze.resolve(strict=True) != run / FREEZE_NAME:
        parser.error("I/O overlay scope differs from the authorized run")
    freeze = validate_freeze(run)
    atomic_write_json(
        run / "fits" / fit_id / "IO_RECOVERY_RUNTIME.json",
        {
            "schema": "rgb_port_io_recovery_runtime_v1",
            "fit_id": fit_id,
            "freeze_sha256": sha256_file(run / FREEZE_NAME),
            "identity_sha256": freeze["identity_sha256"],
            "delegate_module": args.delegate_module,
            "started_utc": datetime.now(UTC).isoformat(),
            "scientific_changes": False,
        },
    )
    with installed(run, fit_id):
        try:
            return int(importlib.import_module(args.delegate_module).main(forwarded))
        except OSError as error:
            # ENOTCONN is raised only by the scoped exhausted-read adapter or by
            # the original trainer. Its regular loop saves before returning 3;
            # a failed startup/prewarm has no new optimizer updates to save.
            if error.errno != errno.ENOTCONN:
                raise
            write_event(
                run,
                fit_id,
                {
                    "scope": "producer_startup_or_prewarm",
                    "message": str(error),
                    "action": "resource_pause_before_update",
                },
            )
            return 3


if __name__ == "__main__":
    raise SystemExit(main())
