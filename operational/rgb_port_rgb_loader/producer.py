"""Apply bounded, exact RGB preparation inside the admitted checkpoint runtime."""

# ruff: noqa: ANN401 -- preserve the frozen dynamically composed producer protocol
from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from e_jepa_ttc.rgb_port.data import RGBProducerSource
from operational.rgb_port import train_producers as training
from operational.rgb_port.accounting import atomic_write_json, sha256_file
from operational.rgb_port_io_recovery.common import retry_input_read, write_event
from operational.rgb_port_loader_v3 import producer as parent
from operational.rgb_port_loader_v3.memory import COMMIT_RESERVE_BYTES, with_requested_reserve

from .collate import installed_collate
from .contracts import MAX_DINO_BYTES, MAX_DINO_SHARDS, NAME, validate
from .guards import resource_guard
from .loader import installed_dino_cache


def main(argv: Sequence[str] | None = None) -> int:
    """Retain source identity, effective T2/T3 reductions and original update loop."""
    values = list(sys.argv[1:] if argv is None else argv)
    if "--help" in values:
        return parent.main(values)
    run = Path(values[values.index("--run") + 1]).resolve(strict=True)
    fit_id = values[values.index("--fit-id") + 1]
    if fit_id not in {"R_A5", "R_C2F"}:
        return parent.main(values)
    validate(run)
    original_fit = training.fit_producer

    def fit(source: Any, recipe: Any, run_root: Path, **kwargs: Any) -> Any:
        if not isinstance(source, RGBProducerSource) or recipe.modality != "rgb":
            raise ValueError("RGB loader requires the original RGB producer source")
        original_restore = training.ProducerCheckpoint.restore
        original_save = training.ProducerCheckpoint.save
        original_fast = training._fast_resource_guard
        started = time.perf_counter()
        source_seconds = 0.0
        source_calls = 0
        origin: dict[str, Any] = {}
        receipt_root = run_root / "fits" / fit_id / "rgb_loader_checkpoints"

        with (
            installed_dino_cache(
                source, max_shards=MAX_DINO_SHARDS, max_bytes=MAX_DINO_BYTES
            ) as cache,
            installed_collate(source),
        ):
            optimized_batch = source.batch

            def batch(indices: Any, modality: str) -> Any:
                nonlocal source_seconds, source_calls
                tick = time.perf_counter()
                try:
                    # The instance collate bypasses the historical class-level
                    # I/O wrapper; explicitly retain its bounded read recovery.
                    return retry_input_read(
                        lambda: optimized_batch(indices, modality),
                        lambda event: write_event(run_root, fit_id, event),
                        scope="rgb_optimized_producer_batch",
                    )
                finally:
                    source_seconds += time.perf_counter() - tick
                    source_calls += 1

            def receipt(state: Any, status: str) -> dict[str, Any]:
                return {
                    "schema": "rgb_port_rgb_loader_runtime_v1",
                    "fit_id": fit_id,
                    "status": status,
                    "freeze_sha256": sha256_file(run / NAME),
                    "checkpoint_sha256": sha256_file(state.path),
                    "completed_updates": state.completed,
                    "origin": origin,
                    "elapsed_s_since_restore": time.perf_counter() - started,
                    "source_prepare_s": source_seconds,
                    "source_calls": source_calls,
                    "dino_cache": cache.snapshot(),
                    "prefetch_depth": 0,
                    "commit_reserve_bytes": COMMIT_RESERVE_BYTES,
                    "objective_changed": False,
                    "geometry_precision": "bf16_unchanged",
                }

            def restore(state: Any, *args: Any, **kw: Any) -> Any:
                nonlocal started, source_seconds, source_calls
                cursor = original_restore(state, *args, **kw)
                origin.update(
                    completed_updates=state.completed, checkpoint_sha256=sha256_file(state.path)
                )
                started = time.perf_counter()
                source_seconds, source_calls = 0.0, 0
                atomic_write_json(
                    receipt_root / f"origin_{state.completed:06d}.json", receipt(state, "RESTORED")
                )
                return cursor

            def save(state: Any, *args: Any, **kw: Any) -> None:
                original_save(state, *args, **kw)
                value = receipt(state, "CHECKPOINTED")
                atomic_write_json(receipt_root / f"checkpoint_{state.completed:06d}.json", value)
                atomic_write_json(run_root / "fits" / fit_id / "RGB_LOADER_RUNTIME.json", value)

            with (
                patch.object(source, "batch", batch),
                patch.object(training.ProducerCheckpoint, "restore", restore),
                patch.object(training.ProducerCheckpoint, "save", save),
                patch.object(
                    training,
                    "_fast_resource_guard",
                    lambda *a, **k: with_requested_reserve(original_fast(*a, **k)),
                ),
                patch.object(
                    training,
                    "_resource_guard",
                    lambda *a, **k: with_requested_reserve(resource_guard(*a, **k)),
                ),
            ):
                # The unchanged fit wraps this same source in EffectiveSource.
                return original_fit(cast(Any, source), recipe, run_root, **kwargs)

    with patch.object(training, "fit_producer", fit):
        return parent.main(values)


if __name__ == "__main__":
    raise SystemExit(main())
