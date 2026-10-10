"""Resume event producers with four shared-memory decoders across both fits."""

# ruff: noqa: ANN401
from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from operational.rgb_port import train_producers as training
from operational.rgb_port.accounting import atomic_write_json, sha256_file
from operational.rgb_port_continuity import producer as continuity
from operational.rgb_port_loader_v3.contracts import NAME, validate
from operational.rgb_port_loader_v3.loader import DepthTwoEventPrefetch, ParallelGroupRowCache


def main(argv: Sequence[str] | None = None) -> int:
    """Wrap the innermost fit while retaining all original checkpoint guards."""
    values = list(sys.argv[1:] if argv is None else argv)
    if "--help" in values:
        return continuity.main(values)
    run = Path(values[values.index("--run") + 1]).resolve(strict=True)
    fit_id = values[values.index("--fit-id") + 1]
    if fit_id not in {"E_A5_MATCHED", "E_C2F_MATCHED"}:
        return continuity.main(values)
    validate(run)
    original_fit = training.fit_producer

    def fit(source: Any, recipe: Any, run_root: Path, **kwargs: Any) -> Any:
        if (
            not isinstance(source, training.EventProducerSource)
            or recipe.modality != "event"
            or recipe.microbatch_size != recipe.effective_batch_size
        ):
            raise ValueError("Loader V3 requires the canonical homogeneous event source")
        cache = ParallelGroupRowCache(source, workers=2)
        source._audit_row_cache = cache
        loader = DepthTwoEventPrefetch(source, batch_size=recipe.effective_batch_size, depth=2)
        original_restore = training.ProducerCheckpoint.restore
        original_save = training.ProducerCheckpoint.save
        started = time.perf_counter()
        origin: dict[str, Any] = {}
        receipt_root = run_root / "fits" / fit_id / "loader_v3_checkpoints"

        def receipt(state: Any, status: str) -> dict[str, Any]:
            return {
                "schema": "rgb_port_loader_v3_runtime_v1",
                "fit_id": fit_id,
                "status": status,
                "freeze_sha256": sha256_file(run / NAME),
                "checkpoint_sha256": sha256_file(state.path),
                "completed_updates": state.completed,
                "origin": origin,
                "elapsed_s_since_restore": time.perf_counter() - started,
                "source_wait_s": loader.wait_seconds,
                "source_prepare_s": loader.prepare_seconds,
                "consumed_batches": loader.consumed,
                "queued_batches_max": loader.outstanding_max,
                "shard_reads": cache.reads,
                "row_hits": cache.hits,
                "row_cache_peak_bytes": cache.peak_bytes,
                "decoder_workers": 2,
                "prefetch_depth": 2,
                "objective_changed": False,
                "geometry_precision": "bf16_unchanged",
            }

        def restore(state: Any, *args: Any, **kw: Any) -> Any:
            nonlocal started
            # All inherited CUDA prewarm and exact-state restoration finish
            # before lookahead starts. Their RNG and cursor checks remain active.
            cursor = original_restore(state, *args, **kw)
            origin.update(
                completed_updates=state.completed, checkpoint_sha256=sha256_file(state.path)
            )
            started = time.perf_counter()
            atomic_write_json(
                receipt_root / f"origin_{state.completed:06d}.json", receipt(state, "RESTORED")
            )
            loader.bind(cursor)
            return cursor

        def save(state: Any, *args: Any, **kw: Any) -> None:
            original_save(state, *args, **kw)
            value = receipt(state, "CHECKPOINTED")
            atomic_write_json(receipt_root / f"checkpoint_{state.completed:06d}.json", value)
            atomic_write_json(run_root / "fits" / fit_id / "LOADER_V3_RUNTIME.json", value)

        if not (run_root / "fits" / fit_id / "CHECKPOINT_POINTER.json").is_file():
            raise ValueError("Loader V3 currently admits only existing durable checkpoints")
        try:
            with (
                patch.object(training.ProducerCheckpoint, "restore", restore),
                patch.object(training.ProducerCheckpoint, "save", save),
            ):
                return original_fit(
                    cast(training.ProducerSource, loader), recipe, run_root, **kwargs
                )
        finally:
            loader.close()
            cache.close()

    with patch.object(training, "fit_producer", fit):
        return continuity.main(values)


if __name__ == "__main__":
    raise SystemExit(main())
