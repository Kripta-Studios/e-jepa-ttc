"""Install exact RGB overlap inside the existing source and checkpoint wrappers."""

# ruff: noqa: ANN401 -- adapt the admitted dynamically composed producer runtime
from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

from operational.rgb_port import train_producers as training
from operational.rgb_port.accounting import atomic_write_json, sha256_file
from operational.rgb_port_revision.loss import EffectiveSource
from operational.rgb_port_rgb_loader import producer as parent

from .compute import installed_compute
from .contracts import (
    COMPUTE_ENABLED,
    DECODER_PENDING_BYTES,
    DECODER_WORKERS,
    DEPTH,
    NAME,
    validate,
)
from .decoder import installed_decoder
from .prefetch import capture_effective_prefetch


def main(argv: Sequence[str] | None = None) -> int:
    """Preserve the training loop, optimizer boundaries and full recovery chain."""
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
        if recipe.modality != "rgb":
            raise ValueError("RGB overlap requires the RGB recipe")
        original_restore = training.ProducerCheckpoint.restore
        original_save = training.ProducerCheckpoint.save
        origin: dict[str, Any] = {}
        started = time.perf_counter()
        receipt_root = run_root / "fits" / fit_id / "rgb_overlap_checkpoints"
        # Close the prefetch coordinator before either its decoder or the outer
        # DINO/collate contexts can be restored. No source runs concurrently with
        # another consumer; decoder workers own independent NPZ files.
        with (
            installed_compute() if COMPUTE_ENABLED else nullcontext(None) as geometry,
            installed_decoder(
                source, workers=DECODER_WORKERS, max_pending_bytes=DECODER_PENDING_BYTES
            ) as decoder,
            capture_effective_prefetch(
                EffectiveSource,
                expected_source=source,
                batch_size=recipe.effective_batch_size,
                depth=DEPTH,
            ) as loader,
        ):

            def receipt(state: Any, status: str) -> dict[str, Any]:
                return {
                    "schema": "rgb_port_rgb_overlap_runtime_v1",
                    "status": status,
                    "fit_id": fit_id,
                    "freeze_sha256": sha256_file(run / NAME),
                    "checkpoint_sha256": sha256_file(state.path),
                    "completed_updates": state.completed,
                    "origin": dict(origin),
                    "elapsed_s_since_restore": time.perf_counter() - started,
                    "prefetch": loader.snapshot(),
                    "decoder": decoder.snapshot(),
                    "geometry": {
                        "enabled": COMPUTE_ENABLED,
                        **(geometry.snapshot() if geometry is not None else {}),
                    },
                    "objective_changed": False,
                    "geometry_precision": "bf16_unchanged",
                    "speculative_preparation_included_in_counters": True,
                }

            def bind(state: Any, cursor: Any, status: str) -> None:
                nonlocal started
                origin.update(
                    completed_updates=state.completed,
                    checkpoint_sha256=sha256_file(state.path),
                )
                started = time.perf_counter()
                loader.bind(cursor)
                atomic_write_json(
                    receipt_root / f"origin_{state.completed:06d}.json", receipt(state, status)
                )

            def restore(state: Any, *args: Any, **kw: Any) -> Any:
                cursor = original_restore(state, *args, **kw)
                bind(state, cursor, "RESTORED")
                return cursor

            def save(state: Any, *args: Any, **kw: Any) -> None:
                original_save(state, *args, **kw)
                if not origin:
                    # Fresh R_C2F creates its exact initial checkpoint before the
                    # first batch, so it has no restore hook to bind the cursor.
                    cursor = kw.get("cursor", args[4] if len(args) > 4 else None)
                    if cursor is None:
                        raise ValueError("Initial checkpoint did not expose its canonical cursor")
                    bind(state, cursor, "INITIALIZED")
                value = receipt(state, "CHECKPOINTED")
                atomic_write_json(receipt_root / f"checkpoint_{state.completed:06d}.json", value)
                atomic_write_json(run_root / "fits" / fit_id / "RGB_OVERLAP_RUNTIME.json", value)

            with (
                patch.object(training.ProducerCheckpoint, "restore", restore),
                patch.object(training.ProducerCheckpoint, "save", save),
            ):
                return original_fit(source, recipe, run_root, **kwargs)

    with patch.object(training, "fit_producer", fit):
        return parent.main(values)


if __name__ == "__main__":
    raise SystemExit(main())
