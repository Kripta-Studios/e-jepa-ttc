"""Resumable frozen EvTTC inference; labels are only read by the separate scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from operational.efficient_context.common import Lease, atomic_json, digest

if TYPE_CHECKING:
    from .models import FrozenModels


def stamp() -> str:
    """Return an auditable wall-clock timestamp."""
    return datetime.now(UTC).isoformat()


def finite_or_none(value: float) -> float | None:
    """Preserve invalid-output coverage in standards-compliant JSON."""
    return float(value) if math.isfinite(float(value)) else None


def measure_query(row: dict, model: FrozenModels, prepare: Callable[[dict], dict]) -> dict:
    """Measure one common input; invalid outputs remain explicit in its receipt."""
    started = time.perf_counter()
    prepared = prepare(row)
    prepared_at = time.perf_counter()
    own = model.predict(
        prepared["own_events"], delta_t_s=prepared["delta_t_s"], valid=prepared["valid"]
    )
    own_at = time.perf_counter()
    garl = (
        model.garl_predict(prepared["garl_events"])
        if prepared["garl_events"] is not None
        else {"ttc": float("nan"), "heights": []}
    )
    ended = time.perf_counter()
    ttc = {name: finite_or_none(value) for name, value in own.items()}
    ttc["public_Garl_event_lhr"] = finite_or_none(cast(float, garl["ttc"]))
    return dict(
        status="PREDICTED"
        if all(v is not None for v in ttc.values())
        else "NONFINITE_OUTPUT_RETAINED",
        ttc=ttc,
        unavailable_inputs={"public_Garl_event_lhr": prepared["garl_unavailable_reason"]}
        if prepared.get("garl_unavailable_reason")
        else {},
        garl_heights=[finite_or_none(x) for x in cast(list[float], garl["heights"])],
        input_tensor_sha256={
            key: hashlib.sha256(np.ascontiguousarray(prepared[key]).tobytes()).hexdigest()
            for key in ("own_events", "garl_events")
            if prepared[key] is not None
        },
        seconds=dict(
            prepare=prepared_at - started,
            own=own_at - prepared_at,
            garl=ended - own_at,
            total=ended - started,
        ),
    )


def run(campaign: Path, output: Path, device: str, limit: int | None = None) -> None:
    """Run every fixed query or resume exact sealed fragments without fitting."""
    import psutil
    import torch

    from .inputs import prepare
    from .models import FrozenModels

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "QUERY_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["rows"]
    if not rows:
        raise ValueError("Empty predeclared evaluation population")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with Lease(output):
        model = FrozenModels(campaign, device=device)
        binding = dict(
            manifest_sha256=digest(manifest_path),
            models=model.bindings,
            sources={str(p): digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            device=device,
            precision="float32",
            optimizer_updates=0,
            no_label_reads=True,
            no_calibration=True,
            keep_all_heads=True,
        )
        binding_path = output / "INFERENCE_FREEZE.json"
        if binding_path.exists():
            if json.loads(binding_path.read_text(encoding="utf-8")) != binding:
                raise ValueError("Frozen inference configuration or sources changed")
        else:
            atomic_json(binding_path, binding)
        binding_sha = digest(binding_path)
        initial = time.perf_counter()
        fresh = 0
        for i, row in enumerate(rows):
            if limit is not None and fresh >= limit:
                break
            path = output / "predictions" / f"query_{i:05d}.json"
            checksum_path = path.with_suffix(".sha256")
            if path.exists() and checksum_path.exists():
                if checksum_path.read_text(encoding="ascii").strip() != digest(path):
                    raise ValueError("Saved prediction fragment hash mismatch")
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved["binding_sha256"] != binding_sha or saved["query_id"] != row["query_id"]:
                    raise ValueError("Saved prediction identity mismatch")
                continue
            if (output / "STOP_REQUEST").exists():
                atomic_json(
                    output / "STATE.json", dict(status="PAUSED_PRESERVED", checked_utc=stamp())
                )
                return
            wait_start = time.monotonic()
            while psutil.virtual_memory().available < 2 * 1024**3:
                atomic_json(
                    output / "STATE.json", dict(status="WAITING_FOR_RAM", checked_utc=stamp())
                )
                if time.monotonic() - wait_start > 300:
                    raise InterruptedError("Insufficient RAM headroom; fragments preserved")
                time.sleep(2)
            started = time.perf_counter()
            try:
                measured = measure_query(row, model, prepare)
            except Exception as error:
                failure = dict(
                    query_id=row["query_id"],
                    binding_sha256=binding_sha,
                    error_type=type(error).__name__,
                    error=str(error),
                    checked_utc=stamp(),
                    status="DEPENDENCY_OR_CONTRACT_FAILURE",
                    elapsed_seconds=time.perf_counter() - started,
                )
                atomic_json(output / "failures" / f"query_{i:05d}_{time.time_ns()}.json", failure)
                atomic_json(output / "STATE.json", failure)
                raise
            result = dict(
                **measured,
                query_id=row["query_id"],
                sequence_id=row["sequence_id"],
                anchor_us=row["anchor_us"],
                binding_sha256=binding_sha,
                optimizer_updates=0,
                finished_utc=stamp(),
            )
            atomic_json(path, result)
            checksum_path.write_text(digest(path) + "\n", encoding="ascii")
            fresh += 1
            atomic_json(
                output / "STATE.json",
                dict(
                    status="RUNNING",
                    completed_queries=i + 1,
                    total_queries=len(rows),
                    fresh_queries_this_process=fresh,
                    elapsed_seconds=time.perf_counter() - initial,
                    checked_utc=stamp(),
                    optimizer_updates=0,
                ),
            )
        fragments = {}
        for i in range(len(rows)):
            name = f"predictions/query_{i:05d}.json"
            p = output / name
            if not p.exists():
                atomic_json(
                    output / "STATE.json",
                    dict(
                        status="PILOT_COMPLETE",
                        fresh_queries=fresh,
                        checked_utc=stamp(),
                        optimizer_updates=0,
                    ),
                )
                return
            expected_path = p.with_suffix(".sha256")
            if not expected_path.exists() or expected_path.read_text(
                encoding="ascii"
            ).strip() != digest(p):
                raise ValueError("Cannot seal an unverified prediction fragment")
            saved = json.loads(p.read_text(encoding="utf-8"))
            if saved["binding_sha256"] != binding_sha or saved["query_id"] != rows[i]["query_id"]:
                raise ValueError("Cannot seal mixed prediction lineage")
            fragments[name] = digest(p)
        atomic_json(
            output / "PREDICTIONS_SEALED.json",
            dict(
                status="COMPLETE",
                queries=len(rows),
                manifest_sha256=digest(manifest_path),
                binding_sha256=binding_sha,
                fragments=fragments,
                checked_utc=stamp(),
                optimizer_updates=0,
            ),
        )
        atomic_json(
            output / "STATE.json",
            dict(
                status="PREDICTIONS_COMPLETE",
                queries=len(rows),
                checked_utc=stamp(),
                optimizer_updates=0,
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.campaign, args.output, args.device, args.limit)
