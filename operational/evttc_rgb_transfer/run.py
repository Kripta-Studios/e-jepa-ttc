"""Resume a separate full-Garl evaluation without changing saved H8 predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import cast

import numpy as np

from operational.efficient_context.common import Lease, atomic_json, digest
from operational.evttc_transfer.run import finite_or_none, stamp


def read(path: Path) -> dict:
    """Read a UTF-8 JSON receipt."""
    return json.loads(path.read_text(encoding="utf-8"))


def verify_baseline(baseline: Path, preservation: dict) -> tuple[dict, list[dict]]:
    """Validate every inherited prediction before reusing any model output."""
    for name, checksum in preservation["key_files"].items():
        if digest(baseline / name) != checksum:
            raise ValueError(f"Preserved baseline changed: {name}")
    manifest = read(baseline / "QUERY_MANIFEST.json")
    seal = read(baseline / "PREDICTIONS_SEALED.json")
    if seal["status"] != "COMPLETE" or seal["queries"] != len(manifest["rows"]):
        raise ValueError("Baseline population is incomplete")
    if seal["binding_sha256"] != digest(baseline / "INFERENCE_FREEZE.json"):
        raise ValueError("Baseline freeze differs from its seal")
    if seal["manifest_sha256"] != digest(baseline / "QUERY_MANIFEST.json"):
        raise ValueError("Baseline manifest differs from its seal")
    predictions = []
    for i, row in enumerate(manifest["rows"]):
        name = f"predictions/query_{i:05d}.json"
        if digest(baseline / name) != seal["fragments"][name]:
            raise ValueError(f"Baseline prediction changed: {name}")
        saved = read(baseline / name)
        if (
            saved["query_id"] != row["query_id"]
            or saved["binding_sha256"] != seal["binding_sha256"]
        ):
            raise ValueError("Baseline identity mismatch")
        predictions.append(saved)
    return manifest, predictions


def run(output: Path, baseline: Path, code_root: Path, device: str, limit: int | None) -> None:
    """Predict full Garl only, save each result atomically, and seal all queries."""
    import psutil
    import torch

    from .inputs import CONTRACT, prepare
    from .model import FullGarl

    output.mkdir(parents=True, exist_ok=True)
    preservation = read(output / "BASELINE_PRESERVATION.json")
    manifest, old_predictions = verify_baseline(baseline, preservation)
    rows = manifest["rows"]
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with Lease(output):
        model = FullGarl(output / "public_garl", code_root, device=device)
        source_paths = list(Path(__file__).parent.glob("*.py"))
        source_paths += list(Path("operational/evttc_transfer").glob("*.py"))
        binding = dict(
            baseline_preservation_sha256=digest(output / "BASELINE_PRESERVATION.json"),
            manifest_sha256=digest(baseline / "QUERY_MANIFEST.json"),
            model=model.bindings,
            input_contract=CONTRACT,
            sources={str(p.resolve()): digest(p) for p in sorted(source_paths)},
            device=device,
            precision="float32",
            optimizer_updates=0,
            historical_labels_already_observed=True,
            labels_not_used_by_inference=True,
            no_model_selection_or_calibration=True,
            own_predictions_reused_exactly=True,
        )
        freeze = output / "INFERENCE_FREEZE.json"
        if freeze.exists() and read(freeze) != binding:
            raise ValueError("Full-Garl freeze changed; preserve old results")
        if not freeze.exists():
            atomic_json(freeze, binding)
        binding_sha = digest(freeze)
        fresh, begin = 0, time.perf_counter()
        for i, (row, old) in enumerate(zip(rows, old_predictions, strict=True)):
            path = output / "predictions" / f"query_{i:05d}.json"
            checksum = path.with_suffix(".sha256")
            if path.exists() and checksum.exists():
                saved = read(path)
                if (
                    checksum.read_text().strip() != digest(path)
                    or saved["binding_sha256"] != binding_sha
                    or saved["query_id"] != row["query_id"]
                ):
                    raise ValueError("Saved full-Garl fragment mismatch")
                continue
            if limit is not None and fresh >= limit:
                break
            if (output / "STOP_REQUEST").exists():
                atomic_json(
                    output / "STATE.json", dict(status="PAUSED_PRESERVED", checked_utc=stamp())
                )
                return
            while psutil.virtual_memory().available < 2 * 1024**3:
                atomic_json(
                    output / "STATE.json", dict(status="WAITING_FOR_RAM", checked_utc=stamp())
                )
                if (output / "STOP_REQUEST").exists():
                    atomic_json(
                        output / "STATE.json", dict(status="PAUSED_PRESERVED", checked_utc=stamp())
                    )
                    return
                time.sleep(2)
            started = time.perf_counter()
            try:
                prepared = prepare(row)
                prepared_at = time.perf_counter()
                event_hash = None
                if prepared["garl_events"] is not None:
                    event_hash = hashlib.sha256(
                        np.ascontiguousarray(prepared["garl_events"]).tobytes()
                    ).hexdigest()
                    if event_hash != old["input_tensor_sha256"].get("garl_events"):
                        raise ValueError("Full-Garl event channels differ from event-only baseline")
                result = (
                    model.predict_sensor(prepared["sensor"])
                    if prepared["sensor"] is not None
                    else {"ttc": float("nan"), "heights": []}
                )
                ended = time.perf_counter()
                value = finite_or_none(cast(float, result["ttc"]))
                measured = dict(
                    status="PREDICTED"
                    if value is not None
                    else "UNAVAILABLE_OR_NONFINITE_RETAINED",
                    query_id=row["query_id"],
                    sequence_id=row["sequence_id"],
                    anchor_us=row["anchor_us"],
                    ttc={**old["ttc"], "public_Garl_rgb_event_full": value},
                    heights=[finite_or_none(x) for x in cast(list[float], result["heights"])],
                    unavailable_reason=prepared["unavailable_reason"]
                    or ("native_nonfinite_ttc" if value is None else None),
                    metadata=prepared["metadata"],
                    event_tensor_sha256=event_hash,
                    sensor_sha256=hashlib.sha256(
                        np.ascontiguousarray(prepared["sensor"]).tobytes()
                    ).hexdigest()
                    if prepared["sensor"] is not None
                    else None,
                    seconds=dict(
                        prepare=prepared_at - started,
                        full_garl=ended - prepared_at,
                        total=ended - started,
                    ),
                    baseline_fragment_sha256=digest(baseline / "predictions" / path.name),
                    binding_sha256=binding_sha,
                    optimizer_updates=0,
                    finished_utc=stamp(),
                )
                atomic_json(path, measured)
                checksum.write_text(digest(path) + "\n", encoding="ascii")
            except Exception as error:
                failure = dict(
                    status="FAILED_PRESERVED",
                    query_id=row["query_id"],
                    binding_sha256=binding_sha,
                    baseline_fragment_sha256=digest(baseline / "predictions" / path.name),
                    error_type=type(error).__name__,
                    error=str(error),
                    checked_utc=stamp(),
                )
                atomic_json(output / "failures" / f"{i:05d}_{time.time_ns()}.json", failure)
                atomic_json(output / "STATE.json", failure)
                raise
            fresh += 1
            atomic_json(
                output / "STATE.json",
                dict(
                    status="RUNNING",
                    completed_queries=i + 1,
                    total_queries=len(rows),
                    fresh_queries=fresh,
                    elapsed_seconds=time.perf_counter() - begin,
                    checked_utc=stamp(),
                ),
            )
        fragments = {}
        for i, row in enumerate(rows):
            name = f"predictions/query_{i:05d}.json"
            path = output / name
            if not path.exists():
                atomic_json(
                    output / "STATE.json",
                    dict(status="PILOT_COMPLETE", fresh_queries=fresh, checked_utc=stamp()),
                )
                return
            saved = read(path)
            if (
                not path.with_suffix(".sha256").exists()
                or path.with_suffix(".sha256").read_text().strip() != digest(path)
                or saved["query_id"] != row["query_id"]
                or saved["binding_sha256"] != binding_sha
            ):
                raise ValueError("Cannot seal invalid full-Garl fragment")
            fragments[name] = digest(path)
        atomic_json(
            output / "PREDICTIONS_SEALED.json",
            dict(
                status="COMPLETE",
                queries=len(rows),
                manifest_sha256=binding["manifest_sha256"],
                binding_sha256=binding_sha,
                fragments=fragments,
                checked_utc=stamp(),
                optimizer_updates=0,
            ),
        )
        atomic_json(
            output / "STATE.json",
            dict(status="PREDICTIONS_COMPLETE", queries=len(rows), checked_utc=stamp()),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.output, args.baseline, args.code_root, args.device, args.limit)
