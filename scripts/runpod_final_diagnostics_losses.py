"""Recover stored training losses from verified CPU checkpoints, never execute a model."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract(candidate: Path, output: Path) -> dict[str, Any]:
    """Export update indices only when the stored loss cursor covers every committed update."""
    paths = json.loads((candidate / "PORTABLE_PATHS.json").read_text())["components"]
    components = {
        name: value for name, value in paths.items() if Path(value["local_path"]).suffix == ".pt"
    }
    for name, value in components.items():
        if not Path(value["local_path"]).is_file():
            value["local_path"] = str(candidate / "checkpoints" / f"{name}.pt")
    expected = {name: value["sha256"] for name, value in sorted(components.items())}
    receipt_path = output / "TRAINING_LOSS_EXTRACTION.json"
    csv_path = output / "TRAINING_LOSS_BY_UPDATE.csv"
    if receipt_path.is_file() and csv_path.is_file():
        existing = json.loads(receipt_path.read_text())
        if existing.get("checkpoint_bindings") == expected and existing.get("csv_sha256") == digest(
            csv_path
        ):
            if all(
                existing["source_stats"][name]
                == {
                    "size_bytes": Path(value["local_path"]).stat().st_size,
                    "mtime_ns": Path(value["local_path"]).stat().st_mtime_ns,
                }
                for name, value in components.items()
            ):
                return existing
    import torch

    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    records, source_stats = {}, {}
    total = 0
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "component",
                "recorded_loss_index",
                "scientific_update",
                "loss",
                "checkpoint_sha256",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for name, value in sorted(components.items()):
            path = Path(value["local_path"])
            if digest(path) != value["sha256"]:
                raise ValueError(f"Preserved checkpoint differs before loss extraction: {name}")
            stat = path.stat()
            source_stats[name] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            payload = torch.load(path, map_location="cpu", weights_only=False)
            cursor = payload.get("cursor", {})
            completed = payload.get("completed_updates", payload.get("committed_updates"))
            losses = cursor.get("losses")
            record = {
                "checkpoint_sha256": value["sha256"],
                "completed_updates": completed,
                "cursor_keys": sorted(cursor),
                "loss_curve_present": isinstance(losses, list),
                "recorded_loss_count": len(losses) if isinstance(losses, list) else 0,
                "cursor_epoch": cursor.get("epoch"),
                "cursor_examples": cursor.get("examples"),
                "cursor_loss_sums": cursor.get("loss_sums"),
                "cursor_pending_curve": cursor.get("pending_curve"),
            }
            if isinstance(losses, list):
                aligned = len(losses) == completed
                record["scientific_update_indices_verified"] = aligned
                for index, loss in enumerate(losses, 1):
                    scalar = float(loss)
                    if not math.isfinite(scalar):
                        raise ValueError(f"Stored loss contains nonfinite value: {name}/{index}")
                    writer.writerow(
                        {
                            "component": name,
                            "recorded_loss_index": index,
                            "scientific_update": index if aligned else None,
                            "loss": scalar,
                            "checkpoint_sha256": value["sha256"],
                        }
                    )
                    total += 1
            else:
                record["missing_reason"] = (
                    "No cursor.losses; epoch aggregates cannot reconstruct the missing curve."
                )
            records[name] = record
            del payload, cursor, losses
            gc.collect()
    receipt = {
        "schema": "ejepa_stored_training_loss_extraction_v1",
        "checkpoint_bindings": expected,
        "source_stats": source_stats,
        "components": records,
        "loss_rows": total,
        "csv_sha256": digest(csv_path),
        "model_forward_calls": 0,
        "optimizer_updates": 0,
        "loaded_device": "cpu",
        "torch_version_used_for_reading": str(torch.__version__),
        "scope": "Stored training loss, not validation/test error or a rerun.",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt
