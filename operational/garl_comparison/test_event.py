"""Run the pinned public event-only Garl checkpoint on official inputs, using CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from filelock import FileLock

from e_jepa_ttc.efficient_context.garl_input import inference_record
from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool
from operational.sota_evidence.codabench import package_predictions
from operational.sota_evidence.run import digest, read
from operational.sota_evidence.submission import build_submission
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.garl_predictions import native_model, release_ttc

COLUMNS = ["sample_token", "sequence_id", "events_path", "event_windows_us", "boxes_xyxy"]


def check_row(row: dict[str, Any]) -> None:
    """Require the explicit official test path and exactly two sensor endpoints."""
    if row["events_path"] != f"data/test/{row['sequence_id']}/events.h5":
        raise ValueError("Expected an official test event path")
    if np.asarray(np.asarray(row["event_windows_us"]).tolist()).shape != (2, 2):
        raise ValueError("Expected two native event windows")
    if np.asarray(np.asarray(row["boxes_xyxy"]).tolist()).shape != (2, 4):
        raise ValueError("Expected two native boxes")


def run(
    inputs: Path,
    raw_root: Path,
    campaign: Path,
    code_root: Path,
    output: Path,
    verified_media: Path,
    maximum_queries: int | None = None,
) -> None:
    """Preserve native TTC and failures; package only complete finite token coverage."""
    output.mkdir(parents=True, exist_ok=True)
    frozen = read(campaign / "DELIVERY_FREEZE.json")
    checkpoint = campaign / "public_garl/paper_event_only_lhr.pth"
    configuration = campaign / "public_garl/configs/ablation/event_lhr.yaml"
    if digest(checkpoint) != frozen["public_checkpoint_sha256"]:
        raise ValueError("Public checkpoint bytes changed")
    if digest(configuration) != frozen["config_sha256"]:
        raise ValueError("Public native configuration changed")
    for item in frozen["native_source_files"]:
        if digest(code_root / item["path"]) != item["sha256"]:
            raise ValueError("Native Garl source changed")
    frame = pd.read_parquet(inputs, columns=COLUMNS)
    plan = read(verified_media / "DOWNLOAD_PLAN.json")
    pins = {item["path"]: item for item in plan["files"]}
    if len(frame) != 6762 or frame.sample_token.duplicated().any():
        raise ValueError("Expected complete unique official test12 inputs")
    binding = {
        "checkpoint_sha256": digest(checkpoint),
        "configuration_sha256": digest(configuration),
        "inputs_sha256": digest(inputs),
        "source_sha256": digest(Path(__file__)),
        "native_sources": frozen["native_source_files"],
        "device": "cpu",
        "threads": 2,
        "precision": "float32",
        "native_delta_s": 0.1,
        "clipping": False,
        "input": "native two-endpoint 40-plane sensor ROI representation",
        "raw_root": str(raw_root.resolve()),
        "test_labels_read": False,
        "optimizer_updates": 0,
        "raw_plan_sha256": digest(verified_media / "DOWNLOAD_PLAN.json"),
        "input_helper_sha256": digest(Path(inference_record.__code__.co_filename)),
    }
    contract = output / "BINDING.json"
    if contract.exists() and read(contract) != binding:
        raise ValueError("Preserve the existing Garl inference identity")
    atomic_json(contract, binding)
    identity = digest(contract)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = native_model(campaign, code_root)
    if float(model.dT) != binding["native_delta_s"]:
        raise ValueError("Native model delta differs")
    pool = ReaderPool()
    fragments = output / "predictions"
    fragments.mkdir(exist_ok=True)
    records = []
    verified_sequences = set()
    started = time.monotonic()
    with torch.inference_mode():
        for number, row in enumerate(frame.to_dict("records")):
            if maximum_queries is not None and number >= maximum_queries:
                break
            check_row(row)
            if row["sequence_id"] not in verified_sequences:
                path = (raw_root / row["events_path"]).resolve()
                pin = pins[row["events_path"]]
                stat = path.stat()
                if stat.st_size != pin["size"]:
                    raise ValueError("Native event source size differs from pinned release")
                raw_binding = {
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": pin["lfs"]["oid"],
                    "path": str(path),
                }
                receipt = verified_media / "raw" / f"{row['sequence_id']}.json"
                if receipt.exists():
                    if read(receipt) != raw_binding:
                        raise ValueError("Verified native event source changed")
                else:
                    if digest(path) != pin["lfs"]["oid"]:
                        raise ValueError("Native event source failed SHA256 verification")
                    atomic_json(receipt, raw_binding)
                verified_sequences.add(row["sequence_id"])
            destination = fragments / f"query_{number:05d}.json"
            if destination.exists():
                record = read(destination)
                if (
                    record["sample_token"] != row["sample_token"]
                    or record["binding_sha256"] != identity
                ):
                    raise ValueError("Prediction identity changed")
            else:
                tick = time.monotonic()
                sensor = inference_record(row, pool, raw_root / "data/test")
                prepared = time.monotonic()
                heights_tensor, _ = model.forward_test(sensor[None])
                heights = heights_tensor.float().cpu().numpy()
                if heights.shape != (1, 2) or not np.isfinite(heights).all():
                    raise ValueError("Invalid native height output")
                prediction = float(release_ttc(heights, float(model.dT))[0])
                record = {
                    "sample_token": row["sample_token"],
                    "sequence_id": row["sequence_id"],
                    "prediction": prediction if np.isfinite(prediction) else None,
                    "status": "FINITE" if np.isfinite(prediction) else "NATIVE_NONFINITE_TTC",
                    "heights": heights[0].tolist(),
                    "binding_sha256": identity,
                    "input_sha256": hashlib.sha256(sensor.numpy().tobytes()).hexdigest(),
                    "cpu_preparation_s": prepared - tick,
                    "cpu_inference_s": time.monotonic() - prepared,
                }
                atomic_json(destination, record)
            records.append(record)
            atomic_json(
                output / "PROGRESS.json",
                {
                    "status": "RUNNING",
                    "completed": len(records),
                    "total": len(frame),
                    "seconds": time.monotonic() - started,
                    "utc": datetime.now(UTC).isoformat(),
                    "gpu_seconds": 0,
                },
            )
    pool.close()
    result = {
        "status": ("PILOT_COMPLETE" if len(records) <= 3 else "PARTIAL_COMPLETE")
        if len(records) != len(frame)
        else "PREDICTED",
        "completed": len(records),
        "total": len(frame),
        "gpu_seconds": 0,
        "cpu_wall_seconds": time.monotonic() - started,
        "official_score_available": False,
    }
    if len(records) == len(frame):
        predictions = pd.DataFrame(records)
        predictions.to_csv(output / "TEST_PREDICTIONS.csv", index=False)
        invalid = int(predictions.prediction.isna().sum())
        if invalid:
            result.update(status="UNSUBMITTABLE_NATIVE_NONFINITE", invalid_predictions=invalid)
        else:
            archive = output / "Garl_event_only_submission.zip"
            if not archive.exists():
                package_predictions(frame.sample_token.tolist(), predictions, checkpoint, archive)
            expected_payload = build_submission(
                frame.sample_token.tolist(), predictions, checkpoint_sha256=digest(checkpoint)
            )
            with zipfile.ZipFile(archive) as stored:
                if stored.namelist() != ["submission.json"] or stored.testzip() is not None:
                    raise ValueError("Existing Garl submission structure or CRC differs")
                if json.loads(stored.read("submission.json")) != expected_payload:
                    raise ValueError("Existing Garl submission differs from the frozen predictions")
            result.update(status="PACKAGED_NOT_SUBMITTED", zip_sha256=digest(archive))
    atomic_json(output / "RESULT.json", result)
    print(json.dumps(result), flush=True)


def main() -> None:
    """Run native CPU inference without competing for the reserved GPU budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "raw-root", "campaign", "code-root", "output", "verified-media"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--maximum-queries", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / "INFERENCE.lock"), timeout=10800):
        run(
            args.inputs,
            args.raw_root,
            args.campaign,
            args.code_root,
            args.output,
            args.verified_media,
            args.maximum_queries,
        )


if __name__ == "__main__":
    main()
