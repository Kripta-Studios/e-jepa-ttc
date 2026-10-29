"""Run one hash-bound CPU probe of the S4 causal event representation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pyarrow.parquet as pq

from e_jepa_ttc.detection.event_data import CausalEventConfig, CausalEventVoxelizer

DEFAULT_SAMPLE_TOKEN = "qGsgzl4Q8B:000001037_000033536"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _worker(args: argparse.Namespace) -> int:
    process = psutil.Process()
    baseline_rss = int(process.memory_info().rss)
    samples = [baseline_rss]
    stop_sampling = threading.Event()

    def sample_self() -> None:
        while not stop_sampling.wait(0.01):
            try:
                samples.append(int(process.memory_info().rss))
            except psutil.Error:
                return

    sampler = threading.Thread(target=sample_self, name="rss-sampler", daemon=True)
    sampler.start()
    print(json.dumps({"kind": "ready", "rss_bytes": baseline_rss}), flush=True)
    started = time.perf_counter()
    config = CausalEventConfig(window_ms=50)
    with CausalEventVoxelizer(args.root, config) as voxelizer:
        result = voxelizer.encode(
            args.events_path,
            sequence_id=args.sequence_id,
            rgb_exposure_end_us=args.cutoff_us,
        )
    samples.append(int(process.memory_info().rss))
    stop_sampling.set()
    sampler.join()
    peak_rss = max(samples)
    payload = {
        "kind": "result",
        "elapsed_cpu_wall_seconds": time.perf_counter() - started,
        "receipt": result.receipt,
        "voxel_finite": bool(np.isfinite(result.voxel).all()),
        "voxel_nonzero_count": int(np.count_nonzero(result.voxel)),
        "resource_sampling": {
            "method": "worker_self_rss_thread",
            "baseline_rss_bytes": baseline_rss,
            "sampled_peak_rss_bytes": peak_rss,
            "sampled_peak_delta_bytes": peak_rss - baseline_rss,
            "sample_interval_ms": 10,
            "sample_count": len(samples),
        },
    }
    print(json.dumps(payload, allow_nan=False), flush=True)
    return 0


def _canonical_row(metadata_path: Path, sample_token: str) -> dict[str, Any]:
    columns = [
        "sample_token",
        "sequence_id",
        "events_path",
        "rgb_exposure_start_timestamp_us",
        "rgb_exposure_end_timestamp_us",
    ]
    table = pq.read_table(
        metadata_path,
        columns=columns,
        filters=[("sample_token", "=", sample_token)],
    )
    if table.num_rows != 1:
        raise ValueError(
            f"Expected exactly one canonical row for {sample_token}, got {table.num_rows}"
        )
    return {name: table[name][0].as_py() for name in columns}


def _run_child(
    *, root: Path, events_path: str, sequence_id: str, cutoff_us: int
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--root",
        str(root),
        "--events-path",
        events_path,
        "--sequence-id",
        sequence_id,
        "--cutoff-us",
        str(cutoff_us),
    ]
    child = subprocess.Popen(  # noqa: S603 - exact current interpreter and script
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if child.stdout is None:
        raise RuntimeError("Probe child stdout pipe was not created")
    ready_line = child.stdout.readline()
    try:
        ready = json.loads(ready_line)
        baseline = int(ready["rss_bytes"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        child.kill()
        raise RuntimeError(
            f"Probe child did not emit a valid ready receipt: {ready_line!r}"
        ) from exc
    remainder, stderr = child.communicate()
    if child.returncode != 0:
        raise RuntimeError(f"Probe child failed ({child.returncode}): {stderr.strip()}")
    lines = [line for line in remainder.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"Probe child emitted unexpected output: {remainder!r}")
    result = json.loads(lines[0])
    sampling = result.get("resource_sampling")
    if not isinstance(sampling, dict) or int(sampling.get("baseline_rss_bytes", -1)) != baseline:
        raise RuntimeError("Probe child resource receipt does not match its ready baseline")
    return result


def _main_probe(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing receipt: {output}")
    metadata_path = root / "data/train.parquet"
    public_manifest_path = args.public_manifest.resolve()
    public_manifest = json.loads(public_manifest_path.read_text(encoding="utf-8"))
    metadata_member = public_manifest["files"]["data/train.parquet"]
    metadata_sha = _sha256(metadata_path)
    if metadata_sha != metadata_member["sha256"]:
        raise ValueError("Local train metadata does not match the public-file manifest")
    row = _canonical_row(metadata_path, args.sample_token)
    events_relative = str(row["events_path"])
    events_path = (root / events_relative).resolve()
    if not events_path.is_relative_to(root) or not events_path.is_file():
        raise ValueError("Canonical event member is absent or escapes the dataset root")
    member = public_manifest["files"].get(events_relative)
    if not isinstance(member, dict) or not member.get("sha256"):
        raise ValueError(f"Public-file manifest lacks a hash for {events_relative}")
    observed_size = events_path.stat().st_size
    if observed_size != int(member["size_bytes"]):
        raise ValueError("Local event member size does not match the public-file manifest")
    child_result = _run_child(
        root=root,
        events_path=events_relative,
        sequence_id=str(row["sequence_id"]),
        cutoff_us=int(row["rgb_exposure_end_timestamp_us"]),
    )
    receipt = {
        "schema": "eap_causal_event_voxel_probe_v3",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "scope": "one CPU 50 ms window; sampled reads; no labels/test/GPU",
        "canonical_sample": {
            "sample_token": str(row["sample_token"]),
            "sequence_id": str(row["sequence_id"]),
            "rgb_exposure_start_timestamp_us": int(row["rgb_exposure_start_timestamp_us"]),
            "rgb_exposure_end_timestamp_us": int(row["rgb_exposure_end_timestamp_us"]),
            "events_path": events_relative,
        },
        "provenance": {
            "train_metadata_sha256_observed": metadata_sha,
            "train_metadata_sha256_expected": metadata_member["sha256"],
            "public_file_manifest_sha256": _sha256(public_manifest_path),
            "event_h5_expected_sha256_from_public_manifest": member["sha256"],
            "event_h5_expected_size_bytes": int(member["size_bytes"]),
            "event_h5_observed_size_bytes": observed_size,
            "event_h5_full_sha256_observed": None,
            "event_h5_verification_scope": (
                "expected public hash recorded; local size verified; this bounded probe did not "
                "read the full 10.2 GB member and therefore does not certify its local full hash"
            ),
            "implementation_module_sha256": _sha256(
                Path(__file__).parents[1] / "src/e_jepa_ttc/detection/event_data.py"
            ),
            "entrypoint_sha256": _sha256(Path(__file__)),
        },
        "resource_sampling": child_result.pop("resource_sampling"),
        "probe": {key: value for key, value in child_result.items() if key != "kind"},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": _sha256(output)}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--public-manifest",
        type=Path,
        default=Path("data/manifests/eap_detection_public_files_v1.json"),
    )
    parser.add_argument("--sample-token", default=DEFAULT_SAMPLE_TOKEN)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--events-path", help=argparse.SUPPRESS)
    parser.add_argument("--sequence-id", help=argparse.SUPPRESS)
    parser.add_argument("--cutoff-us", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.events_path is None or args.sequence_id is None or args.cutoff_us is None:
            parser.error("Internal worker arguments are incomplete")
        return _worker(args)
    if args.output is None:
        parser.error("--output is required")
    return _main_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
