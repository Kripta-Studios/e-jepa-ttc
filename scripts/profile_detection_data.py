"""Profile real D4096 RGB decoding and target preparation with cache parity checks."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset
from e_jepa_ttc.detection.monodgp_eap_adapter import PreparedSample, prepare_sample


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def trace_tokens(tokens: list[str], *, seed: int, count: int) -> list[str]:
    """Select one deterministic prefix from a seeded permutation."""
    if count < 1 or count > len(tokens):
        raise ValueError("Trace count must be within the unique campaign population")
    if len(tokens) != len(set(tokens)):
        raise ValueError("Campaign tokens must be unique")
    result = list(tokens)
    random.Random(seed).shuffle(result)
    return result[:count]


def _update_array(digest: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes())


def prepared_checksum(
    *, raw_image: np.ndarray, member_sha256: str, prepared: PreparedSample
) -> str:
    """Hash image bytes, calibration, targets, native labels and member identity."""
    digest = hashlib.sha256()
    _update_array(digest, "raw_image", raw_image)
    digest.update(member_sha256.encode("ascii") + b"\0")
    for name, tensor in (
        ("image", prepared.image),
        ("calibration", prepared.calibration),
        ("image_size", prepared.image_size),
    ):
        _update_array(digest, name, tensor.detach().cpu().numpy())
    for name in sorted(prepared.target):
        _update_array(
            digest,
            "target:" + name,
            prepared.target[name].detach().cpu().numpy(),
        )
    encoded = json.dumps(
        prepared.native_ground_truth,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest.update(b"native_ground_truth\0" + encoded)
    return digest.hexdigest()


def _peak_working_set_bytes() -> int | None:
    """Return Windows peak resident memory without an optional dependency."""
    if os.name != "nt":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    return int(counters.PeakWorkingSetSize) if ok else None


def _run_pass(
    dataset: EAPRGBDataset, indexes: list[int]
) -> tuple[dict[str, str], float, dict[str, int]]:
    checksums: dict[str, str] = {}
    annotations = 0
    valid_velocity = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for index in indexes:
            sample = dataset[index]
            prepared = prepare_sample(sample, width=640, height=384, include_velocity=True)
            checksums[sample.sample_token] = prepared_checksum(
                raw_image=sample.image,
                member_sha256=sample.member_sha256,
                prepared=prepared,
            )
            annotations += len(sample.annotations)
            valid_velocity += int(prepared.target["velocity_valid"].sum().item())
    return (
        checksums,
        time.perf_counter() - start,
        {"raw_annotations": annotations, "valid_velocity_targets": valid_velocity},
    )


def worker(args: argparse.Namespace) -> int:
    """Run one cache configuration in a clean child process."""
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tokens = manifest["roles"]["train"]["sample_tokens"]
    trace = trace_tokens(tokens, seed=args.seed, count=args.count)
    indexes_by_token = {token: index for index, token in enumerate(tokens)}
    indexes = [indexes_by_token[token] for token in trace]
    dataset = EAPRGBDataset(
        args.data_root,
        tokens,
        max_open_shards=args.max_open_shards,
        max_label_sequences=args.label_cache,
    )
    try:
        cold_hashes, cold_seconds, cold_counts = _run_pass(dataset, indexes)
        warm_hashes, warm_seconds, warm_counts = _run_pass(dataset, indexes)
    finally:
        dataset.close()
    if cold_hashes != warm_hashes or cold_counts != warm_counts:
        raise RuntimeError("Repeated identical trace changed prepared samples")
    result = {
        "schema": "eap_detection_data_profile_worker_v2",
        "label_cache_sequences": args.label_cache,
        "max_open_shards": args.max_open_shards,
        "trace_count": len(trace),
        "trace_sha256": hashlib.sha256("\n".join(trace).encode("utf-8")).hexdigest(),
        "trace_tokens": trace,
        "cold_seconds": cold_seconds,
        "warm_repeat_seconds": warm_seconds,
        "counts": cold_counts,
        "sample_checksums": cold_hashes,
        "checksums_sha256": hashlib.sha256(
            json.dumps(cold_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "peak_working_set_bytes": _peak_working_set_bytes(),
    }
    args.worker_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def orchestrate(args: argparse.Namespace) -> int:
    """Run isolated workers and require exact prepared-sample parity."""
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.manifest_sha256 and sha256_file(args.manifest) != args.manifest_sha256:
        raise ValueError("Campaign manifest SHA-256 differs from the declared input")
    args.output_dir.mkdir(parents=True)
    workers = []
    for cache in (2, 24):
        output = args.output_dir / f"cache{cache}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--data-root",
            str(args.data_root),
            "--manifest",
            str(args.manifest),
            "--seed",
            str(args.seed),
            "--count",
            str(args.count),
            "--max-open-shards",
            str(args.max_open_shards),
            "--label-cache",
            str(cache),
            "--worker-output",
            str(output),
        ]
        subprocess.run(command, check=True)
        workers.append(json.loads(output.read_text(encoding="utf-8")))
    left, right = workers
    parity = {
        "same_trace": left["trace_tokens"] == right["trace_tokens"],
        "same_trace_sha256": left["trace_sha256"] == right["trace_sha256"],
        "same_per_frame_checksums": left["sample_checksums"] == right["sample_checksums"],
        "same_aggregate_checksum": left["checksums_sha256"] == right["checksums_sha256"],
        "same_counts": left["counts"] == right["counts"],
    }
    if not all(parity.values()):
        raise RuntimeError(f"Cache configurations changed prepared data: {parity}")
    summary = {
        "schema": "eap_detection_data_read_profile_v2",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "seed": args.seed,
        "trace_count": args.count,
        "ordering": "random.Random(seed).shuffle(D4096 manifest train tokens); fixed prefix",
        "max_open_shards": args.max_open_shards,
        "pipeline": "EAPRGBDataset.__getitem__ -> PNG decode -> prepare_sample(640x384, velocity)",
        "parity": parity,
        "workers": workers,
        "source": {
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
            "snapshot_aggregate_sha256": args.source_aggregate,
            "snapshot_sha256s_file_sha256": sha256_file(
                Path(__file__).resolve().parent.parent / "SHA256SUMS.txt"
            ),
        },
    }
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "schema": "eap_detection_data_read_profile_receipt_v2",
        "summary_sha256": sha256_file(path),
        "cache2_sha256": sha256_file(args.output_dir / "cache2.json"),
        "cache24_sha256": sha256_file(args.output_dir / "cache24.json"),
        "parity": "PASS",
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--max-open-shards", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-aggregate")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--label-cache", type=int)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.label_cache is None or args.worker_output is None:
            parser.error("worker mode needs --label-cache and --worker-output")
        return worker(args)
    if args.output_dir is None:
        parser.error("orchestrator mode needs --output-dir")
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
