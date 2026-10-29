"""Benchmark receipt-bound indexed TAR reads against the unchanged eAP reader."""

# ruff: noqa: E402 -- direct execution from an uninstalled worktree is supported.

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import psutil
import torch
from train_detection_campaign import PreparedCache

from e_jepa_ttc.detection.campaign_training import load_campaign_config, sha256_file
from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset
from e_jepa_ttc.detection.indexed_rgb_data import IndexedEAPRGBDataset


def _frame(tag: bytes, payload: bytes) -> bytes:
    return len(tag).to_bytes(4, "big") + tag + len(payload).to_bytes(8, "big") + payload


def canonical_structural_bytes(value: object) -> bytes:
    """Encode supported nested structures with explicit domains and boundaries."""
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
        metadata = _frame(b"dtype", str(array.dtype).encode()) + _frame(
            b"shape", repr(array.shape).encode()
        )
        return _frame(b"tensor", metadata + _frame(b"data", array.tobytes()))
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        metadata = _frame(b"dtype", str(array.dtype).encode()) + _frame(
            b"shape", repr(array.shape).encode()
        )
        if array.dtype.hasobject:
            payload = b"".join(
                _frame(b"item", canonical_structural_bytes(item)) for item in array.flat
            )
        else:
            payload = _frame(b"data", array.tobytes())
        return _frame(b"ndarray", metadata + payload)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        payload = _frame(b"class", type(value).__qualname__.encode())
        for field in dataclasses.fields(value):
            payload += _frame(b"field-name", field.name.encode())
            payload += _frame(
                b"field-value", canonical_structural_bytes(getattr(value, field.name))
            )
        return _frame(b"dataclass", payload)
    if isinstance(value, Mapping):
        encoded = sorted(
            (canonical_structural_bytes(key), canonical_structural_bytes(item))
            for key, item in value.items()
        )
        payload = len(encoded).to_bytes(8, "big") + b"".join(
            _frame(b"key", key) + _frame(b"value", item) for key, item in encoded
        )
        return _frame(b"mapping", payload)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        payload = len(value).to_bytes(8, "big") + b"".join(
            _frame(b"item", canonical_structural_bytes(item)) for item in value
        )
        return _frame(type(value).__qualname__.encode(), payload)
    if isinstance(value, bytes):
        return _frame(b"bytes", value)
    scalar_type = f"{type(value).__module__}.{type(value).__qualname__}".encode()
    return _frame(b"scalar", _frame(b"type", scalar_type) + _frame(b"repr", repr(value).encode()))


def structural_sha256(value: object) -> str:
    """Hash nested tensors, arrays, dataclasses and metadata without lossy conversion."""
    return hashlib.sha256(canonical_structural_bytes(value)).hexdigest()


def _load_tokens(path: Path) -> tuple[list[str], str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    selected = [row for row in rows if 2041 <= row.get("update", -1) <= 2048]
    if [row["update"] for row in selected] != list(range(2041, 2049)):
        raise ValueError("Benchmark panel must be exact completed updates 2041-2048")
    tokens = [str(token) for row in selected for token in row["sample_tokens"]]
    if len(tokens) != 32 or len(set(tokens)) != 32:
        raise ValueError("Benchmark panel must contain 32 unique samples")
    return tokens, structural_sha256(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Benchmark output already exists: {args.output}")
    torch.set_num_threads(1)
    run = ROOT / "output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_runs_v1/r0_8192"
    metrics = run / "metrics.jsonl"
    config_path = ROOT / "output/detection_campaign/rgb_screen_v5_seed7_v1/configs/r0.yaml"
    admission_path = ROOT / "output/campaign_data/d4096_v5_readable/media_admission.json"
    inventory_path = run / "code/data/manifests/eap_detection_public_files_v1.json"
    tokens, metric_rows_sha256 = _load_tokens(metrics)
    config = load_campaign_config(config_path)
    admission = json.loads(admission_path.read_text(encoding="utf-8"))["final_media"]
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))["files"]
    member_hashes = {token: admission[token]["member_sha256"] for token in tokens}
    shard_paths = sorted({admission[token]["rgb_shard_path"] for token in tokens})
    shard_hashes = {path: inventory[path]["sha256"] for path in shard_paths}

    started = time.perf_counter()
    old = EAPRGBDataset(
        config.data.root,
        tokens,
        max_open_shards=2,
        max_label_sequences=config.data.label_cache_sequences,
    )
    old_construct = time.perf_counter() - started
    started = time.perf_counter()
    old_raw_hashes = [structural_sha256(old[index]) for index in range(32)]
    old_raw = time.perf_counter() - started
    old_cache = PreparedCache(old, config, dict(member_hashes))
    started = time.perf_counter()
    old_prepared_hashes = [structural_sha256(old_cache.get(index)) for index in range(32)]
    old_prepared = time.perf_counter() - started
    old.close()

    started = time.perf_counter()
    indexed = IndexedEAPRGBDataset(
        config.data.root,
        tokens,
        member_sha256_by_token=member_hashes,
        shard_sha256_by_path=shard_hashes,
        max_open_shards=2,
        max_label_sequences=config.data.label_cache_sequences,
    )
    index_construct = time.perf_counter() - started
    started = time.perf_counter()
    indexed_raw_hashes = [structural_sha256(indexed[index]) for index in range(32)]
    indexed_raw = time.perf_counter() - started
    indexed_cache = PreparedCache(indexed, config, dict(member_hashes))
    started = time.perf_counter()
    indexed_prepared_hashes = [structural_sha256(indexed_cache.get(index)) for index in range(32)]
    indexed_prepared = time.perf_counter() - started
    open_handles = indexed.indexed_reader.open_handle_count
    indexed.close()
    raw_mismatches = [
        index
        for index, pair in enumerate(zip(old_raw_hashes, indexed_raw_hashes, strict=True))
        if pair[0] != pair[1]
    ]
    prepared_mismatches = [
        index
        for index, pair in enumerate(zip(old_prepared_hashes, indexed_prepared_hashes, strict=True))
        if pair[0] != pair[1]
    ]
    if raw_mismatches or prepared_mismatches:
        raise ValueError(
            f"Indexed reader differs from unchanged reader: raw={raw_mismatches}, "
            f"prepared={prepared_mismatches}"
        )
    memory = psutil.Process().memory_info()
    receipt = {
        "schema": "eap_indexed_rgb_benchmark_v1",
        "status": "PASS",
        "scope": "CPU RGB reader and preparation only; no detector/model throughput",
        "timing_interpretation": {
            "passes": "Raw and prepared timings are separate passes and are not additive.",
            "admission": "Indexed construction includes full shard SHA-256 and index build.",
            "cache_control": "Fixed old-then-indexed order is not cold-cache-controlled.",
            "claim": "Diagnostic only; no end-to-end or campaign speedup claim.",
        },
        "gpu_used": False,
        "torch_num_threads": torch.get_num_threads(),
        "population": {"updates": [2041, 2048], "samples": 32, "shards": len(shard_paths)},
        "source_hashes": {
            "metrics_updates_2041_2048": metric_rows_sha256,
            "config": sha256_file(config_path),
            "media_admission": sha256_file(admission_path),
            "frozen_inventory": sha256_file(inventory_path),
            "indexed_reader_implementation": sha256_file(
                ROOT / "src/e_jepa_ttc/detection/indexed_rgb_data.py"
            ),
            "benchmark_implementation": sha256_file(Path(__file__)),
        },
        "population_sha256": structural_sha256(tokens),
        "parity": {
            "raw_samples": "EXACT",
            "prepared_samples": "EXACT",
            "raw_hashes_sha256": structural_sha256(old_raw_hashes),
            "prepared_hashes_sha256": structural_sha256(old_prepared_hashes),
        },
        "seconds": {
            "old_construct": old_construct,
            "old_raw_32": old_raw,
            "old_prepared_32": old_prepared,
            "indexed_construct_and_source_hash": index_construct,
            "indexed_raw_32": indexed_raw,
            "indexed_prepared_32": indexed_prepared,
        },
        "resources": {
            "max_open_handles": 2,
            "observed_open_handles": open_handles,
            "prepared_cache_capacity": config.data.cache_capacity,
            "prepared_cache_hits_for_unique_panel": 0,
            "rss_bytes_at_end": memory.rss,
            "peak_working_set_bytes": getattr(memory, "peak_wset", None),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
