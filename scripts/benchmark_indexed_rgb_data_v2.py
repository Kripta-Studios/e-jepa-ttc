"""Three-trial CPU benchmark of indexed TAR access over all v5 train sequences."""

# ruff: noqa: E402 -- direct execution from an uninstalled worktree is supported.

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import psutil
import torch
from benchmark_indexed_rgb_data import structural_sha256
from train_detection_campaign import PreparedCache

from e_jepa_ttc.detection.campaign_training import load_campaign_config, sha256_file
from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset
from e_jepa_ttc.detection.indexed_rgb_data import IndexedEAPRGBDataset


def _rank(namespace: str, token: str) -> str:
    return hashlib.sha256(f"{namespace}:{token}".encode()).hexdigest()


def deterministic_panel(
    train_tokens: list[str], final_media: dict[str, dict[str, object]]
) -> list[str]:
    """Choose one hash-ranked token per sequence, then hash-fill to 128."""
    by_sequence: dict[str, list[str]] = {}
    for token in train_tokens:
        sequence = str(final_media[token]["sequence_id"])
        by_sequence.setdefault(sequence, []).append(token)
    if len(by_sequence) != 24:
        raise ValueError(f"Expected exactly 24 train sequences, found {len(by_sequence)}")
    anchors = [
        min(tokens, key=lambda token: _rank(f"indexed-v2-anchor:{sequence}", token))
        for sequence, tokens in sorted(by_sequence.items())
    ]
    remaining = sorted(
        set(train_tokens) - set(anchors), key=lambda token: _rank("indexed-v2-fill", token)
    )
    panel = anchors + remaining[: 128 - len(anchors)]
    if len(panel) != 128 or len(set(panel)) != 128:
        raise ValueError("Deterministic v2 panel is not 128 unique tokens")
    if {str(final_media[token]["sequence_id"]) for token in panel} != set(by_sequence):
        raise ValueError("Deterministic v2 panel lost sequence coverage")
    return panel


def _measure_raw(dataset: object, count: int) -> tuple[float, list[str]]:
    started = time.perf_counter()
    hashes = [structural_sha256(dataset[index]) for index in range(count)]  # type: ignore[index]
    return time.perf_counter() - started, hashes


def _measure_prepared(
    dataset: EAPRGBDataset,
    config: object,
    member_hashes: dict[str, str],
    count: int,
) -> tuple[float, list[str]]:
    cache = PreparedCache(dataset, config, dict(member_hashes))  # type: ignore[arg-type]
    started = time.perf_counter()
    hashes = [structural_sha256(cache.get(index)) for index in range(count)]
    elapsed = time.perf_counter() - started
    del cache
    gc.collect()
    return elapsed, hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Benchmark output already exists: {args.output}")
    torch.set_num_threads(1)
    config_path = ROOT / "output/detection_campaign/rgb_screen_v5_seed7_v1/configs/r0.yaml"
    manifest_path = ROOT / "output/campaign_data/d4096_v5_readable/d4096_manifest.json"
    admission_path = ROOT / "output/campaign_data/d4096_v5_readable/media_admission.json"
    inventory_path = (
        ROOT
        / "output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_runs_v1/r0_8192"
        / "code/data/manifests/eap_detection_public_files_v1.json"
    )
    config = load_campaign_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_media = json.loads(admission_path.read_text(encoding="utf-8"))["final_media"]
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))["files"]
    tokens = deterministic_panel(list(manifest["roles"]["train"]["sample_tokens"]), final_media)
    member_hashes = {token: str(final_media[token]["member_sha256"]) for token in tokens}
    shard_paths = sorted({str(final_media[token]["rgb_shard_path"]) for token in tokens})
    shard_hashes = {path: str(inventory[path]["sha256"]) for path in shard_paths}

    started = time.perf_counter()
    old = EAPRGBDataset(
        config.data.root,
        tokens,
        max_open_shards=2,
        max_label_sequences=config.data.label_cache_sequences,
    )
    old_construct = time.perf_counter() - started
    started = time.perf_counter()
    indexed = IndexedEAPRGBDataset(
        config.data.root,
        tokens,
        member_sha256_by_token=member_hashes,
        shard_sha256_by_path=shard_hashes,
        max_open_shards=2,
        max_label_sequences=config.data.label_cache_sequences,
    )
    indexed_construct = time.perf_counter() - started

    # Shared untimed warm-up exercises labels, decode, preparation and direct seeking.
    warm_old_raw = structural_sha256(old[0])
    warm_indexed_raw = structural_sha256(indexed[0])
    warm_old_prepared = _measure_prepared(old, config, member_hashes, 1)[1][0]
    warm_indexed_prepared = _measure_prepared(indexed, config, member_hashes, 1)[1][0]
    if (warm_old_raw, warm_old_prepared) != (warm_indexed_raw, warm_indexed_prepared):
        raise ValueError("Warm-up parity failed")

    trials: list[dict[str, object]] = []
    reference_raw: list[str] | None = None
    reference_prepared: list[str] | None = None
    for trial in range(3):
        order = ("old", "indexed") if trial % 2 == 0 else ("indexed", "old")
        result: dict[str, object] = {"trial": trial + 1, "order": list(order)}
        hashes: dict[str, tuple[list[str], list[str]]] = {}
        for name in order:
            dataset = old if name == "old" else indexed
            raw_seconds, raw_hashes = _measure_raw(dataset, len(tokens))
            prepared_seconds, prepared_hashes = _measure_prepared(
                dataset, config, member_hashes, len(tokens)
            )
            result[name] = {
                "raw_seconds": raw_seconds,
                "prepared_seconds": prepared_seconds,
                "raw_sha256": structural_sha256(raw_hashes),
                "prepared_sha256": structural_sha256(prepared_hashes),
            }
            hashes[name] = raw_hashes, prepared_hashes
        if hashes["old"] != hashes["indexed"]:
            raise ValueError(f"Exact parity failed in trial {trial + 1}")
        if reference_raw is None:
            reference_raw, reference_prepared = hashes["old"]
        elif hashes["old"] != (reference_raw, reference_prepared):
            raise ValueError(f"Repeated-output stability failed in trial {trial + 1}")
        trials.append(result)

    open_handles = indexed.indexed_reader.open_handle_count
    old.close()
    indexed.close()
    memory = psutil.Process().memory_info()
    if getattr(memory, "peak_wset", memory.rss) >= int(1.2 * 1024**3):
        raise ValueError("Peak working set exceeded the preregistered 1.2 GiB bound")
    medians = {
        f"{name}_{stage}_seconds": statistics.median(
            float(trial[name][f"{stage}_seconds"])  # type: ignore[index]
            for trial in trials
        )
        for name in ("old", "indexed")
        for stage in ("raw", "prepared")
    }
    receipt = {
        "schema": "eap_indexed_rgb_benchmark_v2",
        "status": "PASS",
        "scope": "CPU RGB reader and preparation only; no detector/model throughput",
        "timing_interpretation": {
            "passes": "Raw and prepared timings are separate passes and are not additive.",
            "admission": "Indexed construction includes full shard SHA-256 and index build.",
            "cache_control": (
                "Trials follow shared warm-up and alternate order; not cold-cache timings."
            ),
            "claim": "Reader/preparation diagnostic only; no end-to-end campaign speedup claim.",
        },
        "gpu_used": False,
        "torch_num_threads": torch.get_num_threads(),
        "population": {
            "selection": "one SHA-ranked anchor per train sequence, then SHA-ranked fill",
            "samples": len(tokens),
            "sequences": len({str(final_media[token]["sequence_id"]) for token in tokens}),
            "shards": len(shard_paths),
            "sha256": structural_sha256(tokens),
        },
        "source_hashes": {
            "config": sha256_file(config_path),
            "manifest": sha256_file(manifest_path),
            "media_admission": sha256_file(admission_path),
            "frozen_inventory": sha256_file(inventory_path),
            "indexed_reader_implementation": sha256_file(
                ROOT / "src/e_jepa_ttc/detection/indexed_rgb_data.py"
            ),
            "benchmark_implementation": sha256_file(Path(__file__)),
        },
        "admission_seconds": {
            "old_metadata_construct": old_construct,
            "indexed_full_tar_hash_and_index_construct": indexed_construct,
            "note": "Indexed admission is a separate one-time cost, not reader throughput.",
        },
        "warmup": {"raw": "EXACT", "prepared": "EXACT"},
        "trials": trials,
        "median_seconds": medians,
        "median_speedup": {
            "raw": medians["old_raw_seconds"] / medians["indexed_raw_seconds"],
            "prepared": medians["old_prepared_seconds"] / medians["indexed_prepared_seconds"],
        },
        "parity": {
            "each_trial_raw": "EXACT",
            "each_trial_prepared": "EXACT",
            "cross_trial_stability": "EXACT",
        },
        "resources": {
            "max_open_handles": 2,
            "observed_open_handles": open_handles,
            "prepared_cache_capacity": config.data.cache_capacity,
            "prepared_cache_hits_per_unique_trial": 0,
            "rss_bytes_at_end": memory.rss,
            "peak_working_set_bytes": getattr(memory, "peak_wset", None),
            "peak_limit_bytes": int(1.2 * 1024**3),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
