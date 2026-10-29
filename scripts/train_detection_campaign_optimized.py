"""Run a fresh campaign with one receipt-bound indexed RGB store.

This entry point is intentionally incompatible with ordinary campaign
checkpoints: its performance sidecar and indexed sources are part of runtime
provenance. It never enables prefetching and does not migrate checkpoints.
"""

# ruff: noqa: E402 -- direct execution from an uninstalled worktree is supported.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_detection_campaign as ordinary

from e_jepa_ttc.detection.campaign_training import (
    CampaignConfig,
    load_campaign_config,
    sha256_file,
    validate_campaign_manifest,
    validate_manifest_against_metadata,
)
from e_jepa_ttc.detection.indexed_rgb_store import IndexedRGBStore
from e_jepa_ttc.detection.optimized_campaign import (
    PerformanceConfig,
    PhaseMetrics,
    exact_union,
    transactional_stop_due,
)


def _load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return raw


def _build_store(
    config: CampaignConfig, performance: PerformanceConfig
) -> tuple[IndexedRGBStore, dict[str, object]]:
    manifest = validate_campaign_manifest(config)
    metadata = validate_manifest_against_metadata(manifest, config)
    admitted = dict(metadata["media_admission"])
    member_hashes = dict(admitted["data_member_sha256"])
    receipt_path = Path(str(admitted["receipt_path"]))
    receipt = _load_json(receipt_path)
    final_media = receipt.get("final_media")
    if not isinstance(final_media, dict):
        raise ValueError("Media admission lacks final_media ownership")
    full = _load_json(Path(config.data.full_selection_tokens))
    train = tuple(manifest["roles"][config.data.train_role]["sample_tokens"])
    selection = tuple(manifest["roles"][config.data.selection_role]["sample_tokens"])
    pilot = tuple(manifest["historical_panels"]["pilot_selection256"]["sample_tokens"])
    full_tokens = tuple(full["sample_tokens"])
    union = exact_union(train, selection, pilot, full_tokens)
    if set(union) != set(final_media) or set(union) != set(member_hashes):
        raise ValueError("Shared store population differs from admitted campaign union")
    ownership: dict[str, tuple[str, str, str]] = {}
    for token in union:
        row = final_media[token]
        ownership[token] = (
            str(row["rgb_shard_path"]),
            str(row["rgb_member_path"]),
            str(row["member_sha256"]),
        )
        if ownership[token][2] != member_hashes[token]:
            raise ValueError(f"Media hash ledgers disagree: {token}")
    inventory_path = ROOT / "data/manifests/eap_detection_public_files_v1.json"
    inventory = _load_json(inventory_path)["files"]
    shards = {value[0] for value in ownership.values()}
    shard_hashes = {path: str(inventory[path]["sha256"]) for path in shards}
    store = IndexedRGBStore(
        config.data.root,
        ownership,
        shard_hashes,
        max_open_handles=performance.max_open_handles,
    )
    return store, {
        "schema": "eap_detection_indexed_runtime_v1",
        "performance_config_sha256": performance.sha256,
        "indexed_reader_sha256": performance.expected_indexed_reader_sha256,
        "indexed_store_sha256": performance.expected_indexed_store_sha256,
        "union_count": len(union),
        "union_sha256": ordinary.canonical_sha256(list(union)),
        "prefetch_batches": 0,
    }


def run(
    config_path: Path,
    performance_path: Path,
    output: Path,
    resume: Path | None = None,
    stop_after_update: int | None = None,
) -> dict[str, object]:
    """Run with one process-lifetime store and explicit injected dataset views."""
    performance, performance_receipt = PerformanceConfig.load(performance_path)
    reader_path = ROOT / "src/e_jepa_ttc/detection/indexed_rgb_data.py"
    store_path = ROOT / "src/e_jepa_ttc/detection/indexed_rgb_store.py"
    optimized_contract_path = ROOT / "src/e_jepa_ttc/detection/optimized_campaign.py"
    if sha256_file(reader_path) != performance.expected_indexed_reader_sha256:
        raise ValueError("Indexed reader source differs from performance sidecar")
    if sha256_file(store_path) != performance.expected_indexed_store_sha256:
        raise ValueError("Indexed store source differs from performance sidecar")
    config = load_campaign_config(config_path)
    store, extension = _build_store(config, performance)
    extension["performance_config"] = performance_receipt
    phase_metrics = PhaseMetrics(performance.instrument_phases)
    phase_metrics.add("tar_index", float(store.metrics()["index_build_seconds"]))

    def dataset_factory(
        root: str | Path, tokens: list[str] | tuple[str, ...], actual: CampaignConfig
    ) -> Any:
        if Path(root).resolve() != store.root or actual.sha256 != config.sha256:
            raise ValueError("Injected dataset request differs from shared-store campaign")
        return store.dataset_view(
            tokens,
            max_label_sequences=actual.data.label_cache_sequences,
        )

    try:
        def performance_metrics() -> dict[str, object]:
            store_metrics = store.metrics()
            phases: dict[str, object] = dict(phase_metrics.to_dict())
            phases["payload_read"] = {
                "count": store_metrics["read_count"],
                "total_seconds": store_metrics["read_seconds"],
                "bytes": store_metrics["read_bytes"],
            }
            phases["png_decode"] = {
                **phase_metrics.to_dict()["png_decode"],
                "availability": "included_in_prepare_not_separately_instrumented_v1",
            }
            return {"store": store_metrics, "phases": phases, "prefetch_batches": 0}

        return ordinary.run(
            config_path,
            output,
            resume,
            stop_after_update,
            dataset_factory=dataset_factory,
            runtime_extension=extension,
            extra_owned_paths=(
                Path(__file__),
                reader_path,
                store_path,
                optimized_contract_path,
                performance_path,
            ),
            performance_metrics=performance_metrics,
            performance_observer=phase_metrics.add,
            stop_commit_validator=transactional_stop_due,
        )
    finally:
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--performance-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-update", type=int)
    args = parser.parse_args()
    run(args.config, args.performance_config, args.output, args.resume, args.stop_after_update)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
