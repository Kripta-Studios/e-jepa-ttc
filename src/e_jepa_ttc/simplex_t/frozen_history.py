"""Resolve delivery history pools from verified scientific sources and freeze pins."""

from __future__ import annotations

from pathlib import Path

from .campaign_sources import CampaignSources
from .history_bundle import HistoryPoolPins


def frozen_history_pools(
    sources: CampaignSources, record: dict, *, roots: dict[str, Path]
) -> dict[str, HistoryPoolPins]:
    """Select exact configured history manifests, requiring their pre-fit freeze pins.

    Caller has verified the actual freeze and constructed acknowledged sources.
    No metadata payloads, checkpoints or media are opened here. Missing pins
    require a complete pre-fit freeze, never hashes filled in after results.
    """
    work = roots["work"].resolve(strict=True)

    def digest(path: Path) -> str:
        target = path.resolve(strict=True)
        if not target.is_relative_to(work):
            raise ValueError("history source outside companion worktree")
        matches = [
            pin["sha256"]
            for pin in record["files"]
            if pin["root"] in roots
            and (roots[pin["root"]] / pin["relative_path"]).resolve(strict=True) == target
        ]
        if len(matches) != 1:
            raise ValueError("history manifest needs one exact pre-fit freeze pin")
        return matches[0]

    def bind(index: Path, dedup: Path) -> HistoryPoolPins:
        return HistoryPoolPins(index, digest(index), dedup, digest(dedup))

    pools = {
        "D0": bind(
            sources.index_root / "INDEX_MANIFEST.json", sources.dedup_root / "DEDUP_MANIFEST.json"
        )
    }
    flags = record["source_contract"]["availability"]
    for pool, enabled, folds in (
        ("D1", flags["d1"], sources.expansion_folds),
        ("DENSE_OLD", flags["density"], sources.dense_folds),
    ):
        if not enabled:
            continue
        if set(folds) != {0, 1, 2}:
            raise ValueError("all frozen history folds required")
        first = folds[0]
        for fold, row in folds.items():
            if (
                row.index_manifest.resolve() != first.index_manifest.resolve()
                or row.index_manifest_sha256 != first.index_manifest_sha256
                or row.dedup.resolve() != first.dedup.parent.resolve() / f"outer{fold}.npz"
            ):
                raise ValueError("history folds use incompatible indices or dedup roots")
        binding = bind(first.index_manifest, first.dedup.parent / "DEDUP_MANIFEST.json")
        if binding.index_sha256 != first.index_manifest_sha256:
            raise ValueError("history index differs from frozen source binding")
        pools[pool] = binding
    return pools
