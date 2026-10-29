"""Regression tests for the complete eAP to EvTTC orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_eap_evttc_complete import _aggregate_complete

VARIANTS = ("A0_MATCHED_GLOBAL", "A1_MATCHED_DENSE_BLOCK")


def _write_aggregate(path: Path, pairs: list[tuple[int, int]]) -> None:
    rows = []
    for variant in VARIANTS:
        rows.append(
            {
                "variant": variant,
                "complete_for_final_selection": True,
                "run_count": len(pairs),
                "required_run_count": len(pairs),
                "runs": [{"fold": fold, "seed": seed} for fold, seed in pairs],
            }
        )
    path.write_text(
        json.dumps({"all_variants_complete": True, "ranking": rows}),
        encoding="utf-8",
    )


def test_aggregate_complete_requires_exact_requested_pairs(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate.json"
    _write_aggregate(aggregate, [(0, 7)])

    assert _aggregate_complete(aggregate, VARIANTS, folds=[0], seeds=[7])
    assert not _aggregate_complete(aggregate, VARIANTS, folds=[0, 1], seeds=[7])
    assert not _aggregate_complete(aggregate, VARIANTS, folds=[0], seeds=[7, 13])


def test_aggregate_complete_rejects_extra_pairs(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate.json"
    _write_aggregate(aggregate, [(0, 7), (1, 7)])

    assert _aggregate_complete(aggregate, VARIANTS, folds=[0, 1], seeds=[7])
    assert not _aggregate_complete(aggregate, VARIANTS, folds=[0], seeds=[7])
