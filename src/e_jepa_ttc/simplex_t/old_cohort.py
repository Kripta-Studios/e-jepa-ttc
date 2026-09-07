"""Independently assemble OLD evaluation identities from acknowledged producer tables."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .current_inputs import load_current_inputs


def load_old_evaluation_cohort(
    historical_root: Path,
    *,
    ancestry_sha256: str,
    table_index_sha256: str,
    allowed_sequences: set[str],
    resource_ok: Callable[[], bool],
) -> tuple[pd.DataFrame, dict]:
    """Join the three true outer-dev populations, never pooled OOF predictions.

    The caller must bind the supplied root, hashes and sequence set to the owner
    ACK. The current-input loader verifies each query's producer exclusions and
    PAIR ancestry. This loads existing OLD targets for evaluation only: it creates
    no histories, changes no roles and does not authorize fitting or scoring.
    """
    if len(allowed_sequences) != 9:
        raise ValueError("exact nine-sequence OLD role set required")
    root = historical_root.resolve(strict=True)
    index_path = root / "FROZEN_EXPERT_TABLE_INDEX.json"

    def boundary() -> None:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: OLD evaluation cohort")
        if sha256(index_path) != table_index_sha256:
            raise ValueError("historical table index changed")
        if sha256(root / "NESTED_ANCESTRY_AUDIT.json") != ancestry_sha256:
            raise ValueError("historical cohort ancestry changed")

    frames, references = [], []
    for fold in range(3):
        boundary()
        verified = load_current_inputs(
            root,
            fold,
            "outer_dev",
            ancestry_sha256=ancestry_sha256,
            allowed_sequences=allowed_sequences,
        )
        metadata = verified["metadata"]
        reference = verified["reference"]
        path = root / "tables" / f"outer{fold}_outer_dev.csv"
        if sha256(path) != reference["metadata_sha256"]:
            raise ValueError("OLD cohort metadata changed before identity loading")
        columns = ["sample_token", "sequence_id", "track_id", "target_ttc", "outer_fold"]
        frame = pd.read_csv(path, usecols=pd.Index(columns)).loc[:, columns]
        common = ["sample_token", "sequence_id", "target_ttc"]
        if not frame.loc[:, common].equals(metadata.loc[:, common]):
            raise ValueError("OLD identities or targets differ from lineage-validated rows")
        if sha256(path) != reference["metadata_sha256"]:
            raise ValueError("OLD cohort metadata changed during identity loading")
        if not frame.outer_fold.eq(fold).all():
            raise ValueError("OLD metadata changes original outer-fold assignment")
        frames.append(frame)
        references.append(
            {
                "outer_fold": fold,
                "role": "outer_dev",
                "queries": len(frame),
                "metadata_sha256": reference["metadata_sha256"],
                "arrays_sha256": reference["arrays_sha256"],
            }
        )
        del verified
    result = pd.concat(frames, ignore_index=True)
    if (
        len(result) != 8192
        or result.sample_token.duplicated().any()
        or result.isna().to_numpy().any()
        or not np.isfinite(result.target_ttc.to_numpy()).all()
        or set(result.sequence_id) != allowed_sequences
        or set(result.outer_fold) != {0, 1, 2}
        or not result.groupby("sequence_id").outer_fold.nunique().eq(1).all()
    ):
        raise ValueError("OLD outer-dev union is incomplete, overlaps or changes sequence folds")
    boundary()
    return result.sort_values("sample_token").reset_index(drop=True), {
        "schema": "simplex_t_independent_old_cohort_v1",
        "queries": 8192,
        "sequence_count": 9,
        "ancestry_sha256": ancestry_sha256,
        "table_index_sha256": table_index_sha256,
        "folds": references,
        "source_role": "outer_dev",
        "prediction_scores_read": False,
        "optimizer_updates": 0,
        "history_constructed": False,
    }
