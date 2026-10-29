"""Load the predefined H8/H1 pair only from a complete sealed phase publication."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .phase_inference import validated_phase
from .published_predictions import iter_published_predictions


def load_canonical_publication(
    publication: Path,
    *,
    publication_sha256: str,
    endpoints: Path,
    endpoints_sha256: str,
    checkpoint_root: Path,
    freeze_sha256: str,
    availability: dict[str, bool],
    family: str,
    expected_queries: pd.DataFrame,
    validate_authority: Callable[[], None],
    resource_ok: Callable[[], bool],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return canonical seed7 H8 then H1, retaining all OLD8192 queries.

    The caller supplies the independently verified OLD cohort including targets
    and frozen authority. Endpoint completeness is verified with actual compact
    head states before opening predictions; no fit or expert replay is performed.
    Every published table is validated, while only the two predefined arms are
    retained in memory. This does not choose a best arm or authorize a later fit.
    """
    validate_authority()
    if family not in {"TPR", "LATENT"}:
        raise ValueError("canonical scalar or latent family required")
    if (
        not {"sample_token", "sequence_id", "target_ttc"} <= set(expected_queries)
        or len(expected_queries) != 8192
        or expected_queries.sequence_id.nunique() != 9
        or expected_queries.sample_token.duplicated().any()
        or not np.isfinite(expected_queries.target_ttc.to_numpy()).all()
    ):
        raise ValueError("independent OLD8192 target cohort required")
    stage = "T2" if family == "TPR" else "T4"
    _, records = validated_phase(
        endpoints,
        checkpoint_root,
        manifest_sha256=endpoints_sha256,
        freeze_sha256=freeze_sha256,
        stage=stage,
        availability=availability,
        resource_ok=resource_ok,
    )

    def validate() -> None:
        validate_authority()
        if sha256(endpoints) != endpoints_sha256:
            raise ValueError("canonical endpoint seal changed")
        for record in records.values():
            if not resource_ok():
                raise InterruptedError("canonical endpoint integrity resource pause")
            if sha256(record["resolved_checkpoint"]) != record["checkpoint_sha256"]:
                raise ValueError("canonical phase checkpoint changed")

    primary = "D1" if availability["d1"] else "D0"
    names = [f"{family}-{primary}-H{length}-C160" for length in (8, 1)]
    collected: dict[str, list[pd.DataFrame]] = {name: [] for name in names}
    targets = expected_queries.set_index("sample_token").target_ttc
    for spec, frame in iter_published_predictions(
        publication,
        manifest_sha256=publication_sha256,
        endpoint_manifest_sha256=endpoints_sha256,
        freeze_sha256=freeze_sha256,
        stage=stage,
        availability=availability,
        expected_queries=expected_queries,
        validate_prerequisites=validate,
        resource_ok=resource_ok,
    ):
        if not {"target_ttc", "prediction_ttc_s"} <= set(frame) or not np.array_equal(
            frame.target_ttc.to_numpy(), targets.loc[frame.sample_token].to_numpy()
        ):
            raise ValueError("published targets differ from independently verified OLD cohort")
        if spec.name in collected:
            if spec.seed != 7:
                raise ValueError("canonical publication must use seed7")
            collected[spec.name].append(frame)
    result = []
    for name in names:
        parts = collected[name]
        if len(parts) != 3:
            raise ValueError("canonical arm is missing a published fold")
        frame = pd.concat(parts, ignore_index=True)
        if len(frame) != 8192 or frame.sample_token.duplicated().any():
            raise ValueError("canonical arm does not cover OLD8192 exactly once")
        result.append(frame)
    validate()
    if sha256(publication) != publication_sha256:
        raise ValueError("canonical prediction publication changed")
    return result[0], result[1]
