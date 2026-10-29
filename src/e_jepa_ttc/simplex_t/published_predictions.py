"""Read a complete, pinned OLD_DEV publication for registered analyses."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.hashing import compute_file_hash

from .phase_manifest import fit_key
from .registry import FitSpec, registered_graph


def iter_published_predictions(
    manifest: Path,
    *,
    manifest_sha256: str,
    endpoint_manifest_sha256: str,
    freeze_sha256: str,
    stage: str,
    availability: dict[str, bool],
    expected_queries: pd.DataFrame,
    validate_prerequisites: Callable[[], None],
    resource_ok: Callable[[], bool],
) -> Iterator[tuple[FitSpec, pd.DataFrame]]:
    """Check the entire published set before reading the first prediction table.

    The caller verifies freeze authority and supplies its acknowledged OLD cohort.
    No partial phase, identity-based row dropping, altered file or unregistered
    arm is accepted. Tables are yielded individually to bound resident memory.
    """
    validate_prerequisites()
    if any(type(value) is not bool for value in availability.values()):
        raise ValueError("resolved boolean phase availability required")
    for digest in (manifest_sha256, endpoint_manifest_sha256, freeze_sha256):
        if len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            raise ValueError("pinned SHA256 publication contract required")
    if manifest.stat().st_size > 8_388_608 or compute_file_hash(str(manifest)) != manifest_sha256:
        raise ValueError("prediction publication manifest changed or exceeds metadata bound")
    state = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        state.get("schema") != "simplex_t_phase_predictions_v1"
        or state.get("status") != "PREDICTIONS_COMPLETE_NOT_FINAL_ANALYSIS"
    ):
        raise ValueError("complete prediction publication required for analysis")
    contract = state["contract"]
    if (
        contract["manifest_sha256"] != endpoint_manifest_sha256
        or contract["freeze_sha256"] != freeze_sha256
        or contract["stage"] != stage
        or contract["availability"] != availability
    ):
        raise ValueError("prediction publication differs from frozen phase contract")
    identity = ["sample_token", "sequence_id", "track_id", "outer_fold"]
    if (
        not set(identity) <= set(expected_queries)
        or len(expected_queries) != 8192
        or expected_queries.sample_token.duplicated().any()
        or expected_queries.loc[:, identity].isna().to_numpy().any()
        or set(expected_queries.outer_fold) != {0, 1, 2}
    ):
        raise ValueError("complete acknowledged OLD8192 cohort required")
    expected = expected_queries.loc[:, identity].sort_values("sample_token").reset_index(drop=True)
    serialized = expected.to_json(orient="records")
    if not isinstance(serialized, str) or contract["cohort"] != json.loads(serialized):
        raise ValueError("published cohort differs from independently supplied OLD identities")
    selected = {fit_key(s): s for s in registered_graph(**availability) if s.stage == stage}
    if (
        not selected
        or set(state["fits"]) != set(selected)
        or set(contract["dev_source_hashes"]) != set(selected)
    ):
        raise ValueError("publication omits or adds registered fits")
    root = manifest.parent.resolve(strict=True)
    paths = {}
    for key, spec in selected.items():
        record = state["fits"][key]
        canonical = f"{key}/predictions.parquet"
        if record["path"] != canonical or record["rows"] != int(
            (expected.outer_fold == spec.fold).sum()
        ):
            raise ValueError("prediction path or population differs from registered fit")
        path = (root / canonical).resolve(strict=True)
        if not path.is_relative_to(root) or path in paths.values():
            raise ValueError("prediction path escapes root or aliases another fit")
        if not resource_ok():
            raise InterruptedError("prediction integrity resource pause")
        if compute_file_hash(str(path)) != record["sha256"]:
            raise ValueError("published prediction bytes changed before analysis")
        paths[key] = path
    for key, spec in selected.items():
        validate_prerequisites()
        if not resource_ok():
            raise InterruptedError("prediction table resource pause")
        if (
            compute_file_hash(str(manifest)) != manifest_sha256
            or compute_file_hash(str(paths[key])) != state["fits"][key]["sha256"]
        ):
            raise ValueError("prediction publication changed during analysis")
        frame = pd.read_parquet(paths[key])
        required = {*identity, "arm", "seed", "loss", "source_sha256"}
        if not required <= set(frame):
            raise ValueError("prediction analysis fields missing")
        actual = frame.loc[:, identity].sort_values("sample_token").reset_index(drop=True)
        cohort = expected.loc[expected.outer_fold == spec.fold].reset_index(drop=True)
        if not actual.equals(cohort):
            raise ValueError("prediction table query identities changed")
        if (
            not frame.arm.eq(spec.name).all()
            or not frame.seed.eq(spec.seed).all()
            or not frame.source_sha256.eq(contract["dev_source_hashes"][key]).all()
            or not np.isfinite(frame.loss.to_numpy()).all()
            or frame.loss.lt(0).any()
        ):
            raise ValueError("prediction candidate, source or loss invalid")
        yield spec, frame
