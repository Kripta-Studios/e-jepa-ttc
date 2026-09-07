"""Load the unchanged-expert RISK17 comparator against independent OLD identities."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256


def load_risk17_replay(
    manifest: Path,
    *,
    manifest_sha256: str,
    expected_identity: pd.DataFrame,
    expected_bindings: list[dict],
    ancestry_sha256: str,
    validate_prerequisites: Callable[[], None],
) -> pd.DataFrame:
    """Validate output bytes, all-fold source bindings and unchanged expert emission.

    Expected identities and bindings must come from acknowledged historical tables
    and frozen router manifests, not from this replay's own receipt. The mandatory
    validator verifies their authority and current bytes. This reads no scores and
    does not authorize fits, gates or any confirmation access.
    """
    validate_prerequisites()
    if manifest.stat().st_size > 1_048_576 or sha256(manifest) != manifest_sha256:
        raise ValueError("RISK17 replay manifest changed")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        record["status"] != "HISTORICAL_RISK17_OLD_REPLAY_EXACT"
        or record["queries"] != 8192
        or type(record["optimizer_updates"]) is not int
        or record["optimizer_updates"] != 0
        or record["scientific_stage_authorized"] is not False
        or record["ancestry_sha256"] != ancestry_sha256
        or record["bindings"] != expected_bindings
        or len(expected_bindings) != 3
        or {row["fold"] for row in expected_bindings} != {0, 1, 2}
    ):
        raise ValueError("RISK17 replay source or role contract differs")
    for binding in expected_bindings:
        if (
            binding["table"]["role"] != "outer_dev"
            or binding["table"]["outer_fold"] != binding["fold"]
        ):
            raise ValueError("RISK17 requires outer development producers")
    root = manifest.parent.resolve(strict=True)
    payload = (root / "RISK17_OLD.parquet").resolve(strict=True)
    if (
        not payload.is_relative_to(root)
        or payload.stat().st_size > 16_777_216
        or sha256(payload) != record["payload_sha256"]
    ):
        raise ValueError("RISK17 payload changed or exceeds bound")
    frame = pd.read_parquet(payload)
    identity = ["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc"]
    for data in (frame, expected_identity):
        if (
            not set(identity) <= set(data)
            or len(data) != 8192
            or data.sample_token.duplicated().any()
            or data[identity].isna().to_numpy().any()
            or data.sequence_id.nunique() != 9
            or set(data.outer_fold) != {0, 1, 2}
        ):
            raise ValueError("complete independent OLD8192 identity required")
    frame = frame.sort_values("sample_token").reset_index(drop=True)
    expected = expected_identity.sort_values("sample_token").reset_index(drop=True)
    if not frame[identity].equals(expected[identity]):
        raise ValueError("RISK17 query, fold or target identity differs")
    if not frame.arm.eq("S65-RISK17").all() or not frame.seed.eq(7).all():
        raise ValueError("historical RISK17 seed7 comparator required")
    selected = frame.selected_expert.to_numpy()
    experts = frame[["a5_ttc_s", "c2f_ttc_s", "pair_ttc_s"]].to_numpy()
    if (
        selected.dtype.kind not in "iu"
        or not np.isin(selected, [0, 1, 2]).all()
        or not np.isfinite(experts).all()
        or not np.array_equal(frame.prediction_ttc_s.to_numpy(), experts[np.arange(8192), selected])
    ):
        raise ValueError("RISK17 prediction is not the unchanged selected expert")
    validate_prerequisites()
    if sha256(manifest) != manifest_sha256 or sha256(payload) != record["payload_sha256"]:
        raise ValueError("RISK17 publication changed during loading")
    return frame
