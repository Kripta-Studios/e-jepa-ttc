"""Load historical unchanged-expert selectors against independent OLD identities."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .coordination import verified_ack


def load_acknowledged_risk17(
    manifest: Path,
    *,
    manifest_sha256: str,
    ack_path: Path,
    ack_sha256: str,
    table_index_sha256: str,
    ridge_manifest_sha256: str,
    expected_identity: pd.DataFrame,
    resource_ok: Callable[[], bool],
) -> pd.DataFrame:
    """Derive replay bindings from independently frozen historical router inputs.

    The caller binds the supplied table/ridge hashes and OLD cohort to the
    scientific freeze. Router weights and all three outer-dev table pairs are
    rehashed, not inferred from the replay receipt. No router fit or inference
    is repeated. This is suitable for the T5 gate's verified RISK17 callback.
    """
    ack = verified_ack(ack_path, ack_sha256)
    ancestry = ack["producers"]["authoritative_historical_manifest"]
    historical = Path(ancestry["path"]).parent.resolve(strict=True)
    ridge = historical / "frozen_audit/extracted_input/run/stage65"
    index_path = historical / "FROZEN_EXPERT_TABLE_INDEX.json"
    ridge_path = ridge / "ALL_RIDGE_FITS_FROZEN.json"
    pins = {index_path: table_index_sha256, ridge_path: ridge_manifest_sha256}

    def validate() -> None:
        if verified_ack(ack_path, ack_sha256) != ack:
            raise ValueError("RISK17 acknowledged authority changed")
        for path, digest in pins.items():
            if not resource_ok():
                raise InterruptedError("PAUSED_RESOURCE: RISK17 historical bindings")
            if sha256(path) != digest:
                raise ValueError("RISK17 historical binding bytes changed")

    validate()
    tables = json.loads(index_path.read_text(encoding="utf-8"))
    ridge_manifest = json.loads(ridge_path.read_text(encoding="utf-8"))
    if ridge_manifest.get("evidence_type") != "all_router_fits_frozen_before_evaluation":
        raise ValueError("RISK17 router freeze must precede evaluation")
    fits = [row for row in ridge_manifest["fits"] if row["model"] == "S65-RISK17"]
    if len(fits) != 3 or {row["outer_fold"] for row in fits} != {0, 1, 2}:
        raise ValueError("complete historical RISK17 fold set required")
    bindings = []
    for fit in sorted(fits, key=lambda row: row["outer_fold"]):
        fold = fit["outer_fold"]
        if fit["evidence_type"] != "train_only_frozen_router_fit":
            raise ValueError("RISK17 router is not a frozen TRAIN-only fit")
        matches = [
            row for row in tables if row["outer_fold"] == fold and row["role"] == "outer_dev"
        ]
        if len(matches) != 1 or matches[0]["ancestry_sha256"] != ancestry["sha256"]:
            raise ValueError("RISK17 outer table has ambiguous or different ancestry")
        table = matches[0]
        pins[ridge / f"outer{fold}/S65-RISK17.npz"] = fit["sha256"]
        for suffix, key in (("csv", "metadata_sha256"), ("npz", "arrays_sha256")):
            pins[historical / "tables" / f"outer{fold}_outer_dev.{suffix}"] = table[key]
        bindings.append({"fold": fold, "weights_sha256": fit["sha256"], "table": table})
    if set(expected_identity.sequence_id) != set(
        ack["interfaces"]["role_manifest"]["roles"]["original"]
    ):
        raise ValueError("RISK17 expected cohort changes acknowledged OLD roles")
    return load_risk17_replay(
        manifest,
        manifest_sha256=manifest_sha256,
        expected_identity=expected_identity,
        expected_bindings=bindings,
        ancestry_sha256=ancestry["sha256"],
        validate_prerequisites=validate,
    )


def load_selector_replay(
    manifest: Path,
    *,
    comparator: str,
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
    registered = {
        "S65-RISK17": ("HISTORICAL_RISK17_OLD_REPLAY_EXACT", "RISK17_OLD.parquet"),
        "S67-SIMPLEX17": ("HISTORICAL_SIMPLEX17_OLD_REPLAY_EXACT_FP32", "SIMPLEX17_OLD.parquet"),
    }
    if comparator not in registered:
        raise ValueError("unregistered historical selector")
    status, filename = registered[comparator]
    if manifest.stat().st_size > 1_048_576 or sha256(manifest) != manifest_sha256:
        raise ValueError("RISK17 replay manifest changed")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        record["status"] != status
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
    payload = (root / filename).resolve(strict=True)
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
    if not frame.arm.eq(comparator).all() or not frame.seed.eq(7).all():
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


def load_risk17_replay(
    manifest: Path,
    *,
    manifest_sha256: str,
    expected_identity: pd.DataFrame,
    expected_bindings: list[dict],
    ancestry_sha256: str,
    validate_prerequisites: Callable[[], None],
) -> pd.DataFrame:
    """Preserve the explicit RISK17-only interface; never substitute SIMPLEX17."""
    return load_selector_replay(
        manifest,
        comparator="S65-RISK17",
        manifest_sha256=manifest_sha256,
        expected_identity=expected_identity,
        expected_bindings=expected_bindings,
        ancestry_sha256=ancestry_sha256,
        validate_prerequisites=validate_prerequisites,
    )
