"""Resumable, per-fit publication from sealed endpoints to OLD_DEV predictions."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json

from .campaign_sources import CampaignSources
from .development_export import development_frame
from .lifecycle import ExclusiveLease
from .phase_inference import iter_phase_predictions
from .phase_manifest import fit_key
from .registry import registered_graph


def export_phase(
    output: Path,
    *,
    sources: CampaignSources,
    manifest: Path,
    checkpoint_root: Path,
    manifest_sha256: str,
    freeze_sha256: str,
    stage: str,
    availability: dict[str, bool],
    dev_source_hashes: dict[str, str],
    expected_queries: pd.DataFrame,
    validate_prerequisites: Callable[[], None],
    resource_ok: Callable[[], bool],
    resume: bool,
) -> dict:
    """Publish complete fold Parquets, never select or train a candidate.

    Mandatory validation must bind expected_queries and source identities to the
    scientific freeze. Resume recomputes head inference (not expert replay) and
    checks existing Parquet bytes before accepting them. This permits recovery
    after file publication but before the state receipt was durably written.
    Only complete registered phases receive PREDICTIONS_COMPLETE status.
    """
    identity = ["sample_token", "sequence_id", "track_id", "outer_fold"]
    try:
        validate_prerequisites()
        if (
            not set(identity) <= set(expected_queries)
            or len(expected_queries) != 8192
            or expected_queries.sample_token.duplicated().any()
            or expected_queries[identity].isna().to_numpy().any()
            or set(expected_queries.outer_fold) != {0, 1, 2}
        ):
            raise ValueError("complete frozen OLD8192 identity cohort required")
        graph = registered_graph(**availability)
        selected = {fit_key(s): s for s in graph if s.stage == stage}
        if not selected or set(dev_source_hashes) != set(selected):
            raise ValueError("complete registered export source set required")
        if any(spec not in sources.graph for spec in selected.values()):
            raise ValueError("export source adapter omits a registered candidate")
        cohort = (
            expected_queries.loc[:, identity].sort_values("sample_token").reset_index(drop=True)
        )
        serialized_cohort = cohort.to_json(orient="records")
        if not isinstance(serialized_cohort, str):
            raise ValueError("unable to serialize frozen cohort")
        contract = {
            "manifest_sha256": manifest_sha256,
            "freeze_sha256": freeze_sha256,
            "stage": stage,
            "availability": availability,
            "dev_source_hashes": dev_source_hashes,
            "cohort": json.loads(serialized_cohort),
        }
        state_path = output / f"{stage}_PREDICTIONS.json"
        with ExclusiveLease(output / "PREDICTION_WRITER.lock"):
            if state_path.exists():
                if not resume:
                    raise FileExistsError("existing prediction export requires explicit resume")
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if (
                    state.get("schema") != "simplex_t_phase_predictions_v1"
                    or state["contract"] != contract
                ):
                    raise ValueError("prediction resume contract changed")
                if set(state["fits"]) - set(selected):
                    raise ValueError("prediction receipt contains unregistered fits")
            else:
                if resume:
                    raise FileNotFoundError("no prediction export state to resume")
                state = {
                    "schema": "simplex_t_phase_predictions_v1",
                    "contract": contract,
                    "fits": {},
                }
            state["status"] = "PREPARING_PREDICTIONS"
            atomic_json(state_path, state)
            try:
                predictions = iter_phase_predictions(
                    manifest,
                    checkpoint_root,
                    manifest_sha256=manifest_sha256,
                    freeze_sha256=freeze_sha256,
                    stage=stage,
                    availability=availability,
                    dev_source_hashes=dev_source_hashes,
                    validate_prerequisites=validate_prerequisites,
                    dev_source_loader=lambda spec: sources.source(spec, "outer_dev"),
                    resource_ok=resource_ok,
                )
                seen = set()
                for spec, outputs, history in predictions:
                    key = fit_key(spec)
                    if key not in selected or key in seen:
                        raise ValueError("unexpected or duplicated inference fit")
                    seen.add(key)
                    if not resource_ok():
                        raise InterruptedError("prediction publication resource pause")
                    frame = development_frame(
                        sources,
                        spec,
                        outputs,
                        history,
                        expected_source_sha256=dev_source_hashes[key],
                    )
                    actual = (
                        frame.loc[:, identity].sort_values("sample_token").reset_index(drop=True)
                    )
                    expected = cohort.loc[cohort.outer_fold == spec.fold].reset_index(drop=True)
                    if not actual.equals(expected):
                        raise ValueError(
                            "prediction query identities differ from frozen OLD cohort"
                        )
                    validate_prerequisites()
                    if not resource_ok():
                        raise InterruptedError("prediction serialization resource pause")
                    destination = output / key / "predictions.parquet"
                    if not destination.resolve().is_relative_to(output.resolve()):
                        raise ValueError("prediction destination escapes the export root")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            dir=destination.parent, suffix=".parquet", delete=False
                        ) as temp:
                            temporary = Path(temp.name)
                            frame.to_parquet(temp, index=False)
                            temp.flush()
                            os.fsync(temp.fileno())
                        digest = compute_file_hash(str(temporary))
                        receipt = {
                            "path": destination.relative_to(output).as_posix(),
                            "sha256": digest,
                            "rows": len(frame),
                        }
                        if key in state["fits"] and state["fits"][key] != receipt:
                            raise ValueError("recomputed predictions differ from recorded export")
                        if destination.exists():
                            if not resume or compute_file_hash(str(destination)) != digest:
                                raise ValueError(
                                    "existing prediction bytes differ; preserve evidence"
                                )
                        else:
                            # Atomic same-volume publication, refusing an existing name.
                            os.link(temporary, destination)
                    finally:
                        if temporary is not None:
                            temporary.unlink(missing_ok=True)
                    state["fits"][key] = receipt
                    atomic_json(state_path, state)
                if seen != set(selected):
                    raise ValueError("incomplete inference stream cannot complete a phase")
                state["status"] = "PREDICTIONS_COMPLETE_NOT_FINAL_ANALYSIS"
            except InterruptedError:
                state["status"] = "PAUSED_RESOURCE"
            atomic_json(state_path, state)
            return state
    finally:
        sources.release()
