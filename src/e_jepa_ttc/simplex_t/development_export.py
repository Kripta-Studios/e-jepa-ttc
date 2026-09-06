"""Align sealed-head outputs with verified OLD_DEV identities and current experts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.hashing import compute_file_hash

from .arms import resolve_arm
from .campaign_sources import CampaignSources
from .current_inputs import load_current_inputs
from .evaluation import prediction_frame
from .registry import FitSpec


def development_frame(
    sources: CampaignSources,
    spec: FitSpec,
    outputs: dict[str, np.ndarray],
    consumed_history: np.ndarray,
    *,
    expected_source_sha256: str,
) -> pd.DataFrame:
    """Materialize one complete fold after the caller has validated its phase seal.

    Predictions must be from phase_inference.iter_phase_predictions. This adapter
    does not itself authorize evaluation. Original metadata supplies identity and
    labels only; current expert TTC comes from the new coherent FP32 cache, not
    the historical mixed-precision replay comparator. No historical score columns
    are decoded. Historical track IDs remain identifiers, not inferred eAP IDs.
    """
    binding = resolve_arm(spec, sources.graph)
    source = sources.source(spec, "outer_dev")
    if source.identity_sha256 != expected_source_sha256:
        raise ValueError("development export source differs from scientific freeze")
    expected_history = source.history[:, -source.length :]
    if not np.array_equal(consumed_history, expected_history):
        raise ValueError("predicted query history differs from export population")
    table = load_current_inputs(
        sources.historical_root,
        spec.fold,
        "outer_dev",
        ancestry_sha256=sources.ancestry_sha256,
        allowed_sequences=sources.allowed_sequences,
    )
    metadata_path = sources.historical_root / "tables" / f"outer{spec.fold}_outer_dev.csv"
    if compute_file_hash(str(metadata_path)) != table["reference"]["metadata_sha256"]:
        raise ValueError("development metadata bytes changed")
    columns = ["sample_token", "sequence_id", "track_id", "target_ttc"]
    metadata = pd.read_csv(metadata_path, usecols=lambda name: name in columns)
    if set(metadata) != set(columns):
        raise ValueError("missing development metadata columns")
    if metadata[columns].isna().to_numpy().any() or metadata.sample_token.duplicated().any():
        raise ValueError("missing or duplicate development identity")
    for name in ("sample_token", "sequence_id", "target_ttc"):
        if not np.array_equal(metadata[name].to_numpy(), table["metadata"][name].to_numpy()):
            raise ValueError("development metadata order differs from source order")
    if len(metadata) != source.population:
        raise ValueError("development source population changed")
    pin = sources.folds[spec.fold]
    manifest_path = pin.path / "COMPILED.json"
    if compute_file_hash(str(manifest_path)) != pin.sha256:
        raise ValueError("compiled manifest changed before export")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expert_path = pin.path / "expert_ttc.npy"
    if compute_file_hash(str(expert_path)) != manifest["arrays"]["expert_ttc"]:
        raise ValueError("current expert cache changed before export")
    experts = np.load(expert_path, mmap_mode="r", allow_pickle=False)
    current = expected_history[:, -1]
    if np.any(current < 0) or np.any(current >= len(experts)):
        raise ValueError("current expert observation is missing")
    result = prediction_frame(
        metadata,
        np.asarray(experts[current]),
        outputs,
        consumed_history,
        arm=spec.name,
        seed=spec.seed,
        fold=spec.fold,
        output_mode=binding.model.output_mode,
    )
    result["source_sha256"] = expected_source_sha256
    result["context_semantics"] = "RETROSPECTIVE_CURRENT_QUERY_ROI_NOT_VERIFIED_OBJECT_HISTORY"
    result["anchor_us"] = source.anchor_us[current]
    result["roi_available_us"] = source.available_us[current]
    result["roi_age_us"] = source.available_us[current] - source.anchor_us[current]
    valid = consumed_history >= 0
    first = np.argmax(valid, axis=1)
    oldest = consumed_history[np.arange(len(first)), first]
    result["history_span_us"] = source.anchor_us[current] - source.anchor_us[oldest]
    return result
