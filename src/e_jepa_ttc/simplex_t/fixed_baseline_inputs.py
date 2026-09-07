"""Load fixed comparator predictions against independently verified query inputs."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash

from .phase import phase_to_ttc


def load_fixed_baselines(
    manifest: Path,
    *,
    manifest_sha256: str,
    role: str,
    source_sha256: str,
    tokens: np.ndarray,
    sequences: np.ndarray,
    history: np.ndarray,
    validate_prerequisites: Callable[[], None],
) -> dict[str, np.ndarray]:
    """Validate file bytes, role, query order, histories and emitted TTC consistency.

    Expected identities must come from the caller's authoritative source, not
    from this output file. Loading supplies no fit/gate authority and computes
    no score. TRAIN predictions cannot be passed off as OLD_DEV by renaming.
    """
    validate_prerequisites()
    if role not in {"inner_oof", "outer_dev"}:
        raise ValueError("fixed comparator role must be TRAIN or OLD_DEV")
    if manifest.stat().st_size > 1_048_576 or compute_file_hash(str(manifest)) != manifest_sha256:
        raise ValueError("fixed comparator manifest changed")
    report = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        report["status"] != "FIXED_BASELINES_ONE_FOLD_COMPLETE_NOT_SCORED"
        or report["optimizer_updates"] != 0
        or report["scores_computed"] is not False
        or set(report["roles"]) != {"inner_oof", "outer_dev"}
    ):
        raise ValueError("complete unscored fixed comparator report required")
    record = report["roles"][role]
    if record["source_sha256"] != source_sha256 or record["path"] != f"{role}.npz":
        raise ValueError("fixed comparator source or role path changed")
    root = manifest.parent.resolve(strict=True)
    path = (root / record["path"]).resolve(strict=True)
    if not path.is_relative_to(root) or compute_file_hash(str(path)) != record["sha256"]:
        raise ValueError("fixed comparator payload changed or escapes root")
    if (
        tokens.ndim != 1
        or len(np.unique(tokens)) != len(tokens)
        or sequences.shape != tokens.shape
        or history.shape != (len(tokens), 8)
        or record["queries"] != len(tokens)
    ):
        raise ValueError("independent fixed comparator identity schema mismatch")
    names = ("CURRENT_MEDIAN", "EWMA_0P3S_H8")
    emission = report.get("ttc_emission_dtype", "float32")
    if emission not in {"float32", "float64"}:
        raise ValueError("unregistered fixed comparator emission dtype")
    required = {"sample_token", "sequence_id", "history"} | {
        f"{name}_{field}" for name in names for field in ("prediction_phase", "prediction_ttc_s")
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("fixed comparator payload schema changed")
        result = {key: archive[key] for key in archive.files}
    for key, expected in (
        ("sample_token", tokens),
        ("sequence_id", sequences),
        ("history", history),
    ):
        if not np.array_equal(result[key], expected):
            raise ValueError("fixed comparator differs from independent query/history identity")
    for name in names:
        phase, ttc = result[f"{name}_prediction_phase"], result[f"{name}_prediction_ttc_s"]
        if (
            phase.dtype != np.float32
            or ttc.dtype != np.dtype(emission)
            or any(a.shape != tokens.shape or not np.isfinite(a).all() for a in (phase, ttc))
        ):
            raise ValueError("fixed comparator prediction shape, dtype or finiteness changed")
        if not np.array_equal(phase_to_ttc(torch.from_numpy(phase.astype(emission))).numpy(), ttc):
            raise ValueError("fixed comparator emitted TTC differs from its phase")
    validate_prerequisites()
    return result
