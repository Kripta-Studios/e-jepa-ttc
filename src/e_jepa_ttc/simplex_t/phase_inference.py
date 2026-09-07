"""Connect sealed phase endpoints to bounded, role-separated cached inference."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .arms import resolve_arm
from .cache import CachedQueries
from .endpoint import load_endpoint, predict_cached
from .phase_manifest import fit_key
from .registry import FitSpec, registered_graph


def validated_phase(
    manifest_path: Path,
    checkpoint_root: Path,
    *,
    manifest_sha256: str,
    freeze_sha256: str,
    stage: str,
    availability: dict[str, bool],
    resource_ok: Callable[[], bool],
) -> tuple[list[FitSpec], dict[str, dict[str, Any]]]:
    """Validate the complete endpoint set before loading any evaluation population."""
    if sha256(manifest_path) != manifest_sha256:
        raise ValueError("phase manifest differs from pinned endpoint seal")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest["schema"] != "simplex_t_phase_endpoints_v1"
        or manifest["stage"] != stage
        or manifest["scientific_freeze_sha256"] != freeze_sha256
        or manifest["availability"] != availability
        or manifest["scores_read_by_sealer"] is not False
        or any(type(value) is not bool for value in availability.values())
    ):
        raise ValueError("phase seal contract mismatch")
    graph = registered_graph(**availability)
    expected = {fit_key(spec): spec for spec in graph if spec.stage == stage}
    records = manifest["fits"]
    if (
        not expected
        or len(records) != len(expected)
        or {row["key"] for row in records} != set(expected)
        or manifest["optimizer_endpoint_updates"] != 2500 * len(expected)
    ):
        raise ValueError("incomplete or duplicate sealed phase endpoint set")
    root = checkpoint_root.resolve(strict=True)
    paths: set[Path] = set()
    result = {}
    for row in records:
        spec = expected[row["key"]]
        binding = resolve_arm(spec, graph)
        relative = Path(row["checkpoint"])
        path = (root / relative).resolve(strict=True)
        if relative.is_absolute() or not path.is_relative_to(root) or path in paths:
            raise ValueError("aliased or out-of-scope sealed endpoint")
        paths.add(path)
        if row["fit"] != asdict(spec) or row["model"] != asdict(binding.model):
            raise ValueError("sealed endpoint differs from canonical arm")
        if sha256(path) != row["checkpoint_sha256"]:
            raise ValueError("sealed checkpoint bytes changed")
        result[row["key"]] = {**row, "resolved_checkpoint": path}
    # Verify every checkpoint's completed-update identity before any data callback.
    for key, row in result.items():
        if not resource_ok():
            raise InterruptedError("endpoint validation resource pause")
        spec = expected[key]
        model = load_endpoint(
            row["resolved_checkpoint"],
            resolve_arm(spec, graph).model,
            seed=spec.seed,
            freeze_sha256=freeze_sha256,
            train_source_sha256=row["train_source_sha256"],
            endpoint_sha256=row["checkpoint_sha256"],
        )
        del model
    return graph, result


def iter_phase_predictions(
    manifest_path: Path,
    checkpoint_root: Path,
    *,
    manifest_sha256: str,
    freeze_sha256: str,
    stage: str,
    availability: dict[str, bool],
    dev_source_hashes: dict[str, str],
    validate_prerequisites: Callable[[], None],
    dev_source_loader: Callable[[FitSpec], CachedQueries],
    resource_ok: Callable[[], bool],
) -> Iterator[tuple[FitSpec, dict[str, np.ndarray], np.ndarray]]:
    """Yield complete per-fit outputs and histories, never partial query predictions.

    The caller must supply a role-validated OLD_DEV loader (CampaignSources.source
    with role='outer_dev'), pin its identities before freeze, and validate phase
    gates. Publication/aggregation remains the caller's responsibility. No best
    checkpoint selection, expert inference, or optimizer updates occur here.
    """
    validate_prerequisites()
    if not resource_ok():
        raise InterruptedError("phase inference resource pause")
    graph, records = validated_phase(
        manifest_path,
        checkpoint_root,
        manifest_sha256=manifest_sha256,
        freeze_sha256=freeze_sha256,
        stage=stage,
        availability=availability,
        resource_ok=resource_ok,
    )
    if set(dev_source_hashes) != set(records):
        raise ValueError("OLD_DEV source identity set differs from sealed phase")
    for spec in graph:
        if spec.stage != stage:
            continue
        if not resource_ok():
            raise InterruptedError("phase inference resource pause")
        validate_prerequisites()
        if sha256(manifest_path) != manifest_sha256:
            raise ValueError("phase seal changed during inference")
        key = fit_key(spec)
        row = records[key]
        source = dev_source_loader(spec)
        if source.identity_sha256 != dev_source_hashes[key]:
            raise ValueError("OLD_DEV source differs from scientific freeze")
        model = load_endpoint(
            row["resolved_checkpoint"],
            resolve_arm(spec, graph).model,
            seed=spec.seed,
            freeze_sha256=freeze_sha256,
            train_source_sha256=row["train_source_sha256"],
            endpoint_sha256=row["checkpoint_sha256"],
        )
        outputs = predict_cached(model, source, resource_ok=resource_ok)
        history = source.history[:, -source.length :].copy()
        del model, source
        validate_prerequisites()
        if sha256(manifest_path) != manifest_sha256:
            raise ValueError("phase seal changed before prediction publication")
        yield spec, outputs, history
