"""Complete, immutable phase endpoints before any scientific scoring.

Availability and scientific-freeze authorization are validated by the caller.
This module never chooses candidates, computes scores or opens evaluation data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json
from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .arms import resolve_arm
from .endpoint import load_endpoint
from .lifecycle import ExclusiveLease
from .registry import FitSpec, registered_graph


def fit_key(spec: FitSpec) -> str:
    """Unique artifact key including stage, arm, fold and optimization seed."""
    return f"{spec.stage}/{spec.name}/fold{spec.fold}/seed{spec.seed}"


@dataclass(frozen=True)
class EndpointReceipt:
    """Expected training-source and checkpoint bytes, not a best-score pointer."""

    checkpoint: Path
    checkpoint_sha256: str
    train_source_sha256: str


def seal_phase(
    output: Path,
    *,
    checkpoint_root: Path,
    stage: str,
    availability: dict[str, bool],
    freeze_sha256: str,
    receipts: dict[str, EndpointReceipt],
) -> dict[str, Any]:
    """Require every available registered fit in a phase before publishing its list.

    A completed-checkpoint check includes weights and training identity. The
    resulting hash must be pinned by the caller before starting evaluation.
    An existing phase manifest is never replaced, including after a restart.
    """
    if any(type(value) is not bool for value in availability.values()):
        raise ValueError("availability must be resolved booleans")
    graph = registered_graph(**availability)
    expected = {fit_key(spec): spec for spec in graph if spec.stage == stage}
    if not expected or set(receipts) != set(expected):
        raise ValueError("phase endpoint set is missing, extra or unavailable")
    root = checkpoint_root.resolve(strict=True)
    # Check path scope and aliasing before reading any checkpoint bytes.
    paths = {}
    for key, receipt in receipts.items():
        path = receipt.checkpoint.resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("checkpoint outside dedicated campaign root")
        paths[key] = path
    if len(set(paths.values())) != len(paths):
        raise ValueError("distinct fits cannot alias one checkpoint")
    with ExclusiveLease(output.with_suffix(output.suffix + ".lock")):
        if output.exists():
            raise FileExistsError("phase endpoint manifest already frozen")
        records = []
        for key, spec in sorted(expected.items()):
            receipt = receipts[key]
            binding = resolve_arm(spec, graph)
            model = load_endpoint(
                paths[key],
                binding.model,
                seed=spec.seed,
                freeze_sha256=freeze_sha256,
                train_source_sha256=receipt.train_source_sha256,
                endpoint_sha256=receipt.checkpoint_sha256,
            )
            del model  # Only one compact head is live; no model inference is performed.
            records.append(
                {
                    "key": key,
                    "fit": asdict(spec),
                    "checkpoint": paths[key].relative_to(root).as_posix(),
                    "checkpoint_sha256": receipt.checkpoint_sha256,
                    "train_source_sha256": receipt.train_source_sha256,
                    "model": asdict(binding.model),
                }
            )
        manifest = {
            "schema": "simplex_t_phase_endpoints_v1",
            "stage": stage,
            "scientific_freeze_sha256": freeze_sha256,
            "availability": availability,
            "fits": records,
            "optimizer_endpoint_updates": len(records) * 2500,
            "scores_read_by_sealer": False,
        }
        atomic_json(output, manifest)
    return {"path": str(output), "sha256": sha256(output), "fits": len(records)}
