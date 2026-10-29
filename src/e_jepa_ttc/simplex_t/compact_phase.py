"""Export all sealed phase heads, including controls, into safe numeric arrays."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json

from .arms import resolve_arm
from .compact_weights import export_compact_endpoint, load_compact_endpoint
from .endpoint import load_endpoint
from .phase_inference import validated_phase
from .phase_manifest import fit_key
from .scientific_freeze import read_scientific_freeze


def export_compact_phase(
    output: Path,
    *,
    endpoints: Path,
    endpoints_sha256: str,
    checkpoint_root: Path,
    freeze: Path,
    freeze_sha256: str,
    roots: dict[str, Path],
    stage: str,
    availability: dict[str, bool],
    validate_authority_and_qa: Callable[[], None],
    resource_ok: Callable[[], bool],
    resume: bool,
) -> dict:
    """Verify fixed endpoints, export every head and compare exact state tensors.

    Completed per-head manifests are resume boundaries. Unmanifested partial
    output is retained and requires recovery, never silently replaced. This is
    packaging only: there is no optimizer, evaluation, ranking or holdout access.
    """
    work = roots["work"].resolve(strict=True)
    if output.resolve() == work or not output.resolve().is_relative_to(work):
        raise ValueError("compact phase output must remain inside companion")
    if output.exists() and not resume:
        raise FileExistsError("existing compact phase requires explicit resume")
    record = read_scientific_freeze(
        freeze,
        expected_sha256=freeze_sha256,
        roots=roots,
        validate_prerequisites=validate_authority_and_qa,
    )
    frozen_flags = record["source_contract"]["availability"]
    if (
        set(availability) != set(frozen_flags)
        or any(
            type(value) is not bool or (value and not frozen_flags[key])
            for key, value in availability.items()
        )
        or any(availability[key] != frozen_flags[key] for key in ("d1", "density"))
    ):
        raise ValueError("compact phase changes frozen candidate availability")
    graph, records = validated_phase(
        endpoints,
        checkpoint_root,
        manifest_sha256=endpoints_sha256,
        freeze_sha256=freeze_sha256,
        stage=stage,
        availability=availability,
        resource_ok=resource_ok,
    )

    def boundary() -> None:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: compact phase export")
        validate_authority_and_qa()
        if sha256(freeze) != freeze_sha256 or sha256(endpoints) != endpoints_sha256:
            raise ValueError("compact phase freeze or endpoint seal changed")

    boundary()
    output.mkdir(parents=True, exist_ok=resume)
    exported = {}
    for spec in (item for item in graph if item.stage == stage):
        boundary()
        key = fit_key(spec)
        endpoint = records[key]
        train_hash = record["source_identities"][key]["inner_oof"]
        if endpoint["train_source_sha256"] != train_hash:
            raise ValueError("compact endpoint TRAIN source differs from freeze")
        destination = output / key
        manifest = destination / "WEIGHTS.json"
        if destination.exists():
            if not resume or not manifest.is_file():
                raise ValueError("retain unmanifested compact output for recovery")
            digest = sha256(manifest)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = export_compact_endpoint(
                endpoint["resolved_checkpoint"],
                destination,
                spec=spec,
                graph=graph,
                freeze_sha256=freeze_sha256,
                train_source_sha256=train_hash,
                endpoint_sha256=endpoint["checkpoint_sha256"],
                validate_prerequisites=boundary,
                resource_check=boundary,
            )
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            metadata["fit"] != asdict(spec)
            or metadata["scientific_freeze_sha256"] != freeze_sha256
            or metadata["train_source_sha256"] != train_hash
            or metadata["source_endpoint_sha256"] != endpoint["checkpoint_sha256"]
        ):
            raise ValueError("compact manifest changes sealed endpoint identity")
        compact = load_compact_endpoint(destination, manifest_sha256=digest)
        original = load_endpoint(
            endpoint["resolved_checkpoint"],
            resolve_arm(spec, graph).model,
            seed=spec.seed,
            freeze_sha256=freeze_sha256,
            train_source_sha256=train_hash,
            endpoint_sha256=endpoint["checkpoint_sha256"],
        )
        left, right = compact.state_dict(), original.state_dict()
        if set(left) != set(right) or any(not torch.equal(left[k], right[k]) for k in left):
            raise ValueError("compact roundtrip changes exact endpoint tensors")
        del compact, original, left, right
        boundary()
        if sha256(manifest) != digest:
            raise ValueError("compact manifest changed during roundtrip verification")
        exported[key] = {
            "manifest": f"{key}/WEIGHTS.json",
            "sha256": digest,
            "source_endpoint_sha256": endpoint["checkpoint_sha256"],
            "exact_state_roundtrip": True,
        }
    result = {
        "schema": "simplex_t_compact_phase_v1",
        "status": "COMPLETE_COMPACT_PHASE_NOT_T6",
        "stage": stage,
        "freeze_sha256": freeze_sha256,
        "endpoints_sha256": endpoints_sha256,
        "availability": availability,
        "fits": exported,
        "optimizer_updates": 0,
        "holdout_opened": False,
    }
    boundary()
    final = output / "COMPACT_PHASE.json"
    if final.exists():
        if not resume or json.loads(final.read_text(encoding="utf-8")) != result:
            raise ValueError("compact phase final manifest differs; preserve existing evidence")
    else:
        write_new_json(final, result)
    return result
