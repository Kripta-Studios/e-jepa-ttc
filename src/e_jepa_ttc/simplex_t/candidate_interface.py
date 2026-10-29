"""Freeze canonical seed7 endpoint references for a separately authorized comparison."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json

from .arms import resolve_arm
from .phase_inference import validated_phase
from .phase_manifest import fit_key
from .scientific_freeze import read_scientific_freeze
from .stage_gate import CanonicalPublication


def publish_candidate_interface(
    output: Path,
    *,
    freeze: Path,
    freeze_sha256: str,
    roots: dict[str, Path],
    phases: dict[str, CanonicalPublication],
    validate_authority_and_qa: Callable[[], None],
    resource_ok: Callable[[], bool],
) -> dict:
    """Bind the predeclared primary names, not a ranking of exploration/control scores.

    Validate complete phase endpoint states but do not read prediction scores.
    Export seed7 fold-specific references as an interface, not a deployable
    ensemble or a final scientific completion claim. Replicate weights remain
    part of full phase delivery; no best-seed selection occurs here.
    """
    work = roots["work"].resolve(strict=True)
    if output.exists() or output.resolve() == work or not output.resolve().is_relative_to(work):
        raise ValueError("new candidate interface inside companion required")
    record = read_scientific_freeze(
        freeze,
        expected_sha256=freeze_sha256,
        roots=roots,
        validate_prerequisites=validate_authority_and_qa,
    )
    flags = record["source_contract"]["availability"]
    expected = {"T2": record["canonical_scalar"]}
    if flags["latent"]:
        expected["T4"] = record["canonical_latent"]
    if set(phases) != set(expected):
        raise ValueError("complete canonical scalar and technically enabled latent phases required")
    candidates = {}
    checked = []
    for stage, name in expected.items():
        binding = phases[stage]
        if (
            set(binding.availability) != set(flags)
            or any(
                type(value) is not bool or (value and not flags[key])
                for key, value in binding.availability.items()
            )
            or any(binding.availability[key] != flags[key] for key in ("d1", "density", "latent"))
        ):
            raise ValueError("candidate phase changes frozen pool or latent availability")
        graph, endpoints = validated_phase(
            binding.endpoints,
            binding.checkpoint_root,
            manifest_sha256=binding.endpoints_sha256,
            freeze_sha256=freeze_sha256,
            stage=stage,
            availability=binding.availability,
            resource_ok=resource_ok,
        )
        selected = [
            spec for spec in graph if spec.stage == stage and spec.name == name and spec.seed == 7
        ]
        if len(selected) != 3 or {spec.fold for spec in selected} != {0, 1, 2}:
            raise ValueError("canonical seed7 endpoint coverage incomplete")
        rows = []
        for spec in selected:
            if not resource_ok():
                raise InterruptedError("PAUSED_RESOURCE: future candidate interface")
            key = fit_key(spec)
            item = endpoints[key]
            train_hash = record["source_identities"][key]["inner_oof"]
            if item["train_source_sha256"] != train_hash:
                raise ValueError("future candidate TRAIN source differs from frozen source")
            rows.append(
                {
                    "fit": asdict(spec),
                    "model": asdict(resolve_arm(spec, graph).model),
                    "checkpoint": str(item["resolved_checkpoint"]),
                    "checkpoint_sha256": item["checkpoint_sha256"],
                    "train_source_sha256": train_hash,
                }
            )
        candidates[name] = rows
        checked.append((binding.endpoints, binding.endpoints_sha256))
        checked.extend(
            (item["resolved_checkpoint"], item["checkpoint_sha256"]) for item in endpoints.values()
        )
    validate_authority_and_qa()
    for path, digest in [(freeze, freeze_sha256), *checked]:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: candidate input pin verification")
        if sha256(path) != digest:
            raise ValueError("candidate interface input changed")
    result = {
        "schema": "simplex_t_future_candidate_interface_v1",
        "status": "CANONICAL_ENDPOINT_INTERFACE_FROZEN_FUTURE_EXECUTION_REQUIRES_AUTHORIZATION",
        "scientific_freeze_sha256": freeze_sha256,
        "code_commit": record["code_commit"],
        "candidates": candidates,
        "input_contract_references": [
            pin
            for pin in record["files"]
            if pin["category"] in {"config", "roles", "time", "producers", "normalizers", "schemas"}
        ],
        "context_semantics": "RETROSPECTIVE_CURRENT_QUERY_ROI_NOT_VERIFIED_OBJECT_HISTORY",
        "input_requirement": (
            "Frozen producer-family features/tokens with original timing masks and "
            "TRAIN-only normalizers; source and exclusion pins must match"
        ),
        "fold_endpoint_aggregation_authorized": False,
        "seed_policy": (
            "Canonical seed7 endpoint references; no best-seed selection or signed-TTC averaging"
        ),
        "future_prerequisites": [
            "Separate comparison authorization and unexposed evaluation groups",
            "Explicit per-query producer/fold assignment or separately approved aggregation",
            "Matched Garl modality, preprocessing, history/ROI privileges and lineage",
            "Coordinated compute slot and absolute resource admission",
        ],
        "holdout_opened": False,
        "holdout_execution_authorized": False,
        "optimizer_updates": 0,
        "campaign_complete": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(output, result)
    return result
