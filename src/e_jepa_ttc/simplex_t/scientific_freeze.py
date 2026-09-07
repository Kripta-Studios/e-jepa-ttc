"""Immutable freeze publication; semantic prerequisite checks remain mandatory."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json

from .arms import resolve_arm
from .freeze_integrity import FrozenFile, verify_code_commit, verify_files, verify_stage_sources
from .lifecycle import ExclusiveLease
from .phase_manifest import fit_key
from .registry import registered_graph


def _content(
    *,
    files: list[FrozenFile],
    roots: dict[str, Path],
    preparation: FrozenFile,
    technical_ledger: FrozenFile,
    code_commit: str,
) -> dict:
    if len(code_commit) != 40 or set(code_commit) - set("0123456789abcdef"):
        raise ValueError("full source commit required")
    verify_files(files, roots)
    verify_code_commit(files, roots, code_commit)
    if preparation not in files or preparation.category != "normalizers":
        raise ValueError("complete source preparation must be pinned as normalization evidence")
    if technical_ledger not in files or technical_ledger.category != "qa":
        raise ValueError("technical work ledger must be pinned as QA evidence")

    def read(pin: FrozenFile) -> dict:
        path = (roots[pin.root] / pin.relative_path).resolve(strict=True)
        if path.stat().st_size > 8_388_608:
            raise ValueError("freeze metadata exceeds bound")
        return json.loads(path.read_text(encoding="utf-8"))

    prepared, ledger = read(preparation), read(technical_ledger)
    if (
        prepared.get("schema") != "simplex_t_source_preparation_v1"
        or prepared.get("status") != "SOURCE_IDENTITIES_COMPLETE_NOT_SCIENTIFIC_FREEZE"
        or type(prepared.get("optimizer_updates")) is not int
        or prepared.get("optimizer_updates") != 0
        or prepared.get("scientific_freeze") is not False
        or prepared.get("gates_enabled") is not False
    ):
        raise ValueError("complete pre-fit source preparation required")
    contract = prepared["contract"]
    flags = contract["availability"]
    graph = registered_graph(**flags)
    serialized = [{"fit": asdict(s), "model": asdict(resolve_arm(s, graph).model)} for s in graph]
    if contract["fits"] != serialized or set(prepared["identities"]) != {fit_key(s) for s in graph}:
        raise ValueError("preparation graph differs from complete canonical candidates")
    if contract.get("loaded_source_qa") != "simplex_t_full_population_gather_qa_v1" or set(
        prepared.get("source_qa", {})
    ) != set(prepared["identities"]):
        raise ValueError("complete loaded-source QA required before scientific freeze")
    for spec in graph:
        key = fit_key(spec)
        pair = prepared["source_qa"][key]
        binding = resolve_arm(spec, graph)
        if set(pair) != {"inner_oof", "outer_dev"} or set(prepared["identities"][key]) != set(pair):
            raise ValueError("both loaded-source QA roles required")
        for role, qa in pair.items():
            if (
                qa.get("schema") != "simplex_t_full_population_gather_qa_v1"
                or qa.get("source_sha256") != prepared["identities"][key][role]
                or type(qa.get("history_length")) is not int
                or type(qa.get("feature_count")) is not int
                or qa.get("history_length") != binding.history
                or qa.get("feature_count") != binding.model.feature_count
                or type(qa.get("queries")) is not int
                or qa["queries"] < 1
                or type(qa.get("valid_slots")) is not int
                or not qa["queries"] <= qa["valid_slots"] <= qa["queries"] * binding.history
                or qa.get("model_inference") is not False
                or type(qa.get("optimizer_updates")) is not int
                or qa["optimizer_updates"] != 0
                or not isinstance(qa.get("normalizer_ids_sha256"), str)
                or len(qa["normalizer_ids_sha256"]) != 64
                or set(qa["normalizer_ids_sha256"]) - set("0123456789abcdef")
            ):
                raise ValueError("loaded-source QA differs from canonical source")
        if pair["inner_oof"]["normalizer_ids_sha256"] != pair["outer_dev"]["normalizer_ids_sha256"]:
            raise ValueError("TRAIN and OLD source QA use different normalizers")
    for stage in {s.stage for s in graph}:
        subset = {fit_key(s): prepared["identities"][fit_key(s)] for s in graph if s.stage == stage}
        verify_stage_sources(stage=stage, availability=flags, frozen=subset, observed=subset)
    for category, digest in (
        ("config", contract["configuration_sha256"]),
        ("roles", contract["authority_sha256"]),
    ):
        if not any(pin.category == category and pin.sha256 == digest for pin in files):
            raise ValueError("source configuration/authority missing from freeze file pins")
    if ledger.get("schema") != "simplex_t_technical_budget_v1":
        raise ValueError("recognized technical ledger required")
    reservations = ledger["reservations"]
    if not isinstance(reservations, dict) or any(
        type(n) is not int or n < 1 for n in reservations.values()
    ):
        raise ValueError("invalid technical reservations")
    reserved = sum(reservations.values())
    if reserved > 1000:
        raise ValueError("technical work exceeds registered cap")
    primary = "D1" if flags["d1"] else "D0"
    return {
        "schema": "simplex_t_scientific_freeze_v1",
        "status": "FROZEN_INPUTS_STAGE_GATES_REQUIRED",
        "code_commit": code_commit,
        "files": [asdict(pin) for pin in files],
        "preparation": asdict(preparation),
        "technical_ledger": asdict(technical_ledger),
        "source_contract": contract,
        "source_identities": prepared["identities"],
        "canonical_scalar": f"TPR-{primary}-H8-C160",
        "canonical_latent": f"LATENT-{primary}-H8-C160" if flags["latent"] else None,
        "registered_scientific_updates": sum(s.updates for s in graph),
        "technical_reserved_not_execution_claim": reserved,
        "hard_optimizer_update_cap": 250000,
        "practical_stage_gates_enabled_by_freeze": False,
        "holdout_authorized": False,
    }


def publish_scientific_freeze(
    output: Path,
    *,
    files: list[FrozenFile],
    roots: dict[str, Path],
    preparation: FrozenFile,
    technical_ledger: FrozenFile,
    code_commit: str,
    validate_prerequisites: Callable[[], None],
) -> dict:
    """Publish only after actual authority, complete QA and availability validation.

    The callback must verify semantic evidence (including all producer exclusions,
    real replay/resume QA and pre-score pool availability), not merely JSON flags.
    This library does not implement that campaign-specific authority checker.
    """
    validate_prerequisites()
    result = _content(
        files=files,
        roots=roots,
        preparation=preparation,
        technical_ledger=technical_ledger,
        code_commit=code_commit,
    )
    with ExclusiveLease(output.with_suffix(".lock")):
        validate_prerequisites()
        verify_files(files, roots)
        write_new_json(output, result)
    return result


def read_scientific_freeze(
    path: Path,
    *,
    expected_sha256: str,
    roots: dict[str, Path],
    validate_prerequisites: Callable[[], None],
) -> dict:
    """Recheck every pinned input and canonical contract; never open a holdout."""
    validate_prerequisites()
    if path.stat().st_size > 8_388_608 or compute_file_hash(str(path)) != expected_sha256:
        raise ValueError("scientific freeze bytes changed")
    record = json.loads(path.read_text(encoding="utf-8"))
    actual = _content(
        files=[FrozenFile(**pin) for pin in record["files"]],
        roots=roots,
        preparation=FrozenFile(**record["preparation"]),
        technical_ledger=FrozenFile(**record["technical_ledger"]),
        code_commit=record["code_commit"],
    )
    if actual != record:
        raise ValueError("scientific freeze canonical content changed")
    return record
