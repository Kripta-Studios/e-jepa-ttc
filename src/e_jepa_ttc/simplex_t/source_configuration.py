"""Reconstruct campaign sources from pinned configuration and independently authorized roots."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import TypeVar

from e_jepa_ttc.artifacts.hashing import compute_file_hash

from .campaign_sources import CampaignSources, CompiledFold, MatchedBinding
from .dense_file_sources import DenseBinding
from .expansion_sources import ExpansionBinding
from .registry import FitSpec

_Binding = TypeVar("_Binding", bound=ExpansionBinding)


def sources_from_configuration(
    path: Path,
    *,
    expected_sha256: str,
    roots: dict[str, Path],
    graph: list[FitSpec],
    allowed_original_sequences: set[str],
    allowed_expansion_sequences: set[str],
) -> CampaignSources:
    """Parse explicit file bindings without loading cache payloads or authorizing fits.

    Roots and role sets come from the caller's verified owner interfaces. The
    configuration cannot introduce another root or assign protected groups TRAIN.
    Source identities, temporal authority, QA and scientific gates remain separate
    mandatory pre-fit checks. Missing files fail rather than selecting a fallback.
    """

    def digest(value: object) -> str:
        if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
            raise ValueError("configuration requires a canonical SHA256")
        return value

    if path.stat().st_size > 1_048_576 or compute_file_hash(str(path)) != digest(expected_sha256):
        raise ValueError("source configuration size or hash mismatch")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema", "original_sequences", "expansion_sequences", "original"}
    if not required <= set(config) or set(config) - required - {"expansion", "dense", "matched"}:
        raise ValueError("unknown or missing source configuration fields")
    if config["schema"] != "simplex_t_campaign_sources_v1":
        raise ValueError("unregistered source configuration schema")
    for key, authorized in (
        ("original_sequences", allowed_original_sequences),
        ("expansion_sequences", allowed_expansion_sequences),
    ):
        values = config[key]
        if (
            not isinstance(values, list)
            or any(not isinstance(v, str) or not v for v in values)
            or len(set(values)) != len(values)
            or set(values) != authorized
        ):
            raise ValueError("configuration role sets differ from independently authorized sets")
    approved_roots = {name: root.resolve(strict=True) for name, root in roots.items()}

    def resolve(reference: object, *, directory: bool) -> Path:
        if not isinstance(reference, dict) or set(reference) != {"root", "relative_path"}:
            raise ValueError("explicit root-relative source reference required")
        if reference["root"] not in approved_roots or not isinstance(
            reference["relative_path"], str
        ):
            raise ValueError("source references an unapproved root")
        relative = Path(reference["relative_path"])
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise ValueError("source path must be relative without traversal")
        root = approved_roots[reference["root"]]
        target = (root / relative).resolve(strict=True)
        if (
            not target.is_relative_to(root)
            or (target.is_dir() if directory else target.is_file()) is not True
        ):
            raise ValueError("source path escapes approved root or has wrong file kind")
        return target

    original = config["original"]
    if set(original) != {"index_root", "dedup_root", "historical_root", "ancestry_sha256", "folds"}:
        raise ValueError("original source schema mismatch")
    if set(original["folds"]) != {"0", "1", "2"}:
        raise ValueError("all original fold bindings required")
    folds = {}
    for key, value in original["folds"].items():
        if set(value) != {"path", "sha256"}:
            raise ValueError("compiled original fold schema mismatch")
        folds[int(key)] = CompiledFold(
            resolve(value["path"], directory=True), digest(value["sha256"])
        )

    def train_bindings(key: str, cls: type[_Binding]) -> dict[int, _Binding] | None:
        if key not in config:
            return None
        records = config[key]
        if set(records) != {"0", "1", "2"}:
            raise ValueError("all compiled TRAIN fold bindings required")
        names = {field.name for field in fields(cls)}
        result = {}
        for fold, record in records.items():
            if set(record) != names:
                raise ValueError("compiled TRAIN binding schema mismatch")
            result[int(fold)] = cls(
                compiled=resolve(record["compiled"], directory=True),
                compiled_sha256=digest(record["compiled_sha256"]),
                index_manifest=resolve(record["index_manifest"], directory=False),
                index_manifest_sha256=digest(record["index_manifest_sha256"]),
                dedup=resolve(record["dedup"], directory=False),
                pool=resolve(record["pool"], directory=False),
                pool_sha256=digest(record["pool_sha256"]),
                metadata=resolve(record["metadata"], directory=False),
                metadata_sha256=digest(record["metadata_sha256"]),
                labels=resolve(record["labels"], directory=False),
                labels_sha256=digest(record["labels_sha256"]),
                cache_identity_sha256=digest(record["cache_identity_sha256"]),
            )
        return result

    matched = None
    if "matched" in config:
        record = config["matched"]
        if set(record) != {"pool", "pool_sha256", "original_index_manifest_sha256"}:
            raise ValueError("matched source binding schema mismatch")
        matched = MatchedBinding(
            resolve(record["pool"], directory=False),
            digest(record["pool_sha256"]),
            digest(record["original_index_manifest_sha256"]),
        )
    return CampaignSources(
        graph,
        folds,
        index_root=resolve(original["index_root"], directory=True),
        dedup_root=resolve(original["dedup_root"], directory=True),
        historical_root=resolve(original["historical_root"], directory=True),
        ancestry_sha256=digest(original["ancestry_sha256"]),
        allowed_sequences=set(allowed_original_sequences),
        expansion_folds=train_bindings("expansion", ExpansionBinding),
        expansion_sequences=set(allowed_expansion_sequences),
        dense_folds=train_bindings("dense", DenseBinding),
        matched=matched,
    )
