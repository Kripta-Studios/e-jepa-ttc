"""Immutable recipe contract for the single-chain all-TRAIN40 final refit."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA = "e_jepa_ttc_final_refit_recipe_v2"
TRAIN40_QUERIES = 88_744
TRAIN40_GROUPS = 40
ORIGINAL_P = 53_890
ORIGINAL_H = 18_422


def sha256(path: Path) -> str:
    """Return the SHA-256 of a regular file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash JSON with one deterministic representation."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class StageRecipe:
    """One fixed-endpoint stage in the prospective final chain."""

    stage_id: str
    kind: str
    updates: int
    batch_size: int
    weight_decay: float
    depends_on: tuple[str, ...]
    fresh_initialization: bool
    cache_schema: str | None = None


@dataclass(frozen=True)
class FinalRefitRecipe:
    """Complete recipe frozen before any scientific optimizer update."""

    schema: str
    family: str
    seed: int
    train_queries: int
    train_groups: int
    stacking: str
    event_branch: str
    native_train40_rows_sha256: str
    ordered_row_identity_sha256: str
    stages: tuple[StageRecipe, ...]
    source_sha256: dict[str, str]
    external_parent_sha256: dict[str, str]
    recipe_sha256: str

    def payload_without_hash(self) -> dict[str, Any]:
        """Return the canonical identity payload."""
        value = asdict(self)
        value.pop("recipe_sha256")
        return value


def _scaled_updates(base: int, population: int, original: int) -> int:
    return math.ceil(base * population / original)


def ordered_row_identity_sha256(tokens: Sequence[object], groups: Sequence[object]) -> str:
    """Hash the exact native row order and its token-to-group alignment."""
    if len(tokens) != len(groups) or not tokens:
        raise ValueError("Ordered row identity requires aligned nonempty token/group arrays")
    digest = hashlib.sha256(b"final_refit_train40_order_v1\0")
    for token_value, group_value in zip(tokens, groups, strict=True):
        for value in (str(token_value), str(group_value)):
            encoded = value.encode("utf-8")
            if not encoded:
                raise ValueError("Ordered row identity values must be nonempty")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _stages(train_queries: int) -> tuple[StageRecipe, ...]:
    upper_updates = _scaled_updates(2_500, train_queries, ORIGINAL_H)
    pair_updates = _scaled_updates(6_840, train_queries, ORIGINAL_P)
    return (
        StageRecipe("R_A5", "rgb_producer", 18 * math.ceil(train_queries / 32), 32, 1e-4, (), True),
        StageRecipe(
            "R_C2F", "rgb_producer", 18 * math.ceil(train_queries / 32), 32, 1e-4, (), True
        ),
        StageRecipe(
            "PAIR_E_MATCHED",
            "pair_head",
            pair_updates,
            32,
            1e-4,
            ("E_A5",),
            True,
            "final_refit_pair_cache_v1",
        ),
        StageRecipe(
            "PAIR_R",
            "pair_head",
            pair_updates,
            32,
            1e-4,
            ("R_A5",),
            True,
            "final_refit_pair_cache_v1",
        ),
        StageRecipe(
            "E_CTX_MATCHED",
            "temporal_head",
            upper_updates,
            128,
            1e-3,
            ("E_A5", "E_C2F", "PAIR_E_MATCHED"),
            True,
            "final_refit_context_cache_v1",
        ),
        StageRecipe(
            "R_CTX",
            "temporal_head",
            upper_updates,
            128,
            1e-3,
            ("R_A5", "R_C2F", "PAIR_R"),
            True,
            "final_refit_context_cache_v1",
        ),
        StageRecipe(
            "PHASE_ONLY_EVENT_S7_TRAIN40",
            "phase_only_head",
            upper_updates,
            128,
            1e-3,
            ("R_CTX", "E_CTX_MATCHED"),
            True,
            "final_refit_phase_cache_v1",
        ),
    )


def build_recipe(
    *,
    source_files: dict[str, Path],
    external_parent_sha256: dict[str, str],
    native_train40_rows_sha256: str,
    ordered_train40_identity_sha256: str,
    train_queries: int = TRAIN40_QUERIES,
    train_groups: int = TRAIN40_GROUPS,
) -> FinalRefitRecipe:
    """Build the sole sequential all-TRAIN recipe without reading targets."""
    if train_queries != TRAIN40_QUERIES or train_groups != TRAIN40_GROUPS:
        raise ValueError("Final recipe requires the admitted 88,744-query, 40-group TRAIN40 union")
    if not source_files:
        raise ValueError("Recipe must bind its executed sources")
    for label, digest in (
        ("native TRAIN40 rows", native_train40_rows_sha256),
        ("ordered TRAIN40 identity", ordered_train40_identity_sha256),
    ):
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"{label} must be a lowercase SHA-256")
    required_parents = {"E_A5", "E_C2F"}
    if set(external_parent_sha256) != required_parents:
        raise ValueError(f"Event ancestry must bind exactly {sorted(required_parents)}")
    if any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in external_parent_sha256.values()
    ):
        raise ValueError("External parent hashes must be SHA-256 hex strings")
    stages = _stages(train_queries)
    unhashed: dict[str, Any] = {
        "schema": SCHEMA,
        "family": "PHASE_ONLY-derived hybrid full-TRAIN40 RGB+event, seed 7",
        "seed": 7,
        "train_queries": train_queries,
        "train_groups": train_groups,
        "stacking": "declared sequential in-sample all-TRAIN; not OOF and not an evaluation",
        "event_branch": (
            "reuse only exact V12 all-TRAIN40 E_A5/E_C2F producers; rebuild V13/V16-contract "
            "emitted-phase PAIR_E, TRAIN40 normalizer and E_CTX at exposure-scaled endpoints"
        ),
        "native_train40_rows_sha256": native_train40_rows_sha256,
        "ordered_row_identity_sha256": ordered_train40_identity_sha256,
        "stages": [asdict(stage) for stage in stages],
        "source_sha256": {
            name: sha256(path.resolve()) for name, path in sorted(source_files.items())
        },
        "external_parent_sha256": dict(sorted(external_parent_sha256.items())),
    }
    return FinalRefitRecipe(
        **{**unhashed, "stages": stages}, recipe_sha256=canonical_sha256(unhashed)
    )


def write_recipe(path: Path, recipe: FinalRefitRecipe) -> None:
    """Atomically create an immutable recipe; reject changed replacement bytes."""
    value = asdict(recipe)
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError("Refusing to replace an existing final-refit recipe")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def load_recipe(path: Path) -> FinalRefitRecipe:
    """Load and verify a frozen recipe identity."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError("Unknown final-refit recipe schema")
    stages = tuple(
        StageRecipe(**{**item, "depends_on": tuple(item["depends_on"])})
        for item in value.pop("stages")
    )
    recipe = FinalRefitRecipe(stages=stages, **value)
    if canonical_sha256(recipe.payload_without_hash()) != recipe.recipe_sha256:
        raise ValueError("Final-refit recipe identity changed")
    if (
        recipe.seed != 7
        or recipe.train_queries != TRAIN40_QUERIES
        or recipe.train_groups != TRAIN40_GROUPS
        or stages != _stages(TRAIN40_QUERIES)
        or set(recipe.external_parent_sha256) != {"E_A5", "E_C2F"}
        or any(
            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
            for digest in (
                recipe.native_train40_rows_sha256,
                recipe.ordered_row_identity_sha256,
            )
        )
    ):
        raise ValueError("Final-refit stage graph or population differs")
    return recipe


def verify_bound_sources(recipe: FinalRefitRecipe) -> None:
    """Reject source drift for bindings emitted as ``file:<absolute path>``."""
    for name, expected in recipe.source_sha256.items():
        if not name.startswith("file:"):
            continue
        path = Path(name.removeprefix("file:"))
        if sha256(path) != expected:
            raise ValueError(f"Final-refit bound source changed: {path}")


def stage(recipe: FinalRefitRecipe, stage_id: str) -> StageRecipe:
    """Select exactly one declared stage."""
    matches = [item for item in recipe.stages if item.stage_id == stage_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate stage: {stage_id}")
    return matches[0]


__all__ = [
    "FinalRefitRecipe",
    "StageRecipe",
    "build_recipe",
    "canonical_sha256",
    "load_recipe",
    "ordered_row_identity_sha256",
    "sha256",
    "stage",
    "verify_bound_sources",
    "write_recipe",
]
