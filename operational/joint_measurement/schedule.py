"""Execution order can change without changing the registered scientific matrix."""

from __future__ import annotations

from operational.range_transfer.common import digest, read

from .common import RUN, config
from .train import sources


def validate_order(record: dict, fits: list[str], config_hash: str, source_hash: str) -> list[str]:
    """Accept only a permutation of every unchanged primary fit."""
    order = record.get("fits", [])
    if (
        len(order) != len(fits)
        or set(order) != set(fits)
        or record.get("config_hash") != config_hash
        or record.get("scientific_source_hash") != source_hash
        or not str(record.get("reason", "")).strip()
    ):
        raise ValueError("Execution order must preserve the complete admitted matrix")
    return order


def execution_order() -> list[str]:
    """Use the original order unless a bound scheduling decision exists."""
    recipe = config()
    path = RUN / "PRIMARY_EXECUTION_ORDER.json"
    if not path.exists():
        return recipe["fits"]
    return validate_order(read(path), recipe["fits"], digest(recipe), digest(sources()))
