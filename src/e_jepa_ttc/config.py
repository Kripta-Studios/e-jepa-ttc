"""Small, dependency-light configuration helpers for the public CLI.

The project deliberately keeps configuration loading explicit instead of
introducing a second experiment framework.  YAML is converted to ordinary
Python mappings and its canonical JSON representation is hashed so that run
artifacts can record exactly what was used.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is not a mapping."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping without interpolating environment-dependent values."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ConfigError(f"Configuration {source} must contain a YAML mapping.")
    return dict(payload)


def canonical_config_hash(config: Mapping[str, Any]) -> str:
    """Hash a JSON-safe canonical form of a configuration mapping."""

    try:
        encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError) as exc:
        raise ConfigError("Configuration cannot be serialized for hashing.") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = ["ConfigError", "canonical_config_hash", "load_config"]
