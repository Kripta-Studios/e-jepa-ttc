"""Strict YAML/schema loading and matched-arm validation for E-Clock X0."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jsonschema
import yaml


def load_x0_config(path: Path, *, schema_path: Path) -> dict[str, Any]:
    """Load one YAML config with unknown fields rejected by JSON Schema."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(schema, dict):
        raise ValueError("config and schema roots must be mappings")
    jsonschema.Draft202012Validator(schema).validate(payload)
    return payload


def validate_matched_base_dyn(base: dict[str, Any], dynamic: dict[str, Any]) -> None:
    """Prove every declarative field is matched except the scientific switch."""

    if base.get("arm_id") != "X0-BASE-U" or dynamic.get("arm_id") != "X0-DYN-U":
        raise ValueError("matched comparison requires BASE-U and DYN-U")
    allowed_differences = {"arm_id", "scientific_role", "motion_feature_mode"}
    base_common = {key: value for key, value in base.items() if key not in allowed_differences}
    dynamic_common = {
        key: value for key, value in dynamic.items() if key not in allowed_differences
    }
    if base_common != dynamic_common:
        raise ValueError("BASE/DYN declarative training contracts are not matched")
    if base["motion_feature_mode"] != "global_uniform_zeroed_control":
        raise ValueError("BASE motion mode drifted")
    if dynamic["motion_feature_mode"] != "global_uniform":
        raise ValueError("DYN motion mode drifted")


def assert_arm_execution_authorized(config: dict[str, Any]) -> None:
    """Fail closed for the config-only X0-DYN-W arm."""

    if config.get("arm_id") == "X0-DYN-W" or config.get("execution_authorized") is not True:
        raise PermissionError("X0-DYN-W is config/schema/loss-only and cannot execute")


__all__ = ["assert_arm_execution_authorized", "load_x0_config", "validate_matched_base_dyn"]
