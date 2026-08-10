#!/usr/bin/env python
"""Bind a signed train-only A4D calibration artifact into the tracked YAML."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return payload


def run(config_path: Path, calibration_path: Path) -> None:
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(calibration, dict) or not verify_artifact_hash(calibration):
        raise ValueError("calibration artifact signature is invalid")
    if calibration.get("artifact_type") != "a4d_dinov3_temporal_delta_weight_calibration_v1":
        raise ValueError("unexpected calibration artifact type")
    selected = float(calibration.get("selected_weight", float("nan")))
    if not math.isfinite(selected) or not 0.25 <= selected <= 4.0:
        raise ValueError("calibrated temporal weight is outside [0.25, 4.0]")

    config = _read_yaml(config_path)
    training = config.get("training")
    contract = config.get("decision_contract")
    if not isinstance(training, dict) or not isinstance(contract, dict):
        raise ValueError("A4D config lacks training/decision_contract")
    if training.get("representation_supervision") != (
        "dinov3_local_relational_temporal_delta"
    ):
        raise ValueError("config is not the A4D temporal-delta arm")
    if float(training.get("representation_temporal_delta_weight", -1.0)) != 0.0:
        raise ValueError("A4D temporal weight is already frozen; refusing to overwrite")
    if training.get(
        "representation_temporal_delta_calibration_artifact_sha256"
    ) != "REPLACE_AFTER_CALIBRATION":
        raise ValueError("A4D training calibration identity placeholder was modified")
    teacher = config.get("data", {}).get("dinov3_relational_teacher", {})
    if calibration.get("teacher_artifact_sha256") != teacher.get("artifact_sha256"):
        raise ValueError("calibration and A4D config teacher identities differ")

    change = contract.get("representation_change")
    if not isinstance(change, dict) or not isinstance(
        change.get("temporal_delta_calibration"), dict
    ):
        raise ValueError("A4D temporal calibration contract is missing")
    declared = change["temporal_delta_calibration"]
    expected_artifact = (ROOT / str(declared["artifact"])).resolve()
    if expected_artifact != calibration_path.resolve():
        raise ValueError("calibration path differs from A4D decision contract")
    if declared.get("file_sha256") != "REPLACE_AFTER_CALIBRATION" or declared.get(
        "artifact_sha256"
    ) != "REPLACE_AFTER_CALIBRATION":
        raise ValueError("A4D calibration hash placeholders were already modified")

    file_sha = _sha256(calibration_path)
    signed_sha = str(calibration["artifact_sha256"])
    text = config_path.read_text(encoding="utf-8")
    text, count_weight = re.subn(
        r"(?m)^(\s*representation_temporal_delta_weight:\s*)0\.0\s*$",
        rf"\g<1>{selected:.17g}",
        text,
        count=1,
    )
    text, count_training_sha = re.subn(
        (
            r'(?m)^(\s*representation_temporal_delta_calibration_artifact_sha256:\s*)'
            r'"REPLACE_AFTER_CALIBRATION"\s*$'
        ),
        rf'\g<1>"{signed_sha}"',
        text,
        count=1,
    )
    text, count_file = re.subn(
        r'(?m)^(\s*file_sha256:\s*)"REPLACE_AFTER_CALIBRATION"\s*$',
        rf'\g<1>"{file_sha}"',
        text,
        count=1,
    )
    text, count_signed = re.subn(
        r'(?m)^(\s*artifact_sha256:\s*)"REPLACE_AFTER_CALIBRATION"\s*$',
        rf'\g<1>"{signed_sha}"',
        text,
        count=1,
    )
    if (count_weight, count_training_sha, count_file, count_signed) != (1, 1, 1, 1):
        raise ValueError("A4D YAML placeholders do not match the frozen template")
    config_path.write_text(text, encoding="utf-8", newline="\n")

    frozen = _read_yaml(config_path)
    frozen_training = frozen["training"]
    frozen_calibration = frozen["decision_contract"]["representation_change"][
        "temporal_delta_calibration"
    ]
    if not math.isclose(
        float(frozen_training["representation_temporal_delta_weight"]),
        selected,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError("frozen A4D weight did not round-trip")
    if (
        frozen_training["representation_temporal_delta_calibration_artifact_sha256"]
        != signed_sha
    ):
        raise RuntimeError("frozen A4D training calibration identity did not round-trip")
    if frozen_calibration["file_sha256"] != file_sha:
        raise RuntimeError("frozen A4D file hash did not round-trip")
    if frozen_calibration["artifact_sha256"] != signed_sha:
        raise RuntimeError("frozen A4D signed hash did not round-trip")

    print(f"selected_weight={selected:.17g}")
    print(f"calibration_file_sha256={file_sha}")
    print(f"calibration_artifact_sha256={signed_sha}")
    print(f"updated_config={config_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.config.resolve(), args.calibration.resolve())
    except Exception as error:
        parser.exit(2, f"A4D freeze failed: {type(error).__name__}: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
