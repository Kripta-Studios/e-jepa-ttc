"""V8 caches bind protocol by file hash, not a nested artifact object."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.training.scientific_recovery_v8_trainer import (
    resolve_v8_cache_protocol_binding,
    resolve_v8_frozen_manifest_binding,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signed_protocol(path: Path) -> dict[str, str]:
    payload = sign_artifact(
        {
            "artifact_type": "scientific_recovery_v8_temporal_protocol_v1",
            "sample_contract": {"ordered_token_ids_sha256": "a" * 64},
        }
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _signed_freeze(path: Path, *, fold_name: str, fold_sha256: str) -> dict[str, str]:
    payload = sign_artifact(
        {
            "artifact_type": "scientific_recovery_v8_frozen_config_manifest_v1",
            "enabled_seed7_configs": {
                fold_name: {
                    "path": (
                        f"configs/experiment/scientific_recovery_v8_fold_chain/{fold_name}.yaml"
                    ),
                    "sha256": fold_sha256,
                }
            },
        }
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def test_protocol_binding_uses_cache_file_hash_fields(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    payload = _signed_protocol(protocol_path)
    artifact, binding = resolve_v8_cache_protocol_binding(
        {
            "protocol_path": str(protocol_path),
            "protocol_sha256": _sha(protocol_path),
        }
    )
    assert artifact == payload["artifact_sha256"]
    assert binding["artifact_sha256"] == payload["artifact_sha256"]
    assert binding["sha256"] == _sha(protocol_path)


def test_protocol_binding_refuses_file_hash_mismatch(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    _signed_protocol(protocol_path)
    with pytest.raises(ValueError, match="protocol file hash differs"):
        resolve_v8_cache_protocol_binding(
            {
                "protocol_path": str(protocol_path),
                "protocol_sha256": "0" * 64,
            }
        )


def test_protocol_binding_refuses_missing_path_and_nested_object() -> None:
    with pytest.raises(ValueError, match="lacks protocol artifact bindings"):
        resolve_v8_cache_protocol_binding({"row_identity_sha256": "f" * 64})


def test_frozen_manifest_binding_uses_sibling_freeze_file(tmp_path: Path) -> None:
    config_path = tmp_path / "timevol20_3_fold0_seed7.yaml"
    config_path.write_text("experiment:\n  name: timevol20_3_fold0_seed7\n", encoding="utf-8")
    freeze_path = tmp_path / "frozen_manifest.json"
    payload = _signed_freeze(
        freeze_path, fold_name="timevol20_3_fold0_seed7", fold_sha256=_sha(config_path)
    )
    artifact, binding = resolve_v8_frozen_manifest_binding({}, config_path=config_path)
    assert artifact == payload["artifact_sha256"]
    assert binding["artifact_sha256"] == payload["artifact_sha256"]
    assert Path(binding["path"]) == freeze_path


def test_frozen_manifest_binding_refuses_unlisted_fold_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "timevol20_3_fold0_seed7.yaml"
    config_path.write_text("experiment:\n  name: timevol20_3_fold0_seed7\n", encoding="utf-8")
    _signed_freeze(
        tmp_path / "frozen_manifest.json",
        fold_name="timevol20_3_fold0_seed7",
        fold_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="does not list this V8 fold config"):
        resolve_v8_frozen_manifest_binding({}, config_path=config_path)
