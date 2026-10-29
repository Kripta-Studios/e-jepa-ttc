from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.runpod_rgb_context_v8 import (
    ALLOWED_ROOTS,
    canonical,
    digest,
    load_layout,
    validate_equivalence,
)


@pytest.fixture(autouse=True)
def isolated_media(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Construct complete small media identities without local run artifacts."""
    global ARTIFACTS, ORIGINAL
    ARTIFACTS = tmp_path / "media"
    ARTIFACTS.mkdir()
    ORIGINAL = tmp_path / "config.json"
    ORIGINAL.write_text(json.dumps({"schema": "ejepa_rgb_media_scope_config_v1"}))
    for target_rotation, target_role, source_role in (
        (2, "REFINER", "GENERATION"),
        (4, "GENERATION", "REFINER"),
    ):
        plan = {
            "tars": [
                {
                    "relative": f"{source_role}.tar",
                    "members": [{"member": "frame.jpg", "payload_bytes": 32}],
                }
            ]
        }
        target = ARTIFACTS / f"R{target_rotation}_{target_role}_PLAN.json"
        source = ARTIFACTS / f"R3_{source_role}_PLAN.json"
        manifest = ARTIFACTS / f"R3_{source_role}_MANIFEST.json"
        for path in (target, source):
            path.write_text(json.dumps(plan))
        manifest.write_text(json.dumps({"status": "COMPLETE", "tars": plan["tars"]}))
        proof = {
            "schema": "ejepa_rgb_media_scope_reuse_proof_v1",
            "status": "PASS",
            "same_exact_member_names_and_payload_bytes": True,
            "same_logical_tar_paths": True,
            "physical_sidecar_roles_relabelled": False,
            "optimizer_updates": 0,
            "member_count": 1,
            "target": {
                "rotation": target_rotation,
                "role": target_role.lower(),
                "plan_sha256": digest(target),
            },
            "physical": {
                "rotation": 3,
                "role": source_role.lower(),
                "plan_sha256": digest(source),
                "manifest_sha256": digest(manifest),
            },
        }
        proof["receipt_sha256"] = canonical(proof)
        (ARTIFACTS / f"R{target_rotation}_{target_role}_REUSE_R3_{source_role}.json").write_text(
            json.dumps(proof)
        )


def _item(
    rotation: int,
    role: str,
    physical_rotation: int,
    physical_role: str,
    target: str,
    source: str,
    manifest: str,
    proof: str,
) -> dict[str, object]:
    paths = {
        "target_plan": ARTIFACTS / target,
        "source_plan": ARTIFACTS / source,
        "source_manifest": ARTIFACTS / manifest,
        "reuse_proof": ARTIFACTS / proof,
    }
    return {
        "mode": "REUSE",
        "rotation": rotation,
        "role": role,
        "physical_rotation": physical_rotation,
        "physical_role": physical_role,
        "physical_root": "/opt/ejepa_rgb_frames",
        **{f"{name}_path": str(path.resolve()) for name, path in paths.items()},
        **{f"{name}_sha256": digest(path) for name, path in paths.items()},
    }


def layout_value() -> dict[str, object]:
    return {
        "schema": "ejepa_rgb_compositional_media_layout_v8",
        "status": "PASS",
        "original_config_path": str(ORIGINAL.resolve()),
        "original_config_sha256": digest(ORIGINAL),
        "allowed_physical_roots": sorted(str(path) for path in ALLOWED_ROOTS),
        "scopes": [
            _item(
                2,
                "refiner",
                3,
                "generation",
                "R2_REFINER_PLAN.json",
                "R3_GENERATION_PLAN.json",
                "R3_GENERATION_MANIFEST.json",
                "R2_REFINER_REUSE_R3_GENERATION.json",
            ),
            _item(
                4,
                "generation",
                3,
                "refiner",
                "R4_GENERATION_PLAN.json",
                "R3_REFINER_PLAN.json",
                "R3_REFINER_MANIFEST.json",
                "R4_GENERATION_REUSE_R3_REFINER.json",
            ),
        ],
    }


def test_reuse_proofs_have_exact_target_source_member_maps(tmp_path: Path) -> None:
    value = layout_value()
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    layout = load_layout(path, ORIGINAL)
    assert set(layout["scope_map"]) == {"R2_refiner", "R4_generation"}
    for item in layout["scope_map"].values():
        proof = validate_equivalence(item)
        assert proof["physical_sidecar_roles_relabelled"] is False
        assert proof["optimizer_updates"] == 0


def test_layout_rejects_third_root(tmp_path: Path) -> None:
    value = layout_value()
    value["allowed_physical_roots"] = ["/opt/ejepa_rgb_frames", "/tmp/other"]
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="authority differs"):
        load_layout(path, ORIGINAL)


def test_declared_missing_manifest_is_waitable_not_authority_failure(tmp_path: Path) -> None:
    value = layout_value()
    scope = value["scopes"][0]
    scope["source_manifest_path"] = str((tmp_path / "pending_manifest.json").resolve())
    scope["source_manifest_sha256"] = "f" * 64
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    layout = load_layout(path, ORIGINAL)
    assert layout["scope_map"]["R2_refiner"]["source_manifest_sha256"] == "f" * 64


def test_proof_hash_or_target_genealogy_cannot_be_relabelled(tmp_path: Path) -> None:
    item = layout_value()["scopes"][0]
    assert isinstance(item, dict)
    changed = dict(item)
    changed["physical_role"] = "refiner"
    with pytest.raises(ValueError, match="equivalence differs"):
        validate_equivalence(changed)


def test_layout_authority_excludes_scope_mutation_only_from_header() -> None:
    value = layout_value()
    header = {key: item for key, item in value.items() if key != "scopes"}
    before = canonical(header)
    value["scopes"][0]["role"] = "generation"
    assert canonical(header) == before
