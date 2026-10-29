from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from scripts import runpod_rgb_context_v6 as v6
from scripts.runpod_rgb_context_v9 import patched_verify, select_decoded_proof


def legacy_record() -> dict:
    return {
        "decoded_pixel_parity": True,
        "payload_recompressed": False,
        "members": [
            {
                "member": f"frame_{index}.jpg",
                "payload_bytes": 32,
                "compressed_payload_sha256": "a" * 64,
                "decoded_rgb_sha256": "b" * 64,
                "decoded_rgb_shape": [2, 3, 3],
            }
            for index in range(2)
        ],
    }


def test_legacy_and_new_records_select_one_proof() -> None:
    legacy = legacy_record()
    current = {"members": [{"decoded_pixel_parity_sample": True}, {}]}
    assert select_decoded_proof(legacy) == [legacy["members"][0]]
    selected = select_decoded_proof(current)
    assert len(selected) == 1 and selected[0]["decoded_pixel_parity_sample"] is True


def test_legacy_requires_every_member_decoded_closure() -> None:
    legacy = legacy_record()
    legacy["members"][1].pop("decoded_rgb_sha256")
    with pytest.raises(ValueError, match="closure differs"):
        select_decoded_proof(legacy)


def test_zero_flag_new_record_is_not_treated_as_legacy() -> None:
    record = {
        "decoded_pixel_samples": 1,
        "decoded_pixel_parity_basis": "one deterministic decode per TAR",
        "decoded_pixel_parity": True,
        "payload_recompressed": False,
        "members": [],
    }
    with pytest.raises(ValueError, match="members differ"):
        select_decoded_proof(record)


def test_multiple_flags_remain_rejected() -> None:
    record = {
        "members": [
            {"decoded_pixel_parity_sample": True},
            {"decoded_pixel_parity_sample": True},
        ]
    }
    with pytest.raises(ValueError, match="exactly one"):
        select_decoded_proof(record)


def test_ast_patch_is_exact_and_preserves_v6_module() -> None:
    original = v6.verify_scope
    patched = patched_verify(v6)
    assert callable(patched)
    assert v6.verify_scope is original


def test_patched_verify_tracks_dynamic_v8_media_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patched = patched_verify(v6)
    monkeypatch.setattr(v6, "MEDIA_ROOT", tmp_path)
    assert patched({"scope_map": {}}, 4, "refiner") is None
    namespace = inspect.getclosurevars(patched).nonlocals["namespace"]
    assert namespace["MEDIA_ROOT"] == tmp_path
