"""Pure-array tests for physical-only OOF stitching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import operational.final_refit.oof_stitch as stitch_module
from operational.final_refit.oof_stitch import (
    FRAGMENT_SCHEMA,
    NORMALIZER_SCHEMA,
    PHASE17_RAW_EVENT,
    PHASE20_CONTEXT,
    SEMANTICS,
    fit_oof_phase17_normalizer,
    payload_sha256,
    stitch_from_verified_lineage,
    stitch_oof_physical,
    write_oof_stitch,
)
from operational.final_refit.recipe import canonical_sha256

FOLD_SHA = "f" * 64


def _native() -> tuple[list[str], list[str]]:
    return [f"q{index}" for index in range(10)], [f"g{index // 2}" for index in range(10)]


def _fragments(interface: str):
    tokens, groups = _native()
    fragments = []
    for rotation in reversed(range(5)):
        ordinals = np.asarray([2 * rotation + 1, 2 * rotation], np.int64)
        width = 17 if interface == PHASE17_RAW_EVENT else 20
        values = {
            "sample_token": np.asarray([tokens[index] for index in ordinals]),
            "group_id": np.asarray([groups[index] for index in ordinals]),
            "source_ordinal": ordinals,
        }
        if interface == PHASE17_RAW_EVENT:
            values.update(
                {
                    "features": np.stack(
                        [np.full((1, 17), index, np.float32) for index in ordinals]
                    ),
                    "timing": np.zeros((2, 1, 4), np.float32),
                    "valid": np.ones((2, 1), np.bool_),
                    "expert_phase": np.stack(
                        [
                            np.asarray([index, index + 1, index + 2], np.float32)
                            for index in ordinals
                        ]
                    ),
                    "observation_id": np.asarray([[f"o{index}"] for index in ordinals]),
                }
            )
        else:
            values["features"] = np.stack([np.full(20, index, np.float32) for index in ordinals])
            values["rgb_width_logits"] = np.stack(
                [np.asarray([index, -index], np.float32) for index in ordinals]
            )
        receipt = {
            "schema": FRAGMENT_SCHEMA,
            "status": "COMPLETE",
            "interface": interface,
            "feature_semantics": SEMANTICS,
            "width": width,
            "rotation": rotation,
            "role": "generation",
            "generation_manifest_sha256": f"{rotation + 1:x}" * 64,
            "ordered_sample_token_sha256": canonical_sha256(
                values["sample_token"].astype(str).tolist()
            ),
            "group_ids": sorted(set(values["group_id"].astype(str).tolist())),
            "payload_sha256": payload_sha256(values),
            "endpoint_lineages": {
                stage: {
                    "path": f"rotation_{rotation}_{stage}.json",
                    "file_sha256": "a" * 64,
                    "lineage_sha256": "b" * 64,
                }
                for stage in (
                    ("E_A5", "E_C2F", "PAIR_E")
                    if interface == PHASE17_RAW_EVENT
                    else ("E_CTX", "R_CTX")
                )
            },
        }
        fragments.append((values, receipt))
    return fragments


def _admit_test_lineages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stitch_module, "file_sha256", lambda _path: "a" * 64)

    def verify(path: Path):
        stem = path.stem
        rotation = int(stem.split("_")[1])
        stage = "_".join(stem.split("_")[2:])
        return {
            "stage_id": stage,
            "rotation": rotation,
            "lineage_sha256": "b" * 64,
            "training_feature_ancestry": {"generation_rows_excluded_from_all_ancestors": True},
            "deployment_dependencies": {},
        }

    monkeypatch.setattr(stitch_module, "verify_endpoint_lineage", verify)


def test_phase20_stitch_restores_native_order_without_hidden_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _admit_test_lineages(monkeypatch)
    tokens, groups = _native()
    values, receipt = stitch_oof_physical(
        tokens,
        groups,
        _fragments(PHASE20_CONTEXT),
        interface=PHASE20_CONTEXT,
        fold_contract_sha256=FOLD_SHA,
        expected_population=10,
        expected_groups=5,
    )
    assert set(values) == {
        "sample_token",
        "group_id",
        "source_ordinal",
        "features",
        "rgb_width_logits",
    }
    np.testing.assert_array_equal(values["source_ordinal"], np.arange(10))
    np.testing.assert_array_equal(values["features"][:, 0], np.arange(10))
    np.testing.assert_array_equal(values["rgb_width_logits"][:, 0], np.arange(10))
    assert receipt["cross_fold_learned_coordinates"] is False
    assert receipt["head_adapter_status"] == "REQUIRED_NOT_IMPLEMENTED"
    assert len(receipt["fragments"]) == 5


def test_stitch_rejects_endpoint_without_generation_exclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _admit_test_lineages(monkeypatch)
    tokens, groups = _native()
    fragments = _fragments(PHASE17_RAW_EVENT)
    monkeypatch.setattr(
        stitch_module,
        "verify_endpoint_lineage",
        lambda _path: {
            "stage_id": "PAIR_E",
            "rotation": 4,
            "lineage_sha256": "b" * 64,
            "training_feature_ancestry": {"generation_rows_excluded_from_all_ancestors": False},
            "deployment_dependencies": {},
        },
    )
    with pytest.raises(ValueError, match="generation exclusion"):
        stitch_oof_physical(
            tokens,
            groups,
            fragments,
            interface=PHASE17_RAW_EVENT,
            fold_contract_sha256=FOLD_SHA,
            expected_population=10,
            expected_groups=5,
        )


def test_stitch_rejects_latent_or_incomplete_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _admit_test_lineages(monkeypatch)
    tokens, groups = _native()
    fragments = _fragments(PHASE17_RAW_EVENT)
    fragments[0][0]["hidden160"] = np.zeros((2, 160), np.float32)
    with pytest.raises(ValueError, match="fields differ"):
        stitch_oof_physical(
            tokens,
            groups,
            fragments,
            interface=PHASE17_RAW_EVENT,
            fold_contract_sha256=FOLD_SHA,
            expected_population=10,
            expected_groups=5,
        )
    fragments = _fragments(PHASE17_RAW_EVENT)
    fragments[0][0]["source_ordinal"][0] = 0
    fragments[0][1]["payload_sha256"] = payload_sha256(fragments[0][0])
    with pytest.raises(ValueError, match="cover every native row"):
        stitch_oof_physical(
            tokens,
            groups,
            fragments,
            interface=PHASE17_RAW_EVENT,
            fold_contract_sha256=FOLD_SHA,
            expected_population=10,
            expected_groups=5,
        )


def test_phase17_normalizer_rejects_structural_stitch_without_lineage_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _admit_test_lineages(monkeypatch)
    tokens, groups = _native()
    values, receipt = stitch_oof_physical(
        tokens,
        groups,
        _fragments(PHASE17_RAW_EVENT),
        interface=PHASE17_RAW_EVENT,
        fold_contract_sha256=FOLD_SHA,
        expected_population=10,
        expected_groups=5,
    )
    stitch = tmp_path / "phase17_oof.npz"
    write_oof_stitch(stitch, values, receipt)
    with pytest.raises(ValueError, match="lineage authority"):
        fit_oof_phase17_normalizer(stitch, tmp_path / "normalizer.json")


def test_production_loader_requires_independent_lineage_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _admit_test_lineages(monkeypatch)
    fragments = _fragments(PHASE17_RAW_EVENT)
    authority_fragments = []
    for values, fragment in fragments:
        rotation = fragment["rotation"]
        path = tmp_path / f"fragment_{rotation}.npz"
        np.savez_compressed(path, **values)
        authority_fragments.append(
            {
                "rotation": rotation,
                "payload_path": str(path),
                "generation_rows_sha256": fragment["generation_manifest_sha256"],
                "ordered_sample_token_sha256": fragment["ordered_sample_token_sha256"],
                "group_ids": fragment["group_ids"],
                "endpoint_lineages": fragment["endpoint_lineages"],
            }
        )
    authority = {
        "status": "VERIFIED_STRUCTURAL_OOF_STITCH_NOT_SCIENTIFICALLY_ADMITTED",
        "interface": PHASE17_RAW_EVENT,
        "fold_contract_sha256": FOLD_SHA,
        "population": 10,
        "groups": 5,
        "fragments": sorted(authority_fragments, key=lambda item: item["rotation"]),
        "stitch_sha256": "c" * 64,
        "scientific_launch_authorized": False,
        "blind_validation_claim": False,
        "targets_read_for_stitching": False,
    }
    monkeypatch.setattr(stitch_module, "verify_stitch_receipt", lambda _value: authority)
    tokens, groups = _native()
    values, receipt = stitch_from_verified_lineage(authority, tokens, groups)
    np.testing.assert_array_equal(values["source_ordinal"], np.arange(10))
    assert receipt["lineage_authority_verified"] is True
    assert receipt["lineage_authority_sha256"] == "c" * 64
    stitch = tmp_path / "verified_phase17_oof.npz"
    write_oof_stitch(stitch, values, receipt)
    normalizer = fit_oof_phase17_normalizer(stitch, tmp_path / "normalizer.json")
    assert normalizer["schema"] == NORMALIZER_SCHEMA
    assert normalizer["status"] == "STRUCTURAL_NORMALIZER_NOT_SCIENTIFICALLY_ADMITTED"
    assert normalizer["targets_read"] is False
    assert normalizer["unique_observations"] == 10
    np.testing.assert_allclose(normalizer["mean"], np.full(17, 4.5))
