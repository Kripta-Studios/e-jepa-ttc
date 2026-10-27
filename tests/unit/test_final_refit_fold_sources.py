from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from operational.final_refit.fold_producers import (
    build_fold_recipe,
    fit_fold_producer_technical,
)
from operational.final_refit.fold_sources import (
    FoldRGBProducerSource,
    FoldSourceInventory,
    _concat_batches,
    load_fold_inventory,
)
from operational.final_refit.recipe import canonical_sha256, sha256


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _contract(root: Path) -> Path:
    counts = [88_744 // 40 + (index < 88_744 % 40) for index in range(40)]
    assignments = {
        "producer": range(24),
        "refiner": range(24, 32),
        "generation": range(32, 40),
    }
    references: dict[str, Any] = {}
    native = 0
    for role, group_indices in assignments.items():
        rows: list[dict[str, Any]] = []
        for group in group_indices:
            for local in range(counts[group]):
                rows.append(
                    {
                        "rotation": 0,
                        "rotation_role": role,
                        "historical_local_ordinal": local,
                        "source_ordinal": native,
                        "sample_token": f"token-{native}",
                        "group_id": f"group-{group:02d}",
                        "historical_role": ("P", "H", "V")[group % 3],
                    }
                )
                native += 1
        path = root / f"rotation_0_{role}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        references[role] = {
            "path": path.name,
            "sha256": sha256(path),
            "queries": len(rows),
        }
    manifest = {
        "schema": "final_refit_rotation_roles_v1",
        "rotation": 0,
        "status": "VERIFIED_REFERENCE_MANIFESTS",
        "generation_ancestry_excludes_group": True,
        "roles": references,
    }
    manifest_path = root / "rotation_0.json"
    _write(manifest_path, manifest)
    contract = {
        "schema": "final_refit_oof_fold_contract_v1",
        "status": "PASS",
        "population": {"queries": 88_744, "groups": 40},
        "generation_coverage": "EACH_TRAIN40_TOKEN_EXACTLY_ONCE",
        "rotations": [
            {"path": manifest_path.name, "sha256": sha256(manifest_path)} for _ in range(5)
        ],
    }
    contract_path = root / "FOLD_CONTRACT.json"
    _write(contract_path, contract)
    return contract_path


def _batch(tokens: list[str], values: list[float]) -> ObjectEventV4Batch:
    count = len(tokens)
    base = torch.as_tensor(values, dtype=torch.float32).reshape(count, 1)
    return ObjectEventV4Batch(
        events=base.reshape(count, 1, 1, 1, 1),
        delta_t_s=base,
        observable_motion=torch.empty(count, 0),
        visible_heights_px=base,
        target_ttc_s=base[:, 0],
        boxes_xyxy=base.repeat(1, 4).reshape(count, 1, 4),
        common_square_xyxy=base.repeat(1, 4),
        sequence_ids=[f"sequence-{token}" for token in tokens],
        sample_tokens=tokens,
        track_ids=[f"track-{token}" for token in tokens],
        dinov3_relation_targets=base,
        dinov3_relation_valid=torch.ones_like(base, dtype=torch.bool),
    )


def test_fold_inventory_binds_exact_producer_and_forbidden_roles(tmp_path: Path) -> None:
    inventory, rows = load_fold_inventory(_contract(tmp_path), 0)
    assert inventory.producer_population == len(rows)
    assert len(inventory.producer_group_ids) == 24
    assert set(inventory.forbidden) == {"refiner", "generation"}
    assert inventory.inventory_sha256 == canonical_sha256(inventory.payload())
    assert sum(inventory.historical_role_counts.values()) == inventory.producer_population


def test_fold_batch_concatenation_restores_requested_order() -> None:
    merged = _concat_batches(
        [
            ([1], _batch(["one"], [1.0])),
            ([0, 2], _batch(["zero", "two"], [0.0, 2.0])),
        ]
    )
    assert merged.sample_tokens == ["zero", "one", "two"]
    torch.testing.assert_close(merged.target_ttc_s, torch.tensor([0.0, 1.0, 2.0]))


class _FakeNativeRGB:
    frame_counts = (2, 3, 2, 3)
    input_span_us = (10, 20, 30, 40)
    batch_keys = (("P", 2), ("H", 3), ("P", 2), ("H", 3))
    identity = {"identity_sha256": "c" * 64}

    def row(self, index: int) -> dict[str, object]:
        return {
            "sample_token": f"token-{index}",
            "group_id": f"group-{index}",
            "historical_role": ("P", "H", "P", "H")[index],
            "historical_role_ordinal": (0, 0, 1, 1)[index],
        }

    def batch(self, indices: list[int], modality: str) -> ObjectEventV4Batch:
        assert modality == "rgb"
        return _batch([f"token-{index}" for index in indices], [float(index) for index in indices])

    def close(self) -> None:
        return None


def test_fold_rgb_source_uses_native_ordinals_but_returns_manifest_order() -> None:
    payload = {
        "rotation": 0,
        "fold_contract_path": "contract",
        "fold_contract_sha256": "a" * 64,
        "rotation_manifest_path": "manifest",
        "rotation_manifest_sha256": "b" * 64,
        "producer_rows_path": "producer",
        "producer_rows_sha256": "c" * 64,
        "producer_population": 2,
        "producer_group_ids": ("group-1", "group-2"),
        "ordered_sample_token_sha256": "d" * 64,
        "ordered_native_ordinal_sha256": "e" * 64,
        "historical_role_counts": {"P": 1, "H": 1, "V": 0},
        "forbidden": {},
        "inventory_sha256": "f" * 64,
    }
    rows = pd.DataFrame(
        [
            {
                "source_ordinal": 1,
                "sample_token": "token-1",
                "group_id": "group-1",
                "historical_role": "H",
                "historical_local_ordinal": 0,
            },
            {
                "source_ordinal": 2,
                "sample_token": "token-2",
                "group_id": "group-2",
                "historical_role": "P",
                "historical_local_ordinal": 1,
            },
        ]
    )
    source = FoldRGBProducerSource(
        _FakeNativeRGB(),  # type: ignore[arg-type]
        FoldSourceInventory(**payload),
        rows,
    )
    result = source.batch([1, 0], "rgb")
    assert result.sample_tokens == ["token-2", "token-1"]
    torch.testing.assert_close(result.target_ttc_s, torch.tensor([2.0, 1.0]))


class _FakeFoldSource:
    population_size: int = 53
    frame_counts: Sequence[int] = (3,) * population_size
    input_span_us: Sequence[int] = (1,) * population_size
    sampling_groups: Sequence[Sequence[int]] = (tuple(range(population_size)),)
    identity: Mapping[str, Any] = {
        "schema": "final_refit_fold_source_v1",
        "role": "producer_pair",
        "rotation": 2,
        "population_size": population_size,
        "inventory_sha256": "a" * 64,
        "identity_sha256": "b" * 64,
    }

    def batch(self, indices: Sequence[int], modality: str) -> ObjectEventV4Batch:
        raise AssertionError("scientific fail-closed gate must run before data access")

    def close(self) -> None:
        return None


def test_fold_recipe_preserves_seed_architecture_and_exposure() -> None:
    recipe = build_fold_recipe(
        _FakeFoldSource(), "E_A5_MATCHED", Path("configs/rgb_port/producers.json")
    )
    assert recipe.seed == 7
    assert recipe.epochs == 18
    assert recipe.updates == 18 * ((53 + 31) // 32)
    assert recipe.teacher_policy["roles"] == ["producer_pair"]


def test_scientific_fold_producer_remains_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="fail-closed"):
        fit_fold_producer_technical(
            source=_FakeFoldSource(),
            fit_id="E_A5_MATCHED",
            config_path=Path("configs/rgb_port/producers.json"),
            run_root=tmp_path,
            device_name="cpu",
            max_updates_this_call=1,
            scientific=True,
        )
