from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from operational.independent_refit import teacher_cache
from scripts import prepare_independent_phase_teacher


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    source = tmp_path / "source"
    teacher = source / "teacher"
    teacher.mkdir(parents=True)
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"frozen weights")
    freeze = {
        "ordered_tensor_weights_sha256": "state-hash",
        "model_revision": "revision",
        "TTC_targets_read_by_teacher": False,
        "optimizer_updates": 0,
    }
    _write_json(source / "TEACHER_FREEZE.json", freeze)
    freeze_sha = teacher_cache.file_sha256(source / "TEACHER_FREEZE.json")
    tokens = np.asarray([f"g{i // 4}_track_{i:02d}" for i in range(8)])
    targets = (
        np.arange(8 * 2 * 6 * 32 * 32, dtype=np.int64) % 1000
    ).astype(np.float16).reshape(8, 2, 6, 32, 32)
    valid = np.ones(targets.shape, dtype=np.bool_)
    rgb = np.full((8, 2), "a" * 64)
    shard = teacher / "shard_00000.npz"
    np.savez_compressed(
        shard,
        ordinals=np.arange(8, dtype=np.int64),
        tokens=tokens,
        relation_targets=targets,
        relation_valid=valid,
        rgb_sha256=rgb,
    )
    shard_sha = teacher_cache.file_sha256(shard)
    receipt = {
        "status": "VERIFIED",
        "sha256": shard_sha,
        "freeze_sha256": freeze_sha,
        "teacher_weights_sha256": "state-hash",
        "rows": 8,
        "TTC_targets_read": False,
        "optimizer_updates": 0,
    }
    _write_json(shard.with_suffix(".json"), receipt)
    manifest = {
        "schema": "train40_complete_fragment_manifest_v1",
        "status": "COMPLETE_VERIFIED",
        "kind": "TEACHER",
        "row_count": 8,
        "sequence_count": 40,
        "freeze_sha256": freeze_sha,
        "ordered_shards": [
            {
                "name": shard.name,
                "sha256": shard_sha,
                "receipt_sha256": teacher_cache.file_sha256(shard.with_suffix(".json")),
                "start": 0,
                "stop": 8,
                "bytes": shard.stat().st_size,
            }
        ],
    }
    _write_json(source / "TEACHER_MANIFEST.json", manifest)
    admission = {
        "status": "PASSED",
        "complete_rows": 8,
        "all_ordinals_and_tokens_exactly_match": True,
        "TTC_targets_read": False,
        "optimizer_updates": 0,
        "ordered_teacher_weights_sha256": "state-hash",
    }
    _write_json(source / "TEACHER_COMPLETE_ADMISSION.json", admission)
    contract = tmp_path / "population" / "FOLD_CONTRACT.json"
    contract.parent.mkdir()
    _write_json(contract, {"fixture": True})
    selected = [0, 2, 3, 6, 7]
    records = []
    for fit, public in enumerate(selected):
        token = str(tokens[public])
        group = f"g{public // 4}"
        records.append(
            {
                "public_source_ordinal": public,
                "fit_ordinal": fit,
                "sample_token": token,
                "group_id": group,
                "sequence_id": group,
                "track_id": token.rsplit("_", 1)[0],
                "public_track_id": token.rsplit("_", 1)[0],
                "timestamp_us": public * 100_000,
            }
        )
    population = pd.DataFrame.from_records(records, columns=teacher_cache.INDEX_COLUMNS[:8])
    population.to_parquet(contract.parent / "population.parquet", index=False)
    crosswalk = population.copy()
    crosswalk["teacher_ordinal"] = selected
    crosswalk["teacher_shard"] = shard.name
    crosswalk["teacher_shard_index"] = 0
    crosswalk["teacher_local_row"] = selected
    crosswalk["teacher_rgb_sha256_0"] = "a" * 64
    crosswalk["teacher_rgb_sha256_1"] = "a" * 64
    crosswalk["raw_rgb_member_0"] = "rgb/a.png"
    crosswalk["raw_rgb_member_1"] = "rgb/b.png"
    crosswalk_path = contract.parent / "teacher_crosswalk.parquet"
    crosswalk.to_parquet(crosswalk_path, index=False)
    crosswalk_inventory = contract.parent / "teacher_source_inventory.json"
    _write_json(crosswalk_inventory, {"fixture": True})
    crosswalk_receipt = contract.parent / "TEACHER_CROSSWALK.json"
    _write_json(crosswalk_receipt, {"fixture": True})
    _write_json(
        contract.parent / "POPULATION.json", {"sample_token_set_sha256": "token-set"}
    )
    row_locality = tmp_path / "ROW_LOCALITY.json"
    _write_json(
        row_locality,
        {
            "schema": "independent_phase_historical_teacher_row_locality_v1",
            "status": "PASS_ROW_LOCAL_NO_REEXTRACTION_REQUIRED",
            "teacher_source_sha256": teacher_cache.EXPECTED_TEACHER_SOURCE_SHA256,
            "relation_source_sha256": teacher_cache.EXPECTED_RELATION_SOURCE_SHA256,
            "model_config_sha256": "config-hash",
            "weight_file_sha256": teacher_cache.file_sha256(weights),
            "model_revision": "revision",
            "model_eval_mode": True,
            "drop_path_rate": 0.0,
            "batch_global_running_state_keys": 0,
            "layer_norm_parameter_keys": 2,
            "relation_fixture_anchor_neighbor_variants": 2,
            "relation_fixture_max_absolute_error": 0.0,
            "relation_fixture_valid_masks_equal": True,
            "teacher_model_tensors_loaded": False,
            "gpu_used": False,
            "conclusion": "row local",
        },
    )
    teacher_parent = contract.parent / "TEACHER_PARENT.json"
    _write_json(
        teacher_parent,
        {
            "schema": "independent_phase_frozen_teacher_parent_v1",
            "status": "AVAILABLE_FROZEN_NOT_EXTRACTED_FOR_INDEPENDENT_REFIT",
            "kind": "frozen_dinov3_teacher_weights",
            "fold_contract_sha256": teacher_cache.file_sha256(contract),
            "repository": "fixture/repository",
            "revision": "revision",
            "ordered_state_tensor_sha256": "state-hash",
            "weights_file_sha256": teacher_cache.file_sha256(weights),
            "weights_bytes": weights.stat().st_size,
            "weights_path": str(weights.resolve()),
            "query_whitelist": "TRAIN24_ONLY",
            "queries": 5,
            "query_token_set_sha256": "token-set",
            "optimizer_updates": 0,
            "teacher_features_materialized": False,
            "receipt_sha256": "fixture-semantic-receipt",
        },
    )
    pins = teacher_cache.SourcePins(
        manifest_sha256=teacher_cache.file_sha256(source / "TEACHER_MANIFEST.json"),
        freeze_sha256=freeze_sha,
        admission_sha256=teacher_cache.file_sha256(source / "TEACHER_COMPLETE_ADMISSION.json"),
        weight_file_sha256=teacher_cache.file_sha256(weights),
        ordered_state_sha256="state-hash",
        model_revision="revision",
        source_rows=8,
        output_rows=5,
        source_shards=1,
        fold_contract_sha256=teacher_cache.file_sha256(contract),
        row_locality_receipt_sha256=teacher_cache.file_sha256(row_locality),
        model_config_sha256="config-hash",
        teacher_parent_sha256=teacher_cache.file_sha256(teacher_parent),
        teacher_repository="fixture/repository",
        weight_bytes=weights.stat().st_size,
        crosswalk_receipt_sha256=teacher_cache.file_sha256(crosswalk_receipt),
        crosswalk_sha256=teacher_cache.file_sha256(crosswalk_path),
        crosswalk_inventory_sha256=teacher_cache.file_sha256(crosswalk_inventory),
    )
    monkeypatch.setattr(
        teacher_cache,
        "verify_independent_contract",
        lambda _path: {"sample_token_set_sha256": "token-set"},
    )
    monkeypatch.setattr(
        teacher_cache,
        "verify_teacher_crosswalk",
        lambda *_args, **_kwargs: {"status": "PASS", "rows": 5},
    )
    role = population.iloc[:3].copy()
    generation = population.iloc[3:].copy()
    monkeypatch.setattr(
        teacher_cache,
        "load_rotation_role",
        lambda _path, *, rotation, role: (
            generation_frame.copy() if role == "generation" else role_frame.copy()
        ),
    )
    role_frame = role
    generation_frame = generation

    authorization = tmp_path / "AUTHORIZATION.json"
    _write_json(
        authorization,
        {
            "schema": teacher_cache.AUTHORIZATION_SCHEMA,
            "status": "AUTHORIZED",
            "root_reviewed": True,
            "independent_contract_review_status": "SHIP",
            "conversion_only_no_fitting": True,
            "fold_contract_sha256": teacher_cache.file_sha256(contract),
            "population_sha256": teacher_cache.file_sha256(
                contract.parent / "population.parquet"
            ),
            "source_teacher_manifest_sha256": pins.manifest_sha256,
            "row_locality_receipt_sha256": pins.row_locality_receipt_sha256,
            "teacher_parent_sha256": pins.teacher_parent_sha256,
            "teacher_crosswalk_receipt_sha256": pins.crosswalk_receipt_sha256,
            "teacher_crosswalk_sha256": pins.crosswalk_sha256,
            "teacher_crosswalk_inventory_sha256": pins.crosswalk_inventory_sha256,
        },
    )
    return {
        "source": source,
        "weights": weights,
        "pins": pins,
        "contract": contract,
        "authorization": authorization,
        "row_locality": row_locality,
        "population": population,
        "targets": targets,
        "crosswalk": crosswalk_receipt,
    }


def test_prepare_verify_and_role_whitelist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "cache"
    result = teacher_cache.prepare_teacher_cache(
        source_root=fixture["source"],
        weight_path=fixture["weights"],
        fold_contract_path=fixture["contract"],
        authorization_path=fixture["authorization"],
        row_locality_receipt_path=fixture["row_locality"],
        output_dir=output,
        pins=fixture["pins"],
    )
    assert result["rows"] == 5
    manifest = json.loads((output / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["teacher_crosswalk_receipt_sha256"] == fixture[
        "pins"
    ].crosswalk_receipt_sha256
    assert teacher_cache.file_sha256(output / "teacher_crosswalk.parquet") == fixture[
        "pins"
    ].crosswalk_sha256
    converted_index = pd.read_parquet(output / "index.parquet")
    assert converted_index["public_source_ordinal"].tolist() == fixture["population"][
        "public_source_ordinal"
    ].tolist()
    verified = teacher_cache.verify_teacher_cache(
        output / "MANIFEST.json",
        expected_manifest_sha256=result["manifest_sha256"],
        fold_contract_path=fixture["contract"],
        pins=fixture["pins"],
    )
    assert verified["status"] == "PASS"
    cache = teacher_cache.load_teacher_role(
        output / "MANIFEST.json",
        expected_manifest_sha256=result["manifest_sha256"],
        fold_contract_path=fixture["contract"],
        rotation=0,
        role="producer",
        role_receipt_path=tmp_path / "ROLE_RECEIPT.json",
        pins=fixture["pins"],
    )
    token = fixture["population"].iloc[1]["sample_token"]
    assert cache.role_receipt_path.is_file()
    assert cache.role_receipt_sha256 == teacher_cache.file_sha256(cache.role_receipt_path)
    role_receipt = json.loads(cache.role_receipt_path.read_text(encoding="utf-8"))
    assert role_receipt["generation_excluded"] is True
    assert role_receipt["full_train24_exposed"] is False
    relation, valid = cache.get(str(token))
    assert np.array_equal(relation, fixture["targets"][2])
    assert bool(valid.all())
    with pytest.raises(KeyError, match="outside"):
        cache.get(str(fixture["population"].iloc[-1]["sample_token"]))
    with pytest.raises(ValueError, match="producer/refiner"):
        teacher_cache.load_teacher_role(
            output / "MANIFEST.json",
            expected_manifest_sha256=result["manifest_sha256"],
            fold_contract_path=fixture["contract"],
            rotation=0,
            role="generation",
            role_receipt_path=tmp_path / "INVALID_ROLE_RECEIPT.json",
            pins=fixture["pins"],
        )


def test_rejects_missing_root_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    authorization = json.loads(fixture["authorization"].read_text(encoding="utf-8"))
    authorization["root_reviewed"] = False
    _write_json(fixture["authorization"], authorization)
    with pytest.raises(ValueError, match=r"SHIP \+ root authorization"):
        teacher_cache.prepare_teacher_cache(
            source_root=fixture["source"],
            weight_path=fixture["weights"],
            fold_contract_path=fixture["contract"],
            authorization_path=fixture["authorization"],
            row_locality_receipt_path=fixture["row_locality"],
            output_dir=tmp_path / "cache",
            pins=fixture["pins"],
        )


def test_rejects_source_crosswalk_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    population_path = fixture["contract"].parent / "population.parquet"
    population = pd.read_parquet(population_path)
    population.loc[0, "sample_token"] = "forged-token"
    population.to_parquet(population_path, index=False)
    authorization = json.loads(fixture["authorization"].read_text(encoding="utf-8"))
    authorization["population_sha256"] = teacher_cache.file_sha256(population_path)
    _write_json(fixture["authorization"], authorization)
    with pytest.raises(ValueError, match="crosswalk does not preserve TRAIN24 identity"):
        teacher_cache.prepare_teacher_cache(
            source_root=fixture["source"],
            weight_path=fixture["weights"],
            fold_contract_path=fixture["contract"],
            authorization_path=fixture["authorization"],
            row_locality_receipt_path=fixture["row_locality"],
            output_dir=tmp_path / "cache",
            pins=fixture["pins"],
        )


def test_rejects_target_only_source_mutation_after_crosswalk_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    shard = fixture["source"] / "teacher/shard_00000.npz"

    def mutate_after_gate(*_args: object, **_kwargs: object) -> dict[str, object]:
        with np.load(shard, allow_pickle=False) as stored:
            arrays = {key: stored[key].copy() for key in stored.files}
        arrays["relation_targets"][0, 0, 0, 0, 0] += np.float16(1.0)
        np.savez_compressed(shard, **arrays)
        return {"status": "PASS", "rows": 5}

    monkeypatch.setattr(teacher_cache, "verify_teacher_crosswalk", mutate_after_gate)
    with pytest.raises(ValueError, match="Selected teacher source changed before load"):
        teacher_cache.prepare_teacher_cache(
            source_root=fixture["source"],
            weight_path=fixture["weights"],
            fold_contract_path=fixture["contract"],
            authorization_path=fixture["authorization"],
            row_locality_receipt_path=fixture["row_locality"],
            output_dir=tmp_path / "cache",
            pins=fixture["pins"],
        )


def test_rejects_corrupt_converted_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "cache"
    result = teacher_cache.prepare_teacher_cache(
        source_root=fixture["source"],
        weight_path=fixture["weights"],
        fold_contract_path=fixture["contract"],
        authorization_path=fixture["authorization"],
        row_locality_receipt_path=fixture["row_locality"],
        output_dir=output,
        pins=fixture["pins"],
    )
    shard = output / "shards" / "shard_00000.npz"
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="shard changed"):
        teacher_cache.verify_teacher_cache(
            output / "MANIFEST.json",
            expected_manifest_sha256=result["manifest_sha256"],
            fold_contract_path=fixture["contract"],
            pins=fixture["pins"],
        )


def _prepared_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "cache"
    result = teacher_cache.prepare_teacher_cache(
        source_root=fixture["source"],
        weight_path=fixture["weights"],
        fold_contract_path=fixture["contract"],
        authorization_path=fixture["authorization"],
        row_locality_receipt_path=fixture["row_locality"],
        output_dir=output,
        pins=fixture["pins"],
    )
    return fixture, output, result


def test_rejects_forged_index_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, output, _result = _prepared_cache(tmp_path, monkeypatch)
    index_path = output / "index.parquet"
    index = pd.read_parquet(index_path)
    index.loc[0, "cache_row"] = 1
    index.to_parquet(index_path, index=False)
    manifest_path = output / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["index_sha256"] = teacher_cache.file_sha256(index_path)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="shard payload"):
        teacher_cache.verify_teacher_cache(
            manifest_path,
            expected_manifest_sha256=teacher_cache.file_sha256(manifest_path),
            fold_contract_path=fixture["contract"],
            pins=fixture["pins"],
        )


def test_rejects_resealed_public_ordinal_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, output, _result = _prepared_cache(tmp_path, monkeypatch)
    shard = output / "shards/shard_00000.npz"
    with np.load(shard, allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    arrays["public_source_ordinals"][0] += 1
    np.savez_compressed(shard, **arrays)
    receipt_path = shard.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["sha256"] = teacher_cache.file_sha256(shard)
    _write_json(receipt_path, receipt)
    manifest_path = output / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"][0].update(
        sha256=teacher_cache.file_sha256(shard),
        receipt_sha256=teacher_cache.file_sha256(receipt_path),
        bytes=shard.stat().st_size,
    )
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="shard payload"):
        teacher_cache.verify_teacher_cache(
            manifest_path,
            expected_manifest_sha256=teacher_cache.file_sha256(manifest_path),
            fold_contract_path=fixture["contract"],
            pins=fixture["pins"],
        )


def test_rejects_resealed_source_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, output, _result = _prepared_cache(tmp_path, monkeypatch)
    receipt_path = output / "shards/shard_00000.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_shards"][0]["name"] = "not-a-canonical-source"
    _write_json(receipt_path, receipt)
    manifest_path = output / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"][0]["receipt_sha256"] = teacher_cache.file_sha256(receipt_path)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="source-shard reference"):
        teacher_cache.verify_teacher_cache(
            manifest_path,
            expected_manifest_sha256=teacher_cache.file_sha256(manifest_path),
            fold_contract_path=fixture["contract"],
            pins=fixture["pins"],
        )


def test_rejects_authorization_without_exact_row_locality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    authorization = json.loads(fixture["authorization"].read_text(encoding="utf-8"))
    authorization["row_locality_receipt_sha256"] = "0" * 64
    _write_json(fixture["authorization"], authorization)
    with pytest.raises(ValueError, match=r"SHIP \+ root authorization"):
        teacher_cache.prepare_teacher_cache(
            source_root=fixture["source"],
            weight_path=fixture["weights"],
            fold_contract_path=fixture["contract"],
            authorization_path=fixture["authorization"],
            row_locality_receipt_path=fixture["row_locality"],
            output_dir=tmp_path / "cache",
            pins=fixture["pins"],
        )


def test_prepare_requires_postwrite_self_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        teacher_cache,
        "verify_teacher_cache",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("self verify failed")),
    )
    with pytest.raises(ValueError, match="self verify failed"):
        teacher_cache.prepare_teacher_cache(
            source_root=fixture["source"],
            weight_path=fixture["weights"],
            fold_contract_path=fixture["contract"],
            authorization_path=fixture["authorization"],
            row_locality_receipt_path=fixture["row_locality"],
            output_dir=tmp_path / "cache",
            pins=fixture["pins"],
        )


def test_row_locality_audit_detects_exact_batch_invariance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher_source = tmp_path / "teacher.py"
    teacher_source.write_text(
        "\n".join(
            (
                "# .eval()",
                "# .requires_grad_(False)",
                "# hidden_states[2]",
                "# local_cosine_relation_maps(features)",
                "# np.stack([x[0] for x in inputs])",
            )
        ),
        encoding="utf-8",
    )
    relation_source = tmp_path / "relations.py"
    relation_source.write_text(
        "from types import SimpleNamespace\n"
        "import torch\n"
        "def local_cosine_relation_maps(features):\n"
        "    values = torch.nn.functional.normalize(features.float(), dim=-3).sum(dim=-3)\n"
        "    valid = torch.ones_like(values, dtype=torch.bool)\n"
        "    return SimpleNamespace(values=values, valid=valid)\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    _write_json(
        config,
        {
            "architectures": ["DINOv3ConvNextModel"],
            "drop_path_rate": 0.0,
            "layer_norm_eps": 1.0e-6,
            "torch_dtype": "float32",
        },
    )
    weights = tmp_path / "model.safetensors"
    header = json.dumps(
        {"layer_norm.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    weights.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 4)
    monkeypatch.setattr(
        teacher_cache, "EXPECTED_TEACHER_SOURCE_SHA256", teacher_cache.file_sha256(teacher_source)
    )
    monkeypatch.setattr(
        teacher_cache,
        "EXPECTED_RELATION_SOURCE_SHA256",
        teacher_cache.file_sha256(relation_source),
    )
    monkeypatch.setattr(
        teacher_cache, "EXPECTED_WEIGHT_FILE_SHA256", teacher_cache.file_sha256(weights)
    )
    result = teacher_cache.audit_historical_row_locality(
        teacher_source_path=teacher_source,
        relation_source_path=relation_source,
        model_config_path=config,
        weight_path=weights,
    )
    assert result["status"] == "PASS_ROW_LOCAL_NO_REEXTRACTION_REQUIRED"
    assert result["relation_fixture_max_absolute_error"] == 0.0


def test_verify_only_cli_does_not_require_conversion_inputs(tmp_path: Path) -> None:
    args = prepare_independent_phase_teacher.parse_args(
        [
            "--verify-only",
            "--manifest-sha256",
            "a" * 64,
            "--output-dir",
            str(tmp_path / "cache"),
            "--verification-receipt",
            str(tmp_path / "verification.json"),
        ]
    )
    assert args.verify_only is True
    assert args.source_root is None
    assert args.weight_path is None
    assert args.authorization is None


def test_owned_path_guard_rejects_root_and_external_path(tmp_path: Path) -> None:
    root = tmp_path / "independent_phase"
    root.mkdir()
    accepted = prepare_independent_phase_teacher._require_owned_path(
        root / "cache", root, "cache"
    )
    assert accepted == (root / "cache").resolve()
    with pytest.raises(ValueError, match="must stay below"):
        prepare_independent_phase_teacher._require_owned_path(root, root, "root")
    with pytest.raises(ValueError, match="must stay below"):
        prepare_independent_phase_teacher._require_owned_path(
            tmp_path / "external", root, "external"
        )


def test_conversion_cli_requires_conversion_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = Path("output/independent_phase/test_missing_conversion_inputs")
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_independent_phase_teacher.py", "--output-dir", str(output)],
    )
    with pytest.raises(ValueError, match="--source-root.*--weight-path.*--authorization"):
        prepare_independent_phase_teacher.main()
