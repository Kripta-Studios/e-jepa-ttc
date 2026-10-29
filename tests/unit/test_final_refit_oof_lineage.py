from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from operational.final_refit.oof_lineage import (
    STITCH_SCHEMA,
    build_endpoint_lineage,
    build_stitch_receipt,
    canonical_sha256,
    file_sha256,
    verify_endpoint_lineage,
    verify_stitch_receipt,
    write_endpoint_lineage,
)


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fold_contract(root: Path) -> Path:
    blocks = [[f"group-{block}-{index}" for index in range(8)] for block in range(5)]
    rotations = []
    for rotation in range(5):
        generation = set(blocks[rotation])
        refiner = set(blocks[(rotation + 1) % 5])
        producer = set().union(*map(set, blocks)) - generation - refiner
        roles = {
            "producer": sorted(producer),
            "refiner": sorted(refiner),
            "generation": sorted(generation),
        }
        role_refs = {}
        source = 0
        for role, groups in roles.items():
            rows = []
            for group in groups:
                rows.append(
                    {
                        "rotation": rotation,
                        "rotation_role": role,
                        "source_ordinal": int(group.rsplit("-", maxsplit=1)[1])
                        + int(group.split("-")[1]) * 8,
                        "sample_token": f"token-{group}",
                        "group_id": group,
                    }
                )
                source += 1
            path = root / f"rotation_{rotation}_{role}.parquet"
            pd.DataFrame(rows).to_parquet(path, index=False)
            role_refs[role] = {
                "path": path.name,
                "sha256": file_sha256(path),
                "queries": len(rows),
                "groups": len(groups),
            }
        manifest = {
            "schema": "final_refit_rotation_roles_v1",
            "status": "VERIFIED_REFERENCE_MANIFESTS",
            "rotation": rotation,
            "generation_ancestry_excludes_group": True,
            "roles": role_refs,
        }
        manifest_path = root / f"rotation_{rotation}.json"
        _json(manifest_path, manifest)
        rotations.append({"path": manifest_path.name, "sha256": file_sha256(manifest_path)})
    contract = {
        "schema": "final_refit_oof_fold_contract_v1",
        "status": "PASS",
        "training_started": False,
        "optimizer_updates": 0,
        "rotations": rotations,
    }
    path = root / "FOLD_CONTRACT.json"
    _json(path, contract)
    return path


def _endpoint(
    root: Path,
    contract: Path,
    rotation: int,
    name: str,
    parents: dict[str, Path],
) -> Path:
    artifact_root = root / f"rotation_{rotation}_{name}_artifacts"
    artifact_root.mkdir(exist_ok=True)
    checkpoint = artifact_root / "checkpoint.bin"
    checkpoint_receipt = artifact_root / "checkpoint_receipt.json"
    training_cache = artifact_root / "training_cache.bin"
    checkpoint.write_bytes(f"checkpoint-{rotation}-{name}".encode())
    checkpoint_receipt.write_bytes(f"receipt-{rotation}-{name}".encode())
    training_cache.write_bytes(f"cache-{rotation}-{name}".encode())
    value = build_endpoint_lineage(
        fold_contract_path=contract,
        rotation=rotation,
        stage_id=name,
        checkpoint_sha256=file_sha256(checkpoint),
        checkpoint_receipt_sha256=file_sha256(checkpoint_receipt),
        training_cache_sha256=file_sha256(training_cache),
        checkpoint_path=checkpoint,
        checkpoint_receipt_path=checkpoint_receipt,
        training_cache_path=training_cache,
        parent_lineage_paths=parents,
    )
    path = root / f"rotation_{rotation}_{name}.json"
    write_endpoint_lineage(path, value)
    return path


def test_transitive_context_lineage_excludes_generation_from_every_ancestor(tmp_path: Path) -> None:
    contract = _fold_contract(tmp_path)
    a5 = _endpoint(tmp_path, contract, 0, "E_A5", {})
    c2f = _endpoint(tmp_path, contract, 0, "E_C2F", {})
    pair = _endpoint(tmp_path, contract, 0, "PAIR_E", {"E_A5": a5})
    context = _endpoint(
        tmp_path,
        contract,
        0,
        "E_CTX",
        {"E_A5": a5, "E_C2F": c2f, "PAIR_E": pair},
    )
    value = verify_endpoint_lineage(context)
    ancestry = value["training_feature_ancestry"]
    assert ancestry["generation_rows_excluded_from_all_ancestors"] is True
    assert len(ancestry["ancestor_train_group_ids"]) == 32
    assert value["deployment_dependencies"] == {}
    assert value["status"] == "VERIFIED_STRUCTURAL_OOF_ENDPOINT_NOT_SCIENTIFICALLY_ADMITTED"
    assert value["artifact_files_rehashed"] is True
    assert value["blind_validation_claim"] is False


def test_lineage_rejects_generation_group_inserted_into_training_ancestry(tmp_path: Path) -> None:
    contract = _fold_contract(tmp_path)
    endpoint = _endpoint(tmp_path, contract, 0, "E_A5", {})
    value = json.loads(endpoint.read_text(encoding="utf-8"))
    generation = value["training_feature_ancestry"]["generation_group_ids"][0]
    value["training_feature_ancestry"]["ancestor_train_group_ids"].append(generation)
    value["lineage_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "lineage_sha256"}
    )
    with pytest.raises(ValueError, match="Generation rows"):
        verify_endpoint_lineage(value)


def test_lineage_rehashes_artifacts_and_rejects_authorization_flag(tmp_path: Path) -> None:
    contract = _fold_contract(tmp_path)
    endpoint = _endpoint(tmp_path, contract, 0, "E_A5", {})
    value = json.loads(endpoint.read_text(encoding="utf-8"))
    checkpoint = Path(value["artifacts"]["checkpoint"]["path"])
    checkpoint.write_bytes(b"changed-after-lineage")
    with pytest.raises(ValueError, match="artifact bytes changed"):
        verify_endpoint_lineage(value)

    endpoint = _endpoint(tmp_path, contract, 1, "E_A5", {})
    value = json.loads(endpoint.read_text(encoding="utf-8"))
    value["scientific_launch_authorized"] = True
    value["lineage_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "lineage_sha256"}
    )
    with pytest.raises(ValueError, match="identity differs"):
        verify_endpoint_lineage(value)


def test_lineage_rejects_parent_from_another_rotation(tmp_path: Path) -> None:
    contract = _fold_contract(tmp_path)
    wrong_parent = _endpoint(tmp_path, contract, 1, "E_A5", {})
    checkpoint = tmp_path / "wrong_checkpoint.bin"
    receipt = tmp_path / "wrong_receipt.bin"
    cache = tmp_path / "wrong_cache.bin"
    checkpoint.write_bytes(b"checkpoint")
    receipt.write_bytes(b"receipt")
    cache.write_bytes(b"cache")
    with pytest.raises(ValueError, match="another fold|does not match"):
        build_endpoint_lineage(
            fold_contract_path=contract,
            rotation=0,
            stage_id="PAIR_E",
            checkpoint_sha256=file_sha256(checkpoint),
            checkpoint_receipt_sha256=file_sha256(receipt),
            training_cache_sha256=file_sha256(cache),
            checkpoint_path=checkpoint,
            checkpoint_receipt_path=receipt,
            training_cache_path=cache,
            parent_lineage_paths={"E_A5": wrong_parent},
        )


def test_physical_stitch_covers_every_generation_fold_once(tmp_path: Path) -> None:
    contract = _fold_contract(tmp_path)
    lineages = {}
    payloads = {}
    for rotation in range(5):
        a5 = _endpoint(tmp_path, contract, rotation, "E_A5", {})
        pair = _endpoint(tmp_path, contract, rotation, "PAIR_E", {"E_A5": a5})
        c2f = _endpoint(tmp_path, contract, rotation, "E_C2F", {})
        lineages[rotation] = {"E_A5": a5, "E_C2F": c2f, "PAIR_E": pair}
        payload = tmp_path / f"payload_{rotation}.bin"
        payload.write_bytes(f"physical-{rotation}".encode())
        payloads[rotation] = payload
    receipt = build_stitch_receipt(
        fold_contract_path=contract,
        interface="PHASE17_RAW_EVENT",
        fragment_payloads=payloads,
        endpoint_lineages=lineages,
    )
    assert receipt["population"] == 40
    assert receipt["groups"] == 40
    assert verify_stitch_receipt(receipt)["stitch_sha256"] == receipt["stitch_sha256"]


def test_stitch_rejects_generation_token_union_that_is_not_exact(tmp_path: Path) -> None:
    contract = _fold_contract(tmp_path)
    contract_value = json.loads(contract.read_text(encoding="utf-8"))
    manifest_path = tmp_path / "rotation_4.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows_path = tmp_path / manifest["roles"]["generation"]["path"]
    producer_path = tmp_path / manifest["roles"]["producer"]["path"]
    rows = pd.read_parquet(rows_path)
    producer_rows = pd.read_parquet(producer_path)
    rows_token = str(rows.loc[0, "sample_token"])
    producer_token = str(producer_rows.loc[0, "sample_token"])
    rows.loc[0, "sample_token"] = producer_token
    producer_rows.loc[0, "sample_token"] = rows_token
    rows.to_parquet(rows_path, index=False)
    producer_rows.to_parquet(producer_path, index=False)
    manifest["roles"]["generation"]["sha256"] = file_sha256(rows_path)
    manifest["roles"]["producer"]["sha256"] = file_sha256(producer_path)
    _json(manifest_path, manifest)
    contract_value["rotations"][4]["sha256"] = file_sha256(manifest_path)
    _json(contract, contract_value)
    with pytest.raises(ValueError, match="cover the fold universe|overlap by token"):
        build_stitch_receipt(
            fold_contract_path=contract,
            interface="PHASE17_RAW_EVENT",
            fragment_payloads={rotation: tmp_path / f"missing-{rotation}" for rotation in range(5)},
            endpoint_lineages={rotation: {} for rotation in range(5)},
        )


def test_stitch_rejects_latent_interface_and_duplicate_rotation(tmp_path: Path) -> None:
    receipt = {
        "schema": STITCH_SCHEMA,
        "interface": "LATENT128",
        "feature_semantics": "latent_embedding",
        "width": 128,
        "targets_read_for_stitching": False,
        "blind_validation_claim": False,
        "fragments": [],
    }
    receipt["stitch_sha256"] = canonical_sha256(receipt)
    with pytest.raises(ValueError, match="physical interface"):
        verify_stitch_receipt(receipt)
