"""Token-exact crosswalk from raw TRAIN24 identities to frozen TRAIN40 teacher rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from .lineage import _load_rotation_role_unverified, verify_independent_contract
from .population import canonical_sha256, file_sha256, identity_sha256

SCHEMA = "independent_phase_teacher_crosswalk_v3"
PARENT_PINS = {
    "TEACHER_MANIFEST.json": "646d5569671e9d8af9b21c84c12a55584be675ccfc77dc5e996ccea6fb8d38de",
    "TEACHER_COMPLETE_ADMISSION.json": (
        "7c770a0526b43d815b9327dbb01666bf40b329469b091ca40cf7aa43be5a0733"
    ),
    "TEACHER_FREEZE.json": "a80896b73b241b5570861c381497585348196ff06dac180dd9e343af4d753fbb",
}
TEACHER_TENSOR_SHA = "96ad081d5863ba14707d2e3bb734b7fa8dfbf3248cb67ddfcc74d86dd160d628"


def validate_crosswalk_frame(frame: pd.DataFrame, population: pd.DataFrame) -> None:
    """Validate raw identity preservation and teacher coordinates."""
    required = {
        "sample_token",
        "fit_ordinal",
        "public_source_ordinal",
        "teacher_ordinal",
        "teacher_shard",
        "teacher_shard_index",
        "teacher_local_row",
        "teacher_rgb_sha256_0",
        "teacher_rgb_sha256_1",
        "raw_rgb_member_0",
        "raw_rgb_member_1",
    }
    if not required.issubset(frame.columns):
        raise ValueError("Crosswalk schema lacks identity coordinates")
    if (
        len(frame) != len(population)
        or frame["sample_token"].duplicated().any()
        or frame["teacher_ordinal"].duplicated().any()
        or frame["fit_ordinal"].astype(int).tolist() != list(range(len(frame)))
        or not frame[
            [
                "public_source_ordinal",
                "fit_ordinal",
                "sample_token",
                "group_id",
                "sequence_id",
                "track_id",
                "public_track_id",
                "timestamp_us",
            ]
        ]
        .reset_index(drop=True)
        .equals(
            population[
                [
                    "public_source_ordinal",
                    "fit_ordinal",
                    "sample_token",
                    "group_id",
                    "sequence_id",
                    "track_id",
                    "public_track_id",
                    "timestamp_us",
                ]
            ].reset_index(drop=True)
        )
        or not pd.api.types.is_integer_dtype(frame["teacher_ordinal"])
        or any(not 0 <= int(value) <= 88_743 for value in frame["teacher_ordinal"])
        or any(not 0 <= int(value) <= 31 for value in frame["teacher_local_row"])
        or any(
            str(name) != f"shard_{int(index):05d}.npz"
            for name, index in zip(
                frame["teacher_shard"], frame["teacher_shard_index"], strict=True
            )
        )
        or any(
            len(str(value)) != 64 or any(c not in "0123456789abcdef" for c in str(value))
            for value in frame["teacher_rgb_sha256_0"]
        )
        or any(
            len(str(value)) != 64 or any(c not in "0123456789abcdef" for c in str(value))
            for value in frame["teacher_rgb_sha256_1"]
        )
    ):
        raise ValueError("Crosswalk raw/teacher identities differ")


def _role_token_sets(fold_contract_path: Path) -> dict[str, dict[str, Any]]:
    role_sets: dict[str, dict[str, Any]] = {}
    for rotation in range(6):
        for role in ("producer", "refiner"):
            role_frame = _load_rotation_role_unverified(
                fold_contract_path, rotation=rotation, role=role
            )
            role_sets[f"rotation_{rotation}_{role}"] = {
                "rows": len(role_frame),
                "sample_token_set_sha256": identity_sha256(
                    role_frame["sample_token"].tolist(), ordered=False
                ),
            }
    return role_sets


def _verify_physical_rows(frame: pd.DataFrame, teacher_root: Path) -> None:
    resolved_root = teacher_root.resolve(strict=True)
    for shard_name, shard_rows in frame.groupby("teacher_shard", sort=True):
        shard = (resolved_root / str(shard_name)).resolve(strict=True)
        if shard.parent != resolved_root:
            raise ValueError("Crosswalk teacher shard escapes its frozen root")
        with np.load(shard, allow_pickle=False) as archive:
            tokens = archive["tokens"]
            ordinals = archive["ordinals"]
            rgb = archive["rgb_sha256"]
            for row in shard_rows.to_dict(orient="records"):
                local = int(row["teacher_local_row"])
                if (
                    local >= len(tokens)
                    or str(tokens[local]) != str(row["sample_token"])
                    or int(ordinals[local]) != int(row["teacher_ordinal"])
                    or str(rgb[local, 0]) != str(row["teacher_rgb_sha256_0"])
                    or str(rgb[local, 1]) != str(row["teacher_rgb_sha256_1"])
                ):
                    raise ValueError("Crosswalk shard/local row mapping differs")


def build_teacher_crosswalk(
    fold_contract_path: Path, teacher_root: Path, historical_run: Path, output_dir: Path
) -> dict[str, Any]:
    """Read only frozen shard identities/hashes and create a write-once token join."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    verified = verify_independent_contract(fold_contract_path)
    population_root = fold_contract_path.resolve(strict=True).parent
    population = pd.read_parquet(population_root / "population.parquet")
    source = json.loads((population_root / "SOURCE_RECEIPT.json").read_text(encoding="utf-8"))
    groups = sorted(population["group_id"].astype(str).unique())
    raw = (
        ds.dataset(source["sources"]["garl_data"]["path"])
        .to_table(
            columns=["sample_token", "sequence_id", "rgb_member_paths"],
            filter=ds.field("sequence_id").isin(groups),
        )
        .to_pandas()
        .set_index("sample_token", drop=False)
    )

    inventory: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    manifest = json.loads((historical_run / "TEACHER_MANIFEST.json").read_text(encoding="utf-8"))
    manifest_shards = manifest.get("ordered_shards")
    shards = sorted(teacher_root.resolve(strict=True).glob("shard_*.npz"))
    if len(shards) != 2_774:
        raise ValueError("Frozen teacher shard count differs")
    if not isinstance(manifest_shards, list) or len(manifest_shards) != len(shards):
        raise ValueError("Teacher manifest ordered shard inventory differs")
    for shard_index, shard in enumerate(shards):
        receipt_path = shard.with_suffix(".json")
        manifest_ref = manifest_shards[shard_index]
        shard_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        with np.load(shard, allow_pickle=False) as archive:
            tokens = np.asarray(archive["tokens"]).astype(str)
            ordinals = np.asarray(archive["ordinals"], dtype=np.int64)
            rgb = np.asarray(archive["rgb_sha256"]).astype(str)
        if len(tokens) != len(ordinals) or rgb.shape != (len(tokens), 2):
            raise ValueError("Frozen teacher shard identity arrays differ")
        shard_sha = file_sha256(shard)
        receipt_sha = file_sha256(receipt_path)
        if (
            manifest_ref.get("name") != shard.name
            or manifest_ref.get("sha256") != shard_sha
            or manifest_ref.get("receipt_sha256") != receipt_sha
            or manifest_ref.get("start") != int(ordinals[0])
            or manifest_ref.get("stop") != int(ordinals[-1]) + 1
            or manifest_ref.get("bytes") != shard.stat().st_size
            or ordinals.tolist()
            != list(range(int(manifest_ref["start"]), int(manifest_ref["stop"])))
            or shard_receipt.get("status") != "VERIFIED"
            or shard_receipt.get("rows") != len(tokens)
            or shard_receipt.get("freeze_sha256") != PARENT_PINS["TEACHER_FREEZE.json"]
            or shard_receipt.get("teacher_weights_sha256") != TEACHER_TENSOR_SHA
            or shard_receipt.get("optimizer_updates") != 0
            or shard_receipt.get("TTC_targets_read") is not False
            or shard_receipt.get("sha256") != shard_sha
        ):
            raise ValueError("Teacher shard differs from manifest order or scientific receipt")
        inventory.append(
            {
                "shard": shard.name,
                "shard_sha256": shard_sha,
                "receipt": receipt_path.name,
                "receipt_sha256": receipt_sha,
                "rows": len(tokens),
            }
        )
        teacher_rows.extend(
            {
                "sample_token": token,
                "teacher_ordinal": int(ordinal),
                "teacher_shard": shard.name,
                "teacher_shard_index": shard_index,
                "teacher_local_row": local,
                "teacher_rgb_sha256_0": str(rgb[local, 0]),
                "teacher_rgb_sha256_1": str(rgb[local, 1]),
            }
            for local, (token, ordinal) in enumerate(zip(tokens, ordinals, strict=True))
        )
    teacher = pd.DataFrame.from_records(teacher_rows)
    if (
        len(teacher) != 88_744
        or teacher["sample_token"].duplicated().any()
        or teacher["teacher_ordinal"].duplicated().any()
    ):
        raise ValueError("Frozen teacher identity universe is not unique TRAIN40")
    if sorted(teacher["teacher_ordinal"].astype(int).tolist()) != list(range(88_744)):
        raise ValueError("Teacher ordinals do not cover manifest order 0..88743")
    full_raw_tokens = (
        ds.dataset(source["sources"]["garl_data"]["path"])
        .to_table(columns=["sample_token"])
        .column("sample_token")
        .to_pylist()
    )
    if len(full_raw_tokens) != 88_744 or set(map(str, full_raw_tokens)) != set(
        teacher["sample_token"].astype(str)
    ):
        raise ValueError("Full raw Garl token universe differs from frozen teacher")
    selected = population.merge(teacher, on="sample_token", how="left", validate="one_to_one")
    if bool(selected["teacher_ordinal"].isna().any()) or len(selected) != 53_047:
        raise ValueError("TRAIN24 token set is not exactly covered by the frozen teacher")
    selected["raw_rgb_member_0"] = [
        str(raw.loc[token, "rgb_member_paths"][0]) for token in selected["sample_token"]
    ]
    selected["raw_rgb_member_1"] = [
        str(raw.loc[token, "rgb_member_paths"][1]) for token in selected["sample_token"]
    ]
    selected = selected.sort_values("fit_ordinal", kind="stable").reset_index(drop=True)
    if selected["public_source_ordinal"].equals(selected["teacher_ordinal"]):
        raise ValueError("Teacher ordinal unexpectedly aliases raw public ordinal")
    output_dir.mkdir(parents=True)
    crosswalk_path = output_dir / "teacher_crosswalk.parquet"
    selected.to_parquet(crosswalk_path, index=False)
    inventory_path = output_dir / "teacher_source_inventory.json"
    inventory_payload = {
        "schema": "independent_phase_teacher_source_inventory_v1",
        "shards": inventory,
    }
    inventory_payload["receipt_sha256"] = canonical_sha256(inventory_payload)
    inventory_path.write_text(
        json.dumps(inventory_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    parents = {
        name: file_sha256(historical_run / name)
        for name in (
            "TEACHER_MANIFEST.json",
            "TEACHER_COMPLETE_ADMISSION.json",
            "TEACHER_FREEZE.json",
        )
    }
    if parents != PARENT_PINS:
        raise ValueError("Historical teacher parent pins changed")
    role_sets = _role_token_sets(fold_contract_path)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "VERIFIED_TOKEN_EXACT_CROSSWALK_NOT_EXTRACTED",
        "fold_contract_sha256": verified["fold_contract_sha256"],
        "population_sha256": verified["population_sha256"],
        "crosswalk_path": crosswalk_path.name,
        "crosswalk_sha256": file_sha256(crosswalk_path),
        "source_inventory_sha256": file_sha256(inventory_path),
        "historical_parent_sha256": parents,
        "teacher_root": str(teacher_root.resolve()),
        "historical_run": str(historical_run.resolve()),
        "teacher_rows": len(teacher),
        "manifest_order_verified": True,
        "full_raw_train40_token_universe_verified": True,
        "train24_rows": len(selected),
        "sample_token_set_sha256": identity_sha256(
            selected["sample_token"].tolist(), ordered=False
        ),
        "role_token_sets": role_sets,
        "join_key": "sample_token_ONLY",
        "public_source_ordinal_semantics": "raw_Garl_parquet_row",
        "teacher_ordinal_semantics": "frozen_teacher_cache_ordinal",
        "ordinals_equal_required": False,
        "features_read": False,
        "features_extracted": False,
        "gpu_used": False,
        "training_admitted": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (output_dir / "RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def verify_teacher_crosswalk(receipt_path: Path, fold_contract_path: Path) -> dict[str, Any]:
    """Verify receipt, exact TRAIN24 identity and teacher shard/local-row mapping."""
    path = receipt_path.resolve(strict=True)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(receipt)
    claimed = payload.pop("receipt_sha256", None)
    verified = verify_independent_contract(fold_contract_path)
    if (
        claimed != canonical_sha256(payload)
        or receipt.get("schema") != SCHEMA
        or receipt.get("fold_contract_sha256") != verified["fold_contract_sha256"]
        or receipt.get("population_sha256") != verified["population_sha256"]
        or receipt.get("training_admitted") is not False
        or receipt.get("manifest_order_verified") is not True
        or receipt.get("full_raw_train40_token_universe_verified") is not True
        or receipt.get("historical_parent_sha256") != PARENT_PINS
        or receipt.get("teacher_rows") != 88_744
        or receipt.get("train24_rows") != 53_047
        or receipt.get("join_key") != "sample_token_ONLY"
        or receipt.get("ordinals_equal_required") is not False
        or receipt.get("status") != "VERIFIED_TOKEN_EXACT_CROSSWALK_NOT_EXTRACTED"
        or receipt.get("sample_token_set_sha256") != verified["sample_token_set_sha256"]
        or receipt.get("features_read") is not False
        or receipt.get("features_extracted") is not False
        or receipt.get("gpu_used") is not False
    ):
        raise ValueError("Teacher crosswalk receipt differs")
    crosswalk_path = path.parent / str(receipt["crosswalk_path"])
    inventory_path = path.parent / "teacher_source_inventory.json"
    if (
        file_sha256(crosswalk_path) != receipt["crosswalk_sha256"]
        or file_sha256(inventory_path) != receipt["source_inventory_sha256"]
    ):
        raise ValueError("Teacher crosswalk artifacts changed")
    frame = pd.read_parquet(crosswalk_path)
    population = pd.read_parquet(
        fold_contract_path.resolve(strict=True).parent / "population.parquet"
    )
    validate_crosswalk_frame(frame, population)
    population_root = fold_contract_path.resolve(strict=True).parent
    source = json.loads((population_root / "SOURCE_RECEIPT.json").read_text(encoding="utf-8"))
    raw_identity = (
        ds.dataset(source["sources"]["garl_data"]["path"])
        .to_table(columns=["sample_token", "rgb_member_paths"])
        .to_pandas()
    )
    if len(raw_identity) != 88_744 or raw_identity["sample_token"].duplicated().any():
        raise ValueError("Raw Garl identity universe differs during crosswalk verification")
    raw_by_token = raw_identity.set_index("sample_token")["rgb_member_paths"]
    expected_members = [raw_by_token.loc[str(token)] for token in frame["sample_token"]]
    if any(
        not isinstance(members, (list, np.ndarray)) or len(members) != 2
        for members in expected_members
    ):
        raise ValueError("Raw Garl RGB members differ during crosswalk verification")
    if (
        frame["raw_rgb_member_0"].astype(str).tolist()
        != [str(members[0]) for members in expected_members]
        or frame["raw_rgb_member_1"].astype(str).tolist()
        != [str(members[1]) for members in expected_members]
    ):
        raise ValueError("Crosswalk raw RGB member identity differs")
    if receipt.get("role_token_sets") != _role_token_sets(fold_contract_path):
        raise ValueError("Teacher crosswalk role token sets differ from the fold")
    teacher_root = Path(str(receipt["teacher_root"])).resolve(strict=True)
    historical_run = Path(str(receipt["historical_run"])).resolve(strict=True)
    if {name: file_sha256(historical_run / name) for name in PARENT_PINS} != PARENT_PINS:
        raise ValueError("Historical teacher parents changed")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory_payload = dict(inventory)
    inventory_claimed = inventory_payload.pop("receipt_sha256", None)
    if (
        inventory_claimed != canonical_sha256(inventory_payload)
        or len(inventory["shards"]) != 2_774
    ):
        raise ValueError("Teacher source inventory differs")
    total_rows = 0
    for index, item in enumerate(inventory["shards"]):
        if (
            item.get("shard") != f"shard_{index:05d}.npz"
            or item.get("receipt") != f"shard_{index:05d}.json"
            or int(item.get("rows", -1)) < 1
        ):
            raise ValueError("Teacher inventory canonical order differs")
        total_rows += int(item["rows"])
        shard_path = (teacher_root / item["shard"]).resolve(strict=True)
        shard_receipt_path = (teacher_root / item["receipt"]).resolve(strict=True)
        if shard_path.parent != teacher_root or shard_receipt_path.parent != teacher_root:
            raise ValueError("Teacher inventory path escapes its frozen root")
        if (
            file_sha256(shard_path) != item["shard_sha256"]
            or file_sha256(shard_receipt_path) != item["receipt_sha256"]
        ):
            raise ValueError("Teacher source shard or receipt changed")
    if total_rows != 88_744:
        raise ValueError("Teacher inventory row sum differs")
    _verify_physical_rows(frame, teacher_root)
    return {"status": "PASS", "rows": len(frame)}


__all__ = ["build_teacher_crosswalk", "validate_crosswalk_frame", "verify_teacher_crosswalk"]
