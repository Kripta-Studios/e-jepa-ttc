"""Verified TRAIN24-only conversion of the frozen TRAIN40 DINO teacher cache.

The converter reads only the identity-bearing teacher shards and the label-free
independent population.  It never loads the teacher model, sensor data, TTC
labels, or a historical learned PHASE/H8 artifact.  Converted shards are a
write-once dependency; fitting code must use :func:`load_teacher_role` instead
of opening the historical TRAIN40 cache.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import struct
import sys
import tempfile
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .lineage import load_rotation_role, verify_independent_contract
from .teacher_crosswalk import verify_teacher_crosswalk

EXPECTED_SOURCE_MANIFEST_SHA256 = "646d5569671e9d8af9b21c84c12a55584be675ccfc77dc5e996ccea6fb8d38de"
EXPECTED_SOURCE_FREEZE_SHA256 = "a80896b73b241b5570861c381497585348196ff06dac180dd9e343af4d753fbb"
EXPECTED_SOURCE_ADMISSION_SHA256 = (
    "7c770a0526b43d815b9327dbb01666bf40b329469b091ca40cf7aa43be5a0733"
)
EXPECTED_WEIGHT_FILE_SHA256 = "8bab82eaa133954d0557bebf70416e0b667fa417a19ea47b697683b34190633d"
EXPECTED_ORDERED_STATE_SHA256 = "96ad081d5863ba14707d2e3bb734b7fa8dfbf3248cb67ddfcc74d86dd160d628"
EXPECTED_MODEL_REVISION = "e959efa74c867491dcfe3ec3e4f97382e39025b3"
EXPECTED_FOLD_CONTRACT_SHA256 = (
    "ef67d27078a978cdfb94f0f99f660a44d64079fe5e9e2c1887841e751b6d81a4"
)
EXPECTED_ROW_LOCALITY_RECEIPT_SHA256 = (
    "9ed1ff47286f200407c18ea894cf760644238ae9977409ec1ad6b4c4fdbaac7e"
)
EXPECTED_MODEL_CONFIG_SHA256 = (
    "cb33dd99f77a2a73806fa218df80782a15e81422eb6d6fb2816534a3cb620360"
)
EXPECTED_TEACHER_PARENT_SHA256 = (
    "f9cdfb13ba99299de717423cc8d0334ded8e9628895e7a7f4921fcdb13584b75"
)
EXPECTED_TEACHER_REPOSITORY = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
EXPECTED_WEIGHT_BYTES = 784_957_976
EXPECTED_TEACHER_SOURCE_SHA256 = "c03174474ddde846266b85356f7032f5ce18158e97af6cdf8dbf1634e7b77370"
EXPECTED_RELATION_SOURCE_SHA256 = "e778eb51d46514beef2a1700d22d26b08ad3453a3fda628df4fb811afedd57ed"
EXPECTED_SOURCE_ROWS = 88_744
EXPECTED_ROWS = 53_047
EXPECTED_CROSSWALK_RECEIPT_SHA256 = (
    "267e5d5ceb5589f1a251e237e5e7fd8c552b319b28d4e1b6d3ca4859d560233e"
)
EXPECTED_CROSSWALK_SHA256 = "9545f48fdfe5f0cf9222972cfc3d042a4b0cf23daf274fdae0949e98f8569655"
EXPECTED_CROSSWALK_INVENTORY_SHA256 = (
    "e23946fe0d10accb28e28eb3799432a73edf571efc96d95df9ca4b54b1524ba7"
)
SHARD_ROWS = 32

MANIFEST_SCHEMA = "independent_phase_train24_teacher_cache_v2"
AUTHORIZATION_SCHEMA = "independent_phase_teacher_cache_authorization_v2"
INDEX_COLUMNS = [
    "public_source_ordinal",
    "fit_ordinal",
    "sample_token",
    "group_id",
    "sequence_id",
    "track_id",
    "public_track_id",
    "timestamp_us",
    "cache_shard",
    "cache_row",
]
NPZ_KEYS = {
    "public_source_ordinals",
    "fit_ordinals",
    "tokens",
    "relation_targets",
    "relation_valid",
    "rgb_sha256",
}
_ROLES = frozenset({"producer", "refiner"})


def file_sha256(path: Path) -> str:
    """Hash one existing regular file."""
    with path.resolve(strict=True).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@dataclass(frozen=True)
class SourcePins:
    """Immutable identity of the only admitted historical conversion source."""

    manifest_sha256: str = EXPECTED_SOURCE_MANIFEST_SHA256
    freeze_sha256: str = EXPECTED_SOURCE_FREEZE_SHA256
    admission_sha256: str = EXPECTED_SOURCE_ADMISSION_SHA256
    weight_file_sha256: str = EXPECTED_WEIGHT_FILE_SHA256
    ordered_state_sha256: str = EXPECTED_ORDERED_STATE_SHA256
    model_revision: str = EXPECTED_MODEL_REVISION
    source_rows: int = EXPECTED_SOURCE_ROWS
    output_rows: int = EXPECTED_ROWS
    source_shards: int = 2774
    fold_contract_sha256: str = EXPECTED_FOLD_CONTRACT_SHA256
    row_locality_receipt_sha256: str = EXPECTED_ROW_LOCALITY_RECEIPT_SHA256
    model_config_sha256: str = EXPECTED_MODEL_CONFIG_SHA256
    teacher_parent_sha256: str = EXPECTED_TEACHER_PARENT_SHA256
    teacher_repository: str = EXPECTED_TEACHER_REPOSITORY
    weight_bytes: int = EXPECTED_WEIGHT_BYTES
    crosswalk_receipt_sha256: str = EXPECTED_CROSSWALK_RECEIPT_SHA256
    crosswalk_sha256: str = EXPECTED_CROSSWALK_SHA256
    crosswalk_inventory_sha256: str = EXPECTED_CROSSWALK_INVENTORY_SHA256


DEFAULT_SOURCE_PINS = SourcePins()


def audit_historical_row_locality(
    *,
    teacher_source_path: Path,
    relation_source_path: Path,
    model_config_path: Path,
    weight_path: Path,
) -> dict[str, Any]:
    """Prove cached relation targets do not depend on neighbouring batch rows.

    This CPU-only audit does not deserialize model tensors.  It combines exact
    historical source pins, an executable relation-map batching fixture, and a
    safetensors-header/config inspection for batch-global running statistics.
    """
    if file_sha256(teacher_source_path) != EXPECTED_TEACHER_SOURCE_SHA256:
        raise ValueError("Historical teacher source differs")
    if file_sha256(relation_source_path) != EXPECTED_RELATION_SOURCE_SHA256:
        raise ValueError("Historical relation source differs")
    if file_sha256(weight_path) != EXPECTED_WEIGHT_FILE_SHA256:
        raise ValueError("Historical teacher weight file differs")
    teacher_source = teacher_source_path.read_text(encoding="utf-8")
    required_fragments = (
        ".eval()",
        ".requires_grad_(False)",
        "hidden_states[2]",
        "local_cosine_relation_maps(features)",
        "np.stack([x[0] for x in inputs])",
    )
    if any(fragment not in teacher_source for fragment in required_fragments):
        raise ValueError("Historical teacher evaluation chain differs")
    config = _json(model_config_path.resolve(strict=True))
    if (
        config.get("architectures") != ["DINOv3ConvNextModel"]
        or config.get("drop_path_rate") != 0.0
        or config.get("layer_norm_eps") != 1.0e-6
        or config.get("torch_dtype") != "float32"
    ):
        raise ValueError("Historical DINOv3 ConvNeXt config differs")
    with weight_path.resolve(strict=True).open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    tensor_keys = [str(key) for key in header if key != "__metadata__"]
    running_markers = ("running_mean", "running_var", "num_batches_tracked")
    running_state = [key for key in tensor_keys if any(item in key for item in running_markers)]
    if running_state or not any("layer_norm" in key for key in tensor_keys):
        raise ValueError("Historical weights contain unexpected batch-global normalization state")

    module_name = "_audited_historical_dinov3_relational"
    specification = importlib.util.spec_from_file_location(module_name, relation_source_path)
    if specification is None or specification.loader is None:
        raise ValueError("Cannot load historical relation implementation")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
        torch: Any = importlib.import_module("torch")

        generator = torch.Generator(device="cpu").manual_seed(7)
        anchor = torch.randn((1, 384, 8, 8), generator=generator)
        distractor_a = torch.randn((1, 384, 8, 8), generator=generator) * 100.0
        distractor_b = torch.zeros((1, 384, 8, 8))
        isolated = module.local_cosine_relation_maps(anchor)
        batch_a = module.local_cosine_relation_maps(torch.cat((anchor, distractor_a), dim=0))
        batch_b = module.local_cosine_relation_maps(torch.cat((anchor, distractor_b), dim=0))
        value_error = max(
            float((isolated.values[0] - batch_a.values[0]).abs().max().item()),
            float((isolated.values[0] - batch_b.values[0]).abs().max().item()),
        )
        valid_equal = bool(
            torch.equal(isolated.valid[0], batch_a.valid[0])
            and torch.equal(isolated.valid[0], batch_b.valid[0])
        )
    finally:
        sys.modules.pop(module_name, None)
    if value_error != 0.0 or not valid_equal:
        raise ValueError("Historical relation maps depend on neighbouring batch rows")
    return {
        "schema": "independent_phase_historical_teacher_row_locality_v1",
        "status": "PASS_ROW_LOCAL_NO_REEXTRACTION_REQUIRED",
        "teacher_source_sha256": EXPECTED_TEACHER_SOURCE_SHA256,
        "relation_source_sha256": EXPECTED_RELATION_SOURCE_SHA256,
        "model_config_sha256": file_sha256(model_config_path),
        "weight_file_sha256": EXPECTED_WEIGHT_FILE_SHA256,
        "model_revision": EXPECTED_MODEL_REVISION,
        "model_eval_mode": True,
        "drop_path_rate": 0.0,
        "batch_global_running_state_keys": 0,
        "layer_norm_parameter_keys": sum("layer_norm" in key for key in tensor_keys),
        "relation_fixture_anchor_neighbor_variants": 2,
        "relation_fixture_max_absolute_error": value_error,
        "relation_fixture_valid_masks_equal": valid_equal,
        "teacher_model_tensors_loaded": False,
        "gpu_used": False,
        "conclusion": (
            "Filtering cached rows is equivalent to materializing those rows alone: "
            "the frozen eval ConvNeXt has no batch-running state, preprocessing is per image, "
            "and relation normalization/reductions use only channel/spatial axes within a row."
        ),
    }


def _validate_authorization(
    path: Path,
    *,
    fold_contract_path: Path,
    population_path: Path,
    row_locality_receipt_path: Path,
    teacher_crosswalk_receipt_path: Path,
    pins: SourcePins,
) -> dict[str, Any]:
    authorization = _json(path.resolve(strict=True))
    expected_fields = {
        "schema",
        "status",
        "root_reviewed",
        "independent_contract_review_status",
        "conversion_only_no_fitting",
        "fold_contract_sha256",
        "population_sha256",
        "source_teacher_manifest_sha256",
        "row_locality_receipt_sha256",
        "teacher_parent_sha256",
        "teacher_crosswalk_receipt_sha256",
        "teacher_crosswalk_sha256",
        "teacher_crosswalk_inventory_sha256",
    }
    if (
        set(authorization) != expected_fields
        or authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("status") != "AUTHORIZED"
        or authorization.get("root_reviewed") is not True
        or authorization.get("independent_contract_review_status") != "SHIP"
        or authorization.get("conversion_only_no_fitting") is not True
        or authorization.get("fold_contract_sha256") != file_sha256(fold_contract_path)
        or authorization.get("population_sha256") != file_sha256(population_path)
        or authorization.get("source_teacher_manifest_sha256") != pins.manifest_sha256
        or authorization.get("row_locality_receipt_sha256")
        != pins.row_locality_receipt_sha256
        or authorization.get("teacher_parent_sha256") != pins.teacher_parent_sha256
        or authorization.get("teacher_crosswalk_receipt_sha256")
        != pins.crosswalk_receipt_sha256
        or authorization.get("teacher_crosswalk_sha256") != pins.crosswalk_sha256
        or authorization.get("teacher_crosswalk_inventory_sha256")
        != pins.crosswalk_inventory_sha256
        or file_sha256(row_locality_receipt_path) != pins.row_locality_receipt_sha256
        or file_sha256(teacher_crosswalk_receipt_path) != pins.crosswalk_receipt_sha256
    ):
        raise ValueError("Teacher conversion lacks exact SHIP + root authorization")
    return authorization


def _verify_row_locality_receipt(path: Path, pins: SourcePins) -> dict[str, Any]:
    """Verify the exact CPU row-locality evidence required before conversion."""
    if file_sha256(path) != pins.row_locality_receipt_sha256:
        raise ValueError("Historical teacher row-locality receipt differs")
    receipt = _json(path)
    required = {
        "schema": "independent_phase_historical_teacher_row_locality_v1",
        "status": "PASS_ROW_LOCAL_NO_REEXTRACTION_REQUIRED",
        "teacher_source_sha256": EXPECTED_TEACHER_SOURCE_SHA256,
        "relation_source_sha256": EXPECTED_RELATION_SOURCE_SHA256,
        "model_config_sha256": pins.model_config_sha256,
        "weight_file_sha256": pins.weight_file_sha256,
        "model_revision": pins.model_revision,
        "model_eval_mode": True,
        "drop_path_rate": 0.0,
        "batch_global_running_state_keys": 0,
        "relation_fixture_anchor_neighbor_variants": 2,
        "relation_fixture_max_absolute_error": 0.0,
        "relation_fixture_valid_masks_equal": True,
        "teacher_model_tensors_loaded": False,
        "gpu_used": False,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("Historical teacher row-locality evidence does not pass")
    layer_norm_keys = receipt.get("layer_norm_parameter_keys")
    if not isinstance(layer_norm_keys, int) or layer_norm_keys <= 0:
        raise ValueError("Historical teacher row-locality normalization evidence differs")
    if not isinstance(receipt.get("conclusion"), str) or not receipt["conclusion"]:
        raise ValueError("Historical teacher row-locality conclusion is absent")
    return receipt


def _verify_teacher_parent(
    path: Path,
    *,
    weight_path: Path,
    pins: SourcePins,
    sample_token_set_sha256: str,
) -> dict[str, Any]:
    """Verify the distinct physical frozen-weight parent receipt."""
    if file_sha256(path) != pins.teacher_parent_sha256:
        raise ValueError("Frozen teacher parent receipt differs")
    parent = _json(path)
    if (
        parent.get("schema") != "independent_phase_frozen_teacher_parent_v1"
        or parent.get("status")
        != "AVAILABLE_FROZEN_NOT_EXTRACTED_FOR_INDEPENDENT_REFIT"
        or parent.get("kind") != "frozen_dinov3_teacher_weights"
        or parent.get("fold_contract_sha256") != pins.fold_contract_sha256
        or parent.get("repository") != pins.teacher_repository
        or parent.get("revision") != pins.model_revision
        or parent.get("ordered_state_tensor_sha256") != pins.ordered_state_sha256
        or parent.get("weights_file_sha256") != pins.weight_file_sha256
        or parent.get("weights_bytes") != pins.weight_bytes
        or parent.get("query_whitelist") != "TRAIN24_ONLY"
        or parent.get("queries") != pins.output_rows
        or parent.get("query_token_set_sha256") != sample_token_set_sha256
        or parent.get("optimizer_updates") != 0
        or parent.get("teacher_features_materialized") is not False
    ):
        raise ValueError("Frozen teacher parent scientific contract differs")
    resolved_weight = weight_path.resolve(strict=True)
    if (
        Path(str(parent.get("weights_path"))).resolve(strict=True) != resolved_weight
        or resolved_weight.stat().st_size != pins.weight_bytes
        or file_sha256(resolved_weight) != pins.weight_file_sha256
    ):
        raise ValueError("Frozen teacher parent physical weight differs")
    return parent


def _verify_source_header(
    source_root: Path, weight_path: Path, pins: SourcePins
) -> tuple[dict[str, Any], Path]:
    source_root = source_root.resolve(strict=True)
    manifest_path = source_root / "TEACHER_MANIFEST.json"
    freeze_path = source_root / "TEACHER_FREEZE.json"
    admission_path = source_root / "TEACHER_COMPLETE_ADMISSION.json"
    observed = {
        "manifest": file_sha256(manifest_path),
        "freeze": file_sha256(freeze_path),
        "admission": file_sha256(admission_path),
        "weight_file": file_sha256(weight_path),
    }
    expected = {
        "manifest": pins.manifest_sha256,
        "freeze": pins.freeze_sha256,
        "admission": pins.admission_sha256,
        "weight_file": pins.weight_file_sha256,
    }
    if observed != expected:
        raise ValueError(f"Frozen teacher source pin mismatch: {observed}")
    manifest, freeze, admission = map(_json, (manifest_path, freeze_path, admission_path))
    if (
        manifest.get("schema") != "train40_complete_fragment_manifest_v1"
        or manifest.get("status") != "COMPLETE_VERIFIED"
        or manifest.get("kind") != "TEACHER"
        or manifest.get("row_count") != pins.source_rows
        or manifest.get("sequence_count") != 40
        or manifest.get("freeze_sha256") != pins.freeze_sha256
        or freeze.get("ordered_tensor_weights_sha256") != pins.ordered_state_sha256
        or freeze.get("model_revision") != pins.model_revision
        or freeze.get("TTC_targets_read_by_teacher") is not False
        or freeze.get("optimizer_updates") != 0
        or admission.get("status") != "PASSED"
        or admission.get("complete_rows") != pins.source_rows
        or admission.get("all_ordinals_and_tokens_exactly_match") is not True
        or admission.get("TTC_targets_read") is not False
        or admission.get("optimizer_updates") != 0
        or admission.get("ordered_teacher_weights_sha256") != pins.ordered_state_sha256
    ):
        raise ValueError("Frozen teacher scientific contract differs")
    shards = manifest.get("ordered_shards")
    if not isinstance(shards, list) or len(shards) != pins.source_shards:
        raise ValueError("Frozen teacher shard inventory differs")
    return manifest, source_root / "teacher"


def _verified_source_rows(
    manifest: Mapping[str, Any], directory: Path, pins: SourcePins
) -> Iterator[
    tuple[int, str, np.ndarray, np.ndarray, np.ndarray, dict[str, str | int]]
]:
    expected_ordinal = 0
    for reference in manifest["ordered_shards"]:
        if not isinstance(reference, dict):
            raise ValueError("Malformed frozen teacher shard reference")
        name = str(reference.get("name", ""))
        shard_path = (directory / name).resolve(strict=True)
        receipt_path = shard_path.with_suffix(".json")
        if shard_path.parent != directory.resolve(strict=True):
            raise ValueError("Frozen teacher shard escapes its directory")
        if file_sha256(shard_path) != reference.get("sha256") or file_sha256(
            receipt_path
        ) != reference.get("receipt_sha256"):
            raise ValueError(f"Frozen teacher shard/receipt hash mismatch: {name}")
        receipt = _json(receipt_path)
        reference_start = int(reference.get("start", -1))
        reference_stop = int(reference.get("stop", -1))
        if (
            receipt.get("status") != "VERIFIED"
            or receipt.get("sha256") != reference.get("sha256")
            or receipt.get("freeze_sha256") != pins.freeze_sha256
            or receipt.get("teacher_weights_sha256") != pins.ordered_state_sha256
            or receipt.get("TTC_targets_read") is not False
            or receipt.get("optimizer_updates") != 0
            or reference_start != expected_ordinal
            or reference_stop <= reference_start
            or reference.get("bytes") != shard_path.stat().st_size
        ):
            raise ValueError(f"Frozen teacher receipt differs: {name}")
        with np.load(shard_path, allow_pickle=False) as stored:
            if set(stored.files) != {
                "ordinals",
                "tokens",
                "relation_targets",
                "relation_valid",
                "rgb_sha256",
            }:
                raise ValueError(f"Unknown frozen teacher fields: {name}")
            ordinals = stored["ordinals"]
            tokens = stored["tokens"]
            targets = stored["relation_targets"]
            valid = stored["relation_valid"]
            rgb = stored["rgb_sha256"]
            rows = len(ordinals)
            if (
                ordinals.dtype != np.int64
                or not np.array_equal(
                    ordinals, np.arange(expected_ordinal, expected_ordinal + rows)
                )
                or targets.shape != (rows, 2, 6, 32, 32)
                or targets.dtype != np.float16
                or valid.shape != targets.shape
                or valid.dtype != np.bool_
                or rgb.shape != (rows, 2)
                or tokens.shape != (rows,)
                or receipt.get("rows") != rows
                or reference_stop != expected_ordinal + rows
                or not all(
                    len(str(value)) == 64
                    and all(character in "0123456789abcdef" for character in str(value))
                    for value in rgb.flat
                )
                or not bool(np.isfinite(targets).all())
            ):
                raise ValueError(f"Frozen teacher payload differs: {name}")
            for row in range(rows):
                yield (
                    int(ordinals[row]),
                    str(tokens[row]),
                    targets[row].copy(),
                    valid[row].copy(),
                    rgb[row].copy(),
                    {
                        "name": name,
                        "sha256": str(reference["sha256"]),
                        "receipt_sha256": str(reference["receipt_sha256"]),
                        "teacher_ordinal": int(ordinals[row]),
                        "teacher_local_row": row,
                    },
                )
            expected_ordinal += rows
    if expected_ordinal != pins.source_rows:
        raise ValueError("Frozen teacher rows are incomplete")


def _write_output_shard(
    directory: Path,
    rows: list[
        tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, str | int]]
    ],
    shard_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    name = f"shard_{shard_index:05d}.npz"
    path = directory / name
    source_refs = {
        json.dumps(
            {key: row[4][key] for key in ("name", "sha256", "receipt_sha256")},
            sort_keys=True,
        ): {key: row[4][key] for key in ("name", "sha256", "receipt_sha256")}
        for row in rows
    }
    np.savez_compressed(
        path,
        public_source_ordinals=np.asarray(
            [row[0]["public_source_ordinal"] for row in rows], dtype=np.int64
        ),
        fit_ordinals=np.asarray([row[0]["fit_ordinal"] for row in rows], dtype=np.int64),
        tokens=np.asarray([row[0]["sample_token"] for row in rows]),
        relation_targets=np.stack([row[1] for row in rows]),
        relation_valid=np.stack([row[2] for row in rows]),
        rgb_sha256=np.stack([row[3] for row in rows]),
    )
    digest = file_sha256(path)
    receipt = {
        "schema": "independent_phase_teacher_shard_receipt_v1",
        "status": "VERIFIED_CONVERSION",
        "name": name,
        "sha256": digest,
        "rows": len(rows),
        "start_fit_ordinal": int(rows[0][0]["fit_ordinal"]),
        "stop_fit_ordinal": int(rows[-1][0]["fit_ordinal"]) + 1,
        "source_shards": list(source_refs.values()),
        "ttc_values_read": False,
        "optimizer_updates": 0,
    }
    receipt_path = path.with_suffix(".json")
    _write_json(receipt_path, receipt)
    reference = {
        "name": name,
        "sha256": digest,
        "receipt_sha256": file_sha256(receipt_path),
        "start": receipt["start_fit_ordinal"],
        "stop": receipt["stop_fit_ordinal"],
        "rows": len(rows),
        "bytes": path.stat().st_size,
    }
    crosswalk = []
    for cache_row, (identity, *_rest) in enumerate(rows):
        crosswalk.append(
            {
                **{column: identity[column] for column in INDEX_COLUMNS[:8]},
                "cache_shard": name,
                "cache_row": cache_row,
            }
        )
    return reference, crosswalk


def prepare_teacher_cache(
    *,
    source_root: Path,
    weight_path: Path,
    fold_contract_path: Path,
    authorization_path: Path,
    row_locality_receipt_path: Path,
    teacher_crosswalk_receipt_path: Path | None = None,
    output_dir: Path,
    pins: SourcePins = DEFAULT_SOURCE_PINS,
) -> dict[str, Any]:
    """Copy the exact TRAIN24 subset after all immutable gates pass."""
    if teacher_crosswalk_receipt_path is None:
        teacher_crosswalk_receipt_path = (
            fold_contract_path.resolve(strict=True).parent / "TEACHER_CROSSWALK.json"
        )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    if file_sha256(fold_contract_path) != pins.fold_contract_sha256:
        raise ValueError("Independent TRAIN24 v2 fold contract differs from its pin")
    verification = verify_independent_contract(fold_contract_path)
    root = fold_contract_path.resolve(strict=True).parent
    population_path = root / "population.parquet"
    population_receipt = _json(root / "POPULATION.json")
    population = pd.read_parquet(population_path)
    if (
        list(population.columns) != INDEX_COLUMNS[:8]
        or len(population) != pins.output_rows
        or not population["fit_ordinal"].equals(pd.Series(range(pins.output_rows)))
        or not population["public_source_ordinal"].is_monotonic_increasing
        or population["sample_token"].astype(str).duplicated().any()
        or verification.get("sample_token_set_sha256")
        != population_receipt.get("sample_token_set_sha256")
    ):
        raise ValueError("Independent TRAIN24 identity population differs")
    authorization = _validate_authorization(
        authorization_path,
        fold_contract_path=fold_contract_path,
        population_path=population_path,
        row_locality_receipt_path=row_locality_receipt_path,
        teacher_crosswalk_receipt_path=teacher_crosswalk_receipt_path,
        pins=pins,
    )
    _verify_row_locality_receipt(row_locality_receipt_path, pins)
    teacher_parent_path = root / "TEACHER_PARENT.json"
    _verify_teacher_parent(
        teacher_parent_path,
        weight_path=weight_path,
        pins=pins,
        sample_token_set_sha256=verification["sample_token_set_sha256"],
    )
    source_manifest, source_directory = _verify_source_header(source_root, weight_path, pins)
    crosswalk_root = teacher_crosswalk_receipt_path.resolve(strict=True).parent
    crosswalk_path = crosswalk_root / "teacher_crosswalk.parquet"
    crosswalk_inventory_path = crosswalk_root / "teacher_source_inventory.json"
    if (
        file_sha256(teacher_crosswalk_receipt_path) != pins.crosswalk_receipt_sha256
        or file_sha256(crosswalk_path) != pins.crosswalk_sha256
        or file_sha256(crosswalk_inventory_path) != pins.crosswalk_inventory_sha256
    ):
        raise ValueError("Teacher crosswalk artifacts differ from their pins")
    verify_teacher_crosswalk(teacher_crosswalk_receipt_path, fold_contract_path)
    teacher_crosswalk = pd.read_parquet(crosswalk_path).sort_values(
        "fit_ordinal", kind="stable"
    )
    if not teacher_crosswalk[INDEX_COLUMNS[:8]].reset_index(drop=True).equals(
        population[INDEX_COLUMNS[:8]].reset_index(drop=True)
    ):
        raise ValueError("Teacher crosswalk does not preserve TRAIN24 identity")

    output_dir.mkdir(parents=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir()
    crosswalk_records = cast(list[dict[str, Any]], teacher_crosswalk.to_dict("records"))
    shard_refs: list[dict[str, Any]] = []
    crosswalk: list[dict[str, Any]] = []
    manifest_refs = {str(ref["name"]): ref for ref in source_manifest["ordered_shards"]}
    cache: OrderedDict[str, tuple[np.ndarray, ...]] = OrderedDict()
    for start in range(0, len(crosswalk_records), SHARD_ROWS):
        output_rows: list[
            tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, str | int]]
        ] = []
        for identity in crosswalk_records[start : start + SHARD_ROWS]:
            name = str(identity["teacher_shard"])
            reference = manifest_refs.get(name)
            if reference is None:
                raise ValueError("Teacher crosswalk references an unknown frozen shard")
            arrays = cache.get(name)
            if arrays is None:
                source_path = (source_directory / name).resolve(strict=True)
                source_receipt_path = source_path.with_suffix(".json")
                if source_path.parent != source_directory.resolve(strict=True):
                    raise ValueError("Teacher crosswalk source shard escapes its root")
                if (
                    file_sha256(source_path) != reference["sha256"]
                    or file_sha256(source_receipt_path) != reference["receipt_sha256"]
                ):
                    raise ValueError("Selected teacher source changed before load")
                with np.load(source_path, allow_pickle=False) as stored:
                    arrays = tuple(
                        stored[key].copy()
                        for key in (
                            "ordinals",
                            "tokens",
                            "relation_targets",
                            "relation_valid",
                            "rgb_sha256",
                        )
                    )
                if (
                    file_sha256(source_path) != reference["sha256"]
                    or file_sha256(source_receipt_path) != reference["receipt_sha256"]
                ):
                    raise ValueError("Selected teacher source changed during load")
                cache[name] = arrays
                if len(cache) > 8:
                    cache.popitem(last=False)
            else:
                cache.move_to_end(name)
            ordinals, tokens, targets, valid, rgb = arrays
            local = int(identity["teacher_local_row"])
            ordinal = int(identity["teacher_ordinal"])
            if (
                local < 0
                or local >= len(tokens)
                or int(ordinals[local]) != ordinal
                or str(tokens[local]) != str(identity["sample_token"])
                or str(rgb[local, 0]) != str(identity["teacher_rgb_sha256_0"])
                or str(rgb[local, 1]) != str(identity["teacher_rgb_sha256_1"])
                or targets.dtype != np.float16
                or valid.dtype != np.bool_
                or targets.shape[1:] != (2, 6, 32, 32)
                or valid.shape != targets.shape
                or not bool(np.isfinite(targets[local]).all())
            ):
                raise ValueError("Teacher crosswalk selected payload differs")
            source_ref: dict[str, str | int] = {
                "name": name,
                "sha256": str(reference["sha256"]),
                "receipt_sha256": str(reference["receipt_sha256"]),
                "teacher_ordinal": ordinal,
                "teacher_local_row": local,
            }
            output_rows.append(
                (
                    identity,
                    targets[local].copy(),
                    valid[local].copy(),
                    rgb[local].copy(),
                    source_ref,
                )
            )
        reference, rows = _write_output_shard(shard_dir, output_rows, len(shard_refs))
        shard_refs.append(reference)
        crosswalk.extend(rows)
    if len(crosswalk) != pins.output_rows:
        raise ValueError("Frozen teacher does not exactly cover TRAIN24")

    index = pd.DataFrame.from_records(crosswalk, columns=INDEX_COLUMNS)
    index_path = output_dir / "index.parquet"
    pq.write_table(pa.Table.from_pandas(index, preserve_index=False), index_path)
    authorization_sha = file_sha256(authorization_path)
    authorization_copy = output_dir / "AUTHORIZATION.json"
    shutil.copyfile(authorization_path.resolve(strict=True), authorization_copy)
    if file_sha256(authorization_copy) != authorization_sha:
        raise ValueError("Authorization copy changed during conversion")
    row_locality_copy = output_dir / "ROW_LOCALITY.json"
    shutil.copyfile(row_locality_receipt_path.resolve(strict=True), row_locality_copy)
    if file_sha256(row_locality_copy) != pins.row_locality_receipt_sha256:
        raise ValueError("Row-locality receipt copy changed during conversion")
    teacher_parent_copy = output_dir / "TEACHER_PARENT.json"
    shutil.copyfile(teacher_parent_path.resolve(strict=True), teacher_parent_copy)
    if file_sha256(teacher_parent_copy) != pins.teacher_parent_sha256:
        raise ValueError("Frozen teacher parent receipt copy changed during conversion")
    crosswalk_receipt_copy = output_dir / "TEACHER_CROSSWALK.json"
    crosswalk_copy = output_dir / "teacher_crosswalk.parquet"
    crosswalk_inventory_copy = output_dir / "teacher_source_inventory.json"
    shutil.copyfile(teacher_crosswalk_receipt_path, crosswalk_receipt_copy)
    shutil.copyfile(crosswalk_path, crosswalk_copy)
    shutil.copyfile(crosswalk_inventory_path, crosswalk_inventory_copy)
    if (
        file_sha256(crosswalk_receipt_copy) != pins.crosswalk_receipt_sha256
        or file_sha256(crosswalk_copy) != pins.crosswalk_sha256
        or file_sha256(crosswalk_inventory_copy) != pins.crosswalk_inventory_sha256
    ):
        raise ValueError("Teacher crosswalk copies changed during conversion")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "COMPLETE_VERIFIED_CONVERSION",
        "inference_population": "EXACT_TRAIN24_ONLY",
        "rows": pins.output_rows,
        "shard_rows": SHARD_ROWS,
        "shards": shard_refs,
        "index_path": index_path.name,
        "index_sha256": file_sha256(index_path),
        "identity_columns": INDEX_COLUMNS,
        "fold_contract_sha256": file_sha256(fold_contract_path),
        "population_sha256": file_sha256(population_path),
        "population_token_set_sha256": verification["sample_token_set_sha256"],
        "authorization_sha256": authorization_sha,
        "authorization_path": authorization_copy.name,
        "authorization_status": authorization["status"],
        "row_locality_path": row_locality_copy.name,
        "row_locality_receipt_sha256": pins.row_locality_receipt_sha256,
        "teacher_parent_path": teacher_parent_copy.name,
        "teacher_parent_sha256": pins.teacher_parent_sha256,
        "teacher_crosswalk_path": crosswalk_receipt_copy.name,
        "teacher_crosswalk_receipt_sha256": pins.crosswalk_receipt_sha256,
        "teacher_crosswalk_sha256": pins.crosswalk_sha256,
        "teacher_crosswalk_inventory_sha256": pins.crosswalk_inventory_sha256,
        "source": {
            "kind": "frozen_train40_teacher_conversion_only",
            "manifest_sha256": pins.manifest_sha256,
            "freeze_sha256": pins.freeze_sha256,
            "admission_sha256": pins.admission_sha256,
            "weight_file_sha256": pins.weight_file_sha256,
            "ordered_state_tensor_sha256": pins.ordered_state_sha256,
            "model_revision": pins.model_revision,
            "source_rows": pins.source_rows,
        },
        "fit_parent": {
            "kind": "frozen_dinov3_teacher_weights",
            "path": teacher_parent_copy.name,
            "sha256": pins.teacher_parent_sha256,
            "frozen": True,
            "query_whitelist": "TRAIN24_ONLY",
            "queries": pins.output_rows,
            "query_token_set_sha256": verification["sample_token_set_sha256"],
            "repository": pins.teacher_repository,
            "revision": pins.model_revision,
            "weights_file_sha256": pins.weight_file_sha256,
            "ordered_state_tensor_sha256": pins.ordered_state_sha256,
        },
        "forbidden_metadata_present": False,
        "ttc_values_read": False,
        "sensor_payloads_read": False,
        "teacher_model_loaded": False,
        "optimizer_updates": 0,
        "old_cache_live_fit_reader_allowed": False,
    }
    manifest_path = output_dir / "MANIFEST.json"
    _write_json(manifest_path, manifest)
    manifest_sha256 = file_sha256(manifest_path)
    verified = verify_teacher_cache(
        manifest_path,
        expected_manifest_sha256=manifest_sha256,
        fold_contract_path=fold_contract_path,
        pins=pins,
    )
    return {
        **verified,
        "status": "COMPLETE_VERIFIED_CONVERSION",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "shards": len(shard_refs),
        "self_verified_before_publish": True,
    }


def verify_teacher_cache(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    fold_contract_path: Path,
    pins: SourcePins = DEFAULT_SOURCE_PINS,
) -> dict[str, Any]:
    """Rehash the standalone cache and its exact TRAIN24 crosswalk."""
    if file_sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("Teacher cache manifest differs from its caller pin")
    manifest = _json(manifest_path)
    root = manifest_path.resolve(strict=True).parent
    verification = verify_independent_contract(fold_contract_path)
    population_path = fold_contract_path.resolve(strict=True).parent / "population.parquet"
    if file_sha256(fold_contract_path) != pins.fold_contract_sha256:
        raise ValueError("Independent TRAIN24 v2 fold contract differs from its pin")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("status") != "COMPLETE_VERIFIED_CONVERSION"
        or manifest.get("inference_population") != "EXACT_TRAIN24_ONLY"
        or manifest.get("rows") != pins.output_rows
        or manifest.get("shard_rows") != SHARD_ROWS
        or manifest.get("identity_columns") != INDEX_COLUMNS
        or manifest.get("fold_contract_sha256") != file_sha256(fold_contract_path)
        or manifest.get("population_sha256") != file_sha256(population_path)
        or manifest.get("population_token_set_sha256") != verification["sample_token_set_sha256"]
        or manifest.get("forbidden_metadata_present") is not False
        or manifest.get("ttc_values_read") is not False
        or manifest.get("teacher_model_loaded") is not False
        or manifest.get("optimizer_updates") != 0
        or manifest.get("old_cache_live_fit_reader_allowed") is not False
        or manifest.get("sensor_payloads_read") is not False
    ):
        raise ValueError("Teacher cache scientific contract differs")
    source = manifest.get("source")
    if not isinstance(source, dict) or source != {
        "kind": "frozen_train40_teacher_conversion_only",
        "manifest_sha256": pins.manifest_sha256,
        "freeze_sha256": pins.freeze_sha256,
        "admission_sha256": pins.admission_sha256,
        "weight_file_sha256": pins.weight_file_sha256,
        "ordered_state_tensor_sha256": pins.ordered_state_sha256,
        "model_revision": pins.model_revision,
        "source_rows": pins.source_rows,
    }:
        raise ValueError("Teacher cache source identity differs")
    authorization_path = root / str(manifest.get("authorization_path"))
    if authorization_path.name != "AUTHORIZATION.json":
        raise ValueError("Teacher cache authorization path differs")
    if file_sha256(authorization_path) != manifest.get("authorization_sha256"):
        raise ValueError("Teacher cache authorization changed")
    row_locality_path = root / str(manifest.get("row_locality_path"))
    if row_locality_path.name != "ROW_LOCALITY.json":
        raise ValueError("Teacher cache row-locality path differs")
    if (
        manifest.get("row_locality_receipt_sha256")
        != pins.row_locality_receipt_sha256
        or file_sha256(row_locality_path) != pins.row_locality_receipt_sha256
    ):
        raise ValueError("Teacher cache row-locality binding changed")
    _verify_row_locality_receipt(row_locality_path, pins)
    authorization = _validate_authorization(
        authorization_path,
        fold_contract_path=fold_contract_path,
        population_path=population_path,
        row_locality_receipt_path=row_locality_path,
        teacher_crosswalk_receipt_path=root / str(manifest.get("teacher_crosswalk_path")),
        pins=pins,
    )
    if manifest.get("authorization_status") != authorization["status"]:
        raise ValueError("Teacher cache authorization status differs")
    crosswalk_receipt_path = root / str(manifest.get("teacher_crosswalk_path"))
    if (
        crosswalk_receipt_path.name != "TEACHER_CROSSWALK.json"
        or file_sha256(crosswalk_receipt_path) != pins.crosswalk_receipt_sha256
        or file_sha256(root / "teacher_crosswalk.parquet") != pins.crosswalk_sha256
        or file_sha256(root / "teacher_source_inventory.json")
        != pins.crosswalk_inventory_sha256
        or manifest.get("teacher_crosswalk_receipt_sha256")
        != pins.crosswalk_receipt_sha256
        or manifest.get("teacher_crosswalk_sha256") != pins.crosswalk_sha256
        or manifest.get("teacher_crosswalk_inventory_sha256")
        != pins.crosswalk_inventory_sha256
    ):
        raise ValueError("Teacher cache crosswalk binding differs")
    verify_teacher_crosswalk(crosswalk_receipt_path, fold_contract_path)
    teacher_parent_path = root / str(manifest.get("teacher_parent_path"))
    if (
        teacher_parent_path.name != "TEACHER_PARENT.json"
        or manifest.get("teacher_parent_sha256") != pins.teacher_parent_sha256
        or file_sha256(teacher_parent_path) != pins.teacher_parent_sha256
    ):
        raise ValueError("Teacher cache frozen-parent binding differs")
    teacher_parent = _json(teacher_parent_path)
    _verify_teacher_parent(
        teacher_parent_path,
        weight_path=Path(str(teacher_parent.get("weights_path"))),
        pins=pins,
        sample_token_set_sha256=verification["sample_token_set_sha256"],
    )
    expected_parent = {
        "kind": "frozen_dinov3_teacher_weights",
        "path": "TEACHER_PARENT.json",
        "sha256": pins.teacher_parent_sha256,
        "frozen": True,
        "query_whitelist": "TRAIN24_ONLY",
        "queries": pins.output_rows,
        "query_token_set_sha256": verification["sample_token_set_sha256"],
        "repository": pins.teacher_repository,
        "revision": pins.model_revision,
        "weights_file_sha256": pins.weight_file_sha256,
        "ordered_state_tensor_sha256": pins.ordered_state_sha256,
    }
    if manifest.get("fit_parent") != expected_parent:
        raise ValueError("Teacher cache fit-parent identity differs")
    index_path = root / str(manifest.get("index_path"))
    if file_sha256(index_path) != manifest.get("index_sha256"):
        raise ValueError("Teacher cache identity index changed")
    index = pd.read_parquet(index_path)
    population = pd.read_parquet(population_path)
    if (
        list(index.columns) != INDEX_COLUMNS
        or len(index) != pins.output_rows
        or not index[INDEX_COLUMNS[:8]].equals(population[INDEX_COLUMNS[:8]])
        or index["fit_ordinal"].dtype != np.int64
        or index["public_source_ordinal"].dtype != np.int64
        or index["timestamp_us"].dtype != np.int64
    ):
        raise ValueError("Teacher cache crosswalk differs from TRAIN24")
    references = manifest.get("shards")
    expected_shards = (pins.output_rows + SHARD_ROWS - 1) // SHARD_ROWS
    if not isinstance(references, list) or len(references) != expected_shards:
        raise ValueError("Teacher cache canonical shard count differs")
    expected_fit = 0
    for shard_index, reference in enumerate(references):
        if not isinstance(reference, dict) or set(reference) != {
            "name",
            "sha256",
            "receipt_sha256",
            "start",
            "stop",
            "rows",
            "bytes",
        }:
            raise ValueError("Teacher cache shard reference schema differs")
        expected_name = f"shard_{shard_index:05d}.npz"
        expected_rows = min(SHARD_ROWS, pins.output_rows - expected_fit)
        if (
            reference.get("name") != expected_name
            or reference.get("start") != expected_fit
            or reference.get("stop") != expected_fit + expected_rows
            or reference.get("rows") != expected_rows
            or not _is_digest(reference.get("sha256"))
            or not _is_digest(reference.get("receipt_sha256"))
            or not isinstance(reference.get("bytes"), int)
            or reference["bytes"] <= 0
        ):
            raise ValueError("Teacher cache canonical shard reference differs")
        path = (root / "shards" / expected_name).resolve(strict=True)
        if path.parent != (root / "shards").resolve(strict=True):
            raise ValueError("Teacher cache shard escapes its directory")
        receipt_path = path.with_suffix(".json")
        if (
            path.stat().st_size != reference["bytes"]
            or file_sha256(path) != reference["sha256"]
            or file_sha256(receipt_path) != reference["receipt_sha256"]
        ):
            raise ValueError(f"Teacher cache shard changed: {path.name}")
        receipt = _json(receipt_path)
        expected_receipt_fields = {
            "schema",
            "status",
            "name",
            "sha256",
            "rows",
            "start_fit_ordinal",
            "stop_fit_ordinal",
            "source_shards",
            "ttc_values_read",
            "optimizer_updates",
        }
        source_refs = receipt.get("source_shards")
        if (
            set(receipt) != expected_receipt_fields
            or receipt.get("schema") != "independent_phase_teacher_shard_receipt_v1"
            or receipt.get("status") != "VERIFIED_CONVERSION"
            or receipt.get("name") != expected_name
            or receipt.get("sha256") != reference["sha256"]
            or receipt.get("rows") != expected_rows
            or receipt.get("start_fit_ordinal") != expected_fit
            or receipt.get("stop_fit_ordinal") != expected_fit + expected_rows
            or receipt.get("ttc_values_read") is not False
            or receipt.get("optimizer_updates") != 0
            or not isinstance(source_refs, list)
            or not source_refs
        ):
            raise ValueError(f"Teacher cache shard receipt differs: {path.name}")
        for source_ref in source_refs:
            if (
                not isinstance(source_ref, dict)
                or set(source_ref) != {"name", "sha256", "receipt_sha256"}
                or not isinstance(source_ref["name"], str)
                or not source_ref["name"].startswith("shard_")
                or not source_ref["name"].endswith(".npz")
                or not _is_digest(source_ref["sha256"])
                or not _is_digest(source_ref["receipt_sha256"])
            ):
                raise ValueError(f"Teacher cache source-shard reference differs: {path.name}")
        subset = index.iloc[expected_fit : expected_fit + expected_rows]
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != NPZ_KEYS:
                raise ValueError(f"Unknown teacher cache fields: {path.name}")
            public_ordinals = stored["public_source_ordinals"]
            fit_ordinals = stored["fit_ordinals"]
            tokens = stored["tokens"]
            targets = stored["relation_targets"]
            valid = stored["relation_valid"]
            rgb = stored["rgb_sha256"]
            if (
                public_ordinals.shape != (expected_rows,)
                or public_ordinals.dtype != np.int64
                or fit_ordinals.shape != (expected_rows,)
                or fit_ordinals.dtype != np.int64
                or tokens.shape != (expected_rows,)
                or tokens.dtype.kind != "U"
                or targets.shape != (expected_rows, 2, 6, 32, 32)
                or targets.dtype != np.float16
                or not bool(np.isfinite(targets).all())
                or valid.shape != targets.shape
                or valid.dtype != np.bool_
                or rgb.shape != (expected_rows, 2)
                or rgb.dtype.kind != "U"
                or not all(_is_digest(str(value)) for value in rgb.flat)
                or not np.array_equal(
                    public_ordinals, subset["public_source_ordinal"].to_numpy(dtype=np.int64)
                )
                or not np.array_equal(
                    fit_ordinals, subset["fit_ordinal"].to_numpy(dtype=np.int64)
                )
                or tokens.astype(str).tolist() != subset["sample_token"].astype(str).tolist()
                or subset["cache_shard"].astype(str).tolist()
                != [expected_name] * expected_rows
                or subset["cache_row"].to_numpy(dtype=np.int64).tolist()
                != list(range(expected_rows))
            ):
                raise ValueError(f"Teacher cache shard payload differs: {path.name}")
        expected_fit += expected_rows
    if expected_fit != pins.output_rows:
        raise ValueError("Teacher cache shard coverage is incomplete")
    expected_files = {
        f"shard_{index:05d}{suffix}"
        for index in range(expected_shards)
        for suffix in (".npz", ".json")
    }
    observed_files = {path.name for path in (root / "shards").iterdir() if path.is_file()}
    if observed_files != expected_files:
        raise ValueError("Teacher cache shard directory inventory differs")
    return {
        "status": "PASS",
        "rows": expected_fit,
        "manifest_sha256": expected_manifest_sha256,
        "sample_token_set_sha256": verification["sample_token_set_sha256"],
    }


@dataclass(frozen=True)
class TeacherRoleCache:
    """Verified role-scoped view; generation and arbitrary tokens are unavailable."""

    root: Path
    index: pd.DataFrame
    allowed_tokens: frozenset[str]
    role_receipt_path: Path
    role_receipt_sha256: str

    def get(self, sample_token: str) -> tuple[np.ndarray, np.ndarray]:
        """Return copied relation target/valid arrays for an admitted role token."""
        if sample_token not in self.allowed_tokens:
            raise KeyError("Teacher token is outside the admitted producer/refiner role")
        match = self.index[self.index["sample_token"] == sample_token]
        if len(match) != 1:
            raise KeyError("Teacher token is unavailable or duplicated")
        row = match.iloc[0]
        path = self.root / "shards" / str(row["cache_shard"])
        with np.load(path, allow_pickle=False) as stored:
            offset = int(row["cache_row"])
            return (
                stored["relation_targets"][offset].copy(),
                stored["relation_valid"][offset].copy(),
            )


def load_teacher_role(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    fold_contract_path: Path,
    rotation: int,
    role: str,
    role_receipt_path: Path,
    pins: SourcePins = DEFAULT_SOURCE_PINS,
) -> TeacherRoleCache:
    """Open a verified producer/refiner whitelist from the converted cache."""
    if role not in _ROLES:
        raise ValueError("Teacher cache is available only to producer/refiner fitting roles")
    verify_teacher_cache(
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        fold_contract_path=fold_contract_path,
        pins=pins,
    )
    role_rows = load_rotation_role(fold_contract_path, rotation=rotation, role=role)
    generation_rows = load_rotation_role(
        fold_contract_path, rotation=rotation, role="generation"
    )
    allowed = frozenset(role_rows["sample_token"].astype(str))
    generation_tokens = frozenset(generation_rows["sample_token"].astype(str))
    role_groups = sorted(set(role_rows["group_id"].astype(str)))
    generation_groups = sorted(set(generation_rows["group_id"].astype(str)))
    if allowed & generation_tokens or set(role_groups) & set(generation_groups):
        raise ValueError("Teacher fit-role whitelist overlaps its generation holdout")
    index = pd.read_parquet(manifest_path.resolve(strict=True).parent / "index.parquet")
    scoped = cast(
        pd.DataFrame,
        index.loc[index["sample_token"].isin(list(allowed))],
    ).reset_index(drop=True)
    if len(scoped) != len(allowed) or set(scoped["sample_token"].astype(str)) != set(allowed):
        raise ValueError("Teacher cache does not exactly cover the requested fit role")
    ordered_tokens = role_rows.sort_values("fit_ordinal", kind="stable")[
        "sample_token"
    ].astype(str).tolist()
    receipt = {
        "schema": "independent_phase_teacher_role_receipt_v1",
        "status": "PASS_EXACT_ROLE_FILTER",
        "cache_manifest_sha256": expected_manifest_sha256,
        "fold_contract_sha256": file_sha256(fold_contract_path),
        "rotation": rotation,
        "role": role,
        "queries": len(allowed),
        "ordered_token_sha256": _canonical_sha256(ordered_tokens),
        "token_set_sha256": _canonical_sha256(sorted(allowed)),
        "group_ids": role_groups,
        "generation_group_ids": generation_groups,
        "generation_excluded": True,
        "full_train24_exposed": len(allowed) == pins.output_rows,
    }
    if receipt["full_train24_exposed"] is not False:
        raise ValueError("Teacher role loader may not expose the full TRAIN24 cache")
    _write_json(role_receipt_path, receipt)
    receipt_sha256 = file_sha256(role_receipt_path)
    return TeacherRoleCache(
        manifest_path.resolve(strict=True).parent,
        scoped,
        allowed,
        role_receipt_path.resolve(strict=True),
        receipt_sha256,
    )


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "SourcePins",
    "TeacherRoleCache",
    "load_teacher_role",
    "prepare_teacher_cache",
    "verify_teacher_cache",
]
