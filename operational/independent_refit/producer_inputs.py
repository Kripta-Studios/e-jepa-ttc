"""Role-scoped reuse of the frozen raw-derived ObjectEventV4 input cache.

The historical INPUT cache contains deterministic, row-local T3 tensors made
from raw events and the same row's training boxes.  It contains no learned H8,
PAIR, PHASE, checkpoint, or population-normalizer output.  This loader exposes
only history-valid producer tokens and verifies every selected NPZ and receipt
before and after its first load.

The 31-packet independent history is a PHASE history contract.  It is not read
here and is never reshaped into a producer tensor.  Producer inputs are the
frozen native ObjectEventV4 T3 tensors addressed by exact ``sample_token``.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch

INPUT_MANIFEST_SHA256 = "754e8910802eb7c13749dbbe54374a1e513972c318e1269fbb9315640d7dd5fc"
PREPARE_FREEZE_SHA256 = "39911009cffc5198bba0c6fbf18c699fe93e7003418184ffcf67bc24023f44bf"
CROSSWALK_RECEIPT_SHA256 = "267e5d5ceb5589f1a251e237e5e7fd8c552b319b28d4e1b6d3ca4859d560233e"
CROSSWALK_PARQUET_SHA256 = "9545f48fdfe5f0cf9222972cfc3d042a4b0cf23daf274fdae0949e98f8569655"
H5_ADMISSION_SCHEMA = "independent_phase_event_h5_admission_v1"
H5_ADMISSION_STATUS = "PHYSICAL_H5_HASHES_AND_BOUNDS_VERIFIED"

SOURCE_ROWS = 88_744
SOURCE_SHARDS = 2_774
SHARD_ROWS = 32
EVENT_SHAPE = (3, 12, 128, 128)
NPZ_KEYS = {
    "ordinals",
    "events",
    "tokens",
    "target_ttc",
    "visible_heights",
    "boxes",
    "square",
    "motion",
    "delta",
}
FORBIDDEN_PATH_MARKERS = ("h8", "pair_feature", "phase", "normalizer", "checkpoint")

CRITICAL_ROW_LOCAL_SOURCES = {
    "src/e_jepa_ttc/data/eap_representation.py": (
        "1698434a1a5c1e41557e12221c96e545cb5208b57c35541a15599ac5c7fcaa14"
    ),
    "src/e_jepa_ttc/data/event_v4_common_roi.py": (
        "81a13706cc5ef74753e8ed4fb74f9e18e2fdf43fcdccde91a44e659bbfe663ea"
    ),
    "src/e_jepa_ttc/data/event_v4_geometry.py": (
        "0ab7b81fb85122fbaeb9527192675ebc77c33c27e50b5ed9b73bca8dec74d206"
    ),
    "src/e_jepa_ttc/data/garlttc_lhr_cache.py": (
        "70a768846dcaf7a3271f38c7991f65caffc412d0482fff98064ca3a904e154ff"
    ),
    "src/e_jepa_ttc/data/object_event_v4.py": (
        "84f7a1895523bb0c28cfd1f13b47ae4f86d906aed93e26354796df02999587b4"
    ),
    "src/e_jepa_ttc/data/types.py": (
        "79121bf50fc8a060bd8fa8c73bc4c572ed28ad960d6bf8e7476af6b5c450bd48"
    ),
}


def file_sha256(path: Path) -> str:
    """Hash one regular file with bounded memory."""

    with path.resolve(strict=True).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _safe_child(root: Path, name: str) -> Path:
    if Path(name).name != name or any(marker in name.lower() for marker in FORBIDDEN_PATH_MARKERS):
        raise ValueError(f"Unsafe or forbidden input-cache member: {name}")
    resolved_root = root.resolve(strict=True)
    value = (resolved_root / name).resolve(strict=True)
    if value.parent != resolved_root:
        raise ValueError("Input-cache member escapes its root")
    return value


def verify_row_local_prepare_freeze(path: Path) -> dict[str, Any]:
    """Verify the exact raw, per-row ObjectEventV4 preparation recipe."""

    if file_sha256(path) != PREPARE_FREEZE_SHA256:
        raise ValueError("Historical ObjectEventV4 preparation freeze changed")
    value = _json(path)
    config = value.get("config")
    files = value.get("files")
    if not isinstance(config, dict) or not isinstance(files, list):
        raise ValueError("Historical ObjectEventV4 preparation freeze is malformed")
    expected_config = {
        "store_full_frame_events": False,
        "store_garl_event_roi": False,
        "store_jepa_event_roi": False,
        "store_event_v4_common_roi": True,
        "event_v4_margin_fraction": 0.25,
        "event_v4_require_precontext": True,
        "event_v4_precontext_fallback": "shifted_event_window",
        "event_v4_bins_per_polarity": 5,
        "event_v4_storage_dtype": "float32",
        "materialize_splits": ["train"],
        "include_rgb": False,
        "include_masks": False,
        "roi_size": 128,
        "event_pixel_diff": 5,
        "preprocessing_device": "cpu",
        "expected_train_rows": SOURCE_ROWS,
    }
    if any(config.get(key) != expected for key, expected in expected_config.items()):
        raise ValueError("Historical ObjectEventV4 preparation recipe differs")
    source_map = {str(item.get("path")): item.get("sha256") for item in files}
    if any(source_map.get(name) != digest for name, digest in CRITICAL_ROW_LOCAL_SOURCES.items()):
        raise ValueError("Historical row-local ObjectEventV4 source identity differs")
    if value.get("optimizer_updates") != 0 or value.get("lossless_fp32") is not True:
        raise ValueError("Historical input preparation is not a zero-update FP32 transform")
    return value


@dataclass(frozen=True)
class SourceShard:
    """One exact historical input shard."""

    name: str
    sha256: str
    receipt_sha256: str
    start: int
    stop: int
    bytes: int


def verify_input_manifest(
    path: Path, source_root: Path, prepare_freeze_path: Path
) -> tuple[SourceShard, ...]:
    """Verify the immutable INPUT inventory without reading 50+ GiB eagerly."""

    verify_row_local_prepare_freeze(prepare_freeze_path)
    if file_sha256(path) != INPUT_MANIFEST_SHA256:
        raise ValueError("Historical ObjectEventV4 INPUT manifest changed")
    value = _json(path)
    raw = value.get("ordered_shards")
    if (
        value.get("schema") != "train40_complete_fragment_manifest_v1"
        or value.get("status") != "COMPLETE_VERIFIED"
        or value.get("kind") != "INPUT"
        or value.get("row_count") != SOURCE_ROWS
        or value.get("sequence_count") != 40
        or value.get("freeze_sha256") != PREPARE_FREEZE_SHA256
        or value.get("teacher_weights_sha256") is not None
        or value.get("optimizer_updates") != 0
        or not isinstance(raw, list)
        or len(raw) != SOURCE_SHARDS
    ):
        raise ValueError("Historical ObjectEventV4 INPUT manifest contract differs")
    root = source_root.resolve(strict=True)
    if any(marker in root.as_posix().lower() for marker in FORBIDDEN_PATH_MARKERS):
        raise ValueError("Learned/normalized cache cannot be an ObjectEventV4 source")
    result: list[SourceShard] = []
    expected_start = 0
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError("INPUT shard reference is malformed")
        expected_name = f"shard_{index:05d}.npz"
        expected_stop = min(expected_start + SHARD_ROWS, SOURCE_ROWS)
        shard = SourceShard(
            name=str(item.get("name")),
            sha256=str(item.get("sha256")),
            receipt_sha256=str(item.get("receipt_sha256")),
            start=int(item.get("start", -1)),
            stop=int(item.get("stop", -1)),
            bytes=int(item.get("bytes", -1)),
        )
        payload = _safe_child(root, shard.name)
        receipt = _safe_child(root, expected_name.replace(".npz", ".json"))
        if (
            shard.name != expected_name
            or shard.start != expected_start
            or shard.stop != expected_stop
            or shard.bytes <= 0
            or payload.stat().st_size != shard.bytes
            or file_sha256(receipt) != shard.receipt_sha256
        ):
            raise ValueError(f"INPUT shard inventory differs: {expected_name}")
        receipt_value = _json(receipt)
        if (
            receipt_value.get("status") != "VERIFIED"
            or receipt_value.get("sha256") != shard.sha256
            or receipt_value.get("rows") != shard.stop - shard.start
            or receipt_value.get("ordinals") != list(range(shard.start, shard.stop))
            or receipt_value.get("freeze_sha256") != PREPARE_FREEZE_SHA256
            or receipt_value.get("exact_compression_parity") is not True
            or receipt_value.get("optimizer_updates") != 0
        ):
            raise ValueError(f"INPUT shard receipt differs: {expected_name}")
        result.append(shard)
        expected_start = expected_stop
    if expected_start != SOURCE_ROWS:
        raise ValueError("INPUT shard coverage is incomplete")
    return tuple(result)


def verify_h5_admission_for_fit(path: Path) -> dict[str, Any]:
    """Require the real admission verifier immediately before opening the source."""

    value = _json(path)
    if (
        value.get("schema") != H5_ADMISSION_SCHEMA
        or value.get("status") != H5_ADMISSION_STATUS
        or value.get("full_hashes_computed") is not True
        or value.get("stream_bounds_checked") is not True
        or value.get("training_admitted") is not True
        or value.get("admissible_queries") != 52_950
        or value.get("invalid_history_queries") != 97
    ):
        raise RuntimeError("Independent event fitting requires physical H5 admission")
    from .h5_admission import verify_h5_admission

    if verify_h5_admission(path, full_hash=False).get("status") != "PASS":
        raise RuntimeError("Independent event H5 admission verification failed")
    return value


def _load_selected_index(
    *,
    crosswalk_receipt_path: Path,
    crosswalk_path: Path,
    role_mask_path: Path,
    role_mask_sha256: str,
    rotation: int,
) -> tuple[pd.DataFrame, frozenset[str]]:
    if (
        file_sha256(crosswalk_receipt_path) != CROSSWALK_RECEIPT_SHA256
        or file_sha256(crosswalk_path) != CROSSWALK_PARQUET_SHA256
        or file_sha256(role_mask_path) != role_mask_sha256
    ):
        raise ValueError("Producer input identity parent changed")
    crosswalk = pd.read_parquet(
        crosswalk_path,
        columns=[
            "fit_ordinal",
            "sample_token",
            "group_id",
            "sequence_id",
            "track_id",
            "public_track_id",
            "teacher_ordinal",
            "teacher_shard",
            "teacher_local_row",
        ],
    )
    role = pd.read_parquet(role_mask_path)
    required = {
        "rotation",
        "rotation_role",
        "fit_ordinal",
        "sample_token",
        "history_valid",
        "invalid_reason",
    }
    if (
        not required.issubset(role.columns)
        or set(role["rotation"].astype(int)) != {rotation}
        or set(role["rotation_role"].astype(str)) != {"producer"}
        or role["history_valid"].dtype != np.bool_
        or role["sample_token"].astype(str).duplicated().any()
    ):
        raise ValueError("Producer history-valid role mask differs")
    valid_mask = role["history_valid"]
    valid = role.loc[valid_mask, ["fit_ordinal", "sample_token"]]
    invalid_tokens = frozenset(role.loc[~valid_mask, "sample_token"].astype(str))
    selected = valid.merge(
        crosswalk,
        on=["fit_ordinal", "sample_token"],
        how="left",
        validate="one_to_one",
    ).sort_values("fit_ordinal", kind="stable")
    if (
        len(selected) != len(valid)
        or selected["teacher_ordinal"].isna().any()
        or selected["sample_token"].astype(str).duplicated().any()
        or any(
            str(name) != f"shard_{int(ordinal) // SHARD_ROWS:05d}.npz"
            or int(local) != int(ordinal) % SHARD_ROWS
            for name, local, ordinal in zip(
                selected["teacher_shard"],
                selected["teacher_local_row"],
                selected["teacher_ordinal"],
                strict=True,
            )
        )
    ):
        raise ValueError("Producer token-to-INPUT coordinate join differs")
    selected = selected.reset_index(drop=True)
    selected["source_shard"] = selected["teacher_shard"].astype(str)
    selected["source_row"] = selected["teacher_local_row"].astype(np.int64)
    if len(invalid_tokens) != len(role) - len(valid):
        raise ValueError("Invalid-history producer tokens are duplicated")
    return selected, invalid_tokens


def _validate_payload(stored: Mapping[str, np.ndarray], shard: SourceShard) -> None:
    keys = set(stored)
    rows = shard.stop - shard.start
    if keys != NPZ_KEYS:
        raise ValueError("ObjectEventV4 INPUT shard contains unexpected or learned fields")
    if (
        stored["ordinals"].dtype != np.int64
        or stored["ordinals"].tolist() != list(range(shard.start, shard.stop))
        or stored["events"].dtype != np.float32
        or stored["events"].shape != (rows, *EVENT_SHAPE)
        or stored["target_ttc"].dtype != np.float32
        or stored["target_ttc"].shape != (rows,)
        or stored["visible_heights"].shape != (rows, 2)
        or stored["boxes"].shape != (rows, 3, 4)
        or stored["square"].shape != (rows, 4)
        or stored["motion"].ndim != 2
        or stored["motion"].shape[0] != rows
        or stored["delta"].dtype != np.float32
        or stored["delta"].shape != (rows,)
        or stored["tokens"].shape != (rows,)
        or len(set(stored["tokens"].astype(str).tolist())) != rows
    ):
        raise ValueError("ObjectEventV4 INPUT shard dtype/shape contract differs")
    numeric = NPZ_KEYS - {"tokens", "ordinals"}
    if any(not np.isfinite(stored[name]).all() for name in numeric):
        raise ValueError("ObjectEventV4 INPUT shard contains non-finite values")
    if np.any(stored["delta"] <= 0):
        raise ValueError("ObjectEventV4 INPUT shard has non-positive measured delta")


class IndependentEventProducerSource:
    """Bounded, history-valid, producer-role view over frozen T3 inputs."""

    def __init__(
        self,
        *,
        source_root: Path,
        shards: tuple[SourceShard, ...],
        index: pd.DataFrame,
        history_invalid_tokens: frozenset[str],
        teacher_manifest_sha256: str,
        teacher_role_receipt_sha256: str,
        max_cached_shards: int = 2,
    ) -> None:
        if max_cached_shards not in (1, 2):
            raise ValueError("ObjectEventV4 source allows one or two cached shards")
        self.root = source_root.resolve(strict=True)
        self.shards = shards
        self.index = index.reset_index(drop=True)
        self.history_invalid_tokens = history_invalid_tokens
        self.teacher_manifest_sha256 = teacher_manifest_sha256
        self.teacher_role_receipt_sha256 = teacher_role_receipt_sha256
        self.population_size = len(self.index)
        self.max_cached_shards = max_cached_shards
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        groups: list[tuple[int, ...]] = []
        for _, rows in self.index.groupby(
            self.index["teacher_ordinal"].astype(np.int64) // 256, sort=True
        ):
            groups.append(tuple(int(value) for value in rows.index))
        flattened = sorted(value for group in groups for value in group)
        if flattened != list(range(self.population_size)):
            raise ValueError("Producer sampling groups do not partition the valid role")
        self.sampling_groups = tuple(groups)

    def __len__(self) -> int:
        return self.population_size

    @property
    def cached_shard_names(self) -> tuple[str, ...]:
        """Return the currently resident physical INPUT shard names."""

        return tuple(self._cache)

    def _load_shard(self, name: str) -> dict[str, np.ndarray]:
        cached = self._cache.pop(name, None)
        if cached is not None:
            self._cache[name] = cached
            return cached
        index = int(name.removeprefix("shard_").removesuffix(".npz"))
        shard = self.shards[index]
        if shard.name != name:
            raise ValueError("ObjectEventV4 source shard coordinate differs")
        path = _safe_child(self.root, name)
        receipt_path = _safe_child(self.root, name.replace(".npz", ".json"))
        before_payload = file_sha256(path)
        before_receipt = file_sha256(receipt_path)
        if before_payload != shard.sha256 or before_receipt != shard.receipt_sha256:
            raise ValueError("ObjectEventV4 source shard changed before load")
        with np.load(path, allow_pickle=False) as stored:
            payload = {key: stored[key].copy() for key in stored.files}
        if file_sha256(path) != before_payload or file_sha256(receipt_path) != before_receipt:
            raise ValueError("ObjectEventV4 source shard changed during load")
        _validate_payload(payload, shard)
        self._cache[name] = payload
        while len(self._cache) > self.max_cached_shards:
            self._cache.popitem(last=False)
        return payload

    def record(self, index: int) -> dict[str, Any]:
        """Return copied raw-derived inputs and supervised targets for one valid token."""

        if not 0 <= index < self.population_size:
            raise IndexError(index)
        identity = self.index.iloc[index]
        shard = self._load_shard(str(identity["source_shard"]))
        row = int(identity["source_row"])
        token = str(identity["sample_token"])
        if (
            str(shard["tokens"][row]) != token
            or int(shard["ordinals"][row]) != int(identity["teacher_ordinal"])
        ):
            raise ValueError("ObjectEventV4 physical token coordinate differs")
        return {
            "event_v4_common_roi": shard["events"][row].copy(),
            "_event_v4_expected_steps": 3,
            "_event_v4_expected_channels": 12,
            "event_v4_precontext_valid": True,
            "garl_delta_t_s": np.float32(shard["delta"][row]),
            "observable_motion": shard["motion"][row].copy(),
            "garl_visible_heights_px": shard["visible_heights"][row].copy(),
            "ttc_s": np.float32(shard["target_ttc"][row]),
            "event_v4_boxes_xyxy": shard["boxes"][row].copy(),
            "event_v4_common_square_xyxy": shard["square"][row].copy(),
            "sequence_id": str(identity["sequence_id"]),
            "sample_token": token,
            "track_id": str(identity["track_id"]),
            "public_track_id": str(identity["public_track_id"]),
            "source_ordinal": int(identity["teacher_ordinal"]),
        }

    def records(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        """Load one non-empty batch without exposing any out-of-role row."""

        if not indices:
            raise ValueError("Independent producer batch cannot be empty")
        values = [int(index) for index in indices]
        if len(set(values)) != len(values):
            raise ValueError("Independent producer batch indices must be unique")
        return [self.record(index) for index in values]

    def batch(self, indices: Sequence[int], teacher: object) -> ObjectEventV4Batch:
        """Build the existing ObjectEventV4Batch with exact role-scoped DINO targets.

        Torch is imported only when a real admitted fit asks for a batch.  Source
        verification and index construction remain CPU/source-only.
        """

        from .teacher_cache import TeacherRoleCache

        if not isinstance(teacher, TeacherRoleCache):
            raise TypeError("Producer requires a verified nominal TeacherRoleCache")
        if (
            teacher.role_receipt_sha256 != self.teacher_role_receipt_sha256
            or file_sha256(teacher.role_receipt_path) != self.teacher_role_receipt_sha256
            or teacher.allowed_tokens
            != (
                frozenset(self.index["sample_token"].astype(str))
                | self.history_invalid_tokens
            )
        ):
            raise ValueError("Producer teacher role identity differs")
        receipt = _json(teacher.role_receipt_path)
        if (
            receipt.get("schema") != "independent_phase_teacher_role_receipt_v1"
            or receipt.get("status") != "PASS_EXACT_ROLE_FILTER"
            or receipt.get("rotation") != 0
            or receipt.get("role") != "producer"
            or receipt.get("cache_manifest_sha256") != self.teacher_manifest_sha256
            or receipt.get("queries") != len(self) + len(self.history_invalid_tokens)
            or receipt.get("full_train24_exposed") is not False
            or receipt.get("generation_excluded") is not True
        ):
            raise ValueError("Producer teacher role receipt differs")
        records = self.records(indices)
        for record in records:
            targets, valid = teacher.get(str(record["sample_token"]))
            if (
                targets.dtype != np.float16
                or targets.shape != (2, 6, 32, 32)
                or valid.shape != targets.shape
                or valid.dtype != np.bool_
                or not np.isfinite(targets).all()
            ):
                raise ValueError("Independent DINO relation target contract differs")
            record["dinov3_relation_targets"] = targets
            record["dinov3_relation_valid"] = valid
        from e_jepa_ttc.data.object_event_v4 import collate_object_event_v4

        return collate_object_event_v4(records)


def open_event_producer_source(
    *,
    source_manifest_path: Path,
    source_root: Path,
    prepare_freeze_path: Path,
    crosswalk_receipt_path: Path,
    crosswalk_path: Path,
    h5_admission_path: Path,
    plan: Mapping[str, Any],
    teacher_manifest_path: Path,
    teacher_role_receipt_path: Path,
    rotation: int,
    max_cached_shards: int = 2,
) -> IndependentEventProducerSource:
    """Open the real source only after physical H5 admission has passed."""

    from .producer_contract import verify_producer_plan

    verify_producer_plan(plan)
    if rotation != 0 or plan.get("rotation") != 0 or plan.get("component") != "E_A5":
        raise ValueError("Independent source foundation is E_A5 rotation 0 only")
    admission = verify_h5_admission_for_fit(h5_admission_path.resolve(strict=True))
    preflight_path = Path(str(admission["preflight_path"])).resolve(strict=True)
    if file_sha256(preflight_path) != admission.get("preflight_sha256"):
        raise ValueError("H5 admission preflight binding changed")
    preflight = _json(preflight_path)
    role_ref = cast(Mapping[str, Any], preflight["role_masks"])["rotation_0_producer"]
    role_mask_path = (preflight_path.parent / str(role_ref["path"])).resolve(strict=True)
    role_mask_sha256 = str(role_ref["sha256"])
    bound_role = cast(Mapping[str, Any], plan["history_role_mask"])
    if (
        role_mask_path != Path(str(bound_role["path"])).resolve(strict=True)
        or role_mask_sha256 != bound_role.get("sha256")
        or admission.get("history_admissibility_sha256")
        != preflight.get("history_admissibility_sha256")
    ):
        raise ValueError("Physical admission and producer role mask differ")
    teacher_manifest_sha256 = file_sha256(teacher_manifest_path)
    teacher_role_sha256 = file_sha256(teacher_role_receipt_path)
    parent_sha = cast(Mapping[str, str], plan["parent_sha256"])
    if (
        teacher_manifest_sha256 != parent_sha.get("teacher_manifest")
        or teacher_role_sha256 != parent_sha.get("teacher_role")
    ):
        raise ValueError("Producer teacher parents differ from verified plan")
    shards = verify_input_manifest(source_manifest_path, source_root, prepare_freeze_path)
    selected, invalid_tokens = _load_selected_index(
        crosswalk_receipt_path=crosswalk_receipt_path,
        crosswalk_path=crosswalk_path,
        role_mask_path=role_mask_path,
        role_mask_sha256=role_mask_sha256,
        rotation=rotation,
    )
    return IndependentEventProducerSource(
        source_root=source_root,
        shards=shards,
        index=selected,
        history_invalid_tokens=invalid_tokens,
        teacher_manifest_sha256=teacher_manifest_sha256,
        teacher_role_receipt_sha256=teacher_role_sha256,
        max_cached_shards=max_cached_shards,
    )


__all__ = [
    "IndependentEventProducerSource",
    "SourceShard",
    "open_event_producer_source",
    "verify_h5_admission_for_fit",
    "verify_input_manifest",
    "verify_row_local_prepare_freeze",
]
