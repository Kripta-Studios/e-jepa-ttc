"""Source-only contract for the first independent E/R producer fits.

This module deliberately contains no PyTorch, NumPy, pandas, HDF5, or optimizer
imports.  It freezes what can be established from the historical E_A5/R_A5
sources while keeping fitting closed until physical event data and the frozen
row-local ObjectEventV4 INPUT cache have separate, verified admissions.

The independent history has 31 causal observation packets.  The historical
``CausalScaleTTC`` producer consumes a genuine T2/T3 ``ObjectEventV4Batch``.
Those are different contracts: this module never guesses a T2/T3 slice from the
31-packet history.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

SOURCE_COMMIT = "2ad7a05797799caf1bcdc9202c307d244f9fe31d"
TRAIN40_PROTOCOL_SHA256 = "39fac190a2568757511de017fb7e7d55428506382a0066804af8241d4908a66b"
RGB_RECIPE_SHA256 = "9ffb221ebe540ee5fe0013d69be07012954f008ffdd988a15acc2b2d2106f0aa"
FOLD_CONTRACT_SHA256 = "ef67d27078a978cdfb94f0f99f660a44d64079fe5e9e2c1887841e751b6d81a4"
TEACHER_CACHE_MANIFEST_SHA256 = "adadd943e54d0526b982f715ddf589da1d2539070519c8fd9bc9246cbb878408"
RAW_HISTORY_V4_SHA256 = "1a0382a22adfa3b7eed2cf411723ac2f1bb399e381ce4b80ebcca9237c690f8e"
H5_PREFLIGHT_V4_SHA256 = "c9141674d95af89ff67352a09ad5a9c25b7a84942c88c08546f66dbdb0ee4b92"
TEACHER_ROLE_ROT0_PRODUCER_SHA256 = (
    "5a14ea13f4fc4cedc8276fbbbf2588f392811ba7535328b56574d03ada5fccec"
)
PARENT_SHA256 = {
    "train40_protocol": TRAIN40_PROTOCOL_SHA256,
    "rgb_recipe": RGB_RECIPE_SHA256,
    "fold_contract": FOLD_CONTRACT_SHA256,
    "raw_history": RAW_HISTORY_V4_SHA256,
    "h5_preflight": H5_PREFLIGHT_V4_SHA256,
    "teacher_manifest": TEACHER_CACHE_MANIFEST_SHA256,
    "teacher_role": TEACHER_ROLE_ROT0_PRODUCER_SHA256,
}
SOURCE_SHA256 = {
    "src/e_jepa_ttc/models/causal_scale_ttc.py": (
        "0e5802ef7ce98cac37ad414d66ba339bbd62cffe7f9a459fc495875a6feac934"
    ),
    "src/e_jepa_ttc/data/object_event_v4.py": (
        "84f7a1895523bb0c28cfd1f13b47ae4f86d906aed93e26354796df02999587b4"
    ),
    "operational/train40_system/models.py": (
        "3d00e181805b97a24a0ff10531bfe2bb57f2f73f4d3bb861718e5d40d70bd50f"
    ),
    "operational/final_refit/producers.py": (
        "45717d9ca859461d403f938f80388dfcf3bb1ecf292b5f9beb439982397d8c14"
    ),
    "src/e_jepa_ttc/training/causal_scale_eap.py": (
        "0528dc1dd691b9f4650c7bd7c5cfdcbe9ef2a543e7f53646190da7b2fbf65185"
    ),
    "src/e_jepa_ttc/losses/causal_scale_ttc.py": (
        "df7972424c77702e08aec45d4af3cd76a6e09232670fb2ec31ce43f530d801cd"
    ),
}

PLAN_SCHEMA = "independent_phase_producer_plan_v1"
ROLE_SCHEMA = "independent_phase_teacher_role_receipt_v1"
H5_PREFLIGHT_SCHEMA = "independent_phase_event_h5_metadata_preflight_v4"
RAW_HISTORY_SCHEMA = "independent_phase_raw_history_index_v4"

OBSERVATION_PACKETS = 31
OBSERVATION_PACKET_US = 20_000
RAW_OBSERVATION_SPAN_US = 620_000
ENDPOINT_SPAN_US = 600_000
ENDPOINT_TRANSITIONS = 30

Component = Literal["E_A5", "R_A5"]
TrainingMode = Literal["supervised_producer", "representation_only_ssl"]


def canonical_sha256(value: Mapping[str, Any]) -> str:
    """Hash strict canonical JSON without importing the data stack."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file with bounded memory."""

    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SupervisionMask:
    """Declare which training-only fields may reach one objective.

    ``representation_only_ssl`` is label-free: TTC, boxes and raw GT ROI are all
    forbidden.  The historical A5 objective is the supervised producer mode;
    it must not be described as SSL merely because it also distils a frozen
    DINO relation target.
    """

    mode: TrainingMode
    ttc_seconds: bool
    collision_thresholds: bool
    bbox_geometry: bool
    raw_gt_roi_for_preparation: bool
    teacher_relation_targets: bool

    def validate(self) -> None:
        if self.mode not in ("supervised_producer", "representation_only_ssl"):
            raise ValueError("Unknown producer supervision mode")
        if self.mode == "representation_only_ssl" and any(
            (
                self.ttc_seconds,
                self.collision_thresholds,
                self.bbox_geometry,
                self.raw_gt_roi_for_preparation,
            )
        ):
            raise ValueError("SSL supervision mask exposes a TTC label or GT geometry")
        if self.mode == "supervised_producer" and not all(
            (
                self.ttc_seconds,
                self.collision_thresholds,
                self.bbox_geometry,
                self.raw_gt_roi_for_preparation,
                self.teacher_relation_targets,
            )
        ):
            raise ValueError("Historical A5 supervised objective mask is incomplete")


@dataclass(frozen=True)
class TeacherPublicAttributes:
    """Only the frozen teacher attributes exposed to a producer role."""

    target_field: str = "relation_targets"
    target_dtype: str = "float16"
    target_shape_per_query: tuple[int, ...] = (2, 6, 32, 32)
    identity_join: str = "unique_sample_token"
    model_input: bool = False
    task_teacher: bool = False
    ttc_values_read: bool = False
    source_pixels_read_by_fit: bool = False

    def validate(self) -> None:
        if (
            self.target_field != "relation_targets"
            or self.target_dtype != "float16"
            or self.target_shape_per_query != (2, 6, 32, 32)
            or self.identity_join != "unique_sample_token"
            or self.model_input
            or self.task_teacher
            or self.ttc_values_read
            or self.source_pixels_read_by_fit
        ):
            raise ValueError("Teacher public-attribute contract differs")


@dataclass(frozen=True)
class TemporalInputContract:
    """Bind the complete history without pretending it is a producer batch."""

    observation_packets: int = OBSERVATION_PACKETS
    packet_width_us: int = OBSERVATION_PACKET_US
    raw_observation_span_us: int = RAW_OBSERVATION_SPAN_US
    endpoint_span_us: int = ENDPOINT_SPAN_US
    endpoint_transitions: int = ENDPOINT_TRANSITIONS
    observation_intervals: str = "31 contiguous half-open [start,end) packets"
    producer_tensor_contract: str = "genuine ObjectEventV4 T2-or-T3, C=12 event / C=3 RGB"
    history_to_producer_projection: str = "NONE_NATIVE_T3_INPUT_CACHE_JOINED_BY_SAMPLE_TOKEN"

    def validate(self) -> None:
        if asdict(self) != {
            "observation_packets": 31,
            "packet_width_us": 20_000,
            "raw_observation_span_us": 620_000,
            "endpoint_span_us": 600_000,
            "endpoint_transitions": 30,
            "observation_intervals": "31 contiguous half-open [start,end) packets",
            "producer_tensor_contract": ("genuine ObjectEventV4 T2-or-T3, C=12 event / C=3 RGB"),
            "history_to_producer_projection": "NONE_NATIVE_T3_INPUT_CACHE_JOINED_BY_SAMPLE_TOKEN",
        }:
            raise ValueError("Independent temporal input contract differs")


@dataclass(frozen=True)
class ProducerRecipe:
    """Resolved fresh producer recipe; learned TRAIN40 weights are forbidden."""

    component: Component
    modality: Literal["event", "rgb"]
    source_arm: Literal["a5"]
    source_class: str
    in_channels: int
    transport_mode: Literal["legacy"]
    fresh_initialization: bool
    learned_parent_checkpoints: tuple[str, ...]
    seed: int
    epochs: int
    batch_size: int
    optimizer: str
    learning_rate: float
    minimum_learning_rate: float
    weight_decay: float
    scheduler: str
    grad_clip_norm: float
    precision: str
    fixed_last_epoch_selection: bool
    foreground_warmup_epochs: int
    online_augmentation: str
    sampler: str
    model_config: Mapping[str, Any]
    loss_config: Mapping[str, Any]
    training_config: Mapping[str, Any]

    def validate(self) -> None:
        expected_modality = "event" if self.component == "E_A5" else "rgb"
        expected_channels = 12 if self.component == "E_A5" else 3
        if (
            self.modality != expected_modality
            or self.in_channels != expected_channels
            or self.source_arm != "a5"
            or self.source_class != "e_jepa_ttc.models.causal_scale_ttc.CausalScaleTTC"
            or self.transport_mode != "legacy"
            or not self.fresh_initialization
            or self.learned_parent_checkpoints
            or self.seed != 7
            or self.epochs != 18
            or self.batch_size != 32
            or self.optimizer != "AdamW"
            or self.learning_rate != 3e-4
            or self.minimum_learning_rate != 3e-5
            or self.weight_decay != 1e-4
            or self.scheduler != "CosineAnnealingLR"
            or self.grad_clip_norm != 1.0
            or self.precision != "bf16"
            or not self.fixed_last_epoch_selection
            or self.foreground_warmup_epochs != 3
            or self.online_augmentation != "none_in_historical_runner"
            or self.sampler != "shard_grouped_random_v1"
        ):
            raise ValueError("Fresh A5 producer recipe differs")
        _validate_model_config(self.model_config, self.modality, expected_channels)
        _validate_loss_config(self.loss_config)
        _validate_training_config(self.training_config)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def _self_hash(value: Mapping[str, Any]) -> None:
    unsigned = dict(value)
    claimed = unsigned.pop("plan_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise ValueError("Producer plan canonical identity changed")


def _validate_model_config(value: Mapping[str, Any], modality: str, channels: int) -> None:
    expected = {
        "modality": modality,
        "in_channels": channels,
        "hidden_dim": 64,
        "geometry_dim": 128,
        "residual_depth": 2,
        "dropout": 0.05,
        "foreground_decoder": "resize_conv",
        "foreground_fullres_dim": 24,
        "foreground_temporal_smoothing": 0.15,
        "foreground_temporal_smoothing_mode": "causal_left",
        "transport_enabled": True,
        "transport_mode": "legacy",
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("Historical A5 model configuration differs")


def _validate_loss_config(value: Mapping[str, Any]) -> None:
    expected = {
        "log_ratio_nll_weight": 1.0,
        "log_ratio_tail_weight": 2.0,
        "log_ratio_tail_fraction": 0.1,
        "foreground_extent_weight": 1.25,
        "foreground_width_weight": 1.25,
        "foreground_center_weight": 2.5,
        "risk_weight": 0.1,
        "auxiliary_inverse_ttc_weight": 0.05,
        "residual_regularization_weight": 0.1,
        "smooth_l1_beta": 0.02,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("Historical A5 loss configuration differs")


def _validate_training_config(value: Mapping[str, Any]) -> None:
    expected = {
        "seed": 7,
        "epochs": 18,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "minimum_learning_rate": 3e-5,
        "weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "precision": "bf16",
        "foreground_supervision": "bbox_geometry",
        "foreground_warmup_epochs": 3,
        "representation_supervision": "dinov3_local_relational",
        "representation_distillation_weight": 8.0,
        "initialization_mode": "none",
        "initialization_checkpoint": None,
        "checkpoint_selection_mode": "last_epoch",
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("Historical A5 training configuration differs")


def _event_recipe(protocol: Mapping[str, Any]) -> ProducerRecipe:
    arm = cast(Mapping[str, Any], cast(Mapping[str, Any], protocol["producers"])["a5"])
    training = cast(Mapping[str, Any], arm["training_config"])
    recipe = ProducerRecipe(
        component="E_A5",
        modality="event",
        source_arm="a5",
        source_class="e_jepa_ttc.models.causal_scale_ttc.CausalScaleTTC",
        in_channels=12,
        transport_mode="legacy",
        fresh_initialization=True,
        learned_parent_checkpoints=(),
        seed=7,
        epochs=18,
        batch_size=32,
        optimizer="AdamW",
        learning_rate=3e-4,
        minimum_learning_rate=3e-5,
        weight_decay=1e-4,
        scheduler="CosineAnnealingLR",
        grad_clip_norm=1.0,
        precision="bf16",
        fixed_last_epoch_selection=True,
        foreground_warmup_epochs=3,
        online_augmentation="none_in_historical_runner",
        sampler="shard_grouped_random_v1",
        model_config=cast(Mapping[str, Any], arm["model_config"]),
        loss_config=cast(Mapping[str, Any], arm["loss_config"]),
        training_config=training,
    )
    if (
        arm.get("source_weights_reused_as_initialization") is not False
        or arm.get("updates_fixed") != 49_932
        or arm.get("checkpoint_selection") != "fixed_epoch18"
        or arm.get("early_stopping") != "disabled_no_holdout_selection"
    ):
        raise ValueError("Historical TRAIN40 A5 execution contract differs")
    recipe.validate()
    return recipe


def _rgb_recipe(config: Mapping[str, Any]) -> ProducerRecipe:
    arm = cast(Mapping[str, Any], cast(Mapping[str, Any], config["producers"])["R_A5"])
    initialization = cast(Mapping[str, Any], arm["initialization"])
    model = dict(cast(Mapping[str, Any], arm["model_config"]))
    model["deterministic_transport_sampling"] = True
    training = dict(cast(Mapping[str, Any], arm["training_config"]))
    training["deterministic_transport_sampling"] = "ENABLED_FINAL_REFIT_PROSPECTIVE"
    if initialization != {
        "event_checkpoint_loaded": False,
        "kernel_reshape_or_repeat": False,
        "mode": "random",
        "seed": 7,
    }:
        raise ValueError("Historical R_A5 initialization contract differs")
    recipe = ProducerRecipe(
        component="R_A5",
        modality="rgb",
        source_arm="a5",
        source_class="e_jepa_ttc.models.causal_scale_ttc.CausalScaleTTC",
        in_channels=3,
        transport_mode="legacy",
        fresh_initialization=True,
        learned_parent_checkpoints=(),
        seed=7,
        epochs=18,
        batch_size=32,
        optimizer="AdamW",
        learning_rate=3e-4,
        minimum_learning_rate=3e-5,
        weight_decay=1e-4,
        scheduler="CosineAnnealingLR",
        grad_clip_norm=1.0,
        precision="bf16",
        fixed_last_epoch_selection=True,
        foreground_warmup_epochs=3,
        online_augmentation="none_in_historical_runner",
        sampler="shard_grouped_random_v1",
        model_config=model,
        loss_config=cast(Mapping[str, Any], arm["loss_config"]),
        training_config=training,
    )
    recipe.validate()
    return recipe


def _validate_role_receipt(
    value: Mapping[str, Any], *, rotation: int, teacher_manifest_sha256: str
) -> int:
    queries = value.get("queries")
    groups = value.get("group_ids")
    if (
        value.get("schema") != ROLE_SCHEMA
        or value.get("status") != "PASS_EXACT_ROLE_FILTER"
        or value.get("rotation") != rotation
        or value.get("role") != "producer"
        or value.get("cache_manifest_sha256") != teacher_manifest_sha256
        or value.get("fold_contract_sha256") != FOLD_CONTRACT_SHA256
        or value.get("full_train24_exposed") is not False
        or value.get("generation_excluded") is not True
        or not isinstance(queries, int)
        or queries <= 0
        or not isinstance(groups, list)
        or len(groups) != 16
        or len(set(map(str, groups))) != 16
    ):
        raise ValueError("Producer teacher-role receipt differs")
    return queries


def build_producer_plan(
    *,
    component: Component,
    rotation: int,
    train40_protocol_path: Path,
    rgb_recipe_path: Path,
    fold_contract_path: Path,
    raw_history_receipt_path: Path,
    h5_preflight_path: Path,
    teacher_manifest_path: Path,
    teacher_role_receipt_path: Path,
) -> dict[str, Any]:
    """Build a fail-closed plan without loading tensors, H5 payloads, or labels."""

    if component not in {"E_A5", "R_A5"} or rotation != 0:
        raise ValueError("This producer foundation supports rotation-0 E_A5/R_A5 only")
    paths = {
        "train40_protocol": train40_protocol_path.resolve(strict=True),
        "rgb_recipe": rgb_recipe_path.resolve(strict=True),
        "fold_contract": fold_contract_path.resolve(strict=True),
        "raw_history": raw_history_receipt_path.resolve(strict=True),
        "h5_preflight": h5_preflight_path.resolve(strict=True),
        "teacher_manifest": teacher_manifest_path.resolve(strict=True),
        "teacher_role": teacher_role_receipt_path.resolve(strict=True),
    }
    observed = {key: file_sha256(path) for key, path in paths.items()}
    if observed != PARENT_SHA256:
        raise ValueError("A frozen producer-plan parent changed")

    protocol = _json(paths["train40_protocol"])
    rgb_config = _json(paths["rgb_recipe"])
    fold = _json(paths["fold_contract"])
    raw = _json(paths["raw_history"])
    h5 = _json(paths["h5_preflight"])
    teacher = _json(paths["teacher_manifest"])
    role = _json(paths["teacher_role"])
    if (
        fold.get("schema") != "independent_phase_oof_fold_contract_v2"
        or fold.get("status") != "PASS_NOT_TRAINED"
        or fold.get("population") != {"groups": 24, "queries": 53_047}
        or fold.get("training_started") is not False
    ):
        raise ValueError("Independent fold contract differs")
    if (
        raw.get("schema") != RAW_HISTORY_SCHEMA
        or raw.get("packet_endpoints") != 31
        or raw.get("observation_packets") != 31
        or raw.get("packet_intervals") != 30
        or raw.get("raw_observation_span_us") != 620_000
        or raw.get("training_admitted") is not False
    ):
        raise ValueError("Raw-history V4 contract differs")
    if (
        h5.get("schema") != H5_PREFLIGHT_SCHEMA
        or h5.get("status") != "PASS_WITH_INVALID_HISTORY_ROWS_EXCLUDED_NOT_HASHED"
        or h5.get("admissible_queries") != 52_950
        or h5.get("blocked_queries") != 97
        or h5.get("full_hashes_computed") is not False
        or h5.get("training_admitted") is not False
    ):
        raise ValueError("Physical H5 preflight is no longer the expected closed gate")
    if (
        teacher.get("schema") != "independent_phase_train24_teacher_cache_v2"
        or teacher.get("status") != "COMPLETE_VERIFIED_CONVERSION"
        or teacher.get("rows") != 53_047
        or teacher.get("ttc_values_read") is not False
        or teacher.get("sensor_payloads_read") is not False
    ):
        raise ValueError("Independent teacher cache differs")

    queries = _validate_role_receipt(
        role, rotation=rotation, teacher_manifest_sha256=observed["teacher_manifest"]
    )
    role_mask_key = f"rotation_{rotation}_producer"
    role_mask = cast(Mapping[str, Any], cast(Mapping[str, Any], h5["role_masks"])[role_mask_key])
    valid_queries = role_mask.get("valid_rows")
    invalid_queries = role_mask.get("invalid_rows")
    role_mask_path = (paths["h5_preflight"].parent / str(role_mask.get("path"))).resolve()
    if (
        role_mask.get("rows") != queries
        or not isinstance(valid_queries, int)
        or not isinstance(invalid_queries, int)
        or valid_queries + invalid_queries != queries
        or invalid_queries <= 0
        or file_sha256(role_mask_path) != role_mask.get("sha256")
    ):
        raise ValueError("Producer history-valid role mask differs")
    recipe = _event_recipe(protocol) if component == "E_A5" else _rgb_recipe(rgb_config)
    supervision = SupervisionMask(
        mode="supervised_producer",
        ttc_seconds=True,
        collision_thresholds=True,
        bbox_geometry=True,
        raw_gt_roi_for_preparation=True,
        teacher_relation_targets=True,
    )
    teacher_public = TeacherPublicAttributes()
    temporal = TemporalInputContract()
    supervision.validate()
    teacher_public.validate()
    temporal.validate()
    unsigned: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "status": "BLOCKED_PENDING_H5_HASH_AND_INPUT_CACHE_BINDING",
        "training_admitted": False,
        "optimizer_updates": 0,
        "source_commit": SOURCE_COMMIT,
        "component": component,
        "first_executable_component": component == "E_A5",
        "first_component_decision": (
            "E_A5 is the first leaf: it has no learned parent and uses the frozen role-scoped "
            "teacher only as an auxiliary target; R_A5 remains a separate fresh RGB leaf."
        ),
        "rotation": rotation,
        "role": "producer",
        "producer_groups": 16,
        "producer_queries_before_history_mask": queries,
        "producer_queries": valid_queries,
        "history_invalid_queries": invalid_queries,
        "drop_last": False,
        "dropped_queries_per_epoch": 0,
        "final_batch_size": valid_queries % 32 or 32,
        "updates_per_epoch": math.ceil(valid_queries / 32),
        "fixed_updates": 18 * math.ceil(valid_queries / 32),
        "recipe": asdict(recipe),
        "supervision": asdict(supervision),
        "teacher_public_attributes": asdict(teacher_public),
        "temporal_input": asdict(temporal),
        "inference_fields": ["sensor_tensor", "delta_t_s", "observable_motion"],
        "training_only_fields": [
            "ttc_seconds",
            "collision_thresholds",
            "bbox_geometry",
            "raw_gt_roi",
            "relation_targets",
        ],
        "ssl_policy": (
            "Any future representation-only SSL stage must use representation_only_ssl and "
            "must not expose TTC, collision, bbox geometry, or raw GT ROI."
        ),
        "rng": {
            "model_and_sampler_seed": 7,
            "sampler": "shard_grouped_random_v1",
            "shuffle_groups": True,
            "shuffle_rows_within_group": True,
            "resume_requires": [
                "model",
                "optimizer",
                "scheduler",
                "torch_rng",
                "cuda_rng",
                "sampler_generator",
                "epoch",
                "position",
                "ordered_role_identity",
            ],
        },
        "normalization": (
            {
                "learned_normalizers": "FRESH_PRODUCER_ROLE_ONLY",
                "historical_train40_normalizers_allowed": False,
                "fit_statistics_not_yet_computed": True,
            }
            if component == "E_A5"
            else {
                "pixel_conversion": "raw_uint8_to_float32_div255_exactly_once",
                "learned_normalizers": "NONE",
                "historical_train40_normalizers_allowed": False,
                "fit_statistics_not_yet_computed": False,
            }
        ),
        "historical_commands": {
            "event_train40_reference": (
                "python -m operational.train40_system.models --output <run> --arm a5"
            ),
            "rgb_reference": (
                "python -m operational.final_refit.producers --source <descriptor> "
                "--teacher-manifest <manifest> --config configs/rgb_port/producers.json "
                "--run <run> --recipe <recipe> --fit-id R_A5 --device cuda"
            ),
            "independent_launch": "NOT_IMPLEMENTED_OR_AUTHORIZED",
        },
        "blockers": (
            [
                "physical event-H5 full hashes/admission absent",
                "97 prefix-history rows remain excluded, never silently imputed",
                "frozen ObjectEventV4 INPUT cache binding has not passed fit preflight",
                "fresh producer normalizer/statistics receipt absent",
                "root GPU allocation absent",
            ]
            if component == "E_A5"
            else [
                "independent RGB source descriptor absent",
                "fresh P cache binding absent",
                "H/V raw fallback first-batch parity absent",
                "97 prefix-history rows remain excluded, never silently imputed",
                "root GPU allocation absent",
            ]
        ),
        "parent_sha256": observed,
        "parent_paths": {key: str(path) for key, path in paths.items()},
        "executed_source_sha256": dict(SOURCE_SHA256),
        "history_role_mask": {
            "path": str(role_mask_path),
            "sha256": str(role_mask["sha256"]),
            "rows": queries,
            "valid_rows": valid_queries,
            "invalid_rows": invalid_queries,
        },
    }
    unsigned["plan_sha256"] = canonical_sha256(unsigned)
    verify_producer_plan(unsigned)
    return unsigned


def verify_producer_plan(value: Mapping[str, Any]) -> None:
    """Verify a prospective plan without admitting training."""

    _self_hash(value)
    if (
        value.get("schema") != PLAN_SCHEMA
        or value.get("status") != "BLOCKED_PENDING_H5_HASH_AND_INPUT_CACHE_BINDING"
        or value.get("training_admitted") is not False
        or value.get("optimizer_updates") != 0
        or value.get("source_commit") != SOURCE_COMMIT
        or value.get("role") != "producer"
        or value.get("producer_groups") != 16
        or value.get("component") not in {"E_A5", "R_A5"}
        or value.get("rotation") != 0
        or value.get("drop_last") is not False
        or value.get("dropped_queries_per_epoch") != 0
        or value.get("final_batch_size") != (int(value.get("producer_queries", 0)) % 32 or 32)
        or value.get("updates_per_epoch") != math.ceil(int(value.get("producer_queries", 0)) / 32)
        or value.get("fixed_updates") != 18 * math.ceil(int(value.get("producer_queries", 0)) / 32)
    ):
        raise ValueError("Independent producer plan state differs")
    component = value.get("component")
    if component not in {"E_A5", "R_A5"}:
        raise ValueError("Independent producer component differs")
    recipe_dict = dict(cast(Mapping[str, Any], value.get("recipe")))
    recipe_dict["learned_parent_checkpoints"] = tuple(
        recipe_dict.get("learned_parent_checkpoints", ())
    )
    ProducerRecipe(**recipe_dict).validate()
    SupervisionMask(**cast(dict[str, Any], value["supervision"])).validate()
    teacher = dict(cast(Mapping[str, Any], value["teacher_public_attributes"]))
    teacher["target_shape_per_query"] = tuple(teacher["target_shape_per_query"])
    TeacherPublicAttributes(**teacher).validate()
    TemporalInputContract(**cast(dict[str, Any], value["temporal_input"])).validate()
    expected_blockers = (
        {
            "physical event-H5 full hashes/admission absent",
            "97 prefix-history rows remain excluded, never silently imputed",
            "frozen ObjectEventV4 INPUT cache binding has not passed fit preflight",
            "fresh producer normalizer/statistics receipt absent",
            "root GPU allocation absent",
        }
        if component == "E_A5"
        else {
            "independent RGB source descriptor absent",
            "fresh P cache binding absent",
            "H/V raw fallback first-batch parity absent",
            "97 prefix-history rows remain excluded, never silently imputed",
            "root GPU allocation absent",
        }
    )
    if set(value.get("blockers", ())) != expected_blockers:
        raise ValueError("Independent producer blockers differ")
    normalization = value.get("normalization")
    expected_normalization = (
        {
            "learned_normalizers": "FRESH_PRODUCER_ROLE_ONLY",
            "historical_train40_normalizers_allowed": False,
            "fit_statistics_not_yet_computed": True,
        }
        if component == "E_A5"
        else {
            "pixel_conversion": "raw_uint8_to_float32_div255_exactly_once",
            "learned_normalizers": "NONE",
            "historical_train40_normalizers_allowed": False,
            "fit_statistics_not_yet_computed": False,
        }
    )
    if normalization != expected_normalization:
        raise ValueError("Independent producer normalization differs")
    parents = cast(Mapping[str, str], value["parent_sha256"])
    parent_paths = cast(Mapping[str, str], value["parent_paths"])
    if dict(parents) != PARENT_SHA256 or set(parent_paths) != set(PARENT_SHA256):
        raise ValueError("Producer plan parent set differs")
    for name, path in parent_paths.items():
        if file_sha256(Path(path)) != PARENT_SHA256[name]:
            raise ValueError(f"Producer plan parent changed: {name}")
    root = Path(__file__).resolve().parents[2]
    if value.get("executed_source_sha256") != SOURCE_SHA256 or any(
        file_sha256(root / name) != digest for name, digest in SOURCE_SHA256.items()
    ):
        raise ValueError("Producer executed source identity changed")
    role = _json(Path(parent_paths["teacher_role"]))
    _validate_role_receipt(
        role,
        rotation=0,
        teacher_manifest_sha256=TEACHER_CACHE_MANIFEST_SHA256,
    )
    role_mask = cast(Mapping[str, Any], value["history_role_mask"])
    if (
        file_sha256(Path(str(role_mask["path"]))) != role_mask.get("sha256")
        or role_mask.get("rows") != value.get("producer_queries_before_history_mask")
        or role_mask.get("valid_rows") != value.get("producer_queries")
        or role_mask.get("invalid_rows") != value.get("history_invalid_queries")
    ):
        raise ValueError("Producer history-valid mask binding changed")


def write_producer_plan(path: Path, value: Mapping[str, Any]) -> None:
    """Write a new prospective plan; existing evidence is never overwritten."""

    verify_producer_plan(value)
    destination = path.resolve()
    if destination.exists():
        raise FileExistsError(f"Producer plan already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    destination.write_text(payload, encoding="utf-8", newline="\n")
    if hashlib.sha256(destination.read_bytes()).hexdigest() != file_sha256(destination):
        raise AssertionError("Producer plan write verification failed")


__all__ = [
    "ProducerRecipe",
    "SupervisionMask",
    "TeacherPublicAttributes",
    "TemporalInputContract",
    "build_producer_plan",
    "verify_producer_plan",
    "write_producer_plan",
]
