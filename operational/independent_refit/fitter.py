"""Fail-closed independent TRAIN24 fitter foundation.

Only an engineering smoke checkpoint is currently admissible.  It instantiates
the exact historical source class with explicit engineering-default config on
CPU and performs a synthetic forward pass. It never opens sensor H5, fits a
normalizer, reads TTC, or applies an optimizer update. Real fitting remains
blocked by Raw History V4.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from torch import nn

    from e_jepa_ttc.models.causal_scale_ttc import (
        CausalScaleTTCConfig as CausalScaleTTCConfigType,
    )

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

np: Any = importlib.import_module("numpy")
torch: Any = importlib.import_module("torch")
_causal_scale: Any = importlib.import_module("e_jepa_ttc.models.causal_scale_ttc")
CausalScaleTTC: Any = _causal_scale.CausalScaleTTC
CausalScaleTTCConfig: Any = _causal_scale.CausalScaleTTCConfig
_population: Any = importlib.import_module("operational.independent_refit.population")
canonical_sha256: Any = _population.canonical_sha256
file_sha256: Any = _population.file_sha256
identity_sha256: Any = _population.identity_sha256

HISTORICAL_COMMIT = "2ad7a05797799caf1bcdc9202c307d244f9fe31d"
MODEL_SOURCE_SHA256 = "0e5802ef7ce98cac37ad414d66ba339bbd62cffe7f9a459fc495875a6feac934"
DATA_SOURCE_SHA256 = "84f7a1895523bb0c28cfd1f13b47ae4f86d906aed93e26354796df02999587b4"
CHECKPOINT_SCHEMA = "independent_phase_backbone_engineering_smoke_v2"
OBSERVATIONS = 31
PACKET_US = 20_000
RAW_SPAN_US = 620_000
ENDPOINT_SPAN_US = 600_000
TRANSITIONS = 30


@dataclass(frozen=True)
class FoundationIdentity:
    """Externally derived immutable identity bound inside a checkpoint."""

    rotation: int
    train_role: str
    train_group_ids: tuple[str, ...]
    train_token_set_sha256: str
    fold_contract_sha256: str
    raw_history_receipt_sha256: str
    teacher_cache_manifest_sha256: str
    teacher_role_receipt_sha256: str
    source_population_sha256: str

    def __post_init__(self) -> None:
        if self.rotation not in range(6) or self.train_role != "producer":
            raise ValueError("Initial independent backbone is producer-only")
        if len(self.train_group_ids) != 16 or tuple(sorted(set(self.train_group_ids))) != (
            self.train_group_ids
        ):
            raise ValueError("Producer identity must contain exactly 16 sorted groups")
        for name, value in asdict(self).items():
            if name.endswith("sha256") and (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True)
class EngineeringSmokeConfig:
    """Exact CPU-only construction parameters; no training knobs are exposed."""

    modality: Literal["event", "rgb"]
    seed: int = 7
    batch_size: int = 2
    steps: int = 3
    height: int = 32
    width: int = 32

    @property
    def channels(self) -> int:
        return 12 if self.modality == "event" else 3

    def __post_init__(self) -> None:
        if self.modality not in {"event", "rgb"}:
            raise ValueError("Smoke modality must be event or rgb")
        if self.seed < 0 or self.batch_size < 1:
            raise ValueError("Smoke seed/batch size differ")
        if self.steps != 3 or min(self.height, self.width) < 16:
            raise ValueError("Historical producer smoke requires T=3 and spatial size >=16")


def _source_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2]
    return (
        root / "src/e_jepa_ttc/models/causal_scale_ttc.py",
        root / "src/e_jepa_ttc/data/object_event_v4.py",
    )


def verify_historical_sources() -> dict[str, str]:
    """Require byte-exact sources from the inspected historical commit."""
    model, data = _source_paths()
    observed = {"model": file_sha256(model), "data": file_sha256(data)}
    if observed != {"model": MODEL_SOURCE_SHA256, "data": DATA_SOURCE_SHA256}:
        raise ValueError("Historical E/R architecture or input contract changed")
    return observed


def bind_foundation_identity(
    *,
    fold_contract_path: Path,
    raw_history_receipt_path: Path,
    teacher_cache_manifest_path: Path,
    teacher_cache_manifest_sha256: str,
    teacher_role_receipt_path: Path,
    rotation: int,
) -> FoundationIdentity:
    """Derive producer identity from verified fold/history/teacher artifacts."""
    raw_history: Any = importlib.import_module("operational.independent_refit.raw_history")
    lineage: Any = importlib.import_module("operational.independent_refit.lineage")
    teacher_cache: Any = importlib.import_module("operational.independent_refit.teacher_cache")
    raw_history.verify_raw_history_index(raw_history_receipt_path)
    raw_receipt = json.loads(raw_history_receipt_path.read_text(encoding="utf-8"))
    if (
        raw_receipt.get("observation_packets") != OBSERVATIONS
        or raw_receipt.get("raw_observation_span_us") != RAW_SPAN_US
        or raw_receipt.get("packet_endpoints") != OBSERVATIONS
        or raw_receipt.get("endpoint_span_us") != ENDPOINT_SPAN_US
        or raw_receipt.get("packet_intervals") != TRANSITIONS
        or raw_receipt.get("step_us") != PACKET_US
        or raw_receipt.get("training_admitted") is not False
        or raw_receipt.get("stream_bounds_checked") is not False
    ):
        raise ValueError("Raw History V4 temporal/blocking contract differs")
    rows = lineage.load_rotation_role(
        fold_contract_path, rotation=rotation, role="producer"
    )
    cache = teacher_cache.load_teacher_role(
        teacher_cache_manifest_path,
        expected_manifest_sha256=teacher_cache_manifest_sha256,
        fold_contract_path=fold_contract_path,
        rotation=rotation,
        role="producer",
        role_receipt_path=teacher_role_receipt_path,
    )
    tokens = rows["sample_token"].astype(str).tolist()
    if cache.allowed_tokens != frozenset(tokens):
        raise ValueError("Teacher producer role differs from the fold producer identity")
    groups = tuple(sorted(set(rows["group_id"].astype(str).tolist())))
    population_path = fold_contract_path.resolve(strict=True).parent / "population.parquet"
    return FoundationIdentity(
        rotation=rotation,
        train_role="producer",
        train_group_ids=groups,
        train_token_set_sha256=identity_sha256(tokens, ordered=False),
        fold_contract_sha256=file_sha256(fold_contract_path),
        raw_history_receipt_sha256=file_sha256(raw_history_receipt_path),
        teacher_cache_manifest_sha256=file_sha256(teacher_cache_manifest_path),
        teacher_role_receipt_sha256=file_sha256(teacher_role_receipt_path),
        source_population_sha256=file_sha256(population_path),
    )


def state_tensor_sha256(state: Mapping[str, Any]) -> str:
    """Hash ordered tensor names, dtypes, shapes, and exact CPU bytes."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        header = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _model_config(config: EngineeringSmokeConfig) -> CausalScaleTTCConfigType:
    return CausalScaleTTCConfig(modality=config.modality, in_channels=config.channels)


def _seeded_initial_model(
    config: EngineeringSmokeConfig,
) -> tuple[Any, CausalScaleTTCConfigType]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    model_config = _model_config(config)
    return CausalScaleTTC(model_config).cpu().eval(), model_config


def _synthetic_smoke(model: nn.Module, config: EngineeringSmokeConfig) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    inputs = torch.randn(
        (
            config.batch_size,
            config.steps,
            config.channels,
            config.height,
            config.width,
        ),
        generator=generator,
    )
    if config.modality == "rgb":
        inputs = torch.sigmoid(inputs)
    delta_t_s = torch.full(
        (config.batch_size, config.steps - 1), PACKET_US / 1_000_000.0
    )
    with torch.inference_mode():
        output = model(inputs, delta_t_s)
    finite = bool(
        torch.isfinite(output.ttc_mean_seconds).all()
        and torch.isfinite(output.foreground_logits).all()
    )
    if not finite:
        raise FloatingPointError("Historical producer engineering smoke is non-finite")
    return {
        "ttc_shape": list(output.ttc_mean_seconds.shape),
        "foreground_shape": list(output.foreground_logits.shape),
        "finite": finite,
        "output_tensor_sha256": state_tensor_sha256(
            {
                "ttc_mean_seconds": output.ttc_mean_seconds,
                "foreground_logits": output.foreground_logits,
            }
        ),
    }


def write_engineering_smoke_checkpoint(
    path: Path,
    *,
    config: EngineeringSmokeConfig,
    identity: FoundationIdentity,
) -> dict[str, Any]:
    """Construct the exact E/R backbone and write a self-verifying CPU smoke."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite engineering smoke: {path}")
    sources = verify_historical_sources()
    model, model_config = _seeded_initial_model(config)
    smoke = _synthetic_smoke(model, config)
    state = model.state_dict()
    payload: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "status": "ENGINEERING_SMOKE_NO_TRAINING",
        "kind": "independent_phase_backbone_foundation",
        "historical_commit": HISTORICAL_COMMIT,
        "executed_source_sha256": sources,
        "config": asdict(config),
        "config_sha256": canonical_sha256(asdict(config)),
        "model_config": asdict(model_config),
        "model_config_sha256": canonical_sha256(asdict(model_config)),
        "foundation_identity": asdict(identity),
        "foundation_identity_sha256": canonical_sha256(asdict(identity)),
        "state_tensor_sha256": state_tensor_sha256(state),
        "temporal_contract": {
            "observation_packets": OBSERVATIONS,
            "packet_interval_us": PACKET_US,
            "raw_observation_span_us": RAW_SPAN_US,
            "packet_endpoints": OBSERVATIONS,
            "endpoint_span_us": ENDPOINT_SPAN_US,
            "transitions": TRANSITIONS,
            "interval_semantics": "HALF_OPEN_[START,END)",
        },
        "normalizer_status": "NOT_FIT_RAW_H5_ADMISSION_BLOCKED",
        "raw_sensor_payloads_read": False,
        "ttc_values_read": False,
        "teacher_features_read": False,
        "optimizer_updates": 0,
        "gpu_used": False,
        "synthetic_engineering_input": True,
        "smoke": smoke,
    }
    payload["checkpoint_identity_sha256"] = canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save({"identity": payload, "model_state": state}, temporary)
    os.replace(temporary, path)
    return verify_engineering_smoke_checkpoint(path)


def verify_engineering_smoke_checkpoint(path: Path) -> dict[str, Any]:
    """Verify identity from checkpoint internals and recomputed tensor bytes."""
    value = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or set(value) != {"identity", "model_state"}:
        raise ValueError("Engineering checkpoint top-level schema differs")
    identity, state = value["identity"], value["model_state"]
    if not isinstance(identity, dict) or not isinstance(state, dict):
        raise ValueError("Engineering checkpoint internals differ")
    claimed = identity.get("checkpoint_identity_sha256")
    payload = dict(identity)
    payload.pop("checkpoint_identity_sha256", None)
    foundation = FoundationIdentity(**identity["foundation_identity"])
    config = EngineeringSmokeConfig(**identity["config"])
    model_config = CausalScaleTTCConfig(**identity["model_config"])
    temporal_contract = {
        "observation_packets": OBSERVATIONS,
        "packet_interval_us": PACKET_US,
        "raw_observation_span_us": RAW_SPAN_US,
        "packet_endpoints": OBSERVATIONS,
        "endpoint_span_us": ENDPOINT_SPAN_US,
        "transitions": TRANSITIONS,
        "interval_semantics": "HALF_OPEN_[START,END)",
    }
    expected_model, expected_model_config = _seeded_initial_model(config)
    expected_state = expected_model.state_dict()
    expected_smoke = _synthetic_smoke(expected_model, config)
    if (
        identity.get("schema") != CHECKPOINT_SCHEMA
        or identity.get("status") != "ENGINEERING_SMOKE_NO_TRAINING"
        or identity.get("historical_commit") != HISTORICAL_COMMIT
        or identity.get("executed_source_sha256") != verify_historical_sources()
        or identity.get("config_sha256") != canonical_sha256(asdict(config))
        or identity.get("model_config_sha256") != canonical_sha256(asdict(model_config))
        or asdict(model_config) != asdict(expected_model_config)
        or identity.get("foundation_identity_sha256")
        != canonical_sha256(asdict(foundation))
        or claimed != canonical_sha256(payload)
        or identity.get("state_tensor_sha256") != state_tensor_sha256(state)
        or identity.get("state_tensor_sha256") != state_tensor_sha256(expected_state)
        or identity.get("temporal_contract") != temporal_contract
        or identity.get("normalizer_status") != "NOT_FIT_RAW_H5_ADMISSION_BLOCKED"
        or identity.get("raw_sensor_payloads_read") is not False
        or identity.get("ttc_values_read") is not False
        or identity.get("teacher_features_read") is not False
        or identity.get("optimizer_updates") != 0
        or identity.get("gpu_used") is not False
        or identity.get("synthetic_engineering_input") is not True
        or identity.get("smoke") != expected_smoke
    ):
        raise ValueError("Engineering checkpoint internal identity differs")
    if set(state) != set(expected_state) or any(
        not torch.equal(state[name], expected_state[name]) for name in state
    ):
        raise ValueError("Engineering checkpoint architecture state differs")
    return {
        "status": "PASS_ENGINEERING_SMOKE_NO_TRAINING",
        "modality": config.modality,
        "state_tensor_sha256": identity["state_tensor_sha256"],
        "checkpoint_identity_sha256": claimed,
        "optimizer_updates": 0,
    }


def require_real_fit_admission(raw_history_receipt: Path) -> None:
    """Fail closed until physical stream-H5 bounds and hashes are separately admitted."""
    receipt = json.loads(raw_history_receipt.resolve(strict=True).read_text(encoding="utf-8"))
    if (
        receipt.get("training_admitted") is not True
        or receipt.get("stream_bounds_checked") is not True
        or not receipt.get("stream_h5_bounds_receipt")
        or not receipt.get("stream_h5_hash_receipt")
    ):
        raise RuntimeError("Real independent fitting is blocked by raw H5 admission")
    raise RuntimeError("Real independent fitter is not implemented by the engineering smoke")


__all__ = [
    "EngineeringSmokeConfig",
    "FoundationIdentity",
    "bind_foundation_identity",
    "require_real_fit_admission",
    "state_tensor_sha256",
    "verify_engineering_smoke_checkpoint",
    "verify_historical_sources",
    "write_engineering_smoke_checkpoint",
]
