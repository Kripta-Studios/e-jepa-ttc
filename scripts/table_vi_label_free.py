"""Label-free inference primitives for the frozen EvTTC Table VI protocol."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
import yaml

from e_jepa_ttc.data.garl_input_contract import validate_input_schema

IDENTITY_KEYS = ("sequence_id", "sample_token", "track_id", "timestamp_us")
FORBIDDEN_TARGET_KEYS = frozenset(
    {
        "ttc",
        "ttc_s",
        "frame_ttc",
        "target_ttc",
        "target_ttc_s",
        "gt_ttc",
        "future_labels",
        "labels",
        "targets",
        "depth",
        "depth_history",
        "box3d_fcam",
        "box3d_h",
        "category",
        "category_index",
        "class_id",
        "foreground_mask",
        "geometry_target",
        "mask",
        "visible_heights_px",
    }
)


class TTCInferenceModel(Protocol):
    """Small common interface used by the supported checkpoint backends."""

    def predict(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor: ...


@dataclass(frozen=True)
class InferenceSettings:
    """Validated settings resolved from the portable inference config."""

    architecture: str
    input_manifest: Path
    batch_size: int
    device: str
    output_key: str | None
    input_key: str
    normalization_path: Path | None


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_mapping(path: Path) -> dict[str, Any]:
    """Read a JSON/YAML mapping."""

    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.casefold() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping at {path}.")
    return cast(dict[str, Any], value)


def reject_target_fields(value: object, *, location: str) -> None:
    """Reject privileged TTC/depth fields recursively before loading model inputs."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).casefold()
            if key_text in FORBIDDEN_TARGET_KEYS:
                raise ValueError(f"{location} contains forbidden target field: {key}")
            reject_target_fields(child, location=f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            reject_target_fields(child, location=f"{location}[{index}]")


def _resolve(base: Path, raw: object, *, name: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{name} must be a non-empty path string.")
    path = Path(raw)
    return path if path.is_absolute() else (base / path).resolve()


def parse_settings(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    normalization_override: Path | None,
) -> InferenceSettings:
    """Validate and resolve the inference-only configuration."""

    reject_target_fields(config, location="config")
    model_raw = config.get("model", {})
    inference_raw = config.get("inference", config)
    if not isinstance(model_raw, Mapping) or not isinstance(inference_raw, Mapping):
        raise ValueError("config.model and config.inference must be mappings.")
    architecture = str(model_raw.get("architecture", "")).strip()
    supported = {"torchscript", "eap_lhr_jepa_ttc", "e_jepa_tubelet_lhr"}
    if architecture not in supported:
        raise ValueError(
            f"Unsupported model architecture {architecture!r}; expected {sorted(supported)}."
        )
    input_manifest = _resolve(
        config_path.parent,
        inference_raw.get("input_manifest"),
        name="inference.input_manifest",
    )
    batch_size = int(inference_raw.get("batch_size", 1))
    if batch_size <= 0:
        raise ValueError("inference.batch_size must be positive.")
    device = str(inference_raw.get("device", "auto"))
    output_value = model_raw.get("output_key")
    output_key = None if output_value is None else str(output_value)
    input_key = str(model_raw.get("input_key", "inputs"))
    configured_normalization = inference_raw.get("normalization_path")
    normalization_path = normalization_override
    if normalization_path is None and configured_normalization is not None:
        normalization_path = _resolve(
            config_path.parent,
            configured_normalization,
            name="inference.normalization_path",
        )
    return InferenceSettings(
        architecture=architecture,
        input_manifest=input_manifest,
        batch_size=batch_size,
        device=device,
        output_key=output_key,
        input_key=input_key,
        normalization_path=normalization_path,
    )


def canonical_token_schema(payload: Mapping[str, Any], *, location: str) -> tuple[str, ...]:
    """Read and validate the canonical sample-token schema declaration."""

    raw = payload.get("token_schema", list(IDENTITY_KEYS))
    if isinstance(raw, Mapping):
        raw = raw.get("fields")
    if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
        raise ValueError(f"{location}.token_schema must be a list of field names.")
    fields = tuple(raw)
    if fields != IDENTITY_KEYS:
        raise ValueError(
            f"{location}.token_schema must be exactly {list(IDENTITY_KEYS)}, got {list(fields)}."
        )
    return fields


def _expected_sequence_counts(protocol: Mapping[str, Any]) -> dict[str, int]:
    raw = protocol.get("sequences")
    if not isinstance(raw, list) or not raw:
        raise ValueError("protocol.sequences must be a non-empty list.")
    output: dict[str, int] = {}
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise ValueError(f"protocol.sequences[{index}] must be a mapping.")
        sequence_id = str(row.get("local_sequence_id", ""))
        if not sequence_id or sequence_id in output:
            raise ValueError(f"Protocol sequence ID is empty or duplicated: {sequence_id!r}.")
        count_raw = row.get("sample_count", row.get("label_count"))
        if count_raw is None or int(count_raw) <= 0:
            raise ValueError(f"Protocol sequence {sequence_id!r} has no positive sample count.")
        output[sequence_id] = int(count_raw)
    return output


def _identity_columns(archive: Mapping[str, np.ndarray], count: int) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for key in IDENTITY_KEYS:
        if key not in archive:
            raise ValueError(f"Input shard is missing identity column {key!r}.")
        values = np.asarray(archive[key])
        if values.ndim != 1 or values.shape[0] != count:
            raise ValueError(f"Input shard identity column {key!r} must have shape [{count}].")
        output[key] = values
    return output


def _validate_shard_keys(keys: Sequence[str], *, path: Path) -> None:
    for key in keys:
        if key.casefold() in FORBIDDEN_TARGET_KEYS:
            raise ValueError(f"Input shard {path} contains forbidden target field: {key}")


def load_label_free_inputs(
    manifest_path: Path,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], dict[str, int], list[dict[str, object]]]:
    """Load only declared label-free NPZ shards and enforce exact protocol coverage."""

    manifest = read_mapping(manifest_path)
    reject_target_fields(manifest, location="input_manifest")
    canonical_token_schema(manifest, location="input_manifest")
    schema = manifest.get("input_schema")
    if not isinstance(schema, dict):
        raise ValueError("Input manifest has no input_schema mapping.")
    validate_input_schema(cast(dict[str, Any], schema))
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Input manifest must contain a non-empty shards list.")

    arrays: dict[str, list[np.ndarray]] = {}
    identities: list[dict[str, object]] = []
    shard_evidence: list[dict[str, object]] = []
    for index, shard in enumerate(shards):
        if not isinstance(shard, Mapping):
            raise ValueError(f"input_manifest.shards[{index}] must be a mapping.")
        path = _resolve(manifest_path.parent, shard.get("path"), name=f"shards[{index}].path")
        if not path.is_file() or path.suffix.casefold() != ".npz":
            raise ValueError(f"Input shard must be an existing NPZ file: {path}")
        actual_hash = sha256_file(path)
        declared_hash = shard.get("sha256")
        if declared_hash is not None and str(declared_hash).casefold() != actual_hash:
            raise ValueError(f"Input shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive_raw:
            archive = {key: archive_raw[key] for key in archive_raw.files}
        _validate_shard_keys(list(archive), path=path)
        if "sample_token" not in archive:
            raise ValueError(f"Input shard is missing sample_token: {path}")
        count = int(np.asarray(archive["sample_token"]).shape[0])
        declared_count = shard.get("sample_count")
        if declared_count is not None and int(declared_count) != count:
            raise ValueError(f"Input shard sample_count mismatch: {path}")
        columns = _identity_columns(archive, count)
        tensor_keys = sorted(set(archive).difference(IDENTITY_KEYS))
        if not tensor_keys:
            raise ValueError(f"Input shard contains no model tensors: {path}")
        for key in tensor_keys:
            values = np.asarray(archive[key])
            if values.ndim == 0 or values.shape[0] != count:
                raise ValueError(f"Input tensor {key!r} in {path} must start with [{count}].")
            if values.dtype.kind not in "biuf":
                raise ValueError(f"Input tensor {key!r} in {path} must be numeric or boolean.")
            arrays.setdefault(key, []).append(values)
        for row_index in range(count):
            identities.append(
                {
                    "sequence_id": str(columns["sequence_id"][row_index]),
                    "sample_token": str(columns["sample_token"][row_index]),
                    "track_id": str(columns["track_id"][row_index]),
                    "timestamp_us": int(columns["timestamp_us"][row_index]),
                }
            )
        shard_evidence.append(
            {"path": path.as_posix(), "sha256": actual_hash, "sample_count": count}
        )

    merged: dict[str, np.ndarray] = {}
    shard_count = len(shards)
    for key, values in arrays.items():
        if len(values) != shard_count:
            raise ValueError(f"Model input {key!r} is not present in every input shard.")
        try:
            merged[key] = np.concatenate(values, axis=0)
        except ValueError as error:
            raise ValueError(f"Model input {key!r} has incompatible shard shapes.") from error

    identity_tuples = [tuple(row[key] for key in IDENTITY_KEYS) for row in identities]
    if len(set(identity_tuples)) != len(identity_tuples):
        raise ValueError("Input shards contain duplicate sample identities.")
    token_values = [str(row["sample_token"]) for row in identities]
    if len(set(token_values)) != len(token_values):
        raise ValueError("Input shards contain duplicate sample_token values.")

    expected = _expected_sequence_counts(protocol)
    observed: dict[str, int] = {}
    for row in identities:
        sequence_id = str(row["sequence_id"])
        observed[sequence_id] = observed.get(sequence_id, 0) + 1
    if observed != expected:
        raise ValueError(f"Protocol coverage mismatch: expected {expected}, observed {observed}.")
    return merged, identities, observed, shard_evidence


def load_normalization(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    """Load normalization only when an explicit path was supplied."""

    if path is None:
        return None, None
    payload = read_mapping(path)
    reject_target_fields(payload, location="normalization")
    return payload, sha256_file(path)


def _normalization_for(payload: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    inputs = payload.get("inputs", payload)
    if not isinstance(inputs, Mapping):
        raise ValueError("normalization.inputs must be a mapping.")
    if key == "inputs" and "mean" in inputs and "std" in inputs:
        return inputs
    value = inputs.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"Normalization for {key!r} must be a mapping.")
    return value


def tensor_batch(
    arrays: Mapping[str, np.ndarray],
    indices: slice,
    *,
    device: torch.device,
    normalization: Mapping[str, Any] | None,
) -> dict[str, torch.Tensor]:
    """Convert one input batch and apply only explicitly declared normalization."""

    output: dict[str, torch.Tensor] = {}
    for key, values in arrays.items():
        tensor = torch.from_numpy(np.array(values[indices], copy=True)).to(device)
        rule = _normalization_for(normalization, key) if normalization is not None else None
        if rule is not None:
            if "mean" not in rule or "std" not in rule:
                raise ValueError(f"Normalization for {key!r} requires mean and std.")
            tensor = tensor.float()
            mean = torch.as_tensor(rule["mean"], dtype=tensor.dtype, device=device)
            std = torch.as_tensor(rule["std"], dtype=tensor.dtype, device=device)
            if bool((std <= 0).any()):
                raise ValueError(f"Normalization std for {key!r} must be positive.")
            tensor = (tensor - mean) / std
        output[key] = tensor
    return output


def resolve_device(name: str) -> torch.device:
    """Resolve an explicit device without silently requiring CUDA."""

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return device


def _extract_output(value: object, output_key: str | None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        output = value
    elif output_key is not None and isinstance(value, Mapping):
        output = value.get(output_key)
    elif output_key is not None:
        output = getattr(value, output_key, None)
    else:
        output = None
    if not isinstance(output, torch.Tensor):
        raise ValueError(f"Model output does not expose tensor {output_key!r}.")
    return output.reshape(-1)


class _TorchScriptModel:
    def __init__(self, module: torch.jit.ScriptModule, settings: InferenceSettings) -> None:
        self.module = module
        self.settings = settings

    def predict(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.settings.input_key not in batch:
            raise ValueError(f"Input shard has no configured tensor {self.settings.input_key!r}.")
        return _extract_output(
            self.module(batch[self.settings.input_key]),
            self.settings.output_key,
        )


class _EAPLHRModel:
    def __init__(self, module: torch.nn.Module, output_key: str | None) -> None:
        self.module = module
        self.output_key = output_key or "ttc_seconds"

    def predict(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        from e_jepa_ttc.data.garlttc_lhr_cache import observable_motion_from_boxes_torch

        required = {"event_roi_endpoints", "full_event_context", "delta_t_s"}
        missing = sorted(required.difference(batch))
        if missing:
            raise ValueError(f"EAPLHR input shard is missing tensors: {missing}")
        roi = batch["event_roi_endpoints"]
        if roi.ndim != 5:
            raise ValueError("event_roi_endpoints must have shape [B,2,20,128,128].")
        motion = batch.get("observable_motion")
        if motion is None:
            boxes = batch.get("boxes_xyxy")
            if boxes is None:
                raise ValueError("EAPLHR requires observable_motion or boxes_xyxy.")
            motion = observable_motion_from_boxes_torch(boxes.float(), batch["delta_t_s"].float())
        value = self.module(
            full_frame_events=batch["full_event_context"].float(),
            event_roi_pair=roi.flatten(1, 2).float(),
            delta_t_s=batch["delta_t_s"].float(),
            observable_motion=motion.float(),
            jepa_context_motion=(
                batch["jepa_context_motion"].float() if "jepa_context_motion" in batch else None
            ),
            rgb_pair=batch["rgb_endpoints"].float() if "rgb_endpoints" in batch else None,
        )
        return _extract_output(value, self.output_key)


class _TubeletModel:
    def __init__(self, module: torch.nn.Module, settings: InferenceSettings) -> None:
        self.module = module
        self.settings = settings

    def predict(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.settings.input_key not in batch:
            raise ValueError(f"Input shard has no configured tensor {self.settings.input_key!r}.")
        value = self.module(batch[self.settings.input_key].float())
        return _extract_output(value, self.settings.output_key or "ttc_mean_seconds")


def load_model(
    checkpoint_path: Path,
    settings: InferenceSettings,
    device: torch.device,
) -> TTCInferenceModel:
    """Load one explicitly supported frozen inference checkpoint."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if settings.architecture == "torchscript":
        # File objects avoid a Windows TorchScript path bug on non-ASCII user names.
        with checkpoint_path.open("rb") as handle:
            module = torch.jit.load(handle, map_location=device).eval()
        return _TorchScriptModel(module, settings)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a mapping payload.")
    model_config = checkpoint.get("model_config")
    state = checkpoint.get("model_state_dict")
    if not isinstance(model_config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("Checkpoint is missing model_config or model_state_dict.")
    if settings.architecture == "eap_lhr_jepa_ttc":
        from e_jepa_ttc.models.eap_lhr_jepa_ttc import EAPLHRJEPATTC, EAPLHRJEPATTCConfig

        module = EAPLHRJEPATTC(EAPLHRJEPATTCConfig(**dict(model_config))).to(device)
        missing, unexpected = module.load_state_dict(state, strict=False)
        allowed = {
            name for name in missing if name.startswith(("target_roi_encoder.", "jepa_predictor."))
        }
        if set(missing) != allowed or unexpected:
            raise ValueError(f"Incompatible checkpoint: missing={missing}, unexpected={unexpected}")
        module.target_roi_encoder.load_state_dict(module.roi_encoder.state_dict())
        module.eval()
        return _EAPLHRModel(module, settings.output_key)

    from e_jepa_ttc.models.e_jepa_tubelet_lhr import EJEPATubeletLHR, EJEPATubeletLHRConfig

    module = EJEPATubeletLHR(EJEPATubeletLHRConfig(**dict(model_config))).to(device)
    module.load_state_dict(state, strict=True)
    module.eval()
    return _TubeletModel(module, settings)


def infer_predictions(
    model: TTCInferenceModel,
    arrays: Mapping[str, np.ndarray],
    identities: Sequence[Mapping[str, object]],
    *,
    batch_size: int,
    device: torch.device,
    normalization: Mapping[str, Any] | None,
) -> list[dict[str, object]]:
    """Run deterministic frozen inference and return identity-only predictions."""

    predictions: list[dict[str, object]] = []
    with torch.inference_mode():
        for start in range(0, len(identities), batch_size):
            stop = min(start + batch_size, len(identities))
            batch = tensor_batch(
                arrays,
                slice(start, stop),
                device=device,
                normalization=normalization,
            )
            values = model.predict(batch).detach().float().cpu().tolist()
            if len(values) != stop - start:
                raise ValueError("Model output batch size does not match input batch size.")
            for identity, value in zip(identities[start:stop], values, strict=True):
                predicted = float(value)
                if not math.isfinite(predicted):
                    raise ValueError("Model emitted a non-finite predicted_ttc_s.")
                predictions.append({**identity, "predicted_ttc_s": predicted})
    return predictions
