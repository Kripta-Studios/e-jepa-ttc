"""Audited offline initialization for MonoDGP's ResNet-50 backbone."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor

OFFICIAL_FILENAME = "resnet50-0676ba61.pth"
OFFICIAL_SHA256 = "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
OFFICIAL_URL = f"https://download.pytorch.org/models/{OFFICIAL_FILENAME}"
WEIGHTS_ENUM = "torchvision.models.ResNet50_Weights.IMAGENET1K_V1"
IGNORED_CLASSIFIER_KEYS = frozenset(("fc.weight", "fc.bias"))


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _target_backbone(model: torch.nn.Module) -> torch.nn.Module:
    try:
        target = model.get_submodule("backbone.0.body")
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise TypeError("Expected MonoDGP model.backbone[0].body.") from exc
    if not isinstance(target, torch.nn.Module):
        raise TypeError("MonoDGP backbone body must be a torch module.")
    return target


def _validate_source_contract(
    source: Mapping[str, Tensor], target_state: Mapping[str, Tensor]
) -> None:
    missing = sorted(set(target_state).difference(source))
    if missing:
        raise ValueError(f"ImageNet checkpoint is missing backbone keys: {missing}")
    for name, expected in target_state.items():
        value = source[name]
        if not isinstance(value, Tensor):
            raise TypeError(f"Checkpoint value is not a tensor: {name}")
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(
                f"Checkpoint tensor contract differs for {name}: "
                f"{tuple(value.shape)}/{value.dtype} != "
                f"{tuple(expected.shape)}/{expected.dtype}"
            )
    extras = set(source).difference(target_state)
    classifier = extras.intersection(IGNORED_CLASSIFIER_KEYS)
    if classifier != IGNORED_CLASSIFIER_KEYS:
        raise ValueError("Official checkpoint must contain both ResNet-50 classifier tensors.")
    unconsumed = extras.difference(classifier)
    invalid = sorted(
        name
        for name in unconsumed
        if not (
            name.endswith(".num_batches_tracked")
            and f"{name.removesuffix('.num_batches_tracked')}.running_mean" in target_state
        )
    )
    if invalid:
        raise ValueError(f"Unexpected ImageNet checkpoint keys: {invalid}")


def load_imagenet_resnet50_backbone(
    model: torch.nn.Module, weights_path: Path, expected_sha256: str
) -> dict[str, object]:
    """Load the pinned official ImageNet-1K V1 ResNet-50 backbone offline.

    The function never resolves a URL or asks torchvision for weights. It
    accepts only the pinned official digest, loads on CPU with the restricted
    weights-only unpickler, ignores only the two classifier tensors, delegates
    BatchNorm counters to the target module loader, and then compares every
    target parameter and buffer exactly against the checkpoint.
    """
    path = Path(weights_path).resolve(strict=True)
    expected = expected_sha256.lower()
    if expected != OFFICIAL_SHA256:
        raise ValueError("Expected SHA-256 is not the pinned ImageNet-1K V1 digest.")
    observed, byte_count = _sha256_and_size(path)
    if observed != expected:
        raise ValueError(f"ResNet-50 weights SHA-256 mismatch: {observed} != {expected}")

    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, Mapping) or not all(isinstance(name, str) for name in payload):
        raise TypeError("Official ResNet-50 checkpoint must be a string-keyed state dict.")
    source = dict(payload)
    non_tensor = sorted(name for name, value in source.items() if not isinstance(value, Tensor))
    if non_tensor:
        raise TypeError(f"Checkpoint contains non-tensor values: {non_tensor}")
    target = _target_backbone(model)
    before = target.state_dict()
    _validate_source_contract(source, before)

    classifier = sorted(IGNORED_CLASSIFIER_KEYS)
    backbone_state = {name: value for name, value in source.items() if name not in classifier}
    incompatible = target.load_state_dict(backbone_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict backbone load returned incompatible keys: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    target_state = target.state_dict()
    mismatched = []
    for name, actual in target_state.items():
        expected_tensor = source[name]
        if not torch.equal(actual.detach().cpu(), expected_tensor.detach().cpu()):
            mismatched.append(name)
    if mismatched:
        raise RuntimeError(f"Backbone tensors differ after strict load: {mismatched}")

    parameter_names = set(dict(target.named_parameters()))
    buffer_names = set(dict(target.named_buffers()))
    running_statistics = sorted(
        name
        for name in target_state
        if name.endswith(".running_mean") or name.endswith(".running_var")
    )
    frozen_bn_modules = sum(
        type(module).__name__ == "FrozenBatchNorm2d" for module in target.modules()
    )
    return {
        "weights_path": str(path),
        "loaded_filename": path.name,
        "official_filename": OFFICIAL_FILENAME,
        "source_url": OFFICIAL_URL,
        "weights_enum": WEIGHTS_ENUM,
        "weights_enum_sha256": OFFICIAL_SHA256,
        "expected_sha256": expected,
        "observed_sha256": observed,
        "file_bytes": byte_count,
        "source_tensor_count": len(source),
        "loaded_backbone_tensor_count": len(target_state),
        "loaded_parameter_tensor_count": len(parameter_names),
        "loaded_buffer_tensor_count": len(buffer_names),
        "exact_match_tensor_count": len(target_state),
        "ignored_keys": classifier,
        "batch_norm_counters_consumed_by_loader": sum(
            name.endswith(".num_batches_tracked") for name in backbone_state
        ),
        "frozen_batch_norm_module_count": frozen_bn_modules,
        "frozen_batch_norm_running_stat_tensor_count": len(running_statistics),
        "strict_load": True,
        "all_backbone_tensors_exact": True,
        "implicit_download": False,
    }
