"""Portable fusion head with its admitted H normalizers and dynamic RGB fallback.

This export begins at raw producer features. It does not include sensor loaders
or the producer encoders, and cannot establish end-to-end pipeline parity alone.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from e_jepa_ttc.rgb_port.normalization import FrozenNormalizer
from operational.range_transfer.common import freeze, read, sha

from .fusion_features import FeatureIdentity
from .fusion_model import RGBAnchorComplement

INPUT_NAMES = ("rgb_features", "event_features", "rgb_ttc", "event_ttc", "rgb_available")


class NormalizedFusion(nn.Module):
    """Private head plus exact FP32 normalization buffers, with missing-RGB masking."""

    def __init__(
        self,
        model: RGBAnchorComplement,
        rgb_normalizer: FrozenNormalizer,
        event_normalizer: FrozenNormalizer,
        rgb_identity: FeatureIdentity,
        event_identity: FeatureIdentity,
    ) -> None:
        super().__init__()
        self.model = copy.deepcopy(model).cpu().eval().requires_grad_(False)
        for modality, identity, normalizer, width in (
            ("rgb", rgb_identity, rgb_normalizer, model.rgb_features),
            ("event", event_identity, event_normalizer, model.event_features),
        ):
            identity.validate()
            if identity.modality != modality:
                raise ValueError("Portable fusion modality identity mismatch")
            normalizer.validate_endpoint(
                modality=modality,
                fit_role="H",
                schema_sha256=identity.schema_sha256,
                producer_sha256=identity.normalizer_producer_sha256(),
            )
            if (
                normalizer.mean.shape != (width,)
                or normalizer.scale.shape != (width,)
                or normalizer.mean.dtype != np.float32
                or normalizer.scale.dtype != np.float32
                or not np.isfinite(normalizer.mean).all()
                or not np.isfinite(normalizer.scale).all()
                or (normalizer.scale <= 0).any()
            ):
                raise ValueError("Invalid FP32 fusion normalization arrays")
            self.register_buffer(f"{modality}_mean", torch.tensor(normalizer.mean))
            self.register_buffer(f"{modality}_scale", torch.tensor(normalizer.scale))

    def forward(
        self,
        rgb_features: Tensor,
        event_features: Tensor,
        rgb_ttc: Tensor,
        event_ttc: Tensor,
        rgb_available: Tensor,
    ) -> dict[str, Tensor]:
        """Accept unnormalized producer features and preserve the exact event fallback."""
        if (
            rgb_features.ndim != 2
            or event_features.ndim != 2
            or rgb_features.shape[1] != self.model.rgb_features
            or event_features.shape[1] != self.model.event_features
            or rgb_features.dtype != torch.float32
            or event_features.dtype != torch.float32
            or rgb_available.dtype != torch.bool
            or rgb_available.ndim != 1
            or len(rgb_available) != len(rgb_features)
        ):
            raise ValueError("Invalid raw feature normalization contract")
        rgb = (rgb_features - self.rgb_mean) / self.rgb_scale
        rgb = torch.where(rgb_available[:, None], rgb, torch.zeros_like(rgb))
        event = (event_features - self.event_mean) / self.event_scale
        return self.model(rgb, event, rgb_ttc, event_ttc, rgb_available)


def normalizer_record(value: FrozenNormalizer) -> dict:
    """Retain H fitting provenance as well as the actual FP32 transform."""
    return {**asdict(value), "mean": value.mean.tolist(), "scale": value.scale.tolist()}


def tensor_digest(value: Tensor) -> str:
    """Bind exact tensor dtype, shape and bytes, including masked NaN inputs."""
    array = value.detach().cpu().contiguous().numpy()
    checksum = hashlib.sha256()
    checksum.update(str(array.dtype).encode())
    checksum.update(str(array.shape).encode())
    checksum.update(array.tobytes())
    return checksum.hexdigest()


def export_head(
    directory: Path,
    model: RGBAnchorComplement,
    rgb_normalizer: FrozenNormalizer,
    event_normalizer: FrozenNormalizer,
    rgb_identity: FeatureIdentity,
    event_identity: FeatureIdentity,
    examples: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]],
    *,
    binding: dict,
) -> dict:
    """Script/reload raw-feature inference, requiring all three availability cases."""
    coverage = set()
    for values in examples:
        if any(value.device.type != "cpu" for value in values):
            raise ValueError("Portable head examples must be CPU inputs")
        mask = values[-1]
        if mask.dtype != torch.bool or mask.ndim != 1 or not len(mask):
            raise ValueError("Explicit nonempty boolean RGB mask required")
        coverage.add("all" if mask.all() else "none" if not mask.any() else "mixed")
    if coverage != {"all", "none", "mixed"} or not binding:
        raise ValueError("Admit producer/head binding and all, missing, mixed RGB examples")
    wrapper = NormalizedFusion(
        model, rgb_normalizer, event_normalizer, rgb_identity, event_identity
    )
    record = {
        "binding": binding,
        "head_state": {name: tensor_digest(value) for name, value in model.state_dict().items()},
        "head_config": {
            "mode": model.mode,
            "residual_scale": model.residual_scale,
            "rgb_features": model.rgb_features,
            "event_features": model.event_features,
            "hidden": model.network[0].out_features,
        },
        "example_input_hashes": [
            {name: tensor_digest(value) for name, value in zip(INPUT_NAMES, values, strict=True)}
            for values in examples
        ],
        "rgb_identity": asdict(rgb_identity),
        "event_identity": asdict(event_identity),
        "rgb_normalizer": normalizer_record(rgb_normalizer),
        "event_normalizer": normalizer_record(event_normalizer),
        "input_names": list(INPUT_NAMES),
        "availability": "dynamic",
        "precision": "CPU_FP32",
        "rtol": 1e-6,
        "atol": 1e-6,
        "fallback_bit_exact": True,
        "includes_encoders_or_sensor_preprocessing": False,
        "sources": {
            name: sha(Path(__file__).with_name(name))
            for name in ("fusion_export.py", "fusion_model.py")
        },
    }
    freeze(directory / "BINDING.json", record)
    if (directory / "COMPLETE.json").exists():
        receipt = read(directory / "COMPLETE.json")
        if any(sha(directory / name) != expected for name, expected in receipt["files"].items()):
            raise ValueError("Portable fusion artifact changed")
        return receipt
    expected = []
    reference = copy.deepcopy(model).cpu().eval().requires_grad_(False)
    with torch.inference_mode():
        for values in examples:
            rgb, event, rgb_ttc, event_ttc, available = values
            normalized_rgb = np.zeros_like(rgb.numpy())
            normalized_rgb[available.numpy()] = rgb_normalizer.transform(
                rgb.numpy()[available.numpy()]
            )
            normalized_event = event_normalizer.transform(event.numpy())
            target = reference(
                torch.from_numpy(normalized_rgb),
                torch.from_numpy(normalized_event),
                rgb_ttc,
                event_ttc,
                available,
            )
            actual = wrapper(*values)
            for key in target:
                if not torch.equal(target[key], actual[key]):
                    raise ValueError("Embedded normalization differs from the admitted NumPy path")
            expected.append(target)
        scripted = torch.jit.script(wrapper)
        with (directory / "model.torchscript.pt").open("wb") as stream:
            torch.jit.save(scripted, stream)
        with (directory / "model.torchscript.pt").open("rb") as stream:
            loaded = torch.jit.load(stream, map_location="cpu").eval()
        errors = []
        for values, target in zip(examples, expected, strict=True):
            actual = loaded(*values)
            for key in target:
                torch.testing.assert_close(actual[key], target[key], rtol=1e-6, atol=1e-6)
            missing = ~values[-1]
            if not torch.equal(actual["ttc"][missing], values[3][missing]):
                raise ValueError("Portable fusion changed exact event fallback")
            errors.append(float((actual["ttc"] - target["ttc"]).abs().max()))
    np.savez_compressed(
        directory / "example_inputs.npz",
        **{
            f"example_{i:03d}_{name}": value.numpy()
            for i, values in enumerate(examples)
            for name, value in zip(INPUT_NAMES, values, strict=True)
        },
    )
    freeze(
        directory / "example_outputs.json",
        {
            "examples": [
                {name: value.tolist() for name, value in output.items()} for output in expected
            ]
        },
    )
    names = ("BINDING.json", "model.torchscript.pt", "example_inputs.npz", "example_outputs.json")
    result = {
        "status": "PASS",
        "files": {name: sha(directory / name) for name in names},
        "examples": len(examples),
        "maximum_ttc_absolute_error": max(errors),
        "fallback_bit_exact": True,
        "optimizer_updates": 0,
        "scope": "Raw producer features plus admitted H normalization and fusion head only",
    }
    freeze(directory / "COMPLETE.json", result)
    return result
