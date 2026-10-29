"""Frozen native loader for the published Garl RGB+event model.

The adapter accepts already-preprocessed, label-free inputs only.  It does not
download auxiliary encoders, read TTC targets, construct an optimizer, or alter
the published height-ratio conversion.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
import yaml

from operational.efficient_context.common import digest

CHECKPOINT_NAME = "paper_ours_full.pth"
CONFIG_RELATIVE = Path("configs/ablation/ours_full.yaml")
CHECKPOINT_SHA256 = "e96a613a4fb877a1969d57ab562cadba89961fb202f5f2f2f0658f333a0d443e"
CHECKPOINT_BYTES = 973_290_993
CONFIG_SHA256 = "c7dedc9b93d32a416b8e92a6b474cc5a17ffd96b2cdaae3a4554e3cc0ee28454"
HF_REVISION = "b676fcdaf26c04bcf896cdb2b208c9c424e8462a"

_NATIVE_SOURCES = {
    "garl_ttc/models/__init__.py": (
        "039bdb68efbbb07647669670110825e0d3a96ffb4232c62fff59f6d590d834ea"
    ),
    "garl_ttc/models/focal_loss.py": (
        "7e5e53b9efff75b0254ebd13a7361a55be63ec8bd1385770b176f439dd6369c9"
    ),
    "garl_ttc/models/resnet.py": (
        "6fbb6bb08d57f6a1d70fa4df31b362476a2b7af6f5c3d82a09a111da17f2f26b"
    ),
    "garl_ttc/models/ttc_network.py": (
        "7927946d4b9af608f1ed2e57c6cc8b498d3c4be8b9c84305647199ef812d9c14"
    ),
}
_RGB_MEAN = (0.485, 0.456, 0.406)
_RGB_STD = (0.229, 0.224, 0.225)


class _NativeFullModel(Protocol):
    """Small typed boundary around the pinned upstream API."""

    dT: float  # noqa: N815 - pinned upstream API

    def to(self, device: str) -> _NativeFullModel: ...

    def float(self) -> _NativeFullModel: ...

    def eval(self) -> _NativeFullModel: ...

    def requires_grad_(self, requires_grad: bool = True) -> _NativeFullModel: ...

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> Any: ...  # noqa: ANN401 - pinned upstream returns a private Torch tuple

    def forward_test(self, sensor: torch.Tensor) -> tuple[torch.Tensor, Any]: ...


class _NativeConstructor(Protocol):
    def __call__(self, configuration: dict[str, Any], *, is_train: bool) -> _NativeFullModel: ...


def _validate_config(configuration: dict[str, Any]) -> None:
    """Reject a nearby ablation or preprocessing-contract change."""
    dataset, model = configuration["dataset"], configuration["model"]
    expected = {
        "dataset.mode": (dataset["mode"], "image_event"),
        "dataset.sync": (dataset["sync"], "front"),
        "dataset.window_interval": (dataset["window_interval"], 1),
        "dataset.img_mean": (tuple(dataset["img_mean"]), _RGB_MEAN),
        "dataset.img_std": (tuple(dataset["img_std"]), _RGB_STD),
        "model.fusion_style": (model["fusion_style"], "late_fusion"),
        "model.mode": (model["mode"], "height_ratio"),
        "model.input_feat_num_rgb": (model["input_feat_num_rgb"], 6),
        "model.input_feat_num_event": (model["input_feat_num_event"], 40),
        "model.input_feat_size": (tuple(model["input_feat_size"]), (128, 128)),
        "model.frame_num": (model["frame_num"], 2),
        "model.with_decoder": (model["with_decoder"], True),
        "model.num_layers": (model["num_layers"], 50),
    }
    drifted = [name for name, (actual, wanted) in expected.items() if actual != wanted]
    if drifted:
        raise ValueError("Published full Garl configuration changed: " + ", ".join(drifted))


def _constructor(code_root: Path) -> _NativeConstructor:
    """Import the constructor only after its local source bytes are pinned."""
    root = str(code_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    module = importlib.import_module("garl_ttc.models.ttc_network")
    source = Path(module.__file__ or "").resolve()
    if not source.is_relative_to(code_root.resolve()):
        raise ValueError(f"Garl module was imported from an unpinned root: {source}")
    return cast(_NativeConstructor, module.TTCNetwork)


def release_ttc(heights: np.ndarray, delta_t_s: float) -> np.ndarray:
    """Apply the release's native, unclipped height-ratio equation."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return delta_t_s / (1.0 - heights[:, 0] / heights[:, 1])


class FullGarl:
    """Strict inference wrapper for ``paper_ours_full.pth``.

    ``rgb_endpoints`` must be float32 ``[2,3,128,128]`` in endpoint order,
    already converted from RGB uint8 to [0,1] and normalized channel-wise with
    ImageNet mean/std before the common-square crop. ``event_endpoints`` must be
    the release time-volume representation, float32 ``[2,20,128,128]``, without
    additional normalization. The native input order is RGB0, RGB1, event0,
    event1, yielding ``[46,128,128]``.
    """

    def __init__(self, public_dir: Path, code_root: Path, device: str = "cpu") -> None:
        self.public_dir = public_dir.resolve()
        self.code_root = code_root.resolve()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA inference requested but CUDA is unavailable")
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Full Garl inference supports only CPU or CUDA")

        checkpoint = self.public_dir / CHECKPOINT_NAME
        config_path = self.public_dir / CONFIG_RELATIVE
        if checkpoint.stat().st_size != CHECKPOINT_BYTES or digest(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("Published full Garl checkpoint bytes changed")
        if digest(config_path) != CONFIG_SHA256:
            raise ValueError("Published full Garl configuration bytes changed")
        for relative, expected in _NATIVE_SOURCES.items():
            if digest(self.code_root / relative) != expected:
                raise ValueError(f"Pinned native Garl source changed: {relative}")

        configuration = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        _validate_config(configuration)
        # The downloaded full state contains both backbones and the decoder.
        # Removing initializer paths prevents unrelated checkpoint I/O during
        # construction; strict loading below proves the full state is complete.
        removed_initializers = {}
        for key in ("pretrained_ckpt_rgb", "pretrained_ckpt_event"):
            removed_initializers[key] = configuration["model"].pop(key, None)
        constructor = _constructor(self.code_root)
        model = constructor(configuration, is_train=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        incompatible = model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("Strict full Garl state load reported incompatible keys")
        self.model = model.float().eval().requires_grad_(False).to(str(self.device))
        if float(self.model.dT) != 0.1:
            raise ValueError("Published full Garl dT changed")
        self.bindings: dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_bytes": CHECKPOINT_BYTES,
            "config": str(config_path),
            "config_sha256": CONFIG_SHA256,
            "hf_revision": HF_REVISION,
            "native_code_root": str(self.code_root),
            "native_source_sha256": dict(_NATIVE_SOURCES),
            "removed_initializer_paths": removed_initializers,
            "strict_state_load": True,
            "input": {
                "shape": [46, 128, 128],
                "channel_order": ["RGB0[3]", "RGB1[3]", "EVENT0[20]", "EVENT1[20]"],
                "rgb_mean": list(_RGB_MEAN),
                "rgb_std": list(_RGB_STD),
                "event_normalization": "none",
            },
            "native_delta_t_s": 0.1,
            "native_rgb_sync": "front",
            "optimizer_updates": 0,
            "targets_read": False,
            "module_sha256": digest(Path(__file__)),
        }

    @staticmethod
    def compose_sensor(rgb_endpoints: np.ndarray, event_endpoints: np.ndarray) -> np.ndarray:
        """Validate and concatenate the two native modalities in release order."""
        rgb = np.asarray(rgb_endpoints)
        events = np.asarray(event_endpoints)
        if rgb.shape != (2, 3, 128, 128) or rgb.dtype != np.float32:
            raise ValueError("rgb_endpoints must be normalized float32 [2,3,128,128]")
        if events.shape != (2, 20, 128, 128) or events.dtype != np.float32:
            raise ValueError("event_endpoints must be float32 [2,20,128,128]")
        if not np.isfinite(rgb).all() or not np.isfinite(events).all():
            raise ValueError("Full Garl inputs must be finite")
        return np.concatenate((rgb.reshape(6, 128, 128), events.reshape(40, 128, 128)))

    def predict_sensor(self, sensor: np.ndarray) -> dict[str, float | list[float]]:
        """Run one fully composed native sensor tensor without target access."""
        value = np.asarray(sensor)
        if value.shape != (46, 128, 128) or value.dtype != np.float32:
            raise ValueError("Full Garl sensor must be float32 [46,128,128]")
        if not np.isfinite(value).all():
            raise ValueError("Full Garl sensor must be finite")
        tensor = torch.from_numpy(value[None]).to(self.device)
        with torch.inference_mode():
            heights_tensor, _ = self.model.forward_test(tensor)
        heights = heights_tensor.float().cpu().numpy()
        if heights.shape != (1, 2) or not np.isfinite(heights).all():
            raise ValueError("Published full Garl emitted invalid visible heights")
        ttc = release_ttc(heights, float(self.model.dT))
        return {"ttc": float(ttc[0]), "heights": heights[0].tolist()}

    def predict(
        self, rgb_endpoints: np.ndarray, event_endpoints: np.ndarray
    ) -> dict[str, float | list[float]]:
        """Compose both modalities and run the published full checkpoint."""
        return self.predict_sensor(self.compose_sensor(rgb_endpoints, event_endpoints))
