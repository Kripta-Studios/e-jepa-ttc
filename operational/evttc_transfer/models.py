"""Frozen TRAIN40 and published Garl inference for label-free EvTTC inputs.

This module contains inference adapters only.  It never opens TTC targets and
never constructs an optimizer.  The caller is responsible for producing the
sensor tensors described by :class:`FrozenModels`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from operational.efficient_context.common import digest
from operational.simplex_t_shared_route.adapter import extract as canonical_extract
from operational.simplex_t_shared_route.adapter import inputs as canonical_inputs
from operational.train40_system.contracts import read, verified_endpoint
from operational.train40_system.garl_predictions import (
    NativeModel,
    native_model,
    release_ttc,
)

_PUBLIC_GARL_SHA256 = "fcaf9be47e2dafc6f73c6c3ebd102595ae06119dcae78aea698a42627b2b4fef"
_H8_SEEDS = (7, 13, 23)
_H8_LAGS_US = np.arange(350_000, -1, -50_000, dtype=np.int64)


def _array_sha256(value: np.ndarray) -> str:
    """Hash a normalization array together with its dtype and shape."""
    array = np.ascontiguousarray(value)
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode())
    hasher.update(np.asarray(array.shape, dtype="<i8").tobytes())
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def _complete_payload(path: Path, expected_updates: int) -> dict[str, Any]:
    """Load a verified complete checkpoint without restoring training state."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("status") != "COMPLETE"
        or payload.get("committed_updates") != expected_updates
        or not isinstance(payload.get("model_state_dict"), dict)
    ):
        raise ValueError(f"Checkpoint is not a complete endpoint: {path}")
    return cast(dict[str, Any], payload)


class FrozenModels:
    """Strict frozen inference for the three H8 heads and public event-only Garl.

    ``predict`` accepts one label-free history with shape
    ``[8, 3, 12, 128, 128]``.  The eight observations are padded to the original
    canonical producer batch of sixteen before feature extraction. ``valid`` is
    a contiguous suffix mask.  ``availability_lag_s`` records how old the ROI
    metadata was at the current anchor; the EvTTC adapter normally passes zero
    because it supplies an already-observed past box.

    ``garl_predict`` accepts the release-native event tensor with shape
    ``[40, 128, 128]``.  Garl's height-ratio conversion is intentionally left
    unclipped and can therefore return an infinite TTC at its singularity.
    """

    def __init__(self, campaign: Path, device: str = "cpu") -> None:
        self.campaign = campaign.resolve()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA inference requested but CUDA is unavailable")
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Frozen inference supports only CPU or CUDA")

        self.bindings: dict[str, Any] = {
            "campaign": str(self.campaign),
            "device": str(self.device),
            "optimizer_updates": 0,
            "targets_read": False,
            "module_sha256": digest(Path(__file__)),
        }
        self.models = self._load_producers()
        self.heads = self._load_h8_heads()
        self.mean, self.scale = self._load_normalizer()
        self.garl = self._load_public_garl()

    def _load_producers(self) -> dict[str, nn.Module]:
        """Load A5, C2F and PAIR at their exact completed endpoints."""
        from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
        from e_jepa_ttc.training.stage61_pair_head import CachedPairDirectPhase

        protocol_path = self.campaign / "TRAINING_PROTOCOL.json"
        protocol = read(protocol_path)
        result: dict[str, nn.Module] = {}
        bindings: dict[str, Any] = {"protocol_sha256": digest(protocol_path)}
        for arm, expected in (("a5", 49_932), ("c2f", 49_932), ("pair", 6_840)):
            checkpoint, receipt = verified_endpoint(
                self.campaign, f"{arm}_seed7", expected
            )
            payload = _complete_payload(checkpoint, expected)
            if arm == "pair":
                dropout = float(payload["contract"]["model_config"]["dropout"])
                model: nn.Module = CachedPairDirectPhase(dropout=dropout)
            else:
                config = protocol["producers"][arm]["model_config"]
                if payload["contract"]["model_config"] != config:
                    raise ValueError(f"{arm} checkpoint model configuration changed")
                model = CausalScaleTTC(CausalScaleTTCConfig(**config))
            model.load_state_dict(payload["model_state_dict"], strict=True)
            result[arm.upper()] = (
                model.float().eval().requires_grad_(False).to(self.device)
            )
            bindings[arm] = {
                "checkpoint_sha256": receipt["sha256"],
                "committed_updates": receipt["committed_updates"],
                "strict_state_load": True,
            }
        self.bindings["producers"] = bindings
        return result

    def _load_h8_heads(self) -> dict[int, nn.Module]:
        """Load all three fixed H8 seed endpoints strictly."""
        from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner

        result: dict[int, nn.Module] = {}
        bindings: dict[str, Any] = {}
        for seed in _H8_SEEDS:
            checkpoint, receipt = verified_endpoint(
                self.campaign, f"h8_seed{seed}", 2_500
            )
            payload = _complete_payload(checkpoint, 2_500)
            contract = payload["contract"]
            if contract["fit"] != "H8" or contract["seed"] != seed:
                raise ValueError(f"H8 seed {seed} checkpoint identity changed")
            head = TemporalRefiner(TemporalConfig(**contract["model_config"]))
            head.load_state_dict(payload["model_state_dict"], strict=True)
            result[seed] = head.float().eval().requires_grad_(False).to(self.device)
            bindings[str(seed)] = {
                "checkpoint_sha256": receipt["sha256"],
                "committed_updates": receipt["committed_updates"],
                "strict_state_load": True,
            }
        self.bindings["h8_heads"] = bindings
        return result

    def _load_normalizer(self) -> tuple[np.ndarray, np.ndarray]:
        """Read only the H8 mean/scale arrays after verifying the full artifact."""
        manifest_path = self.campaign / "H8_FEATURE_MANIFEST.json"
        manifest = read(manifest_path)
        if manifest["status"] != "COMPLETE_VERIFIED" or manifest["row_count"] != 88_744:
            raise ValueError("Complete H8 feature manifest required")
        feature_path = self.campaign / "H8_FEATURES.npz"
        expected = next(
            (item["sha256"] for item in manifest["files"] if item["path"] == feature_path.name),
            None,
        )
        if expected is None or digest(feature_path) != expected:
            raise ValueError("H8 normalization artifact changed")
        with np.load(feature_path, allow_pickle=False) as stored:
            mean = np.asarray(stored["mean"], dtype=np.float64).copy()
            scale = np.asarray(stored["scale"], dtype=np.float64).copy()
        if mean.shape != (17,) or scale.shape != (17,):
            raise ValueError("H8 normalization shape changed")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("H8 normalization is invalid")
        self.bindings["normalization"] = {
            "artifact_sha256": expected,
            "manifest_sha256": digest(manifest_path),
            "consumed_ids_sha256": manifest["consumed_ids_sha256"],
            "mean_sha256": _array_sha256(mean),
            "scale_sha256": _array_sha256(scale),
            "arrays_read": ["mean", "scale"],
        }
        return mean, scale

    def _load_public_garl(self) -> NativeModel:
        """Verify pinned public bytes and delegate construction to the existing loader."""
        freeze_path = self.campaign / "DELIVERY_FREEZE.json"
        freeze = read(freeze_path)
        checkpoint = self.campaign / "public_garl/paper_event_only_lhr.pth"
        config = self.campaign / "public_garl/configs/ablation/event_lhr.yaml"
        if freeze["public_checkpoint_sha256"] != _PUBLIC_GARL_SHA256:
            raise ValueError("Unexpected public Garl endpoint in delivery freeze")
        if digest(checkpoint) != _PUBLIC_GARL_SHA256:
            raise ValueError("Published Garl checkpoint bytes changed")
        if digest(config) != freeze["config_sha256"]:
            raise ValueError("Published Garl configuration changed")
        code_root = Path(freeze["native_code_root"]).resolve()
        sources: dict[str, str] = {}
        for item in freeze["native_source_files"]:
            path = code_root / item["path"]
            if digest(path) != item["sha256"]:
                raise ValueError(f"Pinned Garl source changed: {item['path']}")
            sources[item["path"]] = item["sha256"]
        model = native_model(self.campaign, code_root).to(str(self.device))
        self.bindings["garl"] = {
            "checkpoint_sha256": _PUBLIC_GARL_SHA256,
            "config_sha256": freeze["config_sha256"],
            "source_sha256": sources,
            "native_code_root": str(code_root),
            "strict_state_load": True,
            "event_only_lhr": True,
        }
        return model

    @staticmethod
    def _own_arrays(
        own_events: np.ndarray, valid: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        events = np.asarray(own_events)
        if events.shape != (8, 3, 12, 128, 128) or events.dtype != np.float32:
            raise ValueError("own_events must be float32 [8,3,12,128,128]")
        if not np.isfinite(events).all():
            raise ValueError("own_events must be finite")
        mask = np.ones(8, dtype=bool) if valid is None else np.asarray(valid)
        if mask.shape != (8,) or mask.dtype != np.bool_:
            raise ValueError("valid must be bool [8]")
        if not mask[-1] or (mask[:-1] & ~mask[1:]).any():
            raise ValueError("valid must be a contiguous suffix including current")
        return events, mask.copy()

    def predict_with_phase(
        self,
        own_events: np.ndarray,
        delta_t_s: float = 0.1,
        valid: np.ndarray | None = None,
        *,
        availability_lag_s: float = 0.0,
    ) -> dict[str, dict[str, float]]:
        """Run the frozen producers, normalizer and heads with no target input."""
        from e_jepa_ttc.simplex_t.phase import phase_to_ttc

        events, mask = self._own_arrays(own_events, valid)
        if not np.isfinite(delta_t_s) or delta_t_s <= 0:
            raise ValueError("delta_t_s must be finite and positive")
        if not np.isfinite(availability_lag_s) or availability_lag_s < 0:
            raise ValueError("availability_lag_s must be finite and nonnegative")

        # The feature cache was produced in canonical batches of sixteen.  Keep
        # that numerical contract even though transfer evaluation supplies H8.
        padded = np.zeros((16, 3, 12, 128, 128), dtype=np.float32)
        padded[-8:] = events
        event_tensor = torch.from_numpy(padded).to(self.device)
        delta = torch.full((16, 2), float(delta_t_s), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            raw = canonical_extract("H8_SEED7", self.models, event_tensor, delta)[-8:]
        raw[~mask] = 0
        anchor_us = 0
        available_us = -int(round(availability_lag_s * 1_000_000))
        xs = canonical_inputs(
            "H8_SEED7",
            raw,
            mask,
            _H8_LAGS_US,
            anchor_us,
            available_us,
            self.mean,
            self.scale,
        )
        xs = tuple(value.to(self.device) for value in xs)
        result: dict[str, dict[str, float]] = {}
        with torch.inference_mode():
            for seed, head in self.heads.items():
                output = head(*xs)
                phase = output["point_phase"]
                ttc = phase_to_ttc(phase)
                if phase.numel() != 1 or not bool(torch.isfinite(ttc).all()):
                    raise ValueError("Frozen H8 head emitted an invalid prediction")
                result[f"H8_seed{seed}"] = {
                    "phase": float(phase.item()),
                    "ttc": float(ttc.item()),
                }
        return result

    def predict(
        self,
        own_events: np.ndarray,
        delta_t_s: float = 0.1,
        valid: np.ndarray | None = None,
        *,
        availability_lag_s: float = 0.0,
    ) -> dict[str, float]:
        """Return native TTC predictions from the three frozen H8 seeds."""
        detailed = self.predict_with_phase(
            own_events,
            delta_t_s,
            valid,
            availability_lag_s=availability_lag_s,
        )
        return {name: values["ttc"] for name, values in detailed.items()}

    def garl_predict(self, sensor: np.ndarray) -> dict[str, float | list[float]]:
        """Run the published event-only LHR checkpoint and native TTC formula."""
        value = np.asarray(sensor)
        if value.shape != (40, 128, 128) or value.dtype != np.float32:
            raise ValueError("Garl sensor must be float32 [40,128,128]")
        if not np.isfinite(value).all():
            raise ValueError("Garl sensor must be finite")
        tensor = torch.from_numpy(value[None]).to(self.device)
        with torch.inference_mode():
            heights_tensor, _ = self.garl.forward_test(tensor)
        heights = heights_tensor.float().cpu().numpy()
        if heights.shape != (1, 2) or not np.isfinite(heights).all():
            raise ValueError("Published Garl emitted invalid visible heights")
        ttc = release_ttc(heights, float(self.garl.dT))
        return {"ttc": float(ttc[0]), "heights": heights[0].tolist()}
