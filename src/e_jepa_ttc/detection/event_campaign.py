"""Explicit event or matched RGB-capacity inputs for a fixed-budget S4 continuation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
import yaml
from torch.nn import functional

from e_jepa_ttc.detection.eap_rgb_data import sha256_file
from e_jepa_ttc.detection.event_detector import EventTemporalDetectorAdapter
from e_jepa_ttc.detection.event_fusion import (
    CausalEventFeatureFusion,
    EventFusionObservables,
    EventFusionSpec,
)
from e_jepa_ttc.detection.event_panel_cache import CausalEventPanelCache
from e_jepa_ttc.detection.temporal_campaign import TemporalCampaignBatch
from e_jepa_ttc.detection.temporal_detector import ProjectedFeatureBackend
from e_jepa_ttc.detection.temporal_training import TemporalTrainingConfig, seeded_fusion


class EventCampaignExtension:
    """Own only sensor input preparation; targets remain with the original trainer."""

    def __init__(self, path: Path, config: TemporalTrainingConfig) -> None:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        expected = {
            "schema",
            "control",
            "input_mode",
            "cache_root",
            "cache_contract_sha256",
            "cache_index_sha256",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("Event training configuration keys differ")
        if raw["schema"] != "eap_detection_event_screen_v1":
            raise ValueError("Unknown event training schema")
        if raw["control"] not in ("F01", "F11"):
            raise ValueError("This extension admits only explicit event-branch controls")
        if raw["input_mode"] not in ("events", "rgb_capacity"):
            raise ValueError("Unknown event/capacity input mode")
        if (raw["control"] == "F11") != (config.temporal.history_limit > 0):
            raise ValueError("Event experiment and RGB history config disagree")
        if config.baseline.evaluation.full_selection_updates:
            raise ValueError("S4 panel cache cannot cover full-selection evaluation")
        self.control = raw["control"]
        self.input_mode = raw["input_mode"]
        self.config = config
        self.cache = CausalEventPanelCache(
            raw["cache_root"],
            contract_sha256=raw["cache_contract_sha256"],
            index_sha256=raw["cache_index_sha256"],
            manifest_sha256=config.baseline.data.expected_manifest_sha256,
        )
        metadata = Path(config.baseline.data.root) / "data/train.parquet"
        if sha256_file(metadata) != self.cache.contract["metadata_sha256"]:
            raise ValueError("Event cache and RGB metadata differ")
        frame = pd.read_parquet(metadata, columns=["sample_token", "rgb_exposure_end_timestamp_us"])
        if frame.sample_token.duplicated().any():
            raise ValueError("RGB exposure identities are not unique")
        self.cutoffs = {
            str(token): int(cutoff)
            for token, cutoff in frame.itertuples(index=False, name=None)
        }
        self.spec = EventFusionSpec(input_height=96, input_width=160)
        self.provenance = {
            "schema": raw["schema"],
            "config_sha256": sha256_file(path),
            "config": raw,
            "spec": asdict(self.spec),
            "rgb_capacity_transform": (
                "bilinear RGB 160x96, channels [RGB,RGB,RGB,mean]; "
                "constant availability count=1,duration=50000; no event values"
            )
            if self.input_mode == "rgb_capacity"
            else None,
        }

    def build_adapter(self, backend: torch.nn.Module) -> EventTemporalDetectorAdapter:
        """Initialize new parameters without consuming the baseline RNG stream."""
        seed = self.config.baseline.seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            event = CausalEventFeatureFusion(self.spec)
        return EventTemporalDetectorAdapter(
            cast(ProjectedFeatureBackend, backend),
            seeded_fusion(seed),
            event,
            producer_version=self.config.temporal.producer_version,
            maximum_history_age_us=self.config.temporal.maximum_history_age_us,
        )

    def model_inputs(self, batch: TemporalCampaignBatch, *, role: str) -> dict[str, Any]:
        """Return sensor-only arguments in RGB order, with no annotation-bearing input."""
        tokens = tuple(identity.frame_id for identity in batch.temporal_context.current)
        if any(
            token not in self.cache.rows or self.cache.rows[token]["role"] != role
            for token in tokens
        ):
            raise ValueError("Sensor batch crosses its frozen population role")
        device = batch.current_images.device
        if self.input_mode == "events":
            inputs = self.cache.load_batch(
                [(token, self.cutoffs[token]) for token in tokens], role=role, device=device
            ).model_kwargs()
        else:
            # Extra learned capacity receives current RGB only, at exactly the event
            # encoder shape. Constant availability carries no event-derived signal.
            rgb = functional.interpolate(
                batch.current_images, size=(96, 160), mode="bilinear", align_corners=False
            )
            voxel = torch.cat((rgb, rgb, rgb, rgb.mean(dim=1, keepdim=True)), dim=1)
            inputs = {
                "event_tokens": tokens,
                "event_voxel": voxel,
                "event_mask": torch.ones(len(tokens), dtype=torch.bool, device=device),
                "event_observables": EventFusionObservables(
                    torch.ones(len(tokens), device=device),
                    torch.full((len(tokens),), 50_000.0, device=device),
                ),
            }
        return {"event_control": self.control, **inputs}
