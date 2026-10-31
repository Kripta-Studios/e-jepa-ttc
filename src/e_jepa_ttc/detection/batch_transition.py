"""Explicit batch transition preserving sample order and LR by image exposure."""

# ruff: noqa: ANN401 -- dynamic arguments mirror the verified frozen checkpoint loader.

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from e_jepa_ttc.detection.campaign_training import (
    UpdateLRScheduler,
    canonical_sha256,
    load_campaign_checkpoint,
    load_campaign_config,
    sha256_file,
)


def build_plan(source_update: int, total_units: int, milestones: list[int]) -> dict[str, Any]:
    """Map physical AdamW steps to the original four-image exposure clock."""
    if not 0 < source_update < min(milestones) or sorted(set(milestones)) != milestones:
        raise ValueError("Transition must precede the registered selection milestones")
    if milestones[-1] != total_units:
        raise ValueError("Final milestone must end the original image budget")
    segments = []
    old, actual = source_update, source_update
    for end in milestones:
        count = (end - old + 1) // 2
        segments.append(
            {
                "start_update": actual,
                "end_update": actual + count,
                "start_units": old,
                "end_units": end,
            }
        )
        old, actual = end, actual + count
    return {
        "schema": "phase_batch8_exposure_plan_v1",
        "source_update": source_update,
        "source_batch_size": 4,
        "batch_size": 8,
        "original_total_updates": total_units,
        "total_exposures": 4 * total_units,
        "total_updates": actual,
        "segments": segments,
        "selection_update_map": {str(s["end_units"]): s["end_update"] for s in segments},
    }


def exposure_units(plan: dict[str, Any], update: int) -> int:
    """Return cumulative exposures divided by four at an actual optimizer step."""
    if not 0 <= update <= plan["total_updates"]:
        raise ValueError("Optimizer step is outside the registered budget")
    if update <= plan["source_update"]:
        return update
    for segment in plan["segments"]:
        if segment["start_update"] < update <= segment["end_update"]:
            return min(
                segment["end_units"],
                segment["start_units"] + 2 * (update - segment["start_update"]),
            )
    raise ValueError("Unmapped optimizer step")


def batch_size_at(plan: dict[str, Any], update: int) -> int:
    """Use batch four only when required to land exactly on a population milestone."""
    if update < 1:
        raise ValueError("Batch updates are one-indexed")
    return 4 * (exposure_units(plan, update) - exposure_units(plan, update - 1))


class ExposureLRScheduler:
    """Preserve the original LR function while reducing the number of AdamW steps."""

    def __init__(
        self,
        optimizer: Any,
        *,
        plan: dict[str, Any],
        total_updates: int,
        kind: str,
        warmup_fraction: float,
        minimum_ratio: float,
    ) -> None:
        if total_updates != plan["total_updates"]:
            raise ValueError("Configuration and batch plan budgets differ")
        self.plan = plan
        self.update = 0
        self.original = UpdateLRScheduler(
            optimizer,
            total_updates=plan["original_total_updates"],
            kind=kind,
            warmup_fraction=warmup_fraction,
            minimum_ratio=minimum_ratio,
        )

    def apply(self, update: int) -> list[float]:
        """Set the original LR at the cumulative image exposure, without LR scaling."""
        result = self.original.apply(exposure_units(self.plan, update))
        self.update = update
        return result

    def state_dict(self) -> dict[str, Any]:
        """Save the physical step, exposure clock and complete original LR state."""
        return {
            "schema": "phase_exposure_scheduler_v1",
            "plan_sha256": canonical_sha256(self.plan),
            "update": self.update,
            "original": self.original.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a bound new state or the specifically registered source state."""
        if state.get("schema") == "phase_exposure_scheduler_v1":
            if state["plan_sha256"] != canonical_sha256(self.plan):
                raise ValueError("Exposure scheduler plan differs")
            update, original = state["update"], state["original"]
            if original["update"] != exposure_units(self.plan, update):
                raise ValueError("Scheduler physical and exposure clocks disagree")
        else:
            update, original = state["update"], state
            if update != self.plan["source_update"]:
                raise ValueError("Only the registered old scheduler state can be migrated")
        self.original.load_state_dict(original)
        self.update = update


def load_transition_checkpoint(path: Path, *, receipt: dict[str, Any], **kwargs: Any) -> Any:
    """Verify the historical producer, then preserve every state through a declared change."""
    if path.resolve() != Path(receipt["source_checkpoint"]).resolve():
        return load_campaign_checkpoint(path, **kwargs)
    if sha256_file(path) != receipt["source_checkpoint_sha256"]:
        raise ValueError("Batch-transition source checkpoint bytes differ")
    loader_path = path.parent.parent / "code/scripts/infer_detection_campaign.py"
    spec = importlib.util.spec_from_file_location("batch_transition_source_verifier", loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(loader_path)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    payload, _, _ = loader.verify_checkpoint_runtime(path)
    old_config = load_campaign_config(Path(receipt["source_config"]))
    old, new = asdict(old_config), asdict(kwargs["config"])
    expected = json.loads(json.dumps(old))
    expected["experiment_id"] = receipt["experiment_id"]
    expected["optimization"].update(
        batch_size=8, updates=receipt["plan"]["total_updates"], checkpoint_interval=512
    )
    expected["evaluation"]["updates"] = [
        receipt["plan"]["selection_update_map"][str(u)] for u in old["evaluation"]["updates"]
    ]
    expected["evaluation"]["full_selection_updates"] = [
        receipt["plan"]["selection_update_map"][str(u)]
        for u in old["evaluation"]["full_selection_updates"]
    ]
    if canonical_sha256(new) != canonical_sha256(expected):
        raise ValueError("Batch transition includes an undeclared configuration change")
    if payload["config_sha256"] != old_config.sha256:
        raise ValueError("Source config no longer matches the original checkpoint")
    if payload["update"] != receipt["plan"]["source_update"] or payload["best_metric"] is not None:
        raise ValueError("Transition checkpoint must precede the first model selection")
    kwargs["config"] = old_config
    kwargs["runtime_provenance"] = payload["runtime_provenance"]
    del payload
    return load_campaign_checkpoint(path, **kwargs)
