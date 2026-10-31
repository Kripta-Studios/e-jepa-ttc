"""Owned producer tensors and stage traces with explicit batch/observation identity."""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTCOutput
from operational.rgb_port import transfer as engine

PRODUCERS = ("E_A5_MATCHED", "E_C2F_MATCHED", "R_A5", "R_C2F")
PAIRS = ("PAIR_E_MATCHED", "PAIR_R")
FIELDS = (
    "pair_log_height_ratio",
    "pair_inverse_ttc",
    "inverse_ttc_mean",
    "ttc_mean_seconds",
    "known_mask",
    "sensor_support",
    "ttc_log_variance",
)


@contextmanager
def capture_calls(models: dict, device: str) -> Iterator[tuple[list[dict], dict]]:
    """Observe the unchanged expert call; the temporary patch has one process owner."""
    original = engine._expert_block
    calls, arrays, handles = [], {}, []
    active = {"index": -1}

    def hook(name: str) -> Callable:
        def observe(
            module: torch.nn.Module, args: tuple, output: torch.Tensor | CausalScaleTTCOutput
        ) -> None:
            del module
            prefix = f"call{active['index']}.{name}"
            if isinstance(output, torch.Tensor):
                arrays[prefix + ".input133"] = args[0].features.detach().cpu().numpy().copy()
                arrays[prefix + ".raw_phase"] = output.detach().cpu().numpy().copy()
            else:
                for field in FIELDS:
                    arrays[prefix + "." + field] = (
                        getattr(output, field).detach().cpu().numpy().copy()
                    )

        return observe

    for name in (*PRODUCERS, *PAIRS):
        if name in models:
            handles.append(models[name].register_forward_hook(hook(name)))

    def expert(
        model_map: dict, prefix: str, pixels: torch.Tensor, delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        index = len(calls)
        active["index"] = index
        key = f"call{index}"
        arrays[key + ".pixels"] = pixels.detach().cpu().numpy().copy()
        arrays[key + ".delta"] = delta.detach().cpu().numpy().copy()
        record = {"call_id": index, "prefix": prefix, "sensor_shape": list(pixels.shape)}
        calls.append(record)
        if device == "cuda":
            torch.cuda.synchronize()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        start = time.perf_counter()
        try:
            raw, experts = original(model_map, prefix, pixels, delta)
            arrays[key + ".raw17"] = raw.detach().cpu().numpy().copy()
            arrays[key + ".expert3"] = experts.detach().cpu().numpy().copy()
            record["status"] = "accepted"
            return raw, experts
        except Exception as error:
            record.update(status="rejected", error=repr(error))
            raise
        finally:
            if device == "cuda":
                end.record()
                torch.cuda.synchronize()
                record["cuda_elapsed_ms"] = begin.elapsed_time(end)
            record["wall_seconds"] = time.perf_counter() - start

    try:
        with patch.object(engine, "_expert_block", expert):
            yield calls, arrays
    finally:
        for handle in handles:
            handle.remove()


def bind_calls(
    calls: list[dict],
    arrays: dict,
    candidates: dict,
    accepted: dict,
    *,
    dt_mode: str = "float64_seconds",
) -> None:
    """Bind every batch member to its actual prepared observation, preserving rejections."""
    offset = {"event": 0, "rgb": 0}
    accepted_anchors = {m: {item["anchor_us"] for item in accepted[m]} for m in offset}
    for call in calls:
        modality = "event" if call["prefix"] == "E" else "rgb"
        key = f"call{call['call_id']}"
        pixels, delta = arrays[key + ".pixels"], arrays[key + ".delta"]
        count = len(pixels)
        items = candidates[modality][offset[modality] : offset[modality] + count]
        if len(items) != count:
            raise ValueError("Producer batch cannot be mapped to prepared observations")
        call["observations"] = []
        for batch_index, source in enumerate(items):
            item = {k: v for k, v in source.items() if k not in {"raw17", "expert3", "accepted"}}
            if dt_mode == "float64_seconds":
                measured = np.asarray(np.diff(item["timestamps_us"]) / 1e6, np.float32)
            elif dt_mode == "float32_microseconds":
                measured = np.diff(item["timestamps_us"]).astype(np.float32) / 1e6
            else:
                raise ValueError("Unknown original clock-conversion path")
            np.testing.assert_array_equal(delta[batch_index], measured)
            if pixels.shape[1] != len(item["timestamps_us"]):
                raise ValueError("Producer temporal dimension differs from source clocks")
            item.update(
                batch_index=batch_index,
                observation_id=f"{modality}:{item['anchor_us']}",
                original_index=offset[modality] + batch_index,
                source_accepted=item["anchor_us"] in accepted_anchors[modality],
            )
            call["observations"].append(item)
        call["dt_construction"] = dt_mode
        offset[modality] += count
    if any(offset[m] != len(candidates[m]) for m in offset):
        raise ValueError("Unmapped prepared observations remain")
