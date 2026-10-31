"""Observe PAIR133 and PHASE17 without copying sensor pixels or producer diagnostics."""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import torch

from operational.phase_timewarp.capture import PAIRS
from operational.rgb_port import transfer as engine


@contextmanager
def capture_compact(models: dict) -> Iterator[tuple[list, dict]]:
    """Keep original forwards/batches unchanged; only copy compact tensors after execution."""
    original = engine._expert_block
    calls: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    active = {"index": -1}

    def hook(name: str) -> Callable:
        def observe(module: torch.nn.Module, args: tuple, output: torch.Tensor) -> None:
            del module
            key = f"call{active['index']}.{name}"
            arrays[key + ".input133"] = args[0].features.detach().cpu().numpy().copy()
            arrays[key + ".raw_phase"] = output.detach().cpu().numpy().copy()

        return observe

    handles = [models[name].register_forward_hook(hook(name)) for name in PAIRS]

    def expert(model_map: dict, prefix: str, pixels: torch.Tensor, delta: torch.Tensor) -> tuple:
        index = len(calls)
        active["index"] = index
        key = f"call{index}"
        arrays[key + ".delta"] = delta.detach().cpu().numpy().copy()
        record = {"call_id": index, "prefix": prefix, "sensor_shape": list(pixels.shape)}
        calls.append(record)
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
            record["wall_seconds"] = time.perf_counter() - start

    try:
        with patch.object(engine, "_expert_block", expert):
            yield calls, arrays
    finally:
        for handle in handles:
            handle.remove()


def bind_compact_road(calls: list, arrays: dict, candidates: dict, accepted: dict) -> None:
    """Map actual batch dimensions to physical observations without needing pixel values."""
    offsets = {"event": 0, "rgb": 0}
    admitted = {m: {item["anchor_us"] for item in accepted[m]} for m in offsets}
    for call in calls:
        modality = "event" if call["prefix"] == "E" else "rgb"
        if call["prefix"] not in ("E", "R"):
            raise ValueError("Unknown producer modality")
        count, temporal = call["sensor_shape"][:2]
        items = candidates[modality][offsets[modality] : offsets[modality] + count]
        if len(items) != count:
            raise ValueError("Unmapped compact producer batch")
        delta = arrays[f"call{call['call_id']}.delta"]
        call["observations"] = []
        for index, source in enumerate(items):
            item = {k: v for k, v in source.items() if k not in {"raw17", "expert3", "accepted"}}
            if len(item["timestamps_us"]) != temporal:
                raise ValueError("Temporal sensor shape differs from observation clocks")
            expected = np.asarray(np.diff(item["timestamps_us"]) / 1e6, np.float32)
            np.testing.assert_array_equal(delta[index], expected)
            item.update(
                batch_index=index,
                observation_id=f"{modality}:{item['anchor_us']}",
                original_index=offsets[modality] + index,
                source_accepted=item["anchor_us"] in admitted[modality],
            )
            call["observations"].append(item)
        call["dt_construction"] = "float64_seconds"
        offsets[modality] += count
        pair = "PAIR_E_MATCHED" if modality == "event" else "PAIR_R"
        if call["status"] != "accepted" and f"call{call['call_id']}.{pair}.input133" in arrays:
            raise ValueError("Original rejection after PAIR needs explicit replay support")
    if any(offsets[m] != len(candidates[m]) for m in offsets):
        raise ValueError("Prepared observations remain unmapped")
