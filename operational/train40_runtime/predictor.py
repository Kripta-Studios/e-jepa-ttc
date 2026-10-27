"""Sensor-to-TTC adapter using opt-in persistent, parity-gated components."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from operational.rgb_port import transfer as oracle_engine
from operational.train40_runtime.context import ContextRuntime
from operational.train40_runtime.inputs import PersistentSensorInputs
from operational.train40_runtime.producers import ProducerRuntime
from scripts.final_train40_transfer import reject_targets


class PersistentTrain40Predictor:
    """Preserve query ROI and causal histories; never cache TTC predictions.

    Loading models and constructing graph signatures are service startup costs.
    The public ``predict`` clock includes raw IO/cache lookup, representation,
    transfers, every model stage and a completed host TTC output.
    """

    def __init__(
        self,
        models: Mapping[str, Any],
        normalizers: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
        device: str = "cuda",
        *,
        graphs: bool = True,
        context_graphs: bool = False,
        batch_rgb: bool = False,
        block_graphs: bool = False,
        reuse_encodings: bool = False,
        parallel_branches: bool = False,
        rgb_workers: int = 1,
        synchronize_stages: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.inputs = PersistentSensorInputs(rgb_workers=rgb_workers)
        producer_options = {"graphs": graphs}
        if block_graphs:
            producer_options["block_graphs"] = True
        if reuse_encodings:
            producer_options["reuse_encodings"] = True
        self.producers = ProducerRuntime(dict(models), self.device, **producer_options)
        self.context = ContextRuntime(
            dict(models), dict(normalizers), self.device, graphs=context_graphs
        )
        self.synchronize_stages = synchronize_stages
        self.batch_rgb = batch_rgb
        if parallel_branches and not (block_graphs and batch_rgb and self.device.type == "cuda"):
            raise ValueError("Parallel branches require CUDA block graphs and batched RGB")
        self.parallel_branches = parallel_branches
        self._warm_signatures: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        self._streams = (
            (torch.cuda.Stream(device=self.device), torch.cuda.Stream(device=self.device))
            if parallel_branches
            else None
        )

    def _stamp(self) -> float:
        if self.synchronize_stages and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def close(self) -> None:
        """Release sensor handles and bounded caches."""
        self.inputs.close()

    def predict(self, row: dict[str, Any]) -> dict[str, Any]:
        """Compute a new query from its raw causal observations and current ROI."""
        start = time.perf_counter()
        reject_targets(row)
        raw = self.inputs.prepare_event(row)
        event_end = self._stamp()
        rgb, reason = self.inputs.raw_rgb_observations(row)
        prepared = self._stamp()
        result = self.infer_prepared(row, torch.from_numpy(raw["own_events"]), rgb, reason)
        result["seconds"].update(
            preprocess=prepared - start,
            event_prepare=event_end - start,
            rgb_prepare=prepared - event_end,
            total=time.perf_counter() - start,
        )
        result.update(
            input_boundary="raw_sensor_io_and_cache_lookup",
            sensor_cache=dict(self.inputs.stats),
            preprocess_detail_seconds=dict(self.inputs.last_profile),
        )
        return result

    def infer_prepared(
        self,
        row: dict[str, Any],
        event_tensor: torch.Tensor,
        rgb: list[tuple[np.ndarray, dict[str, Any]]],
        reason: str = "",
    ) -> dict[str, Any]:
        """Consume prepared event/RGB tensors; caller must time sensor preparation.

        The streaming runner uses this boundary after timed raw packet ingestion.
        This method's own duration is never a sensor-to-TTC latency claim.
        """
        start = time.perf_counter()
        reject_targets(row)
        marks: dict[str, float] = {}
        previous = start

        def finish(name: str) -> None:
            nonlocal previous
            now = self._stamp()
            marks[name] = now - previous
            previous = now

        with torch.inference_mode():
            sensor = event_tensor.to(self.device)
            delta = torch.full((len(oracle_engine.EVENT_LAGS_US), 2), 0.1, device=self.device)
            finish("event_h2d")
            rgb_raw, rgb_experts, metadata = [], [], []
            if self.batch_rgb and rgb:
                # Keep observation order. This opt-in changes producer batch
                # shape and requires its own end-to-end numerical parity gate.
                shapes = {pixels.shape for pixels, _ in rgb}
                if len(shapes) != 1:
                    raise ValueError("Batched RGB requires equal native observation shapes")
                rsensor = torch.stack([torch.from_numpy(pixels) for pixels, _ in rgb]).to(
                    self.device
                )
                rdelta = torch.tensor(
                    [item["delta_t_s"] for _, item in rgb],
                    dtype=torch.float32,
                    device=self.device,
                )
                if self.parallel_branches:
                    signature = (tuple(sensor.shape), tuple(rsensor.shape))
                    if signature not in self._warm_signatures:
                        # Cold capture is serialized, counted, and never overlaps
                        # noncaptured CUDA work from the other branch.
                        self.producers.expert_block("E", sensor, delta)
                        self.producers.expert_block("R", rsensor, rdelta)
                        self._warm_signatures.add(signature)
                    assert self._streams is not None
                    current = torch.cuda.current_stream(self.device)
                    event_stream, rgb_stream = self._streams
                    event_stream.wait_stream(current)
                    rgb_stream.wait_stream(current)
                    with torch.cuda.stream(event_stream):
                        event_ticket = self.producers.expert_block_async("E", sensor, delta)
                    with torch.cuda.stream(rgb_stream):
                        rgb_ticket = self.producers.expert_block_async("R", rsensor, rdelta)
                    current.wait_stream(event_stream)
                    current.wait_stream(rgb_stream)
                    self.producers.validate_async_blocks(event_ticket, rgb_ticket)
                    event_raw, event_experts = event_ticket.raw, event_ticket.experts
                    rraw, rexperts = rgb_ticket.raw, rgb_ticket.experts
                else:
                    event_raw, event_experts = self.producers.expert_block("E", sensor, delta)
                    rraw, rexperts = self.producers.expert_block("R", rsensor, rdelta)
                rgb_raw.extend(rraw.unbind(0))
                rgb_experts.extend(rexperts.unbind(0))
                metadata.extend(item for _, item in rgb)
            else:
                event_raw, event_experts = self.producers.expert_block("E", sensor, delta)
                for pixels, item in rgb:
                    rsensor = torch.from_numpy(pixels)[None].to(self.device)
                    rdelta = torch.tensor(
                        [item["delta_t_s"]], dtype=torch.float32, device=self.device
                    )
                    rraw, rexperts = self.producers.expert_block("R", rsensor, rdelta)
                    rgb_raw.append(rraw[0])
                    rgb_experts.append(rexperts[0])
                    metadata.append(item)
            finish("all_producers_pairs_and_rgb_h2d")
            query_us = int(row["anchor_us"])
            result = self.context.predict(
                event_raw,
                event_experts,
                [query_us - int(lag) for lag in oracle_engine.EVENT_LAGS_US],
                torch.stack(rgb_raw) if rgb_raw else None,
                torch.stack(rgb_experts) if rgb_experts else None,
                metadata,
                query_us,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            finish("context_head_d2h")
        total = time.perf_counter() - start
        return {
            **result,
            "rgb_unavailable_reason": reason,
            "seconds": {
                "total": total,
                **marks,
            },
            "stage_timing": "synchronized" if self.synchronize_stages else "host_spans",
            "input_boundary": "prepared_event_and_rgb_tensors",
            "targets_read": False,
            "optimizer_updates": 0,
        }
