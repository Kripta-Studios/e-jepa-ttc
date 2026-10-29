"""Compose admitted C2F CUDA graphs with the proven four-thread host path."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import psutil

from operational.efficient_context.common import ROOT, Lease, digest
from operational.train40_system import resource_monitor

if TYPE_CHECKING:
    from torch import Tensor, nn

    from operational.train40_system.coordination import DeviceTicket


def scalar_safe_module_tensor_sha256(module: nn.Module) -> str:
    """Hash all ordered state tensors, including zero-dimensional buffers."""
    import torch

    result = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        for value in (name, str(cpu.dtype), repr(tuple(cpu.shape))):
            encoded = value.encode("utf-8")
            result.update(len(encoded).to_bytes(8, "big"))
            result.update(encoded)
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        result.update(len(raw).to_bytes(8, "big"))
        result.update(raw)
    return result.hexdigest()


def planned_prefetch_type(base: type) -> type:
    """Tell the persistent input process the exact epoch order before each take."""

    class PlannedPrefetch(base):
        def take(self, order: Tensor, start: int, batch_size: int) -> DeviceTicket:
            if order is not self.order:
                if self.pending:
                    raise ValueError("Previous epoch still has unconsumed batches")
                if self.order is not None and (
                    self.expected != len(self.order) or self.scheduled != len(self.order)
                ):
                    raise ValueError("Previous epoch was not completely consumed")
            self.source.prepare_order(order, start, batch_size)
            return super().take(order, start, batch_size)

    return PlannedPrefetch


class C2FResourceMonitor(resource_monitor.ResourceMonitor):
    """Single-trainer monitor with bounded repair of dead-process scan races."""

    def _repair_inventory(
        self, allowed: bool, telemetry: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        reasons = list(telemetry.get("reasons", []))
        if "RESOURCE_MONITOR_INVENTORY_INCOMPLETE" not in reasons:
            return allowed, telemetry
        root = psutil.Process()
        fresh: dict[str, Any] | None = None
        for attempt in range(4):
            fresh = resource_monitor._collect_sample(root.pid, root.create_time(), self.output)
            admissible = (
                not fresh.get("inventory_unreadable_python_pids")
                and not fresh.get("other_heavy_processes")
                and int(fresh.get("tree_rss_bytes", resource_monitor.TREE_RSS_LIMIT_BYTES))
                < resource_monitor.TREE_RSS_LIMIT_BYTES
                and int(fresh.get("host_available_bytes", 0))
                >= resource_monitor.HOST_AVAILABLE_MIN_BYTES
                and int(fresh.get("disk_free_bytes", 0)) >= resource_monitor.DISK_FREE_MIN_BYTES
            )
            if admissible:
                reasons.remove("RESOURCE_MONITOR_INVENTORY_INCOMPLETE")
                repaired = {
                    **telemetry,
                    "reasons": reasons,
                    "inventory_repair": {
                        "attempt": attempt + 1,
                        "processes_seen": fresh.get("processes_seen"),
                        "python_cmdline_reads": fresh.get("python_cmdline_reads"),
                    },
                }
                return not reasons, repaired
            if attempt < 3:
                time.sleep(0.01 * (2**attempt))
        return allowed, {**telemetry, "inventory_repair": fresh}

    def guard(self, output: Path) -> tuple[bool, dict[str, Any]]:
        """Retain every frozen check and repair only a transient inventory race."""
        allowed, telemetry = super().guard(output)
        return self._repair_inventory(allowed, telemetry)


def verify_admission(output: Path) -> dict[str, Any]:
    """Authenticate this wrapper, the independent core, and real-CUDA evidence."""
    from operational.train40_system.contracts import read, verify_sources

    path = output / "C2F_GRAPH_REPLAY_FREEZE.json"
    freeze = read(path)
    verify_sources(freeze)
    if freeze["wrapper_engine_sha256"] != digest(Path(__file__)):
        raise ValueError("C2F graph wrapper differs from its freeze")
    core = ROOT / "operational/train40_system/c2f_graph_core.py"
    if freeze["engine_sha256"] != digest(core):
        raise ValueError("C2F graph core differs from its freeze")
    admission = output / "c2f_graph_launch_admission/REAL_GRAPH_LAUNCH_ADMISSION.json"
    if freeze["real_graph_launch_admission_sha256"] != digest(admission):
        raise ValueError("C2F real graph admission differs from its freeze")
    backend_source = Path(str(freeze["backend_source"]))
    if digest(backend_source) != freeze["backend_source_sha256"]:
        raise ValueError("Installed CUDA graph backend differs from its freeze")
    return cast(dict[str, Any], freeze)


def run(output: Path, arm: str) -> None:
    """Run the unchanged C2F science loop with admitted systems bindings."""
    if arm != "c2f":
        raise ValueError("C2F graph replay accepts only the existing C2F arm")
    verify_admission(output)

    import e_jepa_ttc.training.causal_scale_eap as training
    from operational.train40_system import c2f_graph_core
    from operational.train40_system.process_inputs_fast_4 import ProcessInputs

    originals = {
        "SealedInputs": c2f_graph_core.SealedInputs,
        "DevicePrefetch": c2f_graph_core.DevicePrefetch,
        "resource_guard": c2f_graph_core.resource_guard,
        "scan_snapshot": c2f_graph_core.scan_snapshot,
    }
    original_hasher = training._module_tensor_sha256
    sources: list[ProcessInputs] = []
    monitor = C2FResourceMonitor(output)

    def inputs(path: Path) -> ProcessInputs:
        source = ProcessInputs(path)
        sources.append(source)
        return source

    try:
        monitor.start()
        c2f_graph_core.SealedInputs = cast(Any, inputs)
        c2f_graph_core.DevicePrefetch = cast(
            Any, planned_prefetch_type(cast(type, originals["DevicePrefetch"]))
        )
        c2f_graph_core.resource_guard = monitor.guard
        c2f_graph_core.scan_snapshot = monitor.snapshot
        training._module_tensor_sha256 = scalar_safe_module_tensor_sha256
        c2f_graph_core.run(output, arm)
    finally:
        c2f_graph_core.SealedInputs = cast(Any, originals["SealedInputs"])
        c2f_graph_core.DevicePrefetch = cast(Any, originals["DevicePrefetch"])
        c2f_graph_core.resource_guard = cast(Any, originals["resource_guard"])
        c2f_graph_core.scan_snapshot = cast(Any, originals["scan_snapshot"])
        training._module_tensor_sha256 = original_hasher
        try:
            for source in sources:
                source.close()
        finally:
            monitor.close()


if __name__ == "__main__":
    from operational.train40_system.data_audit import OUTPUT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--arm", choices=("c2f",), required=True)
    arguments = parser.parse_args()
    with Lease(arguments.output.resolve()):
        run(arguments.output.resolve(), arguments.arm)
