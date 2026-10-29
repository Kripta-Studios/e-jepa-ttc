"""Native F_TRUE inference with an explicit scoped CUDA-median warning policy."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import torch

from operational.range_transfer.common import freeze, sha, write
from operational.rgb_port.cost_routes import RouteRuntime

from . import fusion_cost
from .common import RUN
from .fusion_cost_native import OUT as NATIVE
from .fusion_cost_native import admit, native_context, qa
from .train import committed_sources

OUT = RUN / "fusion/cost/raw_e2e_native_v2"


@contextmanager
def native_median_policy() -> Iterator[None]:
    """Keep deterministic algorithms enabled but warn for unsupported native CUDA median."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warning = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warning)


class NativeRuntime(RouteRuntime):
    """Preserve the original full-fusion computation, weights, inputs and devices."""

    def full_fusion(self, index: int, fit_id: str = "F_TRUE") -> torch.Tensor:
        """The historical head uses median values; PyTorch also returns unused indices."""
        with native_median_policy():
            return super().full_fusion(index, fit_id)


def run() -> dict:
    """Retain both failed admissions and require unchanged native-TTC parity tolerance."""
    record = admit()
    qa(record)
    source = "operational/joint_measurement/fusion_cost_native_v2.py"
    from .common import ROOT

    sources = {source: sha(ROOT / source)}
    committed_sources(sources)
    admission = {
        "native_source_admission_sha256": sha(NATIVE / "NATIVE_SOURCE_ADMISSION.json"),
        "native_cpu_qa_sha256": sha(NATIVE / "CPU_QA.json"),
        "failed_v1_cost_registration_sha256": sha(NATIVE / "REGISTRATION.json"),
        "sources": sources,
        "torch_policy": (
            "Within historical RouteRuntime.full_fusion only: deterministic algorithms "
            "enabled, warn_only=True; restore both previous flags even after exception"
        ),
        "reason": "CUDA median with indices has no strict deterministic implementation",
        "baseline_computation_changed": False,
        "training_policy_changed": False,
        "optimizer_updates": 0,
        "rtol": 1e-5,
        "atol": 1e-4,
        "claim": "Native inference timings; not a bitwise determinism guarantee",
    }
    freeze(OUT / "RUNTIME_ADMISSION.json", admission)
    with (
        native_context(record),
        patch.object(fusion_cost, "OUT", OUT),
        patch.object(fusion_cost, "RouteRuntime", NativeRuntime),
    ):
        result = fusion_cost.measure()
    receipt = {
        "runtime_admission_sha256": sha(OUT / "RUNTIME_ADMISSION.json"),
        "cost_registration_sha256": sha(OUT / "REGISTRATION.json"),
        "cost_result": result,
    }
    if result["status"] == "COMPLETE":
        freeze(OUT / "RUNTIME_RESULT.json", receipt)
    else:
        write(OUT / "PARTIAL.json", receipt)
    return receipt


if __name__ == "__main__":
    print(run())
