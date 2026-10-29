"""Repeat the bounded C1 graph admission with explicit microbatch step boundaries."""

from __future__ import annotations

import torch

from operational.range_transfer.common import freeze, read, sha

from . import geometry_graph_probe as probe
from .common import ROOT, RUN
from .train import committed_sources

OUT = RUN / "runtime/geometry_graph_probe_v3"
PREVIOUS = RUN / "runtime/geometry_graph_probe_v2"


def register() -> dict:
    """Keep all numerical/speed gates; retain the preceding buffer overwrite failure."""
    prior = read(PREVIOUS / "RESULT.json")
    if prior["status"] != "REJECTED" or "overwritten" not in prior.get("traceback", ""):
        raise ValueError("This probe addresses only the observed graph buffer overwrite")
    record = read(PREVIOUS / "REGISTRATION.json")
    for path, expected in record["sources"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Original graph probe sources changed")
    own = "operational/joint_measurement/geometry_graph_step_probe.py"
    record["sources"][own] = sha(ROOT / own)
    committed_sources(record["sources"])
    record["supersedes_unexecuted_registration_sha256"] = None
    record["prior_rejected_result_sha256"] = sha(PREVIOUS / "RESULT.json")
    record["revision_reason"] = (
        "Declare each model invocation as a CUDA graph step before its matching backward; "
        "same data, FP32 operations, loss, optimizer-free probe and numerical gates"
    )
    freeze(OUT / "REGISTRATION.json", record)
    return record


def install_graph(model: torch.nn.Module) -> None:
    """Mark the independent microbatch boundary without changing parameter identities."""
    original = model.forward
    compiled = torch.compile(original, backend="cudagraphs", dynamic=False, fullgraph=False)

    def forward(inputs: torch.Tensor, delta_t_s: torch.Tensor) -> dict[str, torch.Tensor]:
        torch.compiler.cudagraph_mark_step_begin()
        return compiled(inputs, delta_t_s)

    model._v18_eager_forward = original
    model.forward = forward


if __name__ == "__main__":
    probe.OUT = OUT
    probe.register = register
    probe.install_graph = install_graph
    print(probe.run())
