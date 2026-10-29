"""Optional fixed-shape CUDA-graphs experiment; never changes frozen weights."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn

from operational.efficient_context.common import digest
from operational.evttc_transfer.models import FrozenModels
from operational.train40_system.durable_io import atomic_json
from operational.ttc_revision.benchmark import require_no_other_python_gpu, selected_rows
from operational.ttc_revision.inputs import EventPreparer
from operational.ttc_revision.runtime import H8Runtime


class CompiledH8Runtime(H8Runtime):
    """Use PyTorch's graph backend; keep original modules available for comparison.

    This does not generate custom CUDA kernels or change precision. Compilation
    can split on validation guards; actual benefit must be measured, not assumed.
    Model/data validation and finite-output checks remain in the execution path.
    """

    def __init__(self, frozen: FrozenModels, *, batch: int = 8, on_device: bool = True) -> None:
        if frozen.device.type != "cuda":
            raise ValueError("CUDA graph backend requires a CUDA model")
        compiled = copy.copy(frozen)
        compiled.models = {
            name: cast(
                nn.Module,
                torch.compile(model, backend="cudagraphs", dynamic=False, fullgraph=False),
            )
            if name != "PAIR"
            else model
            for name, model in frozen.models.items()
        }
        compiled.heads = {
            seed: cast(
                nn.Module,
                torch.compile(model, backend="cudagraphs", dynamic=False, fullgraph=False),
            )
            for seed, model in frozen.heads.items()
        }
        super().__init__(compiled, batch=batch, on_device=on_device)


def probe(manifest_path: Path, output: Path) -> None:
    """Measure the wrapper only, and disclose compilation separately from inference."""
    require_no_other_python_gpu()
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "COMPILED_PROBE.json"
    if receipt.exists():
        raise FileExistsError("preserve existing compiler experiment")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = selected_rows(json.loads(manifest_path.read_text(encoding="utf-8"))["rows"])
    frozen = FrozenModels(Path("artifacts/train40_system_20261005"), "cuda")
    eager = H8Runtime(frozen, batch=8, on_device=True)
    compiled = CompiledH8Runtime(frozen)
    canonical = H8Runtime(frozen)
    compiled16 = CompiledH8Runtime(frozen, batch=16, on_device=False)
    record = {
        "status": "RUNNING",
        "manifest_sha256": digest(manifest_path),
        "source_sha256": digest(Path(__file__)),
        "model_bindings": frozen.bindings,
        "scope": (
            "three-head inference wrapper only; prepared CPU tensors; "
            "load and compilation excluded from hot latency"
        ),
    }
    atomic_json(receipt, record)
    measurements, admission = [], []
    generator = np.random.default_rng(20261009)
    try:
        with EventPreparer() as preparer:
            for row in rows:
                value = preparer.prepare(row)["own_events"]
                expected = eager.predict(value)
                begun = time.perf_counter()
                for _ in range(3):
                    actual = compiled.predict(value)
                original16 = canonical.predict(value)
                for _ in range(3):
                    actual16 = compiled16.predict(value)
                warm_seconds = time.perf_counter() - begun
                matched = bool(np.allclose(expected, actual, atol=0.01, rtol=0.0001))
                admission.append(
                    {
                        "query_id": row["query_id"],
                        "admitted": matched,
                        "max_abs_seconds": float(np.max(np.abs(expected - actual))),
                        "canonical16_bit_exact": bool(np.array_equal(original16, actual16)),
                        "canonical16_admitted": bool(
                            np.allclose(original16, actual16, atol=0.01, rtol=0.0001)
                        ),
                        "canonical16_max_abs_seconds": float(np.max(np.abs(original16 - actual16))),
                        "compilation_warmup_and_reference_seconds": warm_seconds,
                    }
                )
                if not matched or not np.allclose(original16, actual16, atol=0.01, rtol=0.0001):
                    raise AssertionError("compiled path failed declared numerical tolerance")
                for iteration in range(5):
                    runners = {
                        "compact_eager_three": eager,
                        "compiled_three": compiled,
                        "canonical_eager_three": canonical,
                        "canonical_compiled_three": compiled16,
                    }
                    for name in generator.permutation(list(runners)):
                        runner = runners[name]
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        runner.predict(value)
                        torch.cuda.synchronize()
                        measurements.append(
                            {
                                "query_id": row["query_id"],
                                "system": name,
                                "iteration": iteration,
                                "wrapper_ms": (time.perf_counter() - start) * 1000,
                            }
                        )
                print(json.dumps(admission[-1]), flush=True)
                atomic_json(receipt, {**record, "admission": admission})
    except Exception as exc:
        atomic_json(
            receipt, {**record, "status": "FAILED", "error": str(exc), "admission": admission}
        )
        raise
    with (output / "COMPILED_LATENCY.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(measurements[0]))
        writer.writeheader()
        writer.writerows(measurements)
    atomic_json(
        receipt,
        {
            **record,
            "status": "COMPLETE",
            "admission": admission,
            "latency_sha256": digest(output / "COMPILED_LATENCY.csv"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/sota_campaign_20261008/dev32_expanded_rgb/QUERY_MANIFEST.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/ttc_revision_20261009/compiled")
    )
    args = parser.parse_args()
    probe(args.manifest, args.output)
