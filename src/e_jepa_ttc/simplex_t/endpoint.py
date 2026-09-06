"""Fixed-endpoint loading and bounded cached-head inference.

Callers must validate the stage endpoint manifest and data roles before calling.
No raw events or frozen expert forwards are performed here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .model import TemporalConfig, TemporalRefiner
from .training import QuerySource, load_checkpoint


def load_endpoint(
    path: Path,
    config: TemporalConfig,
    *,
    seed: int,
    freeze_sha256: str,
    train_source_sha256: str,
    endpoint_sha256: str,
) -> TemporalRefiner:
    """Require the exact fixed checkpoint named by a previously sealed endpoint list."""
    if seed not in {7, 13, 23} or any(
        len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
        for digest in (freeze_sha256, train_source_sha256, endpoint_sha256)
    ):
        raise ValueError("invalid endpoint identity")
    if sha256(path) != endpoint_sha256:
        raise ValueError("checkpoint differs from sealed phase endpoint")
    state = load_checkpoint(path)
    if state["completed_updates"] != 2500 or state["status"] != "COMPLETED":
        raise ValueError("only completed update2500 endpoints may be evaluated")
    expected = {
        "source": train_source_sha256,
        "freeze": freeze_sha256,
        "config": asdict(config),
        "seed": seed,
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "batch": 128,
        "endpoint": 2500,
    }
    identity_hash = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    if state["identity"] != expected or state["identity_sha256"] != identity_hash:
        raise ValueError("endpoint training identity mismatch")
    model = TemporalRefiner(config).float().cpu()
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def predict_cached(
    model: TemporalRefiner,
    source: QuerySource,
    *,
    resource_ok: Callable[[], bool],
) -> dict[str, np.ndarray]:
    """Infer every query in order at FP32 batch128; return only complete outputs.

    A resource interruption raises before publication. The caller may rerun this
    read-only head inference; no optimizer update or partial score is produced.
    This function alone does not establish that the supplied model is an endpoint.
    """
    if source.population < 1 or model.training:
        raise ValueError("nonempty source and evaluation-mode model required")
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("registered CPU FP32 inference required")
    outputs: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, source.population, 128):
            if not resource_ok():
                raise InterruptedError("cached inference paused; no partial endpoint export")
            stop = min(start + 128, source.population)
            x, timing, valid, experts, _, _ = source.gather(torch.arange(start, stop))
            if any(t.device.type != "cpu" for t in (x, timing, valid, experts)) or any(
                t.dtype != torch.float32 for t in (x, timing, experts)
            ):
                raise ValueError("registered CPU FP32 cache inputs required")
            batch = model(x, timing, valid, experts)
            if outputs and set(batch) != set(outputs):
                raise ValueError("inference output schema changed across batches")
            for name, tensor in batch.items():
                values = tensor.numpy()
                if len(values) != stop - start or not np.isfinite(values).all():
                    raise ValueError("nonfinite or misaligned endpoint outputs")
                if name not in outputs:
                    outputs[name] = np.empty((source.population, *values.shape[1:]), values.dtype)
                outputs[name][start:stop] = values
    return outputs
