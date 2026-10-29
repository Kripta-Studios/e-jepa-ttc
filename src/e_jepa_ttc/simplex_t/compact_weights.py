"""Safe-array delivery of new fixed-endpoint heads, without optimizer pickles."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json

from .arms import resolve_arm
from .endpoint import load_endpoint
from .lifecycle import ExclusiveLease
from .model import TemporalRefiner
from .registry import FitSpec


def export_compact_endpoint(
    checkpoint: Path,
    output: Path,
    *,
    spec: FitSpec,
    graph: list[FitSpec],
    freeze_sha256: str,
    train_source_sha256: str,
    endpoint_sha256: str,
    validate_prerequisites: Callable[[], None],
    resource_check: Callable[[], None],
) -> str:
    """Export one validated update2500 head; publish its typed manifest last.

    The prerequisite callback must validate the phase seal and campaign freeze.
    This function does not independently authorize a fit or select a candidate.
    """
    validate_prerequisites()
    resource_check()
    binding = resolve_arm(spec, graph)
    if output.exists():
        raise FileExistsError("compact endpoint output exists")
    model = load_endpoint(
        checkpoint,
        binding.model,
        seed=spec.seed,
        freeze_sha256=freeze_sha256,
        train_source_sha256=train_source_sha256,
        endpoint_sha256=endpoint_sha256,
    )
    arrays = {
        name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()
    }
    if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("only finite FP32 compact head state is supported")
    del model
    with ExclusiveLease(output.with_suffix(".export.lock")):
        output.mkdir()
        resource_check()
        weights = output / "weights.npz"
        with weights.open("xb") as stream:
            np.savez_compressed(stream, **arrays)  # pyright: ignore[reportArgumentType]
        manifest = {
            "schema": "simplex_t_compact_endpoint_v1",
            "fit": asdict(spec),
            "model": asdict(binding.model),
            "history": binding.history,
            "control": binding.control,
            "zero_latent": binding.zero_latent,
            "scientific_freeze_sha256": freeze_sha256,
            "train_source_sha256": train_source_sha256,
            "source_endpoint_sha256": endpoint_sha256,
            "weights_sha256": sha256(weights),
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in arrays.items()
            },
            "optimizer_state_included": False,
            "status": "COMPLETE_COMPACT_ENDPOINT",
        }
        path = output / "WEIGHTS.json"
        write_new_json(path, manifest)
    return sha256(path)


def load_compact_endpoint(output: Path, *, manifest_sha256: str) -> TemporalRefiner:
    """Load verified numeric arrays with allow_pickle=False and strict shapes."""
    path = output / "WEIGHTS.json"
    if sha256(path) != manifest_sha256:
        raise ValueError("compact manifest hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest["schema"] != "simplex_t_compact_endpoint_v1"
        or manifest["status"] != "COMPLETE_COMPACT_ENDPOINT"
        or manifest["optimizer_state_included"] is not False
    ):
        raise ValueError("invalid compact endpoint schema")
    weights = output / "weights.npz"
    if sha256(weights) != manifest["weights_sha256"]:
        raise ValueError("compact weights hash mismatch")
    spec = FitSpec(**manifest["fit"])
    binding = resolve_arm(spec, [spec])
    if (
        manifest["model"] != asdict(binding.model)
        or manifest["history"] != binding.history
        or manifest["control"] != binding.control
        or manifest["zero_latent"] != binding.zero_latent
    ):
        raise ValueError("compact inference contract differs from canonical arm")
    model = TemporalRefiner(binding.model).float().cpu()
    expected = model.state_dict()
    with np.load(weights, allow_pickle=False) as archive:
        if set(archive.files) != set(expected) or set(manifest["arrays"]) != set(expected):
            raise ValueError("compact parameter set mismatch")
        state = {}
        for name, tensor in expected.items():
            value = archive[name]
            if (
                value.dtype != np.float32
                or not np.isfinite(value).all()
                or value.shape != tuple(tensor.shape)
                or manifest["arrays"][name] != {"shape": list(value.shape), "dtype": "float32"}
            ):
                raise ValueError("compact parameter schema mismatch")
            state[name] = torch.from_numpy(value.copy())
    model.load_state_dict(state, strict=True)
    return model.eval()
