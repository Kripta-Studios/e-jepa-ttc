"""Portable inference weights with explicit operators and exact CPU reload checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.range_readout.operators import ResidualSpec
from e_jepa_ttc.range_readout.readout import VectorReadout, wrap
from operational.range_transfer.common import read, sha, write

from .geometry import initialize as init_geometry
from .geometry_model import GeometryReadout
from .model import RepresentationRefiner
from .precision import initialize as init_precision
from .run import INPUTS, OLD, OWNER, RUN, cache
from .spatial_geometry import initialize as init_spatial
from .spatial_model import SpatialReadout
from .temporal_geometry import initialize as init_temporal


def load_export(path: Path) -> nn.Module:
    """Load weights without the original optimizer, parent checkpoint or calibration JSON."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format"] != "v17_explicit_operator_v1":
        raise ValueError("Unsupported export format")
    metadata = payload["metadata"]
    state = payload["model_state_dict"]
    kind = metadata["kind"]
    model: nn.Module
    if kind == "head":
        model = RepresentationRefiner(metadata["bypass_active"])
    elif kind in ("vector_equal_slope", "vector_calibrated_height", "vector_spatial_height"):
        native = CausalScaleTTC(CausalScaleTTCConfig(**metadata["model_config"]))
        spec = ResidualSpec(**metadata["operator"])
        native = wrap(native, spec, reset=False)
        if kind in ("vector_calibrated_height", "vector_spatial_height"):
            calibration = {
                key: state["height_" + key].tolist()
                for key in ("mean", "scale", "coefficient", "intercept")
            }
            calibration["role"] = "P"
            model = (
                SpatialReadout(native, calibration, "SP1")
                if kind == "vector_spatial_height"
                else GeometryReadout(native, calibration, "Q1")
            )
        else:
            model = VectorReadout(native)
    else:
        raise ValueError("Unknown inference operator")
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def save_example(path: Path, values: dict[str, torch.Tensor]) -> None:
    """Preserve exact prepared examples, without targets or mutable ZIP timestamps."""
    arrays = {key: value.detach().cpu().numpy() for key, value in values.items()}
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays) or any(
                not np.array_equal(existing[key], value) for key, value in arrays.items()
            ):
                raise ValueError("Existing portable example changed")
    else:
        np.savez_compressed(path, allow_pickle=False, **arrays)


def main() -> None:
    torch.set_num_threads(1)
    parent = torch.load(
        OWNER / "fits/E_A5_MATCHED/checkpoint_last.pt", map_location="cpu", weights_only=False
    )
    model_config = parent["identity"]["model_config"]
    del parent
    h = cache("H")
    with np.load(RUN / "cache/D_H.npz", allow_pickle=False) as archive:
        dh = {key: archive[key] for key in INPUTS}
    with np.load(OLD / "vector_cache/P/part_000000.npz", allow_pickle=False) as archive:
        raw = {
            key: torch.from_numpy(archive[key][:8].copy())
            for key in archive.files
            if key != "sample_token" and not key.startswith(("target_", "dino_"))
        }
    manifest = {}
    for directory in sorted((RUN / "fits").iterdir()):
        receipt_path = directory / "CHECKPOINT_RECEIPT.json"
        if not receipt_path.exists() or read(receipt_path)["status"] != "COMPLETE":
            continue
        receipt = read(receipt_path)
        name = directory.name
        source = directory / "checkpoint_last.pt"
        if sha(source) != receipt["checkpoint_sha256"]:
            raise ValueError("Completed checkpoint changed")
        arm, seed_text = name.split("-S")
        seed = int(seed_text)
        payload = torch.load(source, map_location="cpu", weights_only=False)
        metadata = {
            "fit": name,
            "seed": seed,
            "source_checkpoint_sha256": sha(source),
            "inference_scope": (
                "cached feature/vector forward; raw events require admitted upstream pipeline"
            ),
        }
        if arm in ("Z", "G", "D"):
            metadata.update(
                kind="head",
                bypass_active=arm != "Z",
                side_transform="temporal_differences_512_zero_pad896"
                if arm == "D"
                else "original_896",
                side_normalizer=read(
                    RUN / ("D_H_NORMALIZER.json" if arm == "D" else "H_SIDE_NORMALIZER.json")
                ),
                phase17_normalizer=read(OWNER / "features/H/E_NORMALIZER.json"),
                decoder="native_phase_to_ttc",
            )
            reference = RepresentationRefiner(arm != "Z")
            values = dh if arm == "D" else h
            inputs = [torch.from_numpy(values[key][:8].copy()) for key in INPUTS]
        else:
            metadata.update(
                kind=(
                    "vector_equal_slope"
                    if arm in ("U0", "U1", "N0")
                    else (
                        "vector_spatial_height"
                        if arm in ("SP0", "SP1")
                        else "vector_calibrated_height"
                    )
                ),
                model_config=model_config,
                operator=payload["identity"]["operator"],
                calibration_branch=(
                    "none_residual_only"
                    if arm in ("U0", "U1", "N0")
                    else "analytic_height_only_original_PAIR_inputs"
                ),
            )
            if arm in ("U0", "U1", "N0"):
                reference, *_ = init_precision(seed, device="cpu")
            elif arm in ("Q0", "Q1"):
                reference, *_ = init_geometry(seed, arm, device="cpu")
            elif arm in ("F0", "F1"):
                reference, *_ = init_temporal(seed, arm, device="cpu")
            elif arm in ("SP0", "SP1"):
                reference, *_ = init_spatial(seed, arm, device="cpu")
            else:
                raise ValueError("Unknown completed fit")
            if arm in ("SP0", "SP1"):
                with np.load(RUN / "spatial_cache/P/part_000000.npz", allow_pickle=False) as arc:
                    spatial = torch.from_numpy(arc["height_features"][:8].copy())
                inputs = [{**raw, "height_features": spatial}]
                metadata["height_features"] = (
                    "matched_A5_dense_64x32x32_mean_pool_8x1_channel_major"
                )
            else:
                inputs = [raw]
        reference.load_state_dict(payload["model_state_dict"], strict=True)
        metadata["input_style"] = "positional" if arm in ("Z", "G", "D") else "dict"
        metadata["input_order"] = list(INPUTS) if arm in ("Z", "G", "D") else sorted(inputs[0])
        reference.eval().requires_grad_(False)
        folder = RUN / "exports" / name
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / "WEIGHTS.pt"
        if destination.exists():
            previous = read(folder / "RECEIPT.json")
            if sha(destination) != previous["weights_sha256"] or previous["source_sha256"] != sha(
                source
            ):
                raise ValueError("Existing export changed")
            previous_payload = torch.load(destination, map_location="cpu", weights_only=True)
            if previous_payload["metadata"] != metadata:
                write(folder / "PREVIOUS_DERIVED_METADATA.json", previous_payload["metadata"])
                torch.save(
                    {
                        "format": "v17_explicit_operator_v1",
                        "metadata": metadata,
                        "model_state_dict": payload["model_state_dict"],
                    },
                    destination,
                )
        else:
            torch.save(
                {
                    "format": "v17_explicit_operator_v1",
                    "metadata": metadata,
                    "model_state_dict": payload["model_state_dict"],
                },
                destination,
            )
        restored = load_export(destination)
        with torch.inference_mode():
            a, b = reference(*inputs), restored(*inputs)
        names = (
            tuple(a)
            if isinstance(a, dict)
            else (
                "ttc_mean_seconds",
                "ttc_log_variance",
                "collision_logits",
                "pair_tokens",
                "pair_log_height_ratio",
            )
        )
        checks = {
            key: torch.equal(a[key], b[key])
            if isinstance(a, dict)
            else torch.equal(getattr(a, key), getattr(b, key))
            for key in names
        }
        if not all(checks.values()):
            raise AssertionError(f"Export parity failure {name}: {checks}")
        example: dict[str, torch.Tensor] = {}
        if arm in ("Z", "G", "D"):
            for key, tensor in zip(INPUTS, inputs, strict=True):
                assert isinstance(tensor, torch.Tensor)
                example[key] = tensor
        else:
            assert isinstance(inputs[0], dict)
            example = inputs[0]
        save_example(folder / "EXAMPLE_INPUT.npz", example)
        save_example(
            folder / "EXAMPLE_OUTPUT.npz",
            a if isinstance(a, dict) else {key: getattr(a, key) for key in names},
        )
        entry = {
            "status": "PASS",
            "source_sha256": sha(source),
            "weights_sha256": sha(destination),
            "exact_cpu_reload_checks": checks,
            "optimizer_updates": 0,
            "raw_inference_parity_claimed": False,
            "example_input_sha256": sha(folder / "EXAMPLE_INPUT.npz"),
            "example_output_sha256": sha(folder / "EXAMPLE_OUTPUT.npz"),
        }
        write(folder / "RECEIPT.json", entry)
        write(folder / "MODEL_METADATA.json", metadata)
        manifest[name] = entry
    write(RUN / "EXPORTS.json", manifest)
    print(f"{len(manifest)} exports verified")


if __name__ == "__main__":
    main()
