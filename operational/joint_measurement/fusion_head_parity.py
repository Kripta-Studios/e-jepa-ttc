"""Isolate historical F_TRUE head numerics with fixed cached inputs and no target access."""

from __future__ import annotations

import argparse
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.range_transfer.common import freeze, read, sha
from operational.rgb_port.train_heads import load_head_endpoint

from .common import ROOT, RUN, gpu_charge, owner
from .compute_guard import guard_idle
from .fusion_cost_native_v2 import native_median_policy
from .train import committed_sources

OUT = RUN / "fusion/cost/F_TRUE_head_diagnostic_v1"
FAILED = RUN / "fusion/cost/raw_e2e_historical_shape_v1"
KEYS = (
    "event_features",
    "event_timing",
    "event_valid",
    "event_expert_phase",
    "rgb_features",
    "rgb_timing",
    "rgb_valid",
)


def register() -> dict:
    """Bind the failed query and its preceding controls before inspecting head outputs."""
    sources = {
        relative: sha(ROOT / relative)
        for relative in (
            "operational/joint_measurement/fusion_head_parity.py",
            "operational/rgb_port/predict_heads.py",
            "operational/rgb_port/train_heads.py",
            "src/e_jepa_ttc/rgb_port/fusion.py",
            "src/e_jepa_ttc/simplex_t/model.py",
            "src/e_jepa_ttc/simplex_t/phase.py",
        )
    }
    committed_sources(sources)
    failed = read(FAILED / "PARITY_FAILURE.json")
    original = read(FAILED / "REGISTRATION.json")
    cache = owner() / "features/V/FUSION.npz"
    receipt = read(cache.with_suffix(".json"))
    if receipt["status"] != "COMPLETE" or sha(cache) != receipt["output_sha256"]:
        raise ValueError("Historical F_TRUE feature cache changed")
    checkpoint = owner() / "fits/F_TRUE/checkpoint_last.pt"
    if sha(checkpoint) != original["historical_parents"]["F_TRUE"]:
        raise ValueError("Historical F_TRUE head changed")
    reference = RUN / "fusion/predictions/V/F_TRUE/predictions.parquet"
    if sha(reference) != read(reference.parent / "COMPLETE.json")["sha256"]:
        raise ValueError("Historical F_TRUE prediction table changed")
    stop = failed["row"] + 1
    record = {
        "sources": sources,
        "failed_parity_sha256": sha(FAILED / "PARITY_FAILURE.json"),
        "cost_registration_sha256": sha(FAILED / "REGISTRATION.json"),
        "indices": original["indices"][:stop],
        "tokens": original["tokens"][:stop],
        "selection": "All queries through the first recorded failure in the original cost order",
        "cache_sha256": sha(cache),
        "checkpoint_sha256": sha(checkpoint),
        "reference_sha256": sha(reference),
        "variants": ["historical_batch128", "repeat128", "padded_batch1", "trimmed_batch1"],
        "devices": ["cpu", "cuda"],
        "outputs": ["raw_location", "point_phase", "direct_ttc", "emitted_ttc"],
        "parity_rtol": original["parity_rtol"],
        "parity_atol": original["parity_atol"],
        "precision": "FP32; TF32 disabled for this head-only diagnostic",
        "labels_read": False,
        "optimizer_updates": 0,
        "interpretation": (
            "Cached features isolate head device/batch/history/emission. No raw-producer "
            "parity, no automatic cost admission, no fitting or tolerance changes."
        ),
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def variants(values: dict[str, np.ndarray], row: int) -> dict[str, tuple[list[torch.Tensor], int]]:
    """Keep padding for historical routes; trim only contiguous valid suffixes for raw shape."""
    if not 0 <= row < len(values["sample_token"]):
        raise ValueError("Query row outside bound cache")
    one = [torch.from_numpy(values[key][row : row + 1].copy()) for key in KEYS]
    trimmed = [value.clone() for value in one]
    for feature, timing, valid in ((0, 1, 2), (4, 5, 6)):
        mask = one[valid][0]
        if mask.dtype != torch.bool or not bool(mask.any()):
            raise ValueError("Present modality needs a valid history suffix")
        start = int(torch.nonzero(mask)[0, 0])
        if not bool(mask[start:].all()):
            raise ValueError("History validity must be a contiguous suffix")
        for key in (feature, timing, valid):
            trimmed[key] = one[key][:, start:].clone()
    begin = row // 128 * 128
    batch = [torch.from_numpy(values[key][begin : begin + 128].copy()) for key in KEYS]
    repeated = [value.repeat(128, *([1] * (value.ndim - 1))) for value in one]
    return {
        "historical_batch128": (batch, row - begin),
        "repeat128": (repeated, 0),
        "padded_batch1": (one, 0),
        "trimmed_batch1": (trimmed, 0),
    }


def run(device: str) -> dict:
    """Run CPU independently; require the exclusive GPU scope for the CUDA contrast."""
    if device not in ("cpu", "cuda"):
        raise ValueError("CPU or CUDA required")
    if device == "cuda":
        guard_idle()
    record = register()
    result_path = OUT / f"{device.upper()}_RESULT.json"
    if result_path.exists():
        result = read(result_path)
        if result["registration_sha256"] != sha(OUT / "REGISTRATION.json"):
            raise ValueError("Diagnostic result registration changed")
        return result
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # Do not load target_phase, mass, group_id or any target column.
    with np.load(owner() / "features/V/FUSION.npz", allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in (*KEYS, "sample_token")}
        if "rgb_available" in archive.files:
            available = archive["rgb_available"].astype(bool)
            values = {key: value[available] for key, value in values.items()}
    tokens = values["sample_token"].astype(str).tolist()
    if len(set(tokens)) != len(tokens):
        raise ValueError("Duplicate historical cache tokens")
    lookup = {token: i for i, token in enumerate(tokens)}
    reference = pd.read_parquet(
        RUN / "fusion/predictions/V/F_TRUE/predictions.parquet",
        columns=["sample_token", "prediction_ttc"],
    )
    if reference.sample_token.duplicated().any():
        raise ValueError("Duplicate reference tokens")
    references = reference.set_index("sample_token").prediction_ttc.to_dict()
    if any(token not in lookup or token not in references for token in record["tokens"]):
        raise ValueError("Fixed query population is missing from the cache or reference")
    measured = []
    scope = (
        gpu_charge("F_TRUE_head_parity_diagnostic", reserve=900)
        if device == "cuda"
        else nullcontext()
    )
    with scope, torch.inference_mode():
        model = load_head_endpoint(owner() / "fits/F_TRUE", "F_TRUE", torch.device(device))
        before = {key: value.clone() for key, value in model.state_dict().items()}
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if device == "cuda" else []
        for index, token in zip(record["indices"], record["tokens"], strict=True):
            for variant, (inputs, offset) in variants(values, lookup[token]).items():
                with native_median_policy() if device == "cuda" else nullcontext():
                    output = model(*[value.to(device) for value in inputs])
                direct = float(output["ttc"][offset].cpu())
                emitted = float(phase_to_ttc(output["point_phase"][offset : offset + 1].cpu())[0])
                target = float(references[token])
                measured.append(
                    {
                        "index": index,
                        "sample_token": token,
                        "variant": variant,
                        "input_shapes": [list(value.shape) for value in inputs],
                        "raw_location": float(output["raw_location"][offset].cpu()),
                        "point_phase": float(output["point_phase"][offset].cpu()),
                        "direct_ttc": direct,
                        "emitted_ttc": emitted,
                        "reference_ttc": target,
                        "direct_difference": abs(direct - target),
                        "emitted_difference": abs(emitted - target),
                        "direct_parity": bool(
                            np.isclose(
                                direct,
                                target,
                                rtol=record["parity_rtol"],
                                atol=record["parity_atol"],
                            )
                        ),
                        "emitted_parity": bool(
                            np.isclose(
                                emitted,
                                target,
                                rtol=record["parity_rtol"],
                                atol=record["parity_atol"],
                            )
                        ),
                    }
                )
        checks = {
            "weights_unchanged": all(
                torch.equal(value, model.state_dict()[key]) for key, value in before.items()
            ),
            "CPU_RNG_unchanged": torch.equal(cpu_rng, torch.get_rng_state()),
            "CUDA_RNG_unchanged": all(
                torch.equal(a, b)
                for a, b in zip(
                    cuda_rng,
                    torch.cuda.get_rng_state_all() if device == "cuda" else [],
                    strict=True,
                )
            ),
            "no_CUDA_in_CPU_route": device == "cuda" or not torch.cuda.is_initialized(),
        }
    if not all(checks.values()):
        raise ValueError(f"Diagnostic mutated state or initialized unexpected CUDA: {checks}")
    result = {
        "status": "COMPLETE_DIAGNOSTIC",
        "device": device,
        "registration_sha256": sha(OUT / "REGISTRATION.json"),
        "checks": checks,
        "measurements": measured,
        "optimizer_updates": 0,
        "cost_admitted": False,
    }
    freeze(result_path, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device", choices=("cpu", "cuda"))
    result = run(parser.parse_args().device)
    print({key: result[key] for key in ("status", "device", "checks", "cost_admitted")})
