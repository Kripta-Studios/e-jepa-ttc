"""Audit reuse of the completed V12 all-TRAIN40 event producers.

The audit is deliberately CPU-only and label-free.  It proves architecture/state
compatibility for A5 and C2F, checks their immutable receipts, and verifies the
canonical 133-D A5 cache that can feed a newly trained PAIR head.  Historical
PAIR/H8 weights are outside this admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.rgb_port.features import pair133
from e_jepa_ttc.simplex_t.expert_phase import expert_phase_from_ttc
from operational.rgb_port.train_producers import producer_features

EXPECTED_UPDATES = 49_932
ARMS = ("a5", "c2f")
FIT_IDS = {"a5": "E_A5_MATCHED", "c2f": "E_C2F_MATCHED"}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one immutable input."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return cast(dict[str, Any], value)


def compare_event_model_configs(
    v12_protocol: Mapping[str, Any], rgb_port_config: Mapping[str, Any]
) -> dict[str, bool]:
    """Require byte-independent JSON equality for both event architectures."""
    result: dict[str, bool] = {}
    for arm in ARMS:
        original = v12_protocol.get("producers", {}).get(arm, {}).get("model_config")
        port = rgb_port_config.get("producers", {}).get(FIT_IDS[arm], {}).get("model_config")
        if not isinstance(original, dict) or not isinstance(port, dict):
            raise ValueError(f"Missing event model config for {arm}")
        result[arm] = original == port
    return result


def load_v12_event_producer(
    run: Path, protocol: Mapping[str, Any], arm: str
) -> tuple[CausalScaleTTC, dict[str, Any]]:
    """Strictly load one immutable V12 all-TRAIN40 event producer on CPU."""
    if arm not in ARMS:
        raise ValueError(f"Unsupported V12 event producer: {arm}")
    fit = run / "fits" / f"{arm}_seed7"
    checkpoint = fit / "checkpoint_last.pt"
    receipt = _read_json(fit / "CHECKPOINT_RECEIPT.json")
    observed_sha = sha256_file(checkpoint)
    if (
        receipt.get("status") != "COMPLETE"
        or int(receipt.get("committed_updates", -1)) != EXPECTED_UPDATES
        or receipt.get("sha256") != observed_sha
    ):
        raise ValueError(f"V12 {arm} endpoint is incomplete or changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("status") != "COMPLETE"
        or int(payload.get("committed_updates", -1)) != EXPECTED_UPDATES
    ):
        raise ValueError(f"V12 {arm} payload is not its completed endpoint")
    config = protocol["producers"][arm]["model_config"]
    model = CausalScaleTTC(CausalScaleTTCConfig(**config))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.float().eval().requires_grad_(False)
    return model, {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": observed_sha,
        "committed_updates": EXPECTED_UPDATES,
        "state_keys": len(payload["model_state_dict"]),
        "strict_state_load": True,
    }


def _projection_parity(model: CausalScaleTTC, events: Tensor, delta: Tensor) -> dict[str, Any]:
    """Compare the V12 model output with the RGB-PORT producer projection."""
    with torch.inference_mode():
        direct = model(events, delta, return_dense_features=False)
        port = producer_features(model, events, delta)
    expected = {
        "token128": direct.pair_tokens[:, -1].float(),
        "prediction_ttc": direct.ttc_mean_seconds.float(),
        "point_phase": torch.from_numpy(
            expert_phase_from_ttc(direct.ttc_mean_seconds.float().numpy())
        ).float(),
        "flow": direct.diagnostics["transport_flow_magnitude"][:, -1].float(),
        "margin": torch.minimum(
            direct.log_height_ratio[:, -1].abs().float() / 0.002,
            direct.sensor_support[:, -1].float() / 0.0001,
        ),
        "log_variance": direct.ttc_log_variance.float(),
        "support": direct.sensor_support.float(),
        "known": direct.known_mask,
    }
    exact: dict[str, bool] = {}
    maxima: dict[str, float] = {}
    for name, reference in expected.items():
        candidate = port[name]
        exact[name] = bool(torch.equal(reference, candidate))
        maxima[name] = (
            0.0
            if reference.dtype == torch.bool
            else float((reference - candidate).abs().max().item())
        )
    return {
        "all_fields_exact": all(exact.values()),
        "field_exact": exact,
        "max_abs_difference": maxima,
        "known_shape": list(port["known"].shape),
        "support_shape": list(port["support"].shape),
    }


def _pair_cache_audit(run: Path, raw_cache: Path, a5: CausalScaleTTC, rows: int) -> dict[str, Any]:
    manifest_path = run / "PAIR_FEATURE_MANIFEST.json"
    manifest = _read_json(manifest_path)
    cache_path = run / "PAIR_FEATURES.npz"
    parent_sha = sha256_file(run / "fits" / "a5_seed7" / "checkpoint_last.pt")
    files = manifest.get("files")
    if (
        manifest.get("status") != "COMPLETE_VERIFIED"
        or int(manifest.get("row_count", -1)) != 88_744
        or manifest.get("parents", {}).get("a5", {}).get("checkpoint_sha256") != parent_sha
        or not isinstance(files, list)
        or len(files) != 1
        or files[0].get("sha256") != sha256_file(cache_path)
    ):
        raise ValueError("V12 PAIR feature cache is incomplete or has changed ancestry")
    with np.load(cache_path, allow_pickle=False) as stored:
        features = np.asarray(stored["features"])
        delta_values = features[:, 128]
        cache_sample = np.asarray(features[:rows], np.float32)
    if features.shape != (88_744, 133) or not np.isfinite(features).all():
        raise ValueError("V12 PAIR cache has a different shape or nonfinite values")
    if not np.array_equal(delta_values, np.full(88_744, np.float32(0.1))):
        raise ValueError("V12 PAIR cache does not retain the registered 100 ms interval")
    with np.load(raw_cache, allow_pickle=False) as stored:
        events = torch.from_numpy(np.asarray(stored["events"][:rows], np.float32))
        dt = torch.from_numpy(np.asarray(stored["delta"][:rows], np.float32))
    delta = dt[:, None].expand(-1, 2).contiguous()
    with torch.inference_mode():
        output = a5(events, delta, return_dense_features=False)
        current = output.sensor_support[:, -1]
        generated = pair133(
            output.pair_tokens[:, -1].float(),
            dt,
            torch.stack((current, torch.minimum(output.sensor_support[:, -2], current)), -1),
        ).numpy()
    difference = np.abs(generated - cache_sample)
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "cache_sha256": sha256_file(cache_path),
        "rows": 88_744,
        "width": 133,
        "all_finite": True,
        "delta_seconds_unique": [0.1],
        "parent_a5_sha256": parent_sha,
        "bounded_cpu_replay_rows": rows,
        "bounded_cpu_replay_allclose_atol_1e_5": bool(
            np.allclose(generated, cache_sample, rtol=0.0, atol=1e-5)
        ),
        "bounded_cpu_replay_max_abs_difference": float(difference.max()),
        "replay_note": (
            "The sealed cache was produced on CUDA; CPU replay is tolerance evidence, "
            "while its manifest/hash/ancestry define the exact reusable bytes."
        ),
    }


def audit_event_reuse(
    *, v12_run: Path, rgb_port_config: Path, raw_cache: Path, sample_rows: int = 2
) -> dict[str, Any]:
    """Run the complete CPU admission and return a truthful JSON-ready receipt."""
    if not 1 <= sample_rows <= 8:
        raise ValueError("sample_rows must be in [1,8]")
    protocol_path = v12_run / "TRAINING_PROTOCOL.json"
    protocol = _read_json(protocol_path)
    port_config = _read_json(rgb_port_config)
    config_equal = compare_event_model_configs(protocol, port_config)
    if not all(config_equal.values()):
        raise ValueError("V12 and RGB-PORT event architectures differ")
    with np.load(raw_cache, allow_pickle=False) as stored:
        events = torch.from_numpy(np.asarray(stored["events"][:sample_rows], np.float32))
        dt = torch.from_numpy(np.asarray(stored["delta"][:sample_rows], np.float32))
    if events.shape != (sample_rows, 3, 12, 128, 128) or dt.shape != (sample_rows,):
        raise ValueError("bounded TRAIN40 event sample schema differs")
    delta = dt[:, None].expand(-1, 2).contiguous()
    endpoints: dict[str, Any] = {}
    models: dict[str, CausalScaleTTC] = {}
    for arm in ARMS:
        model, identity = load_v12_event_producer(v12_run, protocol, arm)
        models[arm] = model
        endpoints[arm] = {**identity, "projection": _projection_parity(model, events, delta)}
        if not endpoints[arm]["projection"]["all_fields_exact"]:
            raise ValueError(f"{arm} RGB-PORT projection differs from the V12 model output")
    pair_cache = _pair_cache_audit(v12_run, raw_cache, models["a5"], sample_rows)
    if not pair_cache["bounded_cpu_replay_allclose_atol_1e_5"]:
        raise ValueError("sealed A5 PAIR feature cache differs from bounded CPU replay")
    known_contract_needs_adapter = any(
        item["projection"]["known_shape"] != item["projection"]["support_shape"]
        for item in endpoints.values()
    )
    return {
        "schema": "final_refit_v12_event_reuse_parity_v1",
        "status": "PASSED_WITH_EXPLICIT_ADAPTER",
        "scope": {
            "admitted": [
                "V12 A5 seed7 all-TRAIN40 producer checkpoint",
                "V12 C2F seed7 all-TRAIN40 producer checkpoint",
                "V12 sealed A5 133-D current-query PAIR feature cache",
            ],
            "excluded": [
                "V12 PAIR endpoint",
                "V12 H8 endpoint and its normalizer",
                "V12 H8 feature cache as a complete new temporal cache",
                "test12 labels or predictions",
            ],
        },
        "inputs": {
            "v12_training_protocol_sha256": sha256_file(protocol_path),
            "rgb_port_config_sha256": sha256_file(rgb_port_config),
            "raw_cache_sha256": sha256_file(raw_cache),
            "sample_rows": sample_rows,
            "targets_loaded": False,
            "optimizer_updates": 0,
            "device": "cpu",
            "precision": "float32",
        },
        "architecture_config_equal": config_equal,
        "endpoints": endpoints,
        "pair_cache": pair_cache,
        "reuse_decision": {
            "producer_weights_safe": True,
            "sealed_pair133_cache_safe": True,
            "historical_pair_or_h8_safe": False,
            "direct_v13_checkpoint_loader_safe": False,
            "known_clock_adapter_required": known_contract_needs_adapter,
            "required_remedy": [
                "Load V12 producer checkpoints through a dedicated strict loader bound to "
                "their V12 receipt and protocol; do not relabel them as V13 checkpoints.",
                "Expand the aggregate producer known flag across the preserved support clock "
                "only at the RGB-PORT observation boundary, or relax that boundary to accept "
                "the aggregate flag; the flag is not a PAIR feature.",
                "Train a fresh PAIR_E and E_CTX with a new TRAIN-only normalizer. Recompute "
                "historical observation blocks needed by E_CTX because the old H8 cache binds "
                "the excluded V12 PAIR phase.",
            ],
        },
    }


def _defaults() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[2]
    v12 = root.parent / "e-jepa-ttc-v12-efficient-context" / "artifacts" / "train40_system_20261005"
    config = root / "configs" / "rgb_port" / "producers.json"
    prepare = _read_json(v12 / "PREPARE_FREEZE.json")
    raw = Path(str(prepare["cache_root"])) / "shard_00000.npz"
    return v12, config, raw


def main(argv: Sequence[str] | None = None) -> int:
    default_v12, default_config, default_raw = _defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v12-run", type=Path, default=default_v12)
    parser.add_argument("--rgb-port-config", type=Path, default=default_config)
    parser.add_argument("--raw-cache", type=Path, default=default_raw)
    parser.add_argument("--sample-rows", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    receipt = audit_event_reuse(
        v12_run=args.v12_run.resolve(strict=True),
        rgb_port_config=args.rgb_port_config.resolve(strict=True),
        raw_cache=args.raw_cache.resolve(strict=True),
        sample_rows=args.sample_rows,
    )
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".pending")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "audit_event_reuse",
    "compare_event_model_configs",
    "load_v12_event_producer",
    "main",
    "sha256_file",
]
