"""Train the frozen-budget RGB-PORT PAIR, temporal, and fusion endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional

from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch
from e_jepa_ttc.rgb_port.fusion import DualClockFusion, fusion_loss
from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner, training_loss
from e_jepa_ttc.training.incremental_residual import deterministic_sequence_grouped_schedule
from e_jepa_ttc.training.stage61_pair_head import CachedPairDirectPhase
from operational.rgb_port.accounting import atomic_write_json, read_json_shared
from operational.rgb_port.train_producers import (
    ProducerCheckpoint,
    _assert_recovery_budget,
    _resource_guard,
)

PAIR_IDS = {"PAIR_E_MATCHED", "PAIR_R"}
TEMPORAL_IDS = {"E_H1_MATCHED", "E_CTX_MATCHED", "R_H1", "R_CTX"}
FUSION_IDS = {"F_TRUE", "F_ZERO"}
FIT_IDS = PAIR_IDS | TEMPORAL_IDS | FUSION_IDS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _frozen_identity(freeze_path: Path, cache: Path, fit_id: str, output: Path) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    required = {
        "source_sha256",
        "git_commit",
        "config_sha256",
        "role_manifest_sha256",
        "split_sha256",
        "parent_sha256",
        "normalizer_sha256",
        "precision",
        "device",
        "output",
        "executed_source_sha256",
    }
    if not required.issubset(freeze):
        raise ValueError(f"freeze lacks bindings: {sorted(required - set(freeze))}")
    if freeze["source_sha256"] != _sha256(cache):
        raise ValueError("sealed cache changed after admission")
    if freeze["precision"] != "float32":
        raise ValueError("RGB-PORT heads require frozen FP32 precision")
    if freeze["output"] != str(output.resolve()):
        raise ValueError("frozen output directory differs")
    sources = freeze["executed_source_sha256"]
    if not isinstance(sources, dict) or not sources:
        raise ValueError("executed source bindings must be a nonempty mapping")
    for source, expected in sources.items():
        if _sha256(Path(source).resolve(strict=True)) != expected:
            raise ValueError(f"executed source changed: {source}")
    result = {
        "schema": "rgb_port_head_identity_v1",
        "fit_id": fit_id,
        "freeze_sha256": _sha256(freeze_path),
        "cache_path": str(cache.resolve()),
        "cache_sha256": _sha256(cache),
        **{name: freeze[name] for name in sorted(required)},
        "seed": 7,
        "torch_version": str(torch.__version__),
    }
    result["identity_sha256"] = _canonical_sha(result)
    return result


class SealedCache:
    """In-memory validated training arrays; V fields are not accepted."""

    def __init__(self, path: Path, fit_id: str) -> None:
        with np.load(path, allow_pickle=False) as archive:
            self.values = {name: archive[name] for name in archive.files}
        if "role" not in self.values or set(
            map(str, np.asarray(self.values["role"]).reshape(-1))
        ) != {"H" if fit_id not in PAIR_IDS else "P"}:
            raise ValueError("cache role differs from the preregistered fit role")
        prohibited = {
            name for name in self.values if "dev" in name.lower() or name.lower().startswith("v_")
        }
        if prohibited:
            raise ValueError(f"development arrays reached trainer: {sorted(prohibited)}")
        self._validate(fit_id)

    def _validate(self, fit_id: str) -> None:
        common = {"target_phase", "mass", "sample_token", "group_id", "role"}
        if fit_id in PAIR_IDS:
            required = common | {"features", "sequence_id"}
        elif fit_id in TEMPORAL_IDS:
            required = common | {"features", "timing", "valid", "expert_phase"}
        else:
            required = common | {
                "event_features",
                "event_timing",
                "event_valid",
                "event_expert_phase",
                "rgb_features",
                "rgb_timing",
                "rgb_valid",
            }
        if set(self.values) != required:
            raise ValueError(f"cache schema differs: {sorted(set(self.values) ^ required)}")
        n = len(self.values["target_phase"])
        if n < 1 or any(len(value) != n for value in self.values.values()):
            raise ValueError("cache arrays are empty or row-misaligned")
        for name in ("target_phase", "mass"):
            value = np.asarray(self.values[name])
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite FP32")
        if np.any(self.values["mass"] < 0):
            raise ValueError("global H/P mass cannot be negative")
        if fit_id in PAIR_IDS and self.values["features"].shape != (n, 133):
            raise ValueError("PAIR cache must be [N,133]")
        if fit_id in TEMPORAL_IDS:
            self._stream("", n)
        if fit_id in FUSION_IDS:
            self._stream("event_", n)
            self._stream("rgb_", n, experts=False)

    def _stream(self, prefix: str, n: int, *, experts: bool = True) -> None:
        features = self.values[f"{prefix}features"]
        timing = self.values[f"{prefix}timing"]
        valid = self.values[f"{prefix}valid"]
        if features.ndim != 3 or features.shape[0] != n or features.shape[-1] != 17:
            raise ValueError(f"{prefix}features must be [N,L,17]")
        if not 1 <= features.shape[1] <= 8 or timing.shape != (*features.shape[:2], 4):
            raise ValueError(f"{prefix}timing differs")
        if valid.shape != features.shape[:2] or valid.dtype != np.bool_:
            raise ValueError(f"{prefix}valid differs")
        if not np.all(valid[:, -1]) or np.any(valid[:, :-1] & ~valid[:, 1:]):
            raise ValueError(f"{prefix}histories are not valid suffixes")
        if experts and self.values[f"{prefix}expert_phase"].shape != (n, 3):
            raise ValueError(f"{prefix}expert phases differ")

    @property
    def population(self) -> int:
        return len(self.values["target_phase"])

    def tensor(self, name: str, ids: Tensor, device: torch.device) -> Tensor:
        return torch.as_tensor(self.values[name][ids.numpy()]).to(device)


def _model(fit_id: str) -> nn.Module:
    if fit_id in PAIR_IDS:
        return CachedPairDirectPhase(dropout=0.05)
    if fit_id in TEMPORAL_IDS:
        return TemporalRefiner(
            TemporalConfig(feature_count=17, hidden=160, backbone="gru", output_mode="residual")
        )
    return DualClockFusion(fit_id)


def _recipe(fit_id: str) -> tuple[int, int, float]:
    return (6840, 32, 1e-4) if fit_id in PAIR_IDS else (2500, 128, 1e-3)


def _lr(fit_id: str, update: int) -> float:
    if fit_id in PAIR_IDS:
        return 3e-4
    if update <= 100:
        return 3e-4 * update / 100
    return 3e-5 + (3e-4 - 3e-5) * (1 + math.cos(math.pi * (update - 100) / 2400)) / 2


def _loss(
    model: nn.Module, fit_id: str, cache: SealedCache, ids: Tensor, device: torch.device
) -> Tensor:
    truth = cache.tensor("target_phase", ids, device).float()
    mass = cache.tensor("mass", ids, device).float()
    if fit_id in PAIR_IDS:
        features = cache.tensor("features", ids, device).float()
        prediction = model(PairFeatureBatch(features))
        return functional.smooth_l1_loss(prediction, truth)
    if fit_id in TEMPORAL_IDS:
        features = cache.tensor("features", ids, device).float()
        timing = cache.tensor("timing", ids, device).float()
        valid = cache.tensor("valid", ids, device).bool()
        experts = cache.tensor("expert_phase", ids, device).float()
        if fit_id in {"E_H1_MATCHED", "R_H1"}:
            features, timing, valid = features[:, -1:], timing[:, -1:], valid[:, -1:]
        output = model(features, timing, valid, experts)
        return training_loss(output, truth, experts, mass, cache.population)
    output = model(
        cache.tensor("event_features", ids, device).float(),
        cache.tensor("event_timing", ids, device).float(),
        cache.tensor("event_valid", ids, device).bool(),
        cache.tensor("event_expert_phase", ids, device).float(),
        cache.tensor("rgb_features", ids, device).float(),
        cache.tensor("rgb_timing", ids, device).float(),
        cache.tensor("rgb_valid", ids, device).bool(),
    )
    return fusion_loss(output, truth, mass, cache.population)


def fit_head(
    *,
    fit_id: str,
    cache_path: Path,
    freeze_path: Path,
    run: Path,
    resume: bool = False,
    device_name: str = "cpu",
    max_updates_this_call: int | None = None,
) -> dict[str, Any]:
    """Run or automatically resume one preregistered fixed-endpoint fit."""
    del resume  # Existing complete-state pointers are always resumed.
    if fit_id not in FIT_IDS or device_name not in {"cpu", "cuda"}:
        raise ValueError("unknown fit ID or device")
    output = run / "fits" / fit_id
    identity = _frozen_identity(freeze_path, cache_path, fit_id, output)
    if identity["device"] != device_name:
        raise ValueError("execution device differs from frozen admission")
    endpoint, batch_size, weight_decay = _recipe(fit_id)
    allowed, resources = _resource_guard(run, checkpoint_reservation=256 * 1024**2)
    if not allowed or (run / "PAUSE").exists() or (output / "STOP_REQUEST").exists():
        receipt_path = output / "CHECKPOINT_RECEIPT.json"
        if receipt_path.is_file():
            return read_json_shared(receipt_path)
        return {
            "schema": "rgb_port_checkpoint_receipt_v1",
            "status": "PAUSED_RESOURCE",
            "fit_id": fit_id,
            "completed_updates": 0,
            "scientific_endpoint": False,
            "identity_sha256": identity["identity_sha256"],
            "resource_snapshot": resources,
        }
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(7)
    torch.use_deterministic_algorithms(True)
    cache = SealedCache(cache_path, fit_id)
    pair_schedule: list[np.ndarray[Any, Any]] | None = None
    if fit_id in PAIR_IDS:
        pair_schedule, schedule_sha256 = deterministic_sequence_grouped_schedule(
            list(map(str, cache.values["sequence_id"])),
            list(map(str, cache.values["sample_token"])),
            seed=7,
            batch_size=batch_size,
            updates=endpoint,
        )
        identity["batch_schedule_sha256"] = schedule_sha256
        identity["identity_sha256"] = _canonical_sha(
            {name: value for name, value in identity.items() if name != "identity_sha256"}
        )
    model = _model(fit_id).float().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr(fit_id, min(step + 1, endpoint)) / 3e-4
    )
    generator = torch.Generator(device="cpu").manual_seed(7)
    state = ProducerCheckpoint(output, identity, endpoint)
    cursor: dict[str, Any] = {"losses": []}
    if state.pointer.is_file():
        cursor = state.restore(model, optimizer, scheduler, generator)
    else:
        state.save(model, optimizer, scheduler, generator, cursor, status="READY")
    provenance_path = output / "RUN_PROVENANCE.json"
    if provenance_path.is_file():
        provenance = read_json_shared(provenance_path)
    else:
        provenance = {
            "schema": "rgb_port_run_provenance_v1",
            "experiment_id": "RGB_PORT_20261008",
            "run_id": fit_id,
            "fit_id": fit_id,
            "git_commit": identity["git_commit"],
            "config_hash": identity["config_sha256"],
            "seed": 7,
            "dataset_manifest_hash": identity["role_manifest_sha256"],
            "split_version": identity["split_sha256"],
            "host": platform.node(),
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "device": device_name,
            "start_time": datetime.now(UTC).isoformat(),
            "end_time": None,
            "status": "RUNNING",
            "checkpoint_path": None,
            "metrics_path": None,
            "identity_sha256": identity["identity_sha256"],
            "source_sha256": identity["source_sha256"],
            "normalizer_sha256": identity["normalizer_sha256"],
        }
        atomic_write_json(provenance_path, provenance)
    started_perf = time.perf_counter()
    call_updates = 0
    last_resource_check = -25
    while state.completed < endpoint:
        if state.completed - last_resource_check >= 25:
            reservation = max(256 * 1024**2, 2 * state.path.stat().st_size)
            allowed, resources = _resource_guard(run, checkpoint_reservation=reservation)
            last_resource_check = state.completed
        if not allowed or (run / "PAUSE").exists() or (output / "STOP_REQUEST").exists():
            state.save(model, optimizer, scheduler, generator, cursor, status="PAUSED_RESOURCE")
            atomic_write_json(output / "RESOURCE_PAUSE.json", resources)
            break
        ids = (
            torch.as_tensor(pair_schedule[state.completed], dtype=torch.long)
            if pair_schedule is not None
            else torch.randint(cache.population, (batch_size,), generator=generator)
        )
        optimizer.param_groups[0]["lr"] = _lr(fit_id, state.completed + 1)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, fit_id, cache, ids, device)
        if not torch.isfinite(loss):
            raise FloatingPointError("head loss became nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        _assert_recovery_budget(run, prospective_pending=1)
        state.begin()
        optimizer.step()
        scheduler.step()
        state.commit()
        cursor["losses"].append(float(loss.detach().cpu()))
        call_updates += 1
        requested_pause = (
            max_updates_this_call is not None and call_updates >= max_updates_this_call
        )
        if state.completed % 100 == 0 or state.completed == endpoint or requested_pause:
            status = "COMPLETE" if state.completed == endpoint else "IN_PROGRESS"
            state.save(model, optimizer, scheduler, generator, cursor, status=status)
        if requested_pause:
            break
    receipt = read_json_shared(state.receipt)
    provenance.update(
        status=receipt["status"],
        end_time=datetime.now(UTC).isoformat() if receipt["status"] == "COMPLETE" else None,
        checkpoint_path=receipt["checkpoint_path"],
        elapsed_seconds_last_call=time.perf_counter() - started_perf,
        recovery_upper=state.recovery_upper,
    )
    atomic_write_json(provenance_path, provenance)
    return receipt


def load_head_endpoint(fit_dir: Path, fit_id: str, device: torch.device) -> nn.Module:
    """Load only a complete, receipt-bound RGB-PORT endpoint."""
    receipt = read_json_shared(fit_dir / "CHECKPOINT_RECEIPT.json")
    checkpoint = Path(receipt["checkpoint_path"])
    endpoint = _recipe(fit_id)[0]
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("completed_updates") != endpoint
        or receipt.get("checkpoint_sha256") != _sha256(checkpoint)
    ):
        raise ValueError("head endpoint is incomplete or changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = _model(fit_id)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).float().eval().requires_grad_(False)


def pair_point_phase(model: nn.Module, features133: Tensor) -> Tensor:
    """Evaluate a frozen PAIR endpoint on its exact FP32 133-D cache input."""
    if features133.ndim != 2 or features133.shape[1] != 133 or features133.dtype != torch.float32:
        raise ValueError("PAIR inference requires FP32[B,133]")
    return model(PairFeatureBatch(features133))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--fit-id", choices=sorted(FIT_IDS), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-updates-this-call", type=int)
    args = parser.parse_args(argv)
    result = fit_head(
        fit_id=args.fit_id,
        cache_path=args.cache.resolve(strict=True),
        freeze_path=args.freeze.resolve(strict=True),
        run=args.run.resolve(),
        resume=args.resume,
        device_name=args.device,
        max_updates_this_call=args.max_updates_this_call,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "COMPLETE" else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FIT_IDS",
    "SealedCache",
    "fit_head",
    "load_head_endpoint",
    "main",
    "pair_point_phase",
]
