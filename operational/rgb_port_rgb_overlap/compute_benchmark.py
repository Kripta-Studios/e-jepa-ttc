"""Read-only zero-update ABBA admission experiment for constant transport masks.

Requires a safely paused producer. It never writes checkpoints, optimizers,
samplers or admission manifests. Strict equality is reported, never inferred
from tolerance, and original-vs-original repeatability is measured separately.
"""

# ruff: noqa: ANN401 -- recursive audit snapshots cover tensor dataclasses and RNG tuples
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import random
import statistics
import subprocess
import time
from collections.abc import Mapping
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.models import local_transport
from e_jepa_ttc.training.causal_scale_eap import _autocast, _foreground_only_loss_config
from operational.rgb_port import train_producers as training
from operational.rgb_port.accounting import read_json_shared, sha256_file
from operational.rgb_port.recipe import ProducerRecipe
from operational.rgb_port.run import make_bound_rgb_producer_source
from operational.rgb_port_revision.loss import EffectiveSource
from operational.rgb_port_rgb_loader.collate import installed_collate
from operational.rgb_port_rgb_loader.contracts import MAX_DINO_BYTES, MAX_DINO_SHARDS
from operational.rgb_port_rgb_loader.loader import installed_dino_cache

from .compute import CandidateValidityCache, installed_compute, local_correlation_match_cached


def snapshot(value: Any) -> Any:
    """Hash every tensor's exact bytes, including BF16, without decimal conversion."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        # contiguous() may preserve a non-unit stride for singleton dimensions.
        # An explicit one-dimensional allocation guarantees safe dtype viewing,
        # including scalar, strided singleton, BF16 and empty diagnostics.
        flat = torch.empty(tensor.numel(), dtype=tensor.dtype, device="cpu")
        flat.copy_(tensor.reshape(-1))
        raw = flat.view(torch.uint8).numpy().tobytes()
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: snapshot(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [snapshot(item) for item in value]
    return value


def _rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state().clone(),
        "cuda": [item.clone() for item in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None,
    }


def _restore_rng(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["cpu"])
    if value["cuda"] is not None:
        torch.cuda.set_rng_state_all(value["cuda"])


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def matcher_parity(device: torch.device) -> list[dict[str, Any]]:
    """Compare local forward/input gradients at real grid sizes in BF16 autocast."""
    records = []
    generator = torch.Generator().manual_seed(7001)
    for batch, radius, size in ((2, 2, 16), (3, 4, 16), (2, 4, 32), (3, 2, 32)):
        left = torch.randn(batch, 64, size, size, generator=generator).to(device, torch.bfloat16)
        right = torch.randn(batch, 64, size, size, generator=generator).to(device, torch.bfloat16)
        cache = CandidateValidityCache()
        outcomes = []
        for variant in ("A1", "B1", "B2", "A2"):
            previous, current = left.clone().requires_grad_(), right.clone().requires_grad_()
            with _autocast(device, "bf16"):
                kwargs: dict[str, Any] = dict(
                    radius=radius, temperature=0.1, return_probability=True
                )
                result = (
                    local_transport.local_correlation_match(previous, current, **kwargs)
                    if variant.startswith("A")
                    else local_correlation_match_cached(previous, current, **kwargs, cache=cache)
                )
                loss = torch.stack(
                    [
                        getattr(result, name).square().mean()
                        for name in ("dx", "dy", "confidence_margin", "entropy", "probability")
                    ]
                ).sum()
            loss.backward()
            outcomes.append(
                snapshot({"output": result, "left_grad": previous.grad, "right_grad": current.grad})
            )
        records.append(
            {
                "batch": batch,
                "radius": radius,
                "grid": size,
                "baseline_repeat_exact": outcomes[0] == outcomes[3],
                "candidate_cold_exact": outcomes[0] == outcomes[1],
                "candidate_hit_exact": outcomes[0] == outcomes[2],
                "cache": cache.snapshot(),
                "outcomes": outcomes,
            }
        )
        cache.clear()
    return records


def main() -> int:
    """Benchmark identical real batches without consuming an optimizer update."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--fit-id", choices=("R_A5", "R_C2F"), default="R_A5")
    args = parser.parse_args()
    if not 1 <= args.batches <= 8:
        raise ValueError("Benchmark requires 1..8 real batches")
    run = args.run.resolve(strict=True)
    fit = run / "fits" / args.fit_id
    pointer = read_json_shared(fit / "CHECKPOINT_POINTER.json")
    journal = read_json_shared(fit / "UPDATE_JOURNAL.json")
    receipt = read_json_shared(fit / "CHECKPOINT_RECEIPT.json")
    if (
        int(journal["completed_updates"]) != int(pointer["completed_updates"])
        or int(journal["pending_update_upper"]) != 0
        or receipt.get("status") not in {"PAUSED_RESOURCE", "COMPLETE", "READY"}
        or (receipt.get("status") != "COMPLETE" and not (fit / "STOP_REQUEST").exists())
    ):
        raise RuntimeError(
            "Pause the producer at a durable checkpoint and retain STOP_REQUEST first"
        )
    checkpoint = fit / "checkpoint_versions" / str(pointer["version"])
    if sha256_file(checkpoint) != pointer["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA differs from pointer")
    protected = {
        str(path): sha256_file(path)
        for path in (
            checkpoint,
            fit / "CHECKPOINT_POINTER.json",
            fit / "CHECKPOINT_RECEIPT.json",
            fit / "UPDATE_JOURNAL.json",
        )
    }
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = payload["identity"]
    recipe = ProducerRecipe(
        **{field.name: identity[field.name] for field in dataclasses.fields(ProducerRecipe)}
    )
    if recipe.identity_sha256 != identity["recipe_identity_sha256"]:
        raise ValueError("Checkpoint recipe identity differs")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_num_threads(2)
    outer_rng = _rng()
    source = None
    result: dict[str, Any] = {
        "schema": "rgb_constant_mask_compute_abba_v1",
        "status": "INCOMPLETE",
        "started_utc": datetime.now(UTC).isoformat(),
        "optimizer_updates": 0,
        "device": str(device),
        "fit_id": args.fit_id,
        "completed_updates": pointer["completed_updates"],
        "checkpoint_sha256": pointer["checkpoint_sha256"],
        "recipe_identity_sha256": recipe.identity_sha256,
        "precision": "bf16",
        "timing_scope": (
            "GPU-ready real inputs; forward/loss/backward/clip; "
            "excludes input, optimizer, snapshot copies"
        ),
        "trials": [],
        "strict_exact_required": True,
    }
    try:
        repository = Path(__file__).resolve().parents[2]
        result["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        result["git_status_porcelain"] = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        source_paths = (
            Path(__file__),
            Path(__file__).with_name("compute.py"),
            Path(local_transport.__file__),
            Path(training.__file__),
            repository / "src/e_jepa_ttc/models/causal_scale_ttc.py",
            repository / "src/e_jepa_ttc/losses/causal_scale_ttc.py",
            repository / "operational/rgb_port_revision/loss.py",
        )
        result["source_sha256"] = {str(path.resolve()): sha256_file(path) for path in source_paths}
        result["environment"] = {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        }
        source = make_bound_rgb_producer_source(run / "P_MANIFEST.json")
        effective = EffectiveSource(
            source, recipe, str(identity["source_identity"]["audit_revision_sha256"])
        )
        cursor = payload["cursor"]
        position = int(cursor["position"])
        order = cursor["order"]
        ids_list = [
            order[start : start + recipe.effective_batch_size].tolist()
            for start in range(
                position,
                position + args.batches * recipe.effective_batch_size,
                recipe.effective_batch_size,
            )
        ]
        if any(not ids for ids in ids_list):
            raise ValueError("Requested batches cross saved epoch end; use fewer batches")
        with (
            installed_dino_cache(source, max_shards=MAX_DINO_SHARDS, max_bytes=MAX_DINO_BYTES),
            installed_collate(source),
        ):
            batches = [effective.batch(ids, "rgb").to(device) for ids in ids_list]
        result["batch_ids"] = ids_list
        result["batch_parts"] = [
            [list(part.events.shape) for part in batch.parts] for batch in batches
        ]
        result["covers_mixed_t2_t3"] = any(
            {int(part.events.shape[1]) for part in batch.parts} == {2, 3} for batch in batches
        )
        model = training.build_producer(recipe).to(device).train()
        full = training.CausalScaleTTCLossConfig(**recipe.loss_config)
        loss_config = _foreground_only_loss_config(full) if int(cursor["epoch"]) <= 3 else full
        initial_state = {
            key: value.detach().cpu().clone() for key, value in payload["model_state_dict"].items()
        }
        trial_rng = {
            "python": payload["python_random_state"],
            "numpy": payload["numpy_random_state"],
            "cpu": payload["torch_rng_state"],
            "cuda": payload["cuda_rng_state_all"],
        }
        result["matcher_parity"] = matcher_parity(device)

        def trial(name: str, *, record: bool) -> dict[str, Any]:
            model.load_state_dict(initial_state, strict=True)
            _restore_rng(trial_rng)
            batch_results = []
            cache_context = installed_compute() if name.startswith("B") else nullcontext(None)
            with cache_context as cache:
                for batch in batches:
                    model.zero_grad(set_to_none=True)
                    training._validate_batch(batch, recipe)
                    outputs = []
                    handle = model.register_forward_hook(
                        lambda module, inputs, output, captured=outputs: captured.append(output)
                    )
                    _sync(device)
                    started = time.perf_counter()
                    try:
                        with _autocast(device, "bf16"):
                            total, components = training._producer_loss(model, batch, loss_config)
                        if not bool(torch.isfinite(total)):
                            raise FloatingPointError("Nonfinite benchmark loss")
                        (total * 1.0).backward()
                        _sync(device)
                        before_clip_s = time.perf_counter() - started
                        # Snapshot outside timing; restores neither grads nor model arithmetic.
                        preclip = (
                            snapshot(
                                {key: parameter.grad for key, parameter in model.named_parameters()}
                            )
                            if record
                            else None
                        )
                        _sync(device)
                        clip_started = time.perf_counter()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            float(recipe.training_config["grad_clip_norm"]),
                            error_if_nonfinite=True,
                        )
                        _sync(device)
                        elapsed = before_clip_s + time.perf_counter() - clip_started
                        if record:
                            batch_results.append(
                                {
                                    "seconds": elapsed,
                                    "outputs": snapshot(outputs),
                                    "loss": snapshot(total),
                                    "components": snapshot(components),
                                    "preclip_gradients": preclip,
                                    "postclip_gradients": snapshot(
                                        {
                                            key: parameter.grad
                                            for key, parameter in model.named_parameters()
                                        }
                                    ),
                                }
                            )
                    finally:
                        handle.remove()
                stats = cache.snapshot() if cache is not None else None
            return {
                "name": name,
                "batches": batch_results,
                "cache": stats,
                "final_state": snapshot(model.state_dict()),
                "torch_rng_after": snapshot(torch.get_rng_state()),
                "cuda_rng_after": snapshot(torch.cuda.get_rng_state_all())
                if device.type == "cuda"
                else None,
            }

        # Warm up both implementations; the subsequent ABBA trials restore all
        # model state and checkpoint RNG, consuming zero optimizer updates.
        trial("A_warm", record=False)
        trial("B_warm", record=False)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        result["trials"] = [trial(name, record=True) for name in ("A1", "B1", "B2", "A2")]

        def comparable(item: dict[str, Any]) -> dict[str, Any]:
            return {
                "batches": [
                    {key: value for key, value in batch.items() if key != "seconds"}
                    for batch in item["batches"]
                ],
                "final_state": item["final_state"],
                "torch_rng_after": item["torch_rng_after"],
                "cuda_rng_after": item["cuda_rng_after"],
            }

        reference = comparable(result["trials"][0])
        result["baseline_repeat_exact"] = reference == comparable(result["trials"][3])
        result["candidate_trials_exact"] = [
            reference == comparable(item) for item in result["trials"][1:3]
        ]
        local_exact = all(
            item["baseline_repeat_exact"]
            and item["candidate_cold_exact"]
            and item["candidate_hit_exact"]
            for item in result["matcher_parity"]
        )
        result["exact_gpu_gradient_parity"] = (
            device.type == "cuda"
            and local_exact
            and result["baseline_repeat_exact"]
            and all(result["candidate_trials_exact"])
        )
        result["median_seconds_per_batch"] = {
            label: statistics.median(
                batch["seconds"]
                for item in result["trials"]
                if item["name"].startswith(label)
                for batch in item["batches"]
            )
            for label in ("A", "B")
        }
        result["speedup_ratio"] = (
            result["median_seconds_per_batch"]["A"] / result["median_seconds_per_batch"]["B"]
        )
        result["cuda_peak_allocated_bytes"] = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        )
        result["cuda_peak_reserved_bytes"] = (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        )
        result["status"] = "PASSED" if result["exact_gpu_gradient_parity"] else "NOT_ADMITTED"
    except BaseException as error:
        result["status"] = "FAILED"
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if source is not None:
            getattr(source, "close", lambda: None)()
        _restore_rng(outer_rng)
        result["protected_files_unchanged"] = all(
            sha256_file(Path(path)) == digest for path, digest in protected.items()
        )
        if not result["protected_files_unchanged"]:
            result["status"] = "NOT_ADMITTED"
        result["ended_utc"] = datetime.now(UTC).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"trials", "matcher_parity"}
            },
            indent=2,
        )
    )
    return 0 if result.get("status") == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
