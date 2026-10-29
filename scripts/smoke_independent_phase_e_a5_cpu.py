"""Run one real E_A5 FP32 CPU forward/loss/backward with zero optimizer updates."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import math
import os
import random
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RSS_LIMIT_BYTES = 3 * 1024**3


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _sha256(path: Path) -> str:
    with path.resolve(strict=True).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _tensor_inventory(output: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in dataclasses.fields(output):
        value = getattr(output, field.name)
        if hasattr(value, "shape") and hasattr(value, "isfinite"):
            if not bool(value.isfinite().all()):
                raise FloatingPointError(f"Non-finite model head: {field.name}")
            result[field.name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        elif isinstance(value, dict):
            diagnostics: dict[str, Any] = {}
            for name, tensor in value.items():
                if not hasattr(tensor, "isfinite") or not bool(tensor.isfinite().all()):
                    raise FloatingPointError(f"Non-finite model diagnostic: {name}")
                diagnostics[str(name)] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }
            result[field.name] = diagnostics
        elif value is not None:
            raise TypeError(f"Unexpected model output field type: {field.name}")
    return result


def _run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import numpy as np
    import pandas as pd
    import psutil
    torch: Any = importlib.import_module("torch")

    from e_jepa_ttc.data.object_event_v4 import collate_object_event_v4
    from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig
    from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
    from e_jepa_ttc.training.causal_scale_eap import _loss
    from operational.independent_refit.fitter import state_tensor_sha256
    from operational.independent_refit.producer_contract import verify_producer_plan
    from operational.independent_refit.producer_inputs import (
        IndependentEventProducerSource,
        SourceShard,
    )
    from operational.independent_refit.producer_train import verify_run_contract
    from operational.independent_refit.teacher_cache import TeacherRoleCache

    started = time.perf_counter()
    evidence = args.source_evidence.resolve(strict=True)
    output_dir = args.output.resolve()
    if output_dir.exists():
        raise FileExistsError(f"CPU engineering smoke output already exists: {output_dir}")
    first_batch_path = evidence / "FIRST_BATCH_RECEIPT.json"
    if _sha256(first_batch_path) != args.first_batch_sha256:
        raise ValueError("Prior real first-batch admission changed")
    first_batch = _json(first_batch_path)
    plan = _json(evidence / "PLAN.json")
    contract = _json(evidence / "RUN_CONTRACT.json")
    if (
        first_batch.get("status") != "PASS_REAL_CPU_FIRST_BATCH_NO_MODEL_NO_UPDATE"
        or args.sample_token not in first_batch.get("sample_tokens", [])
        or first_batch.get("plan_sha256") != plan.get("plan_sha256")
        or first_batch.get("run_contract_sha256") != contract.get("contract_sha256")
    ):
        raise ValueError("Fixed smoke token is not in the admitted real batch")
    verify_producer_plan(plan)
    verify_run_contract(contract)
    parents = cast(Mapping[str, Mapping[str, Any]], contract["parents"])
    launch = _json(evidence / "LAUNCH.json")
    input_root = Path(str(launch["input_root"])).resolve(strict=True)

    crosswalk = pd.read_parquet(Path(str(parents["crosswalk"]["path"])))
    matched = crosswalk.loc[crosswalk["sample_token"].astype(str) == args.sample_token]
    if len(matched) != 1:
        raise ValueError("Fixed smoke token does not have one crosswalk row")
    identity = matched.iloc[0]
    role_mask = pd.read_parquet(Path(str(parents["role_mask"]["path"])))
    role_match = role_mask.loc[role_mask["sample_token"].astype(str) == args.sample_token]
    if len(role_match) != 1 or bool(role_match.iloc[0]["history_valid"]) is not True:
        raise ValueError("Fixed smoke token is not history-valid")

    input_manifest = _json(Path(str(parents["input_manifest"]["path"])))
    shard_refs = cast(list[dict[str, Any]], input_manifest["ordered_shards"])
    shards = tuple(
        SourceShard(
            name=str(item["name"]),
            sha256=str(item["sha256"]),
            receipt_sha256=str(item["receipt_sha256"]),
            start=int(item["start"]),
            stop=int(item["stop"]),
            bytes=int(item["bytes"]),
        )
        for item in shard_refs
    )
    index = pd.DataFrame(
        [
            {
                **identity.to_dict(),
                "source_shard": str(identity["teacher_shard"]),
                "source_row": int(identity["teacher_local_row"]),
            }
        ]
    )
    source = IndependentEventProducerSource(
        source_root=input_root,
        shards=shards,
        index=index,
        history_invalid_tokens=frozenset(),
        teacher_manifest_sha256=str(
            cast(Mapping[str, str], plan["parent_sha256"])["teacher_manifest"]
        ),
        teacher_role_receipt_sha256=str(
            cast(Mapping[str, str], plan["parent_sha256"])["teacher_role"]
        ),
        max_cached_shards=1,
    )
    record = source.record(0)

    teacher_manifest_path = Path(str(parents["teacher_manifest"]["path"]))
    teacher_manifest = _json(teacher_manifest_path)
    teacher_root = teacher_manifest_path.resolve(strict=True).parent
    teacher_index_path = (teacher_root / str(teacher_manifest["index_path"])).resolve(strict=True)
    if _sha256(teacher_index_path) != teacher_manifest["index_sha256"]:
        raise ValueError("Teacher index changed")
    teacher_index = pd.read_parquet(teacher_index_path)
    teacher_row = teacher_index.loc[
        teacher_index["sample_token"].astype(str) == args.sample_token
    ]
    if len(teacher_row) != 1:
        raise ValueError("Teacher target token differs")
    teacher_name = str(teacher_row.iloc[0]["cache_shard"])
    teacher_reference = next(
        item
        for item in cast(list[dict[str, Any]], teacher_manifest["shards"])
        if str(item["name"]) == teacher_name
    )
    teacher_payload = teacher_root / "shards" / teacher_name
    teacher_receipt = teacher_payload.with_suffix(".json")
    if (
        _sha256(teacher_payload) != teacher_reference["sha256"]
        or _sha256(teacher_receipt) != teacher_reference["receipt_sha256"]
    ):
        raise ValueError("Selected teacher payload changed before smoke")
    teacher = TeacherRoleCache(
        root=teacher_root,
        index=teacher_row.reset_index(drop=True),
        allowed_tokens=frozenset({args.sample_token}),
        role_receipt_path=Path(str(parents["teacher_role_receipt"]["path"])),
        role_receipt_sha256=str(parents["teacher_role_receipt"]["sha256"]),
    )
    targets, valid = teacher.get(args.sample_token)
    if (
        targets.dtype != np.float16
        or targets.shape != (2, 6, 32, 32)
        or valid.dtype != np.bool_
        or valid.shape != targets.shape
    ):
        raise ValueError("Selected teacher relation target differs")
    record["dinov3_relation_targets"] = targets
    record["dinov3_relation_valid"] = valid
    batch = collate_object_event_v4([record])
    if list(batch.events.shape) != [1, 3, 12, 128, 128]:
        raise ValueError("Engineering smoke did not retain native T3/C12 input")

    process = psutil.Process()
    peak_rss = [int(process.memory_info().rss)]
    stop = threading.Event()
    exceeded = threading.Event()

    def watch_memory() -> None:
        while not stop.wait(0.02):
            rss = int(process.memory_info().rss)
            peak_rss[0] = max(peak_rss[0], rss)
            if rss > RSS_LIMIT_BYTES:
                exceeded.set()
                import _thread

                _thread.interrupt_main()
                return

    watcher = threading.Thread(target=watch_memory, daemon=True)
    watcher.start()
    try:
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        torch.set_num_threads(1)
        recipe = cast(Mapping[str, Any], plan["recipe"])
        config = CausalScaleTTCConfig(**cast(dict[str, Any], recipe["model_config"]))
        model = CausalScaleTTC(config).cpu().train()
        initial_state_sha256 = state_tensor_sha256(model.state_dict())
        loss_config = CausalScaleTTCLossConfig(
            **cast(dict[str, Any], recipe["loss_config"])
        )
        total, components, model_output = _loss(
            model,
            batch,
            loss_config,
            mask_t0_as_proxy=True,
            foreground_supervision="bbox_geometry",
            representation_supervision="dinov3_local_relational",
            representation_distillation_weight=8.0,
        )
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("Engineering E_A5 total loss is non-finite")
        total.backward()
        output_inventory = _tensor_inventory(model_output)
        component_values = {
            name: float(value.detach()) for name, value in components.items()
        }
        if any(not math.isfinite(value) for value in component_values.values()):
            raise FloatingPointError("Engineering E_A5 component loss is non-finite")
        gradients = [
            (name, parameter.grad)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients or any(
            not bool(torch.isfinite(gradient).all()) for _, gradient in gradients
        ):
            raise FloatingPointError("Engineering E_A5 gradients are absent or non-finite")
        grad_norm = math.sqrt(
            sum(float(gradient.detach().double().square().sum()) for _, gradient in gradients)
        )
        if not math.isfinite(grad_norm) or grad_norm <= 0.0:
            raise FloatingPointError("Engineering E_A5 gradient norm differs")
    finally:
        stop.set()
        watcher.join(timeout=2.0)
    peak_rss[0] = max(peak_rss[0], int(process.memory_info().rss))
    if exceeded.is_set() or peak_rss[0] > RSS_LIMIT_BYTES:
        raise MemoryError("Engineering E_A5 smoke exceeded its 3 GiB RSS guard")
    if (
        _sha256(teacher_payload) != teacher_reference["sha256"]
        or _sha256(teacher_receipt) != teacher_reference["receipt_sha256"]
    ):
        raise ValueError("Selected teacher payload changed during smoke")

    receipt: dict[str, Any] = {
        "schema": "independent_phase_e_a5_cpu_engineering_smoke_v1",
        "status": "PASS_ENGINEERING_FP32_CPU_FORWARD_LOSS_BACKWARD_NO_UPDATE",
        "scientific_fit_claim": False,
        "bf16_gpu_equivalence_claim": False,
        "sample_token": args.sample_token,
        "plan_sha256": plan["plan_sha256"],
        "run_contract_sha256": contract["contract_sha256"],
        "first_batch_receipt_sha256": args.first_batch_sha256,
        "input_shape": list(batch.events.shape),
        "input_dtype": str(batch.events.dtype),
        "model_config": dataclasses.asdict(config),
        "initial_state_sha256": initial_state_sha256,
        "output_inventory": output_inventory,
        "total_loss": float(total.detach()),
        "loss_components": component_values,
        "parameters_with_finite_gradients": len(gradients),
        "trainable_parameters": sum(1 for value in model.parameters() if value.requires_grad),
        "missing_gradient_parameters": sorted(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ),
        "gradient_l2_norm": grad_norm,
        "rss_limit_bytes": RSS_LIMIT_BYTES,
        "peak_rss_bytes": peak_rss[0],
        "wall_seconds": time.perf_counter() - started,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "cuda_used": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "teacher_payload_hash_verified_before_and_after": True,
        "input_payload_hash_verified_by_reader_before_and_after": True,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_dir.mkdir(parents=True)
    _write_new(output_dir / "RECEIPT.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-token", required=True)
    parser.add_argument("--first-batch-sha256", required=True)
    args = parser.parse_args(argv)
    result = _run(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
