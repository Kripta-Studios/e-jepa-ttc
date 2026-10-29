"""Train all A5/C2F parameters on TRAIN40 with the retained architecture and losses."""

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
import torch

from operational.efficient_context.common import Lease, atomic_json, digest
from operational.train40_system.checkpoint import DurableState
from operational.train40_system.data_audit import OUTPUT

if TYPE_CHECKING:
    from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch

_heavy_scan_at = float("-inf")
_other_heavy_pids: list[int] = []


class Inputs:
    """Share one bounded RAM cache across four lossless shard-decode threads."""

    def __init__(self, output: Path) -> None:
        import pandas as pd

        self.output = output
        freeze = json.loads((output / "PREPARE_FREEZE.json").read_text(encoding="utf-8"))
        self.directory = Path(freeze["cache_root"])
        self.metadata = pd.read_parquet(
            output / "TRAIN40_ROWS.parquet", columns=["sequence_id", "sample_token", "track_id"]
        ).to_dict(orient="records")
        self.cache: OrderedDict[int, dict] = OrderedDict()
        self.cache_bytes = 0
        self.limit = 8 * 1024**3
        self.lock = Lock()
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.reads, self.hits = 0, 0

    def shard(self, number: int) -> dict:
        """Decode the canonical event shard and the teacher with the identical row order."""
        import psutil

        with self.lock:
            if number in self.cache:
                self.cache.move_to_end(number)
                self.hits += 1
                return self.cache[number]
        name = f"shard_{number:05d}.npz"
        with np.load(self.directory / name, allow_pickle=False) as stored:
            value = {key: stored[key] for key in stored.files}
        with np.load(self.output / "teacher" / name, allow_pickle=False) as stored:
            if not np.array_equal(value["ordinals"], stored["ordinals"]):
                raise ValueError("Teacher/event ordinals differ")
            if not np.array_equal(value["tokens"], stored["tokens"]):
                raise ValueError("Teacher/event object tokens differ")
            value["relation_targets"] = stored["relation_targets"]
            value["relation_valid"] = stored["relation_valid"]
        size = sum(x.nbytes for x in value.values())
        with self.lock:
            self.reads += 1
            while self.cache and (
                self.cache_bytes + size > self.limit
                or psutil.virtual_memory().available < 4 * 1024**3
            ):
                _, old = self.cache.popitem(last=False)
                self.cache_bytes -= sum(x.nbytes for x in old.values())
            if self.cache_bytes + size <= self.limit:
                self.cache[number] = value
                self.cache_bytes += size
        return value

    def batch(self, ids: list[int]) -> ObjectEventV4Batch:
        """Use the original typed collator, keeping target and teacher outside model inputs."""
        from e_jepa_ttc.data.object_event_v4 import collate_object_event_v4

        futures = {n: self.pool.submit(self.shard, n) for n in sorted({i // 32 for i in ids})}
        shards = {n: f.result() for n, f in futures.items()}
        records = []
        for ordinal in ids:
            value, local = shards[ordinal // 32], ordinal % 32
            if int(value["ordinals"][local]) != ordinal:
                raise ValueError("Event shard ordinal drifted")
            metadata = self.metadata[ordinal]
            if value["tokens"][local] != metadata["sample_token"]:
                raise ValueError("Event shard sample identity drifted")
            records.append(
                {
                    **metadata,
                    "event_v4_common_roi": value["events"][local],
                    "garl_delta_t_s": value["delta"][local],
                    "observable_motion": value["motion"][local],
                    "garl_visible_heights_px": value["visible_heights"][local],
                    "ttc_s": value["target_ttc"][local],
                    "event_v4_boxes_xyxy": value["boxes"][local],
                    "event_v4_common_square_xyxy": value["square"][local],
                    "dinov3_relation_targets": value["relation_targets"][local],
                    "dinov3_relation_valid": value["relation_valid"][local],
                }
            )
        return collate_object_event_v4(records)

    def close(self) -> None:
        """Release decoder threads and cached tensors between the two heavy fits."""
        self.pool.shutdown(wait=True)
        self.cache.clear()
        self.cache_bytes = 0


def epoch_order(size: int, generator: torch.Generator) -> torch.Tensor:
    """Preserve the original shard-grouped shuffle with logical groups of 256 rows."""
    from e_jepa_ttc.training.causal_scale_eap import ShardGroupedRandomSampler

    groups = tuple(tuple(range(start, min(start + 256, size))) for start in range(0, size, 256))
    return torch.tensor(
        list(ShardGroupedRandomSampler(groups, dataset_size=size, generator=generator)),
        dtype=torch.int64,
    )


def resource_guard(output: Path) -> tuple[bool, dict]:
    """Admit each update against joint physical/recovery limits, RAM and the fixed deadline."""
    import psutil

    global _heavy_scan_at, _other_heavy_pids

    authorization = json.loads((output / "AUTHORIZATION.json").read_text(encoding="utf-8"))
    technical = json.loads((output / "TECHNICAL_ACCOUNTING.json").read_text(encoding="utf-8"))
    physical = authorization["previous_physical_execution_upper"]
    physical += technical["synthetic_optimizer_updates"]
    recovery = 0
    for path in (output / "fits").glob("*/UPDATE_JOURNAL.json"):
        journal = json.loads(path.read_text(encoding="utf-8"))
        physical += journal["committed_updates"] + journal["recovery_upper"]
        physical += journal["pending_update_upper"]
        recovery += journal["recovery_upper"]
    process = psutil.Process()
    tree = [process, *process.children(recursive=True)]
    rss = sum(p.memory_info().rss for p in tree if p.is_running())
    available = psutil.virtual_memory().available
    reasons = []
    if time.monotonic() - _heavy_scan_at >= 5:
        related = {
            process.pid,
            *(p.pid for p in process.parents()),
            *(p.pid for p in process.children(recursive=True)),
        }
        others = []
        for candidate in psutil.process_iter(["name", "cmdline"]):
            if candidate.pid in related:
                continue
            line = " ".join(candidate.info["cmdline"] or []).lower()
            if not (candidate.info["name"] or "").lower().startswith("python"):
                continue
            if any(
                marker in line
                for marker in (
                    "operational.train40_system.models",
                    "operational.train40_system.teacher",
                    "operational.efficient_context.garl_train",
                    "operational.efficient_context.garl_heads",
                    "operational.efficient_context.run train",
                )
            ) or ("train" in line and any(f"stage{s}" in line for s in range(70, 77))):
                others.append(candidate.pid)
        _other_heavy_pids = others
        _heavy_scan_at = time.monotonic()
    if _other_heavy_pids:
        reasons.append("ANOTHER_HEAVY_TRAINER")
    if physical >= 240000:
        reasons.append("PHYSICAL_UPDATE_CAP")
    if recovery >= 2000:
        reasons.append("NEW_RECOVERY_RESERVE_EXHAUSTED")
    if rss >= 16_000_000_000 or available < 2 * 1024**3:
        reasons.append("RAM_RESOURCE_BOUNDARY")
    if datetime.now(UTC) >= datetime.fromisoformat(authorization["deadline_utc"]):
        reasons.append("USER_DEADLINE")
    from shutil import disk_usage

    if disk_usage(output).free < 20_000_000_000:
        reasons.append("CHECKPOINT_DISK_RESERVE")
    return not reasons, {
        "physical_updates_upper": physical,
        "new_recovery_upper": recovery,
        "tree_rss_bytes": rss,
        "host_available_bytes": available,
        "reasons": reasons,
        "other_heavy_pids": _other_heavy_pids,
    }


def run(output: Path, arm: str) -> None:
    """Complete eighteen train-only epochs and retain full checkpoints every 100 updates."""
    from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig
    from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
    from e_jepa_ttc.reproducibility import seed_everything
    from e_jepa_ttc.training.causal_scale_eap import (
        CausalScaleEAPTrainingConfig,
        _autocast,
        _foreground_only_loss_config,
        _loss,
        _module_tensor_sha256,
    )

    freeze_path = output / "MODELS_FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if digest(Path(__file__)) != freeze["source_sha256"]:
        raise ValueError("Trainer differs from its tested engineering freeze")
    recipe = json.loads((output / "TRAINING_PROTOCOL.json").read_text(encoding="utf-8"))
    if digest(output / "TRAINING_PROTOCOL.json") != freeze["protocol_sha256"]:
        raise ValueError("Scientific recipe changed")
    for name in (
        "INPUT_PREPARATION_PROGRESS.json",
        "TEACHER_PROGRESS.json",
        "MEDIA_VERIFICATION_PROGRESS.json",
    ):
        progress = json.loads((output / name).read_text(encoding="utf-8"))
        if progress["status"] != "COMPLETE":
            raise ValueError(f"Complete TRAIN40 prerequisite required: {name}")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    seed_everything(7, deterministic=True)
    arm_recipe = recipe["producers"][arm]
    config = CausalScaleEAPTrainingConfig(**arm_recipe["training_config"])
    loss_config = CausalScaleTTCLossConfig(**arm_recipe["loss_config"])
    warmup = _foreground_only_loss_config(loss_config)
    model = CausalScaleTTC(CausalScaleTTCConfig(**arm_recipe["model_config"])).to("cuda")
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError("All learned encoder/producer parameters must be trainable")
    initial_encoder = _module_tensor_sha256(model.encoder)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.minimum_learning_rate
    )
    generator = torch.Generator().manual_seed(7)
    directory = output / "fits" / f"{arm}_seed7"
    contract = {
        "fit": arm,
        "protocol_sha256": digest(output / "TRAINING_PROTOCOL.json"),
        "engineering_freeze_sha256": digest(freeze_path),
        "updates_limit": 49932,
        "recovery_upper_limit": 2000,
    }
    state = DurableState(directory, contract)
    cursor = {
        "epoch": 1,
        "position": 0,
        "order": epoch_order(88744, generator),
        "loss_sums": {},
        "examples": 0,
        "pending_curve": [],
        "initial_encoder_sha256": initial_encoder,
    }
    if state.path.is_file():
        cursor = state.restore(model, optimizer, scheduler, generator)
    else:
        state.save(model, optimizer, scheduler, generator, cursor, status="READY")
    data = Inputs(output)
    status = "RUNNING"
    try:
        model.train()
        while cursor["epoch"] <= config.epochs:
            permitted, telemetry = resource_guard(output)
            if not permitted:
                status = "PAUSED_RESOURCE"
                atomic_json(directory / "RESOURCE_PAUSE.json", telemetry)
                break
            epoch = cursor["epoch"]
            start = cursor["position"]
            ids = cursor["order"][start : start + config.batch_size].tolist()
            begun = time.perf_counter()
            host_batch = data.batch(ids)
            batch = host_batch.to(torch.device("cuda"))
            torch.cuda.synchronize()
            wait_ms = (time.perf_counter() - begun) * 1000
            begun = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with _autocast(torch.device("cuda"), config.precision):
                total, components, _ = _loss(
                    model,
                    batch,
                    warmup if epoch <= config.foreground_warmup_epochs else loss_config,
                    mask_t0_as_proxy=True,
                    foreground_supervision="bbox_geometry",
                    representation_supervision="dinov3_local_relational",
                    representation_distillation_weight=8.0,
                )
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite TRAIN40 training loss")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            state.begin_update()
            optimizer.step()
            state.commit_update()
            torch.cuda.synchronize()
            compute_ms = (time.perf_counter() - begun) * 1000
            losses = {
                name: float(value.detach().float().cpu()) for name, value in components.items()
            }
            losses["total"] = float(total.detach().float().cpu())
            for name, value in losses.items():
                cursor["loss_sums"][name] = cursor["loss_sums"].get(name, 0.0) + value * len(ids)
            cursor["examples"] += len(ids)
            cursor["position"] += len(ids)
            cursor["pending_curve"].append(
                {
                    "update": state.committed,
                    "epoch": epoch,
                    "examples": len(ids),
                    "losses": losses,
                    "data_wait_ms": wait_ms,
                    "compute_ms": compute_ms,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            if cursor["position"] == 88744:
                atomic_json(
                    directory / f"epoch_{epoch:02d}.json",
                    {
                        "epoch": epoch,
                        "updates": state.committed,
                        "examples": cursor["examples"],
                        "losses": {
                            k: v / cursor["examples"] for k, v in cursor["loss_sums"].items()
                        },
                        "encoder_sha256": _module_tensor_sha256(model.encoder),
                        "model_config": arm_recipe["model_config"],
                    },
                )
                scheduler.step()
                cursor["epoch"] += 1
                cursor["position"] = 0
                cursor["loss_sums"] = {}
                cursor["examples"] = 0
                cursor["order"] = epoch_order(88744, generator)
            if state.committed % 100 == 0 or cursor["epoch"] > config.epochs:
                endpoint = "COMPLETE" if cursor["epoch"] > config.epochs else "RUNNING"
                atomic_json(
                    directory / f"curve_{state.committed:06d}.json",
                    {"updates": cursor["pending_curve"], "durable_endpoint": state.committed},
                )
                cursor["pending_curve"] = []
                state.save(model, optimizer, scheduler, generator, cursor, status=endpoint)
                atomic_json(
                    directory / "PROGRESS.json",
                    {
                        "status": endpoint,
                        "committed_updates": state.committed,
                        "durable_updates": state.durable,
                        "total_updates": 49932,
                        "epoch": cursor["epoch"],
                        "position": cursor["position"],
                        "latest": losses,
                        "data_wait_ms": wait_ms,
                        "compute_ms": compute_ms,
                        "cache_bytes": data.cache_bytes,
                        "cache_reads": data.reads,
                        "cache_hits": data.hits,
                        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                        "resources": telemetry,
                    },
                )
        if cursor["epoch"] > config.epochs:
            status = "COMPLETE"
    except BaseException as exc:
        status = "PAUSED_FAILURE"
        atomic_json(
            directory / "FAILURE.json",
            {
                "error": f"{type(exc).__name__}: {exc}",
                "scientific_negative": False,
                "committed_updates": state.committed,
                "durable_updates": state.durable,
            },
        )
        raise
    finally:
        if not state.pending and status != "PAUSED_FAILURE":
            state.save(model, optimizer, scheduler, generator, cursor, status=status)
        data.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--arm", choices=("a5", "c2f"), required=True)
    args = parser.parse_args()
    with Lease(args.output.resolve()):
        run(args.output.resolve(), args.arm)
