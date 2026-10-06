"""Admitted overlap variant of the frozen TRAIN40 encoder trainer."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from operational.efficient_context.common import ROOT, Lease, digest
from operational.train40_system.checkpoint import DurableState
from operational.train40_system.contracts import environment, read, verify_sources
from operational.train40_system.coordination import (
    DevicePrefetch,
    relational_diagnostic,
    scalar_metrics,
)
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json
from operational.train40_system.models import Inputs, epoch_order, resource_guard

SOURCE_REPLACEMENTS = {
    "engineering_admission": "separate coordination freeze; original model freeze retained",
    "input_transfer": "bounded depth-three pinned asynchronous device prefetch",
    "relational_diagnostic": "admitted vectorized callback installed around the original loss",
    "scalar_reporting": "one ordered FP32 host transfer",
    "synchronization": "current compute stream rather than whole-device synchronization",
    "checkpoint_cadence": "original cadence plus every epoch boundary",
    "coordination_pause": "full checkpoint after a committed optimizer boundary",
    "runtime_provenance": "sidecar only; checkpoint contract unchanged",
    "performance_metadata": "perf_counter observations without new GPU synchronization",
}


class SealedInputs(Inputs):
    """Check immutable file identities before consuming a verified input or teacher shard."""

    def __init__(self, output: Path) -> None:
        super().__init__(output)
        self.identities = {}
        for name in ("INPUT", "TEACHER"):
            manifest = read(output / f"{name}_MANIFEST.json")
            directory = Path(manifest["directory"])
            for entry in manifest["ordered_shards"]:
                self.identities[directory / entry["name"]] = entry

    def shard(self, number: int) -> dict:
        """Reject any post-admission input mutation, including when tensors are in RAM."""
        for directory in (self.directory, self.output / "teacher"):
            path = directory / f"shard_{number:05d}.npz"
            entry, stat = self.identities[path], path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (entry["bytes"], entry["mtime_ns"]):
                raise ValueError(f"Sealed TRAIN40 shard changed: {path}")
        return super().shard(number)


def pause_request(output: Path) -> Path | None:
    """Return a user-owned coordination stop marker without consuming or mutating it."""
    for name in ("COORDINATION_PAUSE_REQUEST.json", "STOP_REQUEST"):
        marker = output / name
        if marker.is_file():
            return marker
    return None


def run(output: Path, arm: str) -> None:
    """Complete eighteen train-only epochs and retain full checkpoints every 100 updates."""
    import e_jepa_ttc.training.causal_scale_eap as training
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
    verify_sources(freeze)
    original_engine = ROOT / "operational/train40_system/engine.py"
    if digest(original_engine) != freeze["trainer_sha256"]:
        raise ValueError("Original trainer differs from its tested engineering freeze")
    coordination_path = output / "COORDINATION_FREEZE.json"
    coordination = read(coordination_path)
    verify_sources(coordination)
    if digest(Path(__file__)) != coordination["trainer_sha256"]:
        raise ValueError("Overlap trainer differs from its tested coordination freeze")
    if coordination["models_freeze_sha256"] != digest(freeze_path):
        raise ValueError("Coordination admission does not bind the active model freeze")
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
    effective_training = dict(arm_recipe["training_config"])
    effective_training["representation_teacher_cache_artifact_sha256"] = freeze[
        "teacher_manifest_sha256"
    ]
    config = CausalScaleEAPTrainingConfig(**effective_training)
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
    for kind in ("input", "teacher"):
        manifest_path = output / (kind.upper() + "_MANIFEST.json")
        if digest(manifest_path) != freeze[kind + "_manifest_sha256"]:
            raise ValueError(f"Sealed {kind} population changed")
        manifest = read(manifest_path)
        if manifest["row_count"] != 88744 or manifest["status"] != "COMPLETE_VERIFIED":
            raise ValueError("Whole TRAIN40 manifest required")
    identity_path = directory / "FIT_IDENTITY.json"
    if not identity_path.exists():
        atomic_json(
            identity_path,
            {"environment": environment(), "start_time": datetime.now(UTC).isoformat()},
        )
    contract = {
        "fit": arm,
        "model_config": arm_recipe["model_config"],
        "loss_config": arm_recipe["loss_config"],
        "training_config": effective_training,
        "seed": 7,
        "split_version": "public_TRAIN40_all_train",
        "dataset_manifest_sha256": freeze["input_manifest_sha256"],
        "teacher_manifest_sha256": freeze["teacher_manifest_sha256"],
        "environment_at_start": read(identity_path),
        "protocol_sha256": digest(output / "TRAINING_PROTOCOL.json"),
        "engineering_freeze_sha256": digest(freeze_path),
        "updates_limit": 49932,
        "recovery_upper_limit": 2000,
    }
    atomic_json(
        directory / "COORDINATION_RUNTIME.json",
        {
            "schema": "train40_coordination_runtime_v1",
            "coordination_freeze_sha256": digest(coordination_path),
            "models_freeze_sha256": digest(freeze_path),
            "original_engine_sha256": digest(original_engine),
            "overlap_engine_sha256": digest(Path(__file__)),
            "checkpoint_contract_unchanged": True,
        },
    )
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
    data = SealedInputs(output)
    prefetch = DevicePrefetch(data, depth=3)
    original_diagnostic = training._record_relational_fg_bg_diagnostic
    training._record_relational_fg_bg_diagnostic = relational_diagnostic
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
            ticket = prefetch.take(cursor["order"], start, config.batch_size)
            batch = ticket.consume()
            torch.cuda.current_stream().synchronize()
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
            loss_finished = time.perf_counter()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip_norm, error_if_nonfinite=True
            )
            backward_clip_finished = time.perf_counter()
            journal_begin_started = time.perf_counter()
            state.begin_update()
            journal_begin_finished = time.perf_counter()
            optimizer.step()
            optimizer_submitted = time.perf_counter()
            state.commit_update()
            journal_commit_finished = time.perf_counter()
            torch.cuda.current_stream().synchronize()
            current_stream_sync_finished = time.perf_counter()
            compute_ms = (current_stream_sync_finished - begun) * 1000
            stage_ms = {
                "loss_through_finite_check_cpu_elapsed": (loss_finished - begun) * 1000,
                "backward_clip_cpu_elapsed": (backward_clip_finished - loss_finished) * 1000,
                "journal_reserve_fsync_cpu_elapsed": (
                    journal_begin_finished - journal_begin_started
                )
                * 1000,
                "optimizer_step_cpu_submit_elapsed": (
                    optimizer_submitted - journal_begin_finished
                )
                * 1000,
                "journal_commit_fsync_cpu_elapsed": (
                    journal_commit_finished - optimizer_submitted
                )
                * 1000,
                "post_commit_current_stream_wait_elapsed": (
                    current_stream_sync_finished - journal_commit_finished
                )
                * 1000,
            }
            losses = scalar_metrics(components, total)
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
                    "stage_ms": stage_ms,
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
            marker = pause_request(output)
            if marker is not None:
                status = "PAUSED_COORDINATION"
                state.save(model, optimizer, scheduler, generator, cursor, status=status)
                atomic_json(
                    directory / "COORDINATION_PAUSE.json",
                    {
                        "status": status,
                        "request": marker.name,
                        "committed_updates": state.committed,
                        "durable_updates": state.durable,
                        "lost_updates": 0,
                        "scientific_negative": False,
                    },
                )
                break
            if (
                state.committed % 100 == 0
                or cursor["position"] == 0
                or cursor["epoch"] > config.epochs
            ):
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
                        "stage_ms": stage_ms,
                        "cache_bytes": data.cache_bytes,
                        "cache_reads": data.reads,
                        "cache_hits": data.hits,
                        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(),
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
        training._record_relational_fg_bg_diagnostic = original_diagnostic
        prefetch.close()
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
