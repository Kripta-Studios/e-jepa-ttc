"""FP32 fixed-update head engine with deterministic sampler and exact CPU resume.

The caller must validate real caches, lineage, frozen identities and resource
leases. This engine has no raw loader and cannot make those assertions itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from .model import TemporalConfig, TemporalRefiner, training_loss


class QuerySource(Protocol):
    """Uniform-query gather from prevalidated deduplicated features, never HDF5."""

    identity_sha256: str

    @property
    def population(self) -> int:
        """Read-only query count; both stored attributes and properties conform."""
        ...

    def gather(self, query_ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return features, timing, valid, original experts, target and global mass."""
        ...


class FitJournal(Protocol):
    """Durable work accounting; called only at safe engine boundaries."""

    def start(self, completed: int, checkpoint: Path) -> None: ...

    def before_update(self, completed: int) -> None: ...

    def checkpoint_saved(self, completed: int, checkpoint: Path) -> None: ...


def learning_rate(update: int) -> float:
    """One-based update: warmup100, cosine to exactly 3e-5 at update2500."""
    if not 1 <= update <= 2500:
        raise ValueError("update outside frozen endpoint")
    if update <= 100:
        return 3e-4 * update / 100
    return 3e-5 + (3e-4 - 3e-5) * (1 + math.cos(math.pi * (update - 100) / 2400)) / 2


def state_digest(value: Any) -> str:  # noqa: ANN401 -- typed recursive checkpoint boundary
    """Hash every tensor, scalar and container, independent of torch archive layout."""
    digest = hashlib.sha256()

    def update(item: Any) -> None:  # noqa: ANN401 -- nested optimizer state
        if isinstance(item, Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor:")
            update(str(tensor.dtype))
            update(tuple(tensor.shape))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(f"dict:{len(item)}:".encode())
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"{type(item).__name__}:{len(item)}:".encode())
            for child in item:
                update(child)
        elif item is None or type(item) in {str, int, float, bool}:
            encoded = json.dumps(item, allow_nan=False, ensure_ascii=True).encode()
            digest.update(f"{type(item).__name__}:{len(encoded)}:".encode())
            digest.update(encoded)
        else:
            raise TypeError(f"unsupported checkpoint member: {type(item).__name__}")

    update(value)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Reject incomplete, legacy-unsealed or modified state before loading a model."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "state_sha256" not in state:
        raise ValueError("checkpoint lacks complete-state integrity digest")
    expected = state.pop("state_sha256")
    if state_digest(state) != expected:
        raise ValueError("checkpoint state integrity mismatch")
    count = state.get("completed_updates")
    if type(count) is not int or not 0 <= count <= 2500:
        raise ValueError("invalid checkpoint update count")
    if len(state.get("losses", [])) != count or len(state.get("sampler_hashes", [])) != count:
        raise ValueError("checkpoint progress logs disagree with optimizer update count")
    if state.get("status") == "COMPLETED" and count != 2500:
        raise ValueError("partial checkpoint cannot claim a scientific endpoint")
    return state


def atomic_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Publish a complete checkpoint with durable bytes before atomic replace."""
    if "state_sha256" in state:
        raise ValueError("checkpoint payload already contains an integrity digest")
    sealed = {**state, "state_sha256": state_digest(state)}
    temporary = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as stream:
        torch.save(sealed, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def fit(
    source: QuerySource,
    config: TemporalConfig,
    output: Path,
    *,
    seed: int,
    freeze_sha256: str,
    resource_ok: Callable[[], bool],
    resume: bool = False,
    stop_after: int = 2500,
    journal: FitJournal | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train to an exact endpoint or save a resource pause; never score partials.

    stop_after is for accounted technical/resume probes only. Production callers
    must keep 2500. CUDA is currently limited to accounted technical probes;
    scientific CUDA admission requires a separate validated device recipe.
    """
    if not 1 <= stop_after <= 2500 or seed not in {7, 13, 23} or source.population < 1:
        raise ValueError("invalid registered fit arguments")
    if len(freeze_sha256) != 64 or len(source.identity_sha256) != 64:
        raise ValueError("frozen source and implementation identity required")
    if device not in {"cpu", "cuda:0"}:
        raise ValueError("supported devices are cpu and cuda:0")
    if device != "cpu":
        if stop_after == 2500:
            raise ValueError("CUDA scientific admission pending device benchmark and freeze")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("CUDA probe requires CUBLAS_WORKSPACE_CONFIG=:4096:8 at launch")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no automatic CPU fallback")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() > 2:
        torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = TemporalRefiner(config).float().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    identity = {
        "source": source.identity_sha256,
        "freeze": freeze_sha256,
        "config": asdict(config),
        "seed": seed,
        "device": device,
        "torch_version": str(torch.__version__),
        "batch": 128,
        "endpoint": 2500,
    }
    if device != "cpu":
        identity["cuda_recipe"] = {
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "tf32": False,
            "cublas_workspace": ":4096:8",
        }
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint_last.pt"
    completed = 0
    losses: list[float] = []
    sampler_hashes: list[str] = []
    if resume:
        state = load_checkpoint(checkpoint)
        if state["identity_sha256"] != identity_hash:
            raise ValueError("resume identity mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["sampler_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device != "cpu":
            torch.cuda.set_rng_state(state["cuda_rng"], device=device)
        completed = state["completed_updates"]
        losses = state["losses"]
        sampler_hashes = state["sampler_hashes"]
    elif checkpoint.exists():
        raise FileExistsError("existing fit requires explicit resume")
    if completed > stop_after:
        raise ValueError("cannot rewind fixed fit")
    journal_ready = False

    def save(status: str) -> dict[str, Any]:
        state = {
            "identity": identity,
            "identity_sha256": identity_hash,
            "completed_updates": completed,
            "status": status,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sampler_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "losses": losses,
            "sampler_hashes": sampler_hashes,
        }
        if device != "cpu":
            state["cuda_rng"] = torch.cuda.get_rng_state(device)
        atomic_checkpoint(checkpoint, state)
        if journal is not None and journal_ready:
            journal.checkpoint_saved(completed, checkpoint)
        return {
            "status": status,
            "completed_updates": completed,
            "scientific_endpoint": completed == 2500,
            "identity_sha256": identity_hash,
        }

    if journal is not None:
        if not checkpoint.exists():
            save("IN_PROGRESS")  # A durable update0 restart point precedes any reservation.
        journal.start(completed, checkpoint)
        journal_ready = True

    model.train()
    for update in range(completed + 1, stop_after + 1):
        if not resource_ok():
            return save("PAUSED_RESOURCE")
        if journal is not None:
            journal.before_update(completed)
        ids = torch.randint(source.population, (128,), generator=generator)
        x, timing, valid, experts, truth, mass = source.gather(ids)
        if any(value.device.type != "cpu" for value in (x, timing, valid, experts, truth, mass)):
            raise ValueError("CPU recipe only")
        for value in (x, timing, experts, truth, mass):
            if value.dtype != torch.float32:
                raise ValueError("FP32 training tensors required")
        if device != "cpu":
            x, timing, valid, experts, truth, mass = (
                value.to(device) for value in (x, timing, valid, experts, truth, mass)
            )
        optimizer.param_groups[0]["lr"] = learning_rate(update)
        optimizer.zero_grad(set_to_none=True)
        loss = training_loss(
            model(x, timing, valid, experts),
            truth,
            experts,
            mass,
            source.population,
            selector_only=config.output_mode == "selector",
        )
        if not torch.isfinite(loss):
            raise ArithmeticError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        completed = update
        losses.append(float(loss.detach()))
        sampler_hashes.append(hashlib.sha256(ids.numpy().tobytes()).hexdigest())
        if update % 100 == 0:
            save("IN_PROGRESS")
        if update % 25 == 0:
            print(json.dumps({"update": update, "loss": losses[-1]}), flush=True)
    return save("COMPLETED" if completed == 2500 else "TECHNICAL_PARTIAL")
