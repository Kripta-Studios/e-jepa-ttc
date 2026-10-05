"""Full optimizer-boundary resume state and conservative physical-update accounting."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from operational.efficient_context.common import digest
from operational.train40_system.durable_io import atomic_json, replace


def payload_digest(value: object) -> str:
    """Hash the full resume state independent of archive layout and tensor device."""
    result = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            result.update(b"tensor:")
            visit(str(item.dtype))
            visit(tuple(item.shape))
            result.update(array.tobytes())
        elif isinstance(item, np.ndarray):
            result.update(b"numpy:")
            visit(str(item.dtype))
            visit(tuple(item.shape))
            result.update(item.tobytes())
        elif isinstance(item, dict):
            result.update(f"dict:{len(item)}:".encode())
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            result.update(f"{type(item).__name__}:{len(item)}:".encode())
            for child in item:
                visit(child)
        elif item is None or type(item) in (str, int, float, bool):
            encoded = json.dumps(item, allow_nan=False).encode()
            result.update(f"{type(item).__name__}:{len(encoded)}:".encode())
            result.update(encoded)
        else:
            raise TypeError(f"Unsupported checkpoint state: {type(item).__name__}")

    visit(value)
    return result.hexdigest()


class DurableState:
    """Keep model, optimizer, scheduler, sampler and every RNG in an atomic checkpoint."""

    def __init__(self, directory: Path, contract: dict) -> None:
        self.directory = directory
        self.contract = contract
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "checkpoint_last.pt"
        self.journal_path = directory / "UPDATE_JOURNAL.json"
        self.committed = 0
        self.durable = 0
        self.recovery_upper = 0
        self.pending = False

    def journal(self) -> None:
        """Publish conservative optimizer accounting before and after each update."""
        atomic_json(
            self.journal_path,
            {
                "contract": self.contract,
                "committed_updates": self.committed,
                "durable_updates": self.durable,
                "recovery_upper": self.recovery_upper,
                "pending_update_upper": int(self.pending),
            },
        )

    def begin_update(self) -> None:
        """Reserve at most one potentially executed update before optimizer.step()."""
        if self.pending:
            raise RuntimeError("Another optimizer update is already pending")
        if self.committed >= self.contract["updates_limit"]:
            raise RuntimeError("Fixed scientific update endpoint reached")
        self.pending = True
        self.journal()

    def commit_update(self) -> None:
        """Record exactly one completed optimizer update."""
        if not self.pending:
            raise RuntimeError("No optimizer update was reserved")
        self.committed += 1
        self.pending = False
        self.journal()

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        sampler: torch.Generator,
        cursor: dict,
        *,
        status: str,
    ) -> None:
        """Publish a complete checkpoint followed by its independently hashable receipt."""
        if self.pending:
            raise RuntimeError("Checkpoint requested during an incomplete optimizer update")
        payload = {
            "artifact_type": "train40_complete_resume_state_v1",
            "contract": self.contract,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "sampler_generator_state": sampler.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "numpy_random_state": np.random.get_state(),
            "python_random_state": random.getstate(),
            "cursor": cursor,
            "committed_updates": self.committed,
            "recovery_upper": self.recovery_upper,
            "status": status,
        }
        temporary = self.path.with_suffix(".pending.pt")
        payload["full_state_sha256"] = payload_digest(payload)
        torch.save(payload, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        replace(temporary, self.path)
        self.durable = self.committed
        self.journal()
        atomic_json(
            self.directory / "CHECKPOINT_RECEIPT.json",
            {
                "status": status,
                "sha256": digest(self.path),
                "committed_updates": self.committed,
                "recovery_upper": self.recovery_upper,
                "full_optimizer_scheduler_sampler_and_all_RNG": True,
            },
        )

    def restore(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        sampler: torch.Generator,
    ) -> dict:
        """Restore full state and account for all possibly lost optimizer updates."""
        saved = torch.load(self.path, map_location="cpu", weights_only=False)
        expected_state_sha = saved.pop("full_state_sha256", None)
        if expected_state_sha != payload_digest(saved):
            raise ValueError("Complete checkpoint state integrity differs")
        if saved["artifact_type"] != "train40_complete_resume_state_v1":
            raise ValueError("Wrong checkpoint schema")
        if saved["contract"] != self.contract:
            raise ValueError("Checkpoint source/config/data contract differs")
        journal = json.loads(self.journal_path.read_text(encoding="utf-8"))
        if journal["contract"] != self.contract:
            raise ValueError("Update accounting contract differs")
        self.committed = int(saved["committed_updates"])
        self.durable = self.committed
        lost_upper = max(0, int(journal["committed_updates"]) - self.committed)
        lost_upper += int(journal["pending_update_upper"])
        self.recovery_upper = (
            max(int(saved["recovery_upper"]), int(journal["recovery_upper"])) + lost_upper
        )
        if self.recovery_upper > self.contract["recovery_upper_limit"]:
            raise RuntimeError("Recovery budget would be exceeded; checkpoint preserved")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        sampler.set_state(saved["sampler_generator_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if saved["cuda_rng_state_all"] is not None:
            if not torch.cuda.is_available():
                raise ValueError("CUDA RNG checkpoint requires a CUDA host")
            torch.cuda.set_rng_state_all(saved["cuda_rng_state_all"])
        np.random.set_state(saved["numpy_random_state"])
        random.setstate(saved["python_random_state"])
        self.pending = False
        self.journal()
        return saved["cursor"]
