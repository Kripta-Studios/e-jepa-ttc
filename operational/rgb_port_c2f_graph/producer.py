"""Checkpoint-safe CUDA-graph overlay for RGB-PORT E_C2F_MATCHED."""

# ruff: noqa: ANN401

from __future__ import annotations

import argparse
import copy
import importlib
import random
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

import operational.rgb_port.train_producers as training
from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file
from operational.rgb_port_acceleration.producer import ForwardDispatcher, _same
from operational.rgb_port_c2f_graph.contracts import (
    FIT_ID,
    validate_c2f_graph_freeze,
    validate_c2f_graph_lineage,
)

DELEGATE = "operational.rgb_port_concurrent.producer"


class Runtime:
    """Install the admitted graph only after an exact full-state restore."""

    def __init__(
        self,
        source: training.ProducerSource,
        recipe: Any,
        run: Path,
        freeze_path: Path,
        freeze: Mapping[str, Any],
    ) -> None:
        if str(recipe.fit_id) != FIT_ID or int(recipe.microbatch_size) != 32:
            raise ValueError("C2F graph overlay requires E_C2F_MATCHED microbatch 32")
        self.source, self.recipe, self.run = source, recipe, run
        self.freeze_path, self.freeze = freeze_path, dict(freeze)
        self.freeze_file_sha = sha256_file(freeze_path)
        self.fit = run / "fits" / FIT_ID
        self.root = self.fit / "c2f_graph_checkpoints"
        self.pending = self.root / "PENDING_EXECUTION_RECEIPT.json"
        self.runtime_path = self.fit / "C2F_GRAPH_RUNTIME.json"
        self.prewarm_path = self.fit / "C2F_GRAPH_PREWARM_QA.json"
        self.dispatcher: ForwardDispatcher | None = None
        self.counters = {"warmup_backward_passes": 0}

    def _common(self) -> dict[str, Any]:
        return {
            "fit_id": FIT_ID,
            "c2f_graph_freeze_sha256": self.freeze["identity_sha256"],
            "c2f_graph_freeze_file_sha256": self.freeze_file_sha,
            "source_sha256": self.freeze["source_sha256"],
            "torch_backend_source_sha256": self.freeze["torch_backend_source_sha256"],
            "backend": "cudagraphs",
            "shape": {"batch_size": 32, "frames": 3},
            "optimizer_updates": 0,
        }

    def _write_runtime(self, state: Any, status: str, exact: Mapping[str, bool]) -> None:
        atomic_write_json(
            self.runtime_path,
            {
                "schema": "rgb_port_c2f_graph_runtime_v1",
                "status": status,
                **self._common(),
                "completed_updates": state.completed,
                "last_saved_update": state.completed,
                "checkpoint_sha256": sha256_file(state.path),
                "last_checkpoint_sha256": sha256_file(state.path),
                "post_warm_exact": dict(exact),
                "counters": {
                    **self.counters,
                    **(self.dispatcher.snapshot() if self.dispatcher else {}),
                },
            },
        )

    def before_save(self, state: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.pending,
            {
                "schema": "rgb_port_c2f_graph_pending_v1",
                "status": "PENDING",
                **self._common(),
                "start_update": state.durable,
                "end_update": state.completed,
                "checkpoint_version": f"checkpoint_{state.completed:06d}.pt",
                "checkpoint_identity_sha256": state.identity_sha256,
            },
        )

    def _receipt(
        self, state: Any, status: str, snapshot: Path, snapshot_sha: str
    ) -> dict[str, Any]:
        return {
            "schema": "rgb_port_c2f_graph_checkpoint_receipt_v1",
            "status": status,
            **self._common(),
            "completed_updates": state.completed,
            "checkpoint_version": f"checkpoint_{state.completed:06d}.pt",
            "checkpoint_sha256": sha256_file(state.path),
            "checkpoint_identity_sha256": state.identity_sha256,
            "runtime_snapshot": snapshot.name,
            "runtime_snapshot_sha256": snapshot_sha,
        }

    def _seal(self, state: Any, status: str) -> None:
        if self.dispatcher is None:
            raise RuntimeError("C2F graph checkpoint cannot precede graph prewarm")
        current = read_json_shared(self.runtime_path)
        current.update(
            status="ACTIVE",
            completed_updates=state.completed,
            last_saved_update=state.completed,
            checkpoint_sha256=sha256_file(state.path),
            last_checkpoint_sha256=sha256_file(state.path),
            counters={**self.counters, **self.dispatcher.snapshot()},
        )
        atomic_write_json(self.runtime_path, current)
        snapshot = self.root / f"runtime_{state.completed:06d}.json"
        if snapshot.exists() and read_json_shared(snapshot) != current:
            raise RuntimeError("Immutable C2F graph runtime snapshot differs")
        if not snapshot.exists():
            atomic_write_json(snapshot, current)
        atomic_write_json(
            self.root / f"checkpoint_{state.completed:06d}.json",
            self._receipt(state, status, snapshot, sha256_file(snapshot)),
        )
        self.pending.unlink(missing_ok=True)

    def after_save(self, state: Any) -> None:
        self._seal(state, "COMPLETE")

    def _repair_or_verify(self, state: Any) -> None:
        origin = self.freeze["origin"]
        if state.completed == int(origin["completed_updates"]):
            if (
                sha256_file(state.path) != origin["checkpoint_sha256"]
                or state.identity_sha256 != origin["identity_sha256"]
            ):
                raise RuntimeError("C2F graph native origin differs")
            return
        receipt = self.root / f"checkpoint_{state.completed:06d}.json"
        if receipt.exists():
            validate_c2f_graph_lineage(self.run, self.freeze_path)
            return
        if not self.pending.exists():
            raise RuntimeError("C2F graph checkpoint lacks receipt and pending proof")
        pending = read_json_shared(self.pending)
        pointer = read_json_shared(state.pointer)
        if (
            pending.get("schema") != "rgb_port_c2f_graph_pending_v1"
            or pending.get("status") != "PENDING"
            or any(pending.get(k) != v for k, v in self._common().items())
            or pending.get("end_update") != state.completed
            or pending.get("checkpoint_version") != pointer.get("version")
            or pending.get("checkpoint_identity_sha256") != state.identity_sha256
            or pointer.get("checkpoint_sha256") != sha256_file(state.path)
        ):
            raise RuntimeError("C2F graph pending proof cannot repair checkpoint")

    def after_restore(
        self,
        state: Any,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        generator: torch.Generator,
        cursor: dict[str, Any],
        original_restore: Any,
    ) -> dict[str, Any]:
        self._repair_or_verify(state)
        protected = {
            path: sha256_file(path)
            for path in (state.path, state.pointer, state.receipt, state.journal)
        }
        parameter_ids = {name: id(value) for name, value in model.named_parameters()}
        state_keys = tuple(model.state_dict())
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        scheduler_before = copy.deepcopy(scheduler.state_dict())
        generator_before = generator.get_state().clone()
        cursor_before = copy.deepcopy(cursor)
        python_before, numpy_before = random.getstate(), np.random.get_state()
        cpu_before = torch.get_rng_state().clone()
        cuda_before = [value.clone() for value in torch.cuda.get_rng_state_all()]
        original_forward = model.forward
        compiled = torch.compile(
            original_forward, backend="cudagraphs", dynamic=False, fullgraph=False
        )
        self.dispatcher = ForwardDispatcher(original_forward, compiled)
        model.forward = self.dispatcher  # type: ignore[method-assign]
        order = cast(Tensor, cursor["order"])
        position = int(cursor["position"])
        ids = order[position : position + 32].tolist()
        if len(ids) != 32:
            ids = order[:32].tolist()
        batch = training._to_device(self.source.batch(ids, "event"), torch.device("cuda"))
        training._validate_batch(batch, self.recipe)
        loss_config = training.CausalScaleTTCLossConfig(**self.recipe.loss_config)
        if int(cursor["epoch"]) <= 3:
            from e_jepa_ttc.training.causal_scale_eap import _foreground_only_loss_config

            loss_config = _foreground_only_loss_config(loss_config)
        from e_jepa_ttc.training.causal_scale_eap import _autocast

        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with _autocast(torch.device("cuda"), "bf16"):
                total, _ = training._producer_loss(
                    cast(training.CausalScaleTTC, model), batch, loss_config
                )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            self.counters["warmup_backward_passes"] += 1
        torch.cuda.synchronize()
        restored = original_restore(state, model, optimizer, scheduler, generator)
        optimizer.zero_grad(set_to_none=True)
        exact = {
            "parameter_identities": parameter_ids
            == {name: id(value) for name, value in model.named_parameters()},
            "state_keys": state_keys == tuple(model.state_dict()),
            "optimizer": _same(optimizer_before, optimizer.state_dict()),
            "scheduler": _same(scheduler_before, scheduler.state_dict()),
            "sampler": torch.equal(generator_before, generator.get_state()),
            "cursor": _same(cursor_before, restored),
            "python_rng": _same(python_before, random.getstate()),
            "numpy_rng": _same(numpy_before, np.random.get_state()),
            "cpu_rng": torch.equal(cpu_before, torch.get_rng_state()),
            "cuda_rng": all(
                torch.equal(a, b)
                for a, b in zip(cuda_before, torch.cuda.get_rng_state_all(), strict=True)
            ),
            "protected_files": protected == {p: sha256_file(p) for p in protected},
        }
        if not all(exact.values()):
            raise RuntimeError(f"C2F graph post-warm restore was not exact: {exact}")
        self._write_runtime(state, "READY", exact)
        prewarm = {
            "schema": "rgb_port_c2f_graph_prewarm_v1",
            "status": "PASSED",
            **self._common(),
            "completed_updates": state.completed,
            "checkpoint_sha256": sha256_file(state.path),
            "checkpoint_identity_sha256": state.identity_sha256,
            "post_warm_exact": exact,
            "counters": {**self.counters, **self.dispatcher.snapshot()},
        }
        atomic_write_json(self.prewarm_path, prewarm)
        if self.pending.exists():
            self._seal(state, "RECOVERED_FROM_PENDING")
        return restored


@contextmanager
def installed(runtime: Runtime) -> Iterator[None]:
    original_restore = training.ProducerCheckpoint.restore
    original_save = training.ProducerCheckpoint.save

    def restore(state: Any, model: Any, optimizer: Any, scheduler: Any, generator: Any) -> Any:
        cursor = original_restore(state, model, optimizer, scheduler, generator)
        return runtime.after_restore(
            state, model, optimizer, scheduler, generator, cursor, original_restore
        )

    def save(state: Any, *args: Any, **kwargs: Any) -> None:
        runtime.before_save(state)
        original_save(state, *args, **kwargs)
        runtime.after_save(state)

    cast(Any, training.ProducerCheckpoint).restore = restore
    cast(Any, training.ProducerCheckpoint).save = save
    try:
        yield
    finally:
        cast(Any, training.ProducerCheckpoint).restore = original_restore
        cast(Any, training.ProducerCheckpoint).save = original_save


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-freeze", type=Path, required=True)
    parser.add_argument("--delegate-module", choices=[DELEGATE], required=True)
    parser.add_argument("forward", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.forward or args.forward[0] != "--":
        parser.error("concurrent producer arguments must follow --")
    forwarded = args.forward[1:]
    if "--fit-id" not in forwarded or forwarded[forwarded.index("--fit-id") + 1] != FIT_ID:
        parser.error("C2F graph overlay is restricted to E_C2F_MATCHED")
    if "--device" not in forwarded or forwarded[forwarded.index("--device") + 1] != "cuda":
        parser.error("C2F graph overlay requires --device cuda")
    freeze_path = args.graph_freeze.resolve(strict=True)
    freeze = validate_c2f_graph_freeze(freeze_path)
    original_fit = training.fit_producer

    def fit(source: Any, recipe: Any, run: Path, **kwargs: Any) -> dict[str, Any]:
        runtime = Runtime(source, recipe, run, freeze_path, freeze)
        with installed(runtime):
            return original_fit(source, recipe, run, **kwargs)

    training.fit_producer = fit
    try:
        delegate = importlib.import_module(args.delegate_module)
        return int(delegate.main(forwarded))
    finally:
        training.fit_producer = original_fit


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Runtime", "installed", "main"]

